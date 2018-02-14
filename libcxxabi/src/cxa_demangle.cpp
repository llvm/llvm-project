//===-------------------------- cxa_demangle.cpp --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// FIXME: (possibly) incomplete list of features that clang mangles that this
// file does not yet support:
//   - enable_if attribute
//   - decomposition declarations
//   - C++ modules TS

#define _LIBCPP_NO_EXCEPTIONS

#include "__cxxabi_config.h"

#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>

#ifdef _MSC_VER
// snprintf is implemented in VS 2015
#if _MSC_VER < 1900
#define snprintf _snprintf_s
#endif
#endif

namespace __cxxabiv1
{

namespace
{

enum
{
    unknown_error = -4,
    invalid_args = -3,
    invalid_mangled_name,
    memory_alloc_failure,
    success
};

class StringView {
  const char *First;
  const char *Last;

public:
  template <size_t N>
  StringView(const char (&Str)[N]) : First(Str), Last(Str + N - 1) {}
  StringView(const char *First_, const char *Last_) : First(First_), Last(Last_) {}
  StringView() : First(nullptr), Last(nullptr) {}

  StringView substr(size_t From, size_t To) {
    if (To >= size())
      To = size() - 1;
    if (From >= size())
      From = size() - 1;
    return StringView(First + From, First + To);
  }

  StringView dropFront(size_t N) const {
    if (N >= size())
      N = size() - 1;
    return StringView(First + N, Last);
  }

  bool startsWith(StringView Str) const {
    if (Str.size() > size())
      return false;
    return std::equal(Str.begin(), Str.end(), begin());
  }

  const char &operator[](size_t Idx) const { return *(begin() + Idx); }

  const char *begin() const { return First; }
  const char *end() const { return Last; }
  size_t size() const { return static_cast<size_t>(Last - First); }
};

bool operator==(const StringView &LHS, const StringView &RHS) {
  return LHS.size() == RHS.size() &&
         std::equal(LHS.begin(), LHS.end(), RHS.begin());
}

// Stream that AST nodes write their string representation into after the AST
// has been parsed.
class OutputStream {
  char *Buffer;
  size_t CurrentPosition;
  size_t BufferCapacity;

  // Ensure there is at least n more positions in buffer.
  void grow(size_t N) {
    if (N + CurrentPosition >= BufferCapacity) {
      BufferCapacity *= 2;
      if (BufferCapacity < N + CurrentPosition)
        BufferCapacity = N + CurrentPosition;
      Buffer = static_cast<char *>(std::realloc(Buffer, BufferCapacity));
    }
  }

public:
  OutputStream(char *StartBuf, size_t Size)
      : Buffer(StartBuf), CurrentPosition(0), BufferCapacity(Size) {}

  OutputStream &operator+=(StringView R) {
    size_t Size = R.size();
    if (Size == 0)
      return *this;
    grow(Size);
    memmove(Buffer + CurrentPosition, R.begin(), Size);
    CurrentPosition += Size;
    return *this;
  }

  OutputStream &operator+=(char C) {
    grow(1);
    Buffer[CurrentPosition++] = C;
    return *this;
  }

  // Offset of position in buffer, used for building stream_string_view.
  typedef unsigned StreamPosition;

  // StringView into a stream, used for caching the ast nodes.
  class StreamStringView {
    StreamPosition First, Last;

    friend class OutputStream;

  public:
    StreamStringView() : First(0), Last(0) {}

    StreamStringView(StreamPosition First_, StreamPosition Last_)
        : First(First_), Last(Last_) {}

    bool empty() const { return First == Last; }
  };

  OutputStream &operator+=(StreamStringView &s) {
    size_t Sz = static_cast<size_t>(s.Last - s.First);
    if (Sz == 0)
      return *this;
    grow(Sz);
    memmove(Buffer + CurrentPosition, Buffer + s.First, Sz);
    CurrentPosition += Sz;
    return *this;
  }

  StreamPosition getCurrentPosition() const {
    return static_cast<StreamPosition>(CurrentPosition);
  }

  StreamStringView makeStringViewFromPastPosition(StreamPosition Pos) {
    return StreamStringView(Pos, getCurrentPosition());
  }

  char back() const {
    return CurrentPosition ? Buffer[CurrentPosition - 1] : '\0';
  }

  bool empty() const { return CurrentPosition == 0; }

  char *getBuffer() { return Buffer; }
  char *getBufferEnd() { return Buffer + CurrentPosition - 1; }
  size_t getBufferCapacity() { return BufferCapacity; }
};

// Base class of all AST nodes. The AST is built by the parser, then is
// traversed by the printLeft/Right functions to produce a demangled string.
class Node {
public:
  enum Kind : unsigned char {
    KDotSuffix,
    KVendorExtQualType,
    KQualType,
    KConversionOperatorType,
    KPostfixQualifiedType,
    KNameType,
    KAbiTagAttr,
    KObjCProtoName,
    KPointerType,
    KLValueReferenceType,
    KRValueReferenceType,
    KPointerToMemberType,
    KArrayType,
    KFunctionType,
    KTopLevelFunctionDecl,
    KFunctionQualType,
    KFunctionRefQualType,
    KLiteralOperator,
    KSpecialName,
    KCtorVtableSpecialName,
    KQualifiedName,
    KEmptyName,
    KVectorType,
    KTemplateParams,
    KNameWithTemplateArgs,
    KGlobalQualifiedName,
    KStdQualifiedName,
    KExpandedSpecialSubstitution,
    KSpecialSubstitution,
    KCtorDtorName,
    KDtorName,
    KUnnamedTypeName,
    KLambdaTypeName,
    KExpr,
  };

  const Kind K;

private:
  // If this Node has any RHS part, potentally many Nodes further down.
  const unsigned HasRHSComponent : 1;
  const unsigned HasFunction : 1;
  const unsigned HasArray : 1;

public:
  Node(Kind K_, bool HasRHS_ = false, bool HasFunction_ = false,
       bool HasArray_ = false)
      : K(K_), HasRHSComponent(HasRHS_), HasFunction(HasFunction_),
        HasArray(HasArray_) {}

  bool hasRHSComponent() const { return HasRHSComponent; }
  bool hasArray() const { return HasArray; }
  bool hasFunction() const { return HasFunction; }

  void print(OutputStream &s) const {
    printLeft(s);
    if (hasRHSComponent())
      printRight(s);
  }

  // Print the "left" side of this Node into OutputStream.
  virtual void printLeft(OutputStream &) const = 0;

  // Print the "right". This distinction is necessary to represent C++ types
  // that appear on the RHS of their subtype, such as arrays or functions.
  // Since most types don't have such a component, provide a default
  // implemenation.
  virtual void printRight(OutputStream &) const {}

  virtual StringView getBaseName() const { return StringView(); }

  // Silence compiler warnings, this dtor will never be called.
  virtual ~Node() = default;
};

class NodeArray {
  Node **Elements;
  size_t NumElements;

public:
  NodeArray() : NumElements(0) {}
  NodeArray(Node **Elements_, size_t NumElements_)
      : Elements(Elements_), NumElements(NumElements_) {}

  bool empty() const { return NumElements == 0; }
  size_t size() const { return NumElements; }

  void printWithSeperator(OutputStream &S, StringView Seperator) const {
    for (size_t Idx = 0; Idx != NumElements; ++Idx) {
      if (Idx)
        S += Seperator;
      Elements[Idx]->print(S);
    }
  }
};

class DotSuffix final : public Node {
  const Node *Prefix;
  const StringView Suffix;

public:
  DotSuffix(Node *Prefix_, StringView Suffix_)
      : Node(KDotSuffix), Prefix(Prefix_), Suffix(Suffix_) {}

  void printLeft(OutputStream &s) const override {
    Prefix->print(s);
    s += " (";
    s += Suffix;
    s += ")";
  }
};

class VendorExtQualType final : public Node {
  const Node *Ext;
  const Node *Ty;

public:
  VendorExtQualType(Node *Ext_, Node *Ty_)
      : Node(KVendorExtQualType), Ext(Ext_), Ty(Ty_) {}

  void printLeft(OutputStream &S) const override {
    Ext->print(S);
    S += " ";
    Ty->printLeft(S);
  }

  void printRight(OutputStream &S) const override { Ty->printRight(S); }
};

enum Qualifiers {
  QualNone = 0,
  QualConst = 0x1,
  QualVolatile = 0x2,
  QualRestrict = 0x4,
};

void addQualifiers(Qualifiers &Q1, Qualifiers Q2) {
  Q1 = static_cast<Qualifiers>(Q1 | Q2);
}

class QualType : public Node {
protected:
  const Qualifiers Quals;
  const Node *Child;

  void printQuals(OutputStream &S) const {
    if (Quals & QualConst)
      S += " const";
    if (Quals & QualVolatile)
      S += " volatile";
    if (Quals & QualRestrict)
      S += " restrict";
  }

public:
  QualType(Node *Child_, Qualifiers Quals_)
      : Node(KQualType, Child_->hasRHSComponent(), Child_->hasFunction(),
             Child_->hasArray()),
        Quals(Quals_), Child(Child_) {}

  QualType(Node::Kind ChildKind_, Node *Child_, Qualifiers Quals_)
      : Node(ChildKind_, Child_->hasRHSComponent(), Child_->hasFunction(),
             Child_->hasArray()),
        Quals(Quals_), Child(Child_) {}

  void printLeft(OutputStream &S) const override {
    Child->printLeft(S);
    printQuals(S);
  }

  void printRight(OutputStream &S) const override { Child->printRight(S); }
};

class ConversionOperatorType final : public Node {
  const Node *Ty;

public:
  ConversionOperatorType(Node *Ty_) : Node(KConversionOperatorType), Ty(Ty_) {}

  void printLeft(OutputStream &S) const override {
    S += "operator ";
    Ty->print(S);
  }
};

class PostfixQualifiedType final : public Node {
  const Node *Ty;
  const StringView Postfix;

public:
  PostfixQualifiedType(Node *Ty_, StringView Postfix_)
      : Node(KPostfixQualifiedType), Ty(Ty_), Postfix(Postfix_) {}

  void printLeft(OutputStream &s) const override {
    Ty->printLeft(s);
    s += Postfix;
  }

  void printRight(OutputStream &S) const override { Ty->printRight(S); }
};

class NameType final : public Node {
  const StringView Name;

public:
  NameType(StringView Name_) : Node(KNameType), Name(Name_) {}

  StringView getName() const { return Name; }
  StringView getBaseName() const override { return Name; }

  void printLeft(OutputStream &s) const override { s += Name; }
};

class AbiTagAttr final : public Node {
  const Node* Base;
  StringView Tag;
public:
  AbiTagAttr(const Node* Base_, StringView Tag_)
      : Node(KAbiTagAttr), Base(Base_), Tag(Tag_) {}

  void printLeft(OutputStream &S) const override {
    Base->printLeft(S);
    S += "[abi:";
    S += Tag;
    S += "]";
  }
};

class ObjCProtoName : public Node {
  Node *Ty;
  Node *Protocol;

  friend class PointerType;

public:
  ObjCProtoName(Node *Ty_, Node *Protocol_)
      : Node(KObjCProtoName), Ty(Ty_), Protocol(Protocol_) {}

  bool isObjCObject() const {
    return Ty->K == KNameType &&
           static_cast<NameType *>(Ty)->getName() == "objc_object";
  }

  void printLeft(OutputStream &S) const override {
    Ty->printLeft(S);
    S += "<";
    Protocol->printLeft(S);
    S += ">";
  }
};

class PointerType final : public Node {
  const Node *Pointee;

public:
  PointerType(Node *Pointee_)
      : Node(KPointerType, Pointee_->hasRHSComponent()), Pointee(Pointee_) {}

  void printLeft(OutputStream &s) const override {
    // We rewrite objc_object<SomeProtocol>* into id<SomeProtocol>.
    if (Pointee->K != KObjCProtoName ||
        !static_cast<const ObjCProtoName *>(Pointee)->isObjCObject()) {
      Pointee->printLeft(s);
      if (Pointee->hasArray())
        s += " ";
      if (Pointee->hasArray() || Pointee->hasFunction())
        s += "(";
      s += "*";
    } else {
      const auto *objcProto = static_cast<const ObjCProtoName *>(Pointee);
      s += "id<";
      objcProto->Protocol->print(s);
      s += ">";
    }
  }

  void printRight(OutputStream &s) const override {
    if (Pointee->K != KObjCProtoName ||
        !static_cast<const ObjCProtoName *>(Pointee)->isObjCObject()) {
      if (Pointee->hasArray() || Pointee->hasFunction())
        s += ")";
      Pointee->printRight(s);
    }
  }
};

class LValueReferenceType final : public Node {
  const Node *Pointee;

public:
  LValueReferenceType(Node *Pointee_)
      : Node(KLValueReferenceType, Pointee_->hasRHSComponent()),
        Pointee(Pointee_) {}

  void printLeft(OutputStream &s) const override {
    Pointee->printLeft(s);
    if (Pointee->hasArray())
      s += " ";
    if (Pointee->hasArray() || Pointee->hasFunction())
      s += "(&";
    else
      s += "&";
  }
  void printRight(OutputStream &s) const override {
    if (Pointee->hasArray() || Pointee->hasFunction())
      s += ")";
    Pointee->printRight(s);
  }
};

class RValueReferenceType final : public Node {
  const Node *Pointee;

public:
  RValueReferenceType(Node *Pointee_)
      : Node(KRValueReferenceType, Pointee_->hasRHSComponent()),
        Pointee(Pointee_) {}

  void printLeft(OutputStream &s) const override {
    Pointee->printLeft(s);
    if (Pointee->hasArray())
      s += " ";
    if (Pointee->hasArray() || Pointee->hasFunction())
      s += "(&&";
    else
      s += "&&";
  }

  void printRight(OutputStream &s) const override {
    if (Pointee->hasArray() || Pointee->hasFunction())
      s += ")";
    Pointee->printRight(s);
  }
};

class PointerToMemberType final : public Node {
  const Node *ClassType;
  const Node *MemberType;

public:
  PointerToMemberType(Node *ClassType_, Node *MemberType_)
      : Node(KPointerToMemberType, MemberType_->hasRHSComponent()),
        ClassType(ClassType_), MemberType(MemberType_) {}

  void printLeft(OutputStream &s) const override {
    MemberType->printLeft(s);
    if (MemberType->hasArray() || MemberType->hasFunction())
      s += "(";
    else
      s += " ";
    ClassType->print(s);
    s += "::*";
  }

  void printRight(OutputStream &s) const override {
    if (MemberType->hasArray() || MemberType->hasFunction())
      s += ")";
    MemberType->printRight(s);
  }
};

class NodeOrString {
  const void *First;
  const void *Second;

public:
  /* implicit */ NodeOrString(StringView Str) {
    const char *FirstChar = Str.begin();
    const char *SecondChar = Str.end();
    if (SecondChar == nullptr) {
      assert(FirstChar == SecondChar);
      ++FirstChar, ++SecondChar;
    }
    First = static_cast<const void *>(FirstChar);
    Second = static_cast<const void *>(SecondChar);
  }

  /* implicit */ NodeOrString(Node *N)
      : First(static_cast<const void *>(N)), Second(nullptr) {}
  NodeOrString() : First(nullptr), Second(nullptr) {}

  bool isString() const { return Second && First; }
  bool isNode() const { return First && !Second; }
  bool isEmpty() const { return !First && !Second; }

  StringView asString() const {
    assert(isString());
    return StringView(static_cast<const char *>(First),
                      static_cast<const char *>(Second));
  }

  const Node *asNode() const {
    assert(isNode());
    return static_cast<const Node *>(First);
  }
};

class ArrayType final : public Node {
  Node *Base;
  NodeOrString Dimension;

public:
  ArrayType(Node *Base_, NodeOrString Dimension_)
      : Node(KArrayType, true, false, true), Base(Base_), Dimension(Dimension_) {}

  // Incomplete array type.
  ArrayType(Node *Base_) : Node(KArrayType, true, false, true), Base(Base_) {}

  void printLeft(OutputStream &S) const override { Base->printLeft(S); }

  void printRight(OutputStream &S) const override {
    if (S.back() != ']')
      S += " ";
    S += "[";
    if (Dimension.isString())
      S += Dimension.asString();
    else if (Dimension.isNode())
      Dimension.asNode()->print(S);
    S += "]";
    Base->printRight(S);
  }
};

class FunctionType final : public Node {
  Node *Ret;
  NodeArray Params;

public:
  FunctionType(Node *Ret_, NodeArray Params_)
      : Node(KFunctionType, true, true), Ret(Ret_), Params(Params_) {}

  // Handle C++'s ... quirky decl grammer by using the left & right
  // distinction. Consider:
  //   int (*f(float))(char) {}
  // f is a function that takes a float and returns a pointer to a function
  // that takes a char and returns an int. If we're trying to print f, start
  // by printing out the return types's left, then print our parameters, then
  // finally print right of the return type.
  void printLeft(OutputStream &S) const override {
    Ret->printLeft(S);
    S += " ";
  }

  void printRight(OutputStream &S) const override {
    S += "(";
    Params.printWithSeperator(S, ", ");
    S += ")";
    Ret->printRight(S);
  }
};

class TopLevelFunctionDecl final : public Node {
  const Node *Ret;
  const Node *Name;
  NodeArray Params;

public:
  TopLevelFunctionDecl(Node *Ret_, Node *Name_, NodeArray Params_)
      : Node(KTopLevelFunctionDecl, true, true), Ret(Ret_), Name(Name_),
        Params(Params_) {}

  void printLeft(OutputStream &S) const override {
    if (Ret) {
      Ret->printLeft(S);
      if (!Ret->hasRHSComponent())
        S += " ";
    }
    Name->print(S);
  }

  void printRight(OutputStream &S) const override {
    S += "(";
    Params.printWithSeperator(S, ", ");
    S += ")";
    if (Ret)
      Ret->printRight(S);
  }
};

enum FunctionRefQual : unsigned char {
  FrefQualNone,
  FrefQualLValue,
  FrefQualRValue,
};

class FunctionRefQualType : public Node {
  Node *Fn;
  FunctionRefQual Quals;

  friend class FunctionQualType;

public:
  FunctionRefQualType(Node *Fn_, FunctionRefQual Quals_)
      : Node(KFunctionRefQualType, true, true), Fn(Fn_), Quals(Quals_) {}

  void printQuals(OutputStream &S) const {
    if (Quals == FrefQualLValue)
      S += " &";
    else
      S += " &&";
  }

  void printLeft(OutputStream &S) const override { Fn->printLeft(S); }

  void printRight(OutputStream &S) const override {
    Fn->printRight(S);
    printQuals(S);
  }
};

class FunctionQualType final : public QualType {
public:
  FunctionQualType(Node *Child_, Qualifiers Quals_)
      : QualType(KFunctionQualType, Child_, Quals_) {}

  void printLeft(OutputStream &S) const override { Child->printLeft(S); }

  void printRight(OutputStream &S) const override {
    if (Child->K == KFunctionRefQualType) {
      auto *RefQuals = static_cast<const FunctionRefQualType *>(Child);
      RefQuals->Fn->printRight(S);
      printQuals(S);
      RefQuals->printQuals(S);
    } else {
      Child->printRight(S);
      printQuals(S);
    }
  }
};

class LiteralOperator : public Node {
  const Node *OpName;

public:
  LiteralOperator(Node *OpName_) : Node(KLiteralOperator), OpName(OpName_) {}

  void printLeft(OutputStream &S) const override {
    S += "operator\"\" ";
    OpName->print(S);
  }
};

class SpecialName final : public Node {
  const StringView Special;
  const Node *Child;

public:
  SpecialName(StringView Special_, Node *Child_)
      : Node(KSpecialName), Special(Special_), Child(Child_) {}

  void printLeft(OutputStream &S) const override {
    S += Special;
    Child->print(S);
  }
};

class CtorVtableSpecialName final : public Node {
  const Node *FirstType;
  const Node *SecondType;

public:
  CtorVtableSpecialName(Node *FirstType_, Node *SecondType_)
      : Node(KCtorVtableSpecialName), FirstType(FirstType_),
        SecondType(SecondType_) {}

  void printLeft(OutputStream &S) const override {
    S += "construction vtable for ";
    FirstType->print(S);
    S += "-in-";
    SecondType->print(S);
  }
};

class QualifiedName final : public Node {
  // qualifier::name
  const Node *Qualifier;
  const Node *Name;

  mutable OutputStream::StreamStringView Cache;

public:
  QualifiedName(Node *Qualifier_, Node *Name_)
      : Node(KQualifiedName), Qualifier(Qualifier_), Name(Name_) {}

  StringView getBaseName() const override { return Name->getBaseName(); }

  void printLeft(OutputStream &S) const override {
    if (!Cache.empty()) {
      S += Cache;
      return;
    }

    OutputStream::StreamPosition Start = S.getCurrentPosition();
    if (Qualifier->K != KEmptyName) {
      Qualifier->print(S);
      S += "::";
    }
    Name->print(S);
    Cache = S.makeStringViewFromPastPosition(Start);
  }
};

class EmptyName : public Node {
public:
  EmptyName() : Node(KEmptyName) {}
  void printLeft(OutputStream &) const override {}
};

class VectorType final : public Node {
  const Node *BaseType;
  const NodeOrString Dimension;
  const bool IsPixel;

public:
  VectorType(NodeOrString Dimension_)
      : Node(KVectorType), BaseType(nullptr), Dimension(Dimension_),
        IsPixel(true) {}
  VectorType(Node *BaseType_, NodeOrString Dimension_)
      : Node(KVectorType), BaseType(BaseType_), Dimension(Dimension_),
        IsPixel(false) {}

  void printLeft(OutputStream &S) const override {
    if (IsPixel) {
      S += "pixel vector[";
      S += Dimension.asString();
      S += "]";
    } else {
      BaseType->print(S);
      S += " vector[";
      if (Dimension.isNode())
        Dimension.asNode()->print(S);
      else if (Dimension.isString())
        S += Dimension.asString();
      S += "]";
    }
  }
};

class TemplateParams final : public Node {
  NodeArray Params;

  mutable OutputStream::StreamStringView Cache;

public:
  TemplateParams(NodeArray Params_) : Node(KTemplateParams), Params(Params_) {}

  void printLeft(OutputStream &S) const override {
    if (!Cache.empty()) {
      S += Cache;
      return;
    }

    OutputStream::StreamPosition Start = S.getCurrentPosition();

    S += "<";
    Params.printWithSeperator(S, ", ");
    if (S.back() == '>')
      S += " ";
    S += ">";

    Cache = S.makeStringViewFromPastPosition(Start);
  }
};

class NameWithTemplateArgs final : public Node {
  // name<template_args>
  Node *Name;
  Node *TemplateArgs;

public:
  NameWithTemplateArgs(Node *Name_, Node *TemplateArgs_)
      : Node(KNameWithTemplateArgs), Name(Name_), TemplateArgs(TemplateArgs_) {}

  StringView getBaseName() const override { return Name->getBaseName(); }

  void printLeft(OutputStream &S) const override {
    Name->print(S);
    TemplateArgs->print(S);
  }
};

class GlobalQualifiedName final : public Node {
  Node *Child;

public:
  GlobalQualifiedName(Node *Child_) : Node(KGlobalQualifiedName), Child(Child_) {}

  StringView getBaseName() const override { return Child->getBaseName(); }

  void printLeft(OutputStream &S) const override {
    S += "::";
    Child->print(S);
  }
};

class StdQualifiedName final : public Node {
  Node *Child;

public:
  StdQualifiedName(Node *Child_) : Node(KStdQualifiedName), Child(Child_) {}

  StringView getBaseName() const override { return Child->getBaseName(); }

  void printLeft(OutputStream &S) const override {
    S += "std::";
    Child->print(S);
  }
};

enum class SpecialSubKind {
  allocator,
  basic_string,
  string,
  istream,
  ostream,
  iostream,
};

class ExpandedSpecialSubstitution final : public Node {
  SpecialSubKind SSK;

public:
  ExpandedSpecialSubstitution(SpecialSubKind SSK_)
      : Node(KExpandedSpecialSubstitution), SSK(SSK_) {}

  StringView getBaseName() const override {
    switch (SSK) {
    case SpecialSubKind::allocator:
      return StringView("allocator");
    case SpecialSubKind::basic_string:
      return StringView("basic_string");
    case SpecialSubKind::string:
      return StringView("basic_string");
    case SpecialSubKind::istream:
      return StringView("basic_istream");
    case SpecialSubKind::ostream:
      return StringView("basic_ostream");
    case SpecialSubKind::iostream:
      return StringView("basic_iostream");
    }
    _LIBCPP_UNREACHABLE();
  }

  void printLeft(OutputStream &S) const override {
    switch (SSK) {
    case SpecialSubKind::allocator:
      S += "std::basic_string<char, std::char_traits<char>, "
           "std::allocator<char> >";
      break;
    case SpecialSubKind::basic_string:
    case SpecialSubKind::string:
      S += "std::basic_string<char, std::char_traits<char>, "
           "std::allocator<char> >";
      break;
    case SpecialSubKind::istream:
      S += "std::basic_istream<char, std::char_traits<char> >";
      break;
    case SpecialSubKind::ostream:
      S += "std::basic_ostream<char, std::char_traits<char> >";
      break;
    case SpecialSubKind::iostream:
      S += "std::basic_iostream<char, std::char_traits<char> >";
      break;
    }
  }
};

class SpecialSubstitution final : public Node {
public:
  SpecialSubKind SSK;

  SpecialSubstitution(SpecialSubKind SSK_)
      : Node(KSpecialSubstitution), SSK(SSK_) {}

  StringView getBaseName() const override {
    switch (SSK) {
    case SpecialSubKind::allocator:
      return StringView("allocator");
    case SpecialSubKind::basic_string:
      return StringView("basic_string");
    case SpecialSubKind::string:
      return StringView("string");
    case SpecialSubKind::istream:
      return StringView("istream");
    case SpecialSubKind::ostream:
      return StringView("ostream");
    case SpecialSubKind::iostream:
      return StringView("iostream");
    }
    _LIBCPP_UNREACHABLE();
  }

  void printLeft(OutputStream &S) const override {
    switch (SSK) {
    case SpecialSubKind::allocator:
      S += "std::allocator";
      break;
    case SpecialSubKind::basic_string:
      S += "std::basic_string";
      break;
    case SpecialSubKind::string:
      S += "std::string";
      break;
    case SpecialSubKind::istream:
      S += "std::istream";
      break;
    case SpecialSubKind::ostream:
      S += "std::ostream";
      break;
    case SpecialSubKind::iostream:
      S += "std::iostream";
      break;
    }
  }
};

class CtorDtorName final : public Node {
  const Node *Basename;
  const bool IsDtor;

public:
  CtorDtorName(Node *Basename_, bool IsDtor_)
      : Node(KCtorDtorName), Basename(Basename_), IsDtor(IsDtor_) {}

  void printLeft(OutputStream &S) const override {
    if (IsDtor)
      S += "~";
    S += Basename->getBaseName();
  }
};

class DtorName : public Node {
  const Node *Base;

public:
  DtorName(Node *Base_) : Node(KDtorName), Base(Base_) {}

  void printLeft(OutputStream &S) const override {
    S += "~";
    Base->printLeft(S);
  }
};

class UnnamedTypeName : public Node {
  const StringView Count;

public:
  UnnamedTypeName(StringView Count_) : Node(KUnnamedTypeName), Count(Count_) {}

  void printLeft(OutputStream &S) const override {
    S += "'unnamed";
    S += Count;
    S += "\'";
  }
};

class LambdaTypeName : public Node {
  NodeArray Params;
  StringView Count;

public:
  LambdaTypeName(NodeArray Params_, StringView Count_)
      : Node(KLambdaTypeName), Params(Params_), Count(Count_) {}

  void printLeft(OutputStream &S) const override {
    S += "\'lambda";
    S += Count;
    S += "\'(";
    Params.printWithSeperator(S, ", ");
    S += ")";
  }
};

// -- Expression Nodes --

struct Expr : public Node {
  Expr() : Node(KExpr) {}
};

class BinaryExpr : public Expr {
  const Node *LHS;
  const StringView InfixOperator;
  const Node *RHS;

public:
  BinaryExpr(Node *LHS_, StringView InfixOperator_, Node *RHS_)
      : LHS(LHS_), InfixOperator(InfixOperator_), RHS(RHS_) {}

  void printLeft(OutputStream &S) const override {
    // might be a template argument expression, then we need to disambiguate
    // with parens.
    if (InfixOperator == ">")
      S += "(";

    S += "(";
    LHS->print(S);
    S += ") ";
    S += InfixOperator;
    S += " (";
    RHS->print(S);
    S += ")";

    if (InfixOperator == ">")
      S += ")";
  }
};

class ArraySubscriptExpr : public Expr {
  const Node *Op1;
  const Node *Op2;

public:
  ArraySubscriptExpr(Node *Op1_, Node *Op2_) : Op1(Op1_), Op2(Op2_) {}

  void printLeft(OutputStream &S) const override {
    S += "(";
    Op1->print(S);
    S += ")[";
    Op2->print(S);
    S += "]";
  }
};

class PostfixExpr : public Expr {
  const Node *Child;
  const StringView Operand;

public:
  PostfixExpr(Node *Child_, StringView Operand_)
      : Child(Child_), Operand(Operand_) {}

  void printLeft(OutputStream &S) const override {
    S += "(";
    Child->print(S);
    S += ")";
    S += Operand;
  }
};

class ConditionalExpr : public Expr {
  const Node *Cond;
  const Node *Then;
  const Node *Else;

public:
  ConditionalExpr(Node *Cond_, Node *Then_, Node *Else_)
      : Cond(Cond_), Then(Then_), Else(Else_) {}

  void printLeft(OutputStream &S) const override {
    S += "(";
    Cond->print(S);
    S += ") ? (";
    Then->print(S);
    S += ") : (";
    Else->print(S);
    S += ")";
  }
};

class MemberExpr : public Expr {
  const Node *LHS;
  const StringView Kind;
  const Node *RHS;

public:
  MemberExpr(Node *LHS_, StringView Kind_, Node *RHS_)
      : LHS(LHS_), Kind(Kind_), RHS(RHS_) {}

  void printLeft(OutputStream &S) const override {
    LHS->print(S);
    S += Kind;
    RHS->print(S);
  }
};

class EnclosingExpr : public Expr {
  const StringView Prefix;
  const Node *Infix;
  const StringView Postfix;

public:
  EnclosingExpr(StringView Prefix_, Node *Infix_, StringView Postfix_)
      : Prefix(Prefix_), Infix(Infix_), Postfix(Postfix_) {}

  void printLeft(OutputStream &S) const override {
    S += Prefix;
    Infix->print(S);
    S += Postfix;
  }
};

class CastExpr : public Expr {
  // cast_kind<to>(from)
  const StringView CastKind;
  const Node *To;
  const Node *From;

public:
  CastExpr(StringView CastKind_, Node *To_, Node *From_)
      : CastKind(CastKind_), To(To_), From(From_) {}

  void printLeft(OutputStream &S) const override {
    S += CastKind;
    S += "<";
    To->printLeft(S);
    S += ">(";
    From->printLeft(S);
    S += ")";
  }
};

class SizeofParamPackExpr : public Expr {
  NodeArray Args;

public:
  SizeofParamPackExpr(NodeArray Args_) : Args(Args_) {}

  void printLeft(OutputStream &S) const override {
    S += "sizeof...(";
    Args.printWithSeperator(S, ", ");
    S += ")";
  }
};

class CallExpr : public Expr {
  const Node *Callee;
  NodeArray Args;

public:
  CallExpr(Node *Callee_, NodeArray Args_) : Callee(Callee_), Args(Args_) {}

  void printLeft(OutputStream &S) const override {
    Callee->print(S);
    S += "(";
    Args.printWithSeperator(S, ", ");
    S += ")";
  }
};

class NewExpr : public Expr {
  // new (expr_list) type(init_list)
  NodeArray ExprList;
  Node *Type;
  NodeArray InitList;
  bool IsGlobal; // ::operator new ?
  bool IsArray;  // new[] ?
public:
  NewExpr(NodeArray ExprList_, Node *Type_, NodeArray InitList_, bool IsGlobal_,
          bool IsArray_)
      : ExprList(ExprList_), Type(Type_), InitList(InitList_), IsGlobal(IsGlobal_),
        IsArray(IsArray_) {}

  void printLeft(OutputStream &S) const override {
    if (IsGlobal)
      S += "::operator ";
    S += "new";
    if (IsArray)
      S += "[]";
    if (!ExprList.empty()) {
      S += "(";
      ExprList.printWithSeperator(S, ", ");
      S += ")";
    }
    Type->print(S);
    if (!InitList.empty()) {
      S += "(";
      InitList.printWithSeperator(S, ", ");
      S += ")";
    }
  }
};

class DeleteExpr : public Expr {
  Node *Op;
  bool IsGlobal;
  bool IsArray;

public:
  DeleteExpr(Node *Op_, bool IsGlobal_, bool IsArray_)
      : Op(Op_), IsGlobal(IsGlobal_), IsArray(IsArray_) {}

  void printLeft(OutputStream &S) const override {
    if (IsGlobal)
      S += "::";
    S += "delete";
    if (IsArray)
      S += "[] ";
    Op->print(S);
  }
};

class PrefixExpr : public Expr {
  StringView Prefix;
  Node *Child;

public:
  PrefixExpr(StringView Prefix_, Node *Child_) : Prefix(Prefix_), Child(Child_) {}

  void printLeft(OutputStream &S) const override {
    S += Prefix;
    S += "(";
    Child->print(S);
    S += ")";
  }
};

class FunctionParam : public Expr {
  StringView Number;

public:
  FunctionParam(StringView Number_) : Number(Number_) {}

  void printLeft(OutputStream &S) const override {
    S += "fp";
    S += Number;
  }
};

class ExprList : public Expr {
  NodeArray SubExprs;

public:
  ExprList(NodeArray SubExprs_) : SubExprs(SubExprs_) {}

  void printLeft(OutputStream &S) const override {
    S += "(";
    SubExprs.printWithSeperator(S, ", ");
    S += ")";
  }
};

class ConversionExpr : public Expr {
  NodeArray Expressions;
  NodeArray Types;

public:
  ConversionExpr(NodeArray Expressions_, NodeArray Types_)
      : Expressions(Expressions_), Types(Types_) {}

  void printLeft(OutputStream &S) const override {
    S += "(";
    Expressions.printWithSeperator(S, ", ");
    S += ")(";
    Types.printWithSeperator(S, ", ");
    S += ")";
  }
};

class ThrowExpr : public Expr {
  const Node *Op;

public:
  ThrowExpr(Node *Op_) : Op(Op_) {}

  void printLeft(OutputStream &S) const override {
    S += "throw ";
    Op->print(S);
  }
};

class BoolExpr : public Expr {
  bool Value;

public:
  BoolExpr(bool Value_) : Value(Value_) {}

  void printLeft(OutputStream &S) const override {
    S += Value ? StringView("true") : StringView("false");
  }
};

class IntegerCastExpr : public Expr {
  // ty(integer)
  Node *Ty;
  StringView Integer;

public:
  IntegerCastExpr(Node *Ty_, StringView Integer_) : Ty(Ty_), Integer(Integer_) {}

  void printLeft(OutputStream &S) const override {
    S += "(";
    Ty->print(S);
    S += ")";
    S += Integer;
  }
};

class IntegerExpr : public Expr {
  StringView Type;
  StringView Value;

public:
  IntegerExpr(StringView Type_, StringView Value_) : Type(Type_), Value(Value_) {}

  void printLeft(OutputStream &S) const override {
    if (Type.size() > 3) {
      S += "(";
      S += Type;
      S += ")";
    }

    if (Value[0] == 'n') {
      S += "-";
      S += Value.dropFront(1);
    } else
      S += Value;

    if (Type.size() <= 3)
      S += Type;
  }
};

template <class Float> struct FloatData;

template <class Float> class FloatExpr : public Expr {
  const StringView Contents;

public:
  FloatExpr(StringView Contents_) : Contents(Contents_) {}

  void printLeft(OutputStream &s) const override {
    const char *first = Contents.begin();
    const char *last = Contents.end() + 1;

    const size_t N = FloatData<Float>::mangled_size;
    if (static_cast<std::size_t>(last - first) > N) {
      last = first + N;
      union {
        Float value;
        char buf[sizeof(Float)];
      };
      const char *t = first;
      char *e = buf;
      for (; t != last; ++t, ++e) {
        unsigned d1 = isdigit(*t) ? static_cast<unsigned>(*t - '0')
                                  : static_cast<unsigned>(*t - 'a' + 10);
        ++t;
        unsigned d0 = isdigit(*t) ? static_cast<unsigned>(*t - '0')
                                  : static_cast<unsigned>(*t - 'a' + 10);
        *e = static_cast<char>((d1 << 4) + d0);
      }
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
      std::reverse(buf, e);
#endif
      char num[FloatData<Float>::max_demangled_size] = {0};
      int n = snprintf(num, sizeof(num), FloatData<Float>::spec, value);
      s += StringView(num, num + n);
    }
  }
};

class BumpPointerAllocator {
  struct BlockMeta {
    BlockMeta* Next;
    size_t Current;
  };

  static constexpr size_t AllocSize = 4096;
  static constexpr size_t UsableAllocSize = AllocSize - sizeof(BlockMeta);

  alignas(16) char InitialBuffer[AllocSize];
  BlockMeta* BlockList = nullptr;

  void grow() {
    char* NewMeta = new char[AllocSize];
    BlockList = new (NewMeta) BlockMeta{BlockList, 0};
  }

  void* allocateMassive(size_t NBytes) {
    NBytes += sizeof(BlockMeta);
    BlockMeta* NewMeta = reinterpret_cast<BlockMeta*>(new char[NBytes]);
    BlockList->Next = new (NewMeta) BlockMeta{BlockList->Next, 0};
    return static_cast<void*>(NewMeta + 1);
  }

public:
  BumpPointerAllocator()
      : BlockList(new (InitialBuffer) BlockMeta{nullptr, 0}) {}

  void* allocate(size_t N) {
    N = (N + 15u) & ~15u;
    if (N + BlockList->Current >= UsableAllocSize) {
      if (N > UsableAllocSize)
        return allocateMassive(N);
      grow();
    }
    BlockList->Current += N;
    return static_cast<void*>(reinterpret_cast<char*>(BlockList + 1) +
                              BlockList->Current - N);
  }

  ~BumpPointerAllocator() {
    while (BlockList) {
      BlockMeta* Tmp = BlockList;
      BlockList = BlockList->Next;
      if (reinterpret_cast<char*>(Tmp) != InitialBuffer)
        delete[] reinterpret_cast<char*>(Tmp);
    }
  }
};

template <class T, size_t N>
class PODSmallVector {
  static_assert(std::is_pod<T>::value,
                "T is required to be a plain old data type");

  T* First;
  T* Last;
  T* Cap;
  T Inline[N];

  bool isInline() const { return First == Inline; }

  void clearInline() {
    First = Inline;
    Last = Inline;
    Cap = Inline + N;
  }

  void reserve(size_t NewCap) {
    size_t S = size();
    if (isInline()) {
      auto* Tmp = static_cast<T*>(std::malloc(NewCap * sizeof(T)));
      std::copy(First, Last, Tmp);
      First = Tmp;
    } else
      First = static_cast<T*>(std::realloc(First, NewCap * sizeof(T)));
    Last = First + S;
    Cap = First + NewCap;
  }

public:
  PODSmallVector() : First(Inline), Last(First), Cap(Inline + N) {}

  PODSmallVector(const PODSmallVector&) = delete;
  PODSmallVector& operator=(const PODSmallVector&) = delete;

  PODSmallVector(PODSmallVector&& Other) : PODSmallVector() {
    if (Other.isInline()) {
      std::copy(Other.begin(), Other.end(), First);
      Last = First + Other.size();
      Other.clear();
      return;
    }

    First = Other.First;
    Last = Other.Last;
    Cap = Other.Cap;
    Other.clearInline();
  }

  PODSmallVector& operator=(PODSmallVector&& Other) {
    if (Other.isInline()) {
      if (!isInline()) {
        std::free(First);
        clearInline();
      }
      std::copy(Other.begin(), Other.end(), First);
      Last = First + Other.size();
      Other.clear();
      return *this;
    }

    if (isInline()) {
      First = Other.First;
      Last = Other.Last;
      Cap = Other.Cap;
      Other.clearInline();
      return *this;
    }

    std::swap(First, Other.First);
    std::swap(Last, Other.Last);
    std::swap(Cap, Other.Cap);
    Other.clear();
    return *this;
  }

  void push_back(const T& Elem) {
    if (Last == Cap)
      reserve(size() * 2);
    *Last++ = Elem;
  }

  void pop_back() {
    assert(Last != First && "Popping empty vector!");
    --Last;
  }

  void dropBack(size_t Index) {
    assert(Index <= size() && "dropBack() can't expand!");
    Last = First + Index;
  }

  T* begin() { return First; }
  T* end() { return Last; }

  bool empty() const { return First == Last; }
  size_t size() const { return static_cast<size_t>(Last - First); }
  T& back() {
    assert(Last != First && "Calling back() on empty vector!");
    return *(Last - 1);
  }
  T& operator[](size_t Index) {
    assert(Index < size() && "Invalid access!");
    return *(begin() + Index);
  }
  void clear() { Last = First; }

  ~PODSmallVector() {
    if (!isInline())
      std::free(First);
  }
};

// Substitution table. This type is used to track the substitutions that are
// known by the parser.
template <size_t Size>
class SubstitutionTable {
  // Substitutions hold the actual entries in the table, and PackIndices tells
  // us which entries are members of which pack. For example, if the
  // substitutions we're tracking are: {int, {float, FooBar}, char}, with
  // {float, FooBar} being a parameter pack, we represent the substitutions as:
  // Substitutions: int, float, FooBar, char
  // PackIndices:     0,             1,    3
  // So, PackIndicies[I] holds the offset of the begin of the Ith pack, and
  // PackIndices[I + 1] holds the offset of the end.
  PODSmallVector<Node*, Size> Substitutions;
  PODSmallVector<unsigned, Size> PackIndices;

public:
  // Add a substitution that represents a single name to the table. This is
  // modeled as a parameter pack with just one element.
  void pushSubstitution(Node* Entry) {
    pushPack();
    pushSubstitutionIntoPack(Entry);
  }

  // Add a new empty pack to the table. Subsequent calls to
  // pushSubstitutionIntoPack() will add to this pack.
  void pushPack() {
    PackIndices.push_back(static_cast<unsigned>(Substitutions.size()));
  }
  void pushSubstitutionIntoPack(Node* Entry) {
    assert(!PackIndices.empty() && "No pack to push substitution into!");
    Substitutions.push_back(Entry);
  }

  // Remove the last pack from the table.
  void popPack() {
    unsigned Last = PackIndices.back();
    PackIndices.pop_back();
    Substitutions.dropBack(Last);
  }

  // For use in a range-for loop.
  struct NodeRange {
    Node** First;
    Node** Last;
    Node** begin() { return First; }
    Node** end() { return Last; }
  };

  // Retrieve the Nth substitution. This is represented as a range, as the
  // substitution could be referring to a parameter pack.
  NodeRange nthSubstitution(size_t N) {
    assert(PackIndices[N] <= Substitutions.size());
    // The Nth parameter pack starts at offset PackIndices[N], and ends at
    // PackIndices[N + 1].
    Node** Begin = Substitutions.begin() + PackIndices[N];
    Node** End;
    if (N + 1 != PackIndices.size()) {
      assert(PackIndices[N + 1] <= Substitutions.size());
      End = Substitutions.begin() + PackIndices[N + 1];
    } else
      End = Substitutions.end();
    assert(Begin <= End);
    return NodeRange{Begin, End};
  }

  size_t size() const { return PackIndices.size(); }
  bool empty() const { return PackIndices.empty(); }
  void clear() {
    Substitutions.clear();
    PackIndices.clear();
  }
};

struct Db
{
    // Name stack, this is used by the parser to hold temporary names that were
    // parsed. The parser colapses multiple names into new nodes to construct
    // the AST. Once the parser is finished, names.size() == 1.
    PODSmallVector<Node*, 32> Names;

    // Substitution table. Itanium supports name substitutions as a means of
    // compression. The string "S42_" refers to the 42nd entry in this table.
    SubstitutionTable<32> Subs;

    // Template parameter table. Like the above, but referenced like "T42_".
    // This has a smaller size compared to Subs and Names because it can be
    // stored on the stack.
    SubstitutionTable<4> TemplateParams;

    Qualifiers CV = QualNone;
    FunctionRefQual RefQuals = FrefQualNone;
    unsigned EncodingDepth = 0;
    bool ParsedCtorDtorCV = false;
    bool TagTemplates = true;
    bool FixForwardReferences = false;
    bool TryToParseTemplateArgs = true;

    BumpPointerAllocator ASTAllocator;

    template <class T, class... Args> T* make(Args&& ...args)
    {
        return new (ASTAllocator.allocate(sizeof(T)))
            T(std::forward<Args>(args)...);
    }

    template <class It> NodeArray makeNodeArray(It begin, It end)
    {
        size_t sz = static_cast<size_t>(end - begin);
        void* mem = ASTAllocator.allocate(sizeof(Node*) * sz);
        Node** data = new (mem) Node*[sz];
        std::copy(begin, end, data);
        return NodeArray(data, sz);
    }

    NodeArray popTrailingNodeArray(size_t FromPosition)
    {
        assert(FromPosition <= Names.size());
        NodeArray res = makeNodeArray(
            Names.begin() + (long)FromPosition, Names.end());
        Names.dropBack(FromPosition);
        return res;
    }
};

const char* parse_type(const char* first, const char* last, Db& db);
const char* parse_encoding(const char* first, const char* last, Db& db);
const char* parse_name(const char* first, const char* last, Db& db,
                       bool* ends_with_template_args = 0);
const char* parse_expression(const char* first, const char* last, Db& db);
const char* parse_template_args(const char* first, const char* last, Db& db);
const char* parse_operator_name(const char* first, const char* last, Db& db);
const char* parse_unqualified_name(const char* first, const char* last, Db& db);
const char* parse_decltype(const char* first, const char* last, Db& db);

// <number> ::= [n] <non-negative decimal integer>

const char*
parse_number(const char* first, const char* last)
{
    if (first != last)
    {
        const char* t = first;
        if (*t == 'n')
            ++t;
        if (t != last)
        {
            if (*t == '0')
            {
                first = t+1;
            }
            else if ('1' <= *t && *t <= '9')
            {
                first = t+1;
                while (first != last && std::isdigit(*first))
                    ++first;
            }
        }
    }
    return first;
}

template <class Float>
struct FloatData;

template <>
struct FloatData<float>
{
    static const size_t mangled_size = 8;
    static const size_t max_demangled_size = 24;
    static constexpr const char* spec = "%af";
};

constexpr const char* FloatData<float>::spec;

template <>
struct FloatData<double>
{
    static const size_t mangled_size = 16;
    static const size_t max_demangled_size = 32;
    static constexpr const char* spec = "%a";
};

constexpr const char* FloatData<double>::spec;

template <>
struct FloatData<long double>
{
#if defined(__mips__) && defined(__mips_n64) || defined(__aarch64__) || \
    defined(__wasm__)
    static const size_t mangled_size = 32;
#elif defined(__arm__) || defined(__mips__) || defined(__hexagon__)
    static const size_t mangled_size = 16;
#else
    static const size_t mangled_size = 20;  // May need to be adjusted to 16 or 24 on other platforms
#endif
    static const size_t max_demangled_size = 40;
    static constexpr const char* spec = "%LaL";
};

constexpr const char* FloatData<long double>::spec;

template <class Float>
const char*
parse_floating_number(const char* first, const char* last, Db& db)
{
    const size_t N = FloatData<Float>::mangled_size;
    if (static_cast<std::size_t>(last - first) <= N)
        return first;
    last = first + N;
    const char* t = first;
    for (; t != last; ++t)
    {
        if (!isxdigit(*t))
            return first;
    }
    if (*t == 'E')
    {
        db.Names.push_back(
            db.make<FloatExpr<Float>>(StringView(first, t)));
        first = t + 1;
    }
    return first;
}

// <positive length number> ::= [0-9]*
const char*
parse_positive_integer(const char* first, const char* last, size_t* out)
{
    if (first != last)
    {
        char c = *first;
        if (isdigit(c) && first+1 != last)
        {
            const char* t = first+1;
            size_t n = static_cast<size_t>(c - '0');
            for (c = *t; isdigit(c); c = *t)
            {
                n = n * 10 + static_cast<size_t>(c - '0');
                if (++t == last)
                    return first;
            }
            *out = n;
            first = t;
        }
    }
    return first;
}

// extension
// <abi-tag-seq> ::= <abi-tag>*
// <abi-tag>     ::= B <positive length number> <identifier>
const char*
parse_abi_tag_seq(const char* first, const char* last, Db& db)
{
    while (first != last && *first == 'B' && first+1 != last)
    {
        size_t length;
        const char* t = parse_positive_integer(first+1, last, &length);
        if (t == first+1)
            return first;
        if (static_cast<size_t>(last - t) < length || db.Names.empty())
            return first;
        db.Names.back() = db.make<AbiTagAttr>(
            db.Names.back(), StringView(t, t + length));
        first = t + length;
    }
    return first;
}

// <source-name> ::= <positive length number> <identifier> [<abi-tag-seq>]
const char*
parse_source_name(const char* first, const char* last, Db& db)
{
    if (first != last)
    {
        size_t length;
        const char* t = parse_positive_integer(first, last, &length);
        if (t == first)
            return first;
        if (static_cast<size_t>(last - t) >= length)
        {
            StringView r(t, t + length);
            if (r.substr(0, 10) == "_GLOBAL__N")
                db.Names.push_back(db.make<NameType>("(anonymous namespace)"));
            else
                db.Names.push_back(db.make<NameType>(r));
            first = t + length;
            first = parse_abi_tag_seq(first, last, db);
        }
    }
    return first;
}

// <substitution> ::= S <seq-id> _
//                ::= S_
// <substitution> ::= Sa # ::std::allocator
// <substitution> ::= Sb # ::std::basic_string
// <substitution> ::= Ss # ::std::basic_string < char,
//                                               ::std::char_traits<char>,
//                                               ::std::allocator<char> >
// <substitution> ::= Si # ::std::basic_istream<char,  std::char_traits<char> >
// <substitution> ::= So # ::std::basic_ostream<char,  std::char_traits<char> >
// <substitution> ::= Sd # ::std::basic_iostream<char, std::char_traits<char> >

const char*
parse_substitution(const char* first, const char* last, Db& db)
{
    if (last - first >= 2)
    {
        if (*first == 'S')
        {
            switch (first[1])
            {
            case 'a':
                db.Names.push_back(
                    db.make<SpecialSubstitution>(
                        SpecialSubKind::allocator));
                first += 2;
                break;
            case 'b':
                db.Names.push_back(
                    db.make<SpecialSubstitution>(SpecialSubKind::basic_string));
                first += 2;
                break;
            case 's':
                db.Names.push_back(
                    db.make<SpecialSubstitution>(
                        SpecialSubKind::string));
                first += 2;
                break;
            case 'i':
                db.Names.push_back(db.make<SpecialSubstitution>(SpecialSubKind::istream));
                first += 2;
                break;
            case 'o':
                db.Names.push_back(db.make<SpecialSubstitution>(SpecialSubKind::ostream));
                first += 2;
                break;
            case 'd':
                db.Names.push_back(db.make<SpecialSubstitution>(SpecialSubKind::iostream));
                first += 2;
                break;
            case '_':
                if (!db.Subs.empty())
                {
                    for (Node* n : db.Subs.nthSubstitution(0))
                        db.Names.push_back(n);
                    first += 2;
                }
                break;
            default:
                if (std::isdigit(first[1]) || std::isupper(first[1]))
                {
                    size_t sub = 0;
                    const char* t = first+1;
                    if (std::isdigit(*t))
                        sub = static_cast<size_t>(*t - '0');
                    else
                        sub = static_cast<size_t>(*t - 'A') + 10;
                    for (++t; t != last && (std::isdigit(*t) || std::isupper(*t)); ++t)
                    {
                        sub *= 36;
                        if (std::isdigit(*t))
                            sub += static_cast<size_t>(*t - '0');
                        else
                            sub += static_cast<size_t>(*t - 'A') + 10;
                    }
                    if (t == last || *t != '_')
                        return first;
                    ++sub;
                    if (sub < db.Subs.size())
                    {
                        for (Node* n : db.Subs.nthSubstitution(sub))
                            db.Names.push_back(n);
                        first = t+1;
                    }
                }
                break;
            }
        }
    }
    return first;
}

// <builtin-type> ::= v    # void
//                ::= w    # wchar_t
//                ::= b    # bool
//                ::= c    # char
//                ::= a    # signed char
//                ::= h    # unsigned char
//                ::= s    # short
//                ::= t    # unsigned short
//                ::= i    # int
//                ::= j    # unsigned int
//                ::= l    # long
//                ::= m    # unsigned long
//                ::= x    # long long, __int64
//                ::= y    # unsigned long long, __int64
//                ::= n    # __int128
//                ::= o    # unsigned __int128
//                ::= f    # float
//                ::= d    # double
//                ::= e    # long double, __float80
//                ::= g    # __float128
//                ::= z    # ellipsis
//                ::= Dd   # IEEE 754r decimal floating point (64 bits)
//                ::= De   # IEEE 754r decimal floating point (128 bits)
//                ::= Df   # IEEE 754r decimal floating point (32 bits)
//                ::= Dh   # IEEE 754r half-precision floating point (16 bits)
//                ::= Di   # char32_t
//                ::= Ds   # char16_t
//                ::= Da   # auto (in dependent new-expressions)
//                ::= Dc   # decltype(auto)
//                ::= Dn   # std::nullptr_t (i.e., decltype(nullptr))
//                ::= u <source-name>    # vendor extended type

const char*
parse_builtin_type(const char* first, const char* last, Db& db)
{
    if (first != last)
    {
        switch (*first)
        {
        case 'v':
            db.Names.push_back(db.make<NameType>("void"));
            ++first;
            break;
        case 'w':
            db.Names.push_back(db.make<NameType>("wchar_t"));
            ++first;
            break;
        case 'b':
            db.Names.push_back(db.make<NameType>("bool"));
            ++first;
            break;
        case 'c':
            db.Names.push_back(db.make<NameType>("char"));
            ++first;
            break;
        case 'a':
            db.Names.push_back(db.make<NameType>("signed char"));
            ++first;
            break;
        case 'h':
            db.Names.push_back(db.make<NameType>("unsigned char"));
            ++first;
            break;
        case 's':
            db.Names.push_back(db.make<NameType>("short"));
            ++first;
            break;
        case 't':
            db.Names.push_back(db.make<NameType>("unsigned short"));
            ++first;
            break;
        case 'i':
            db.Names.push_back(db.make<NameType>("int"));
            ++first;
            break;
        case 'j':
            db.Names.push_back(db.make<NameType>("unsigned int"));
            ++first;
            break;
        case 'l':
            db.Names.push_back(db.make<NameType>("long"));
            ++first;
            break;
        case 'm':
            db.Names.push_back(db.make<NameType>("unsigned long"));
            ++first;
            break;
        case 'x':
            db.Names.push_back(db.make<NameType>("long long"));
            ++first;
            break;
        case 'y':
            db.Names.push_back(db.make<NameType>("unsigned long long"));
            ++first;
            break;
        case 'n':
            db.Names.push_back(db.make<NameType>("__int128"));
            ++first;
            break;
        case 'o':
            db.Names.push_back(db.make<NameType>("unsigned __int128"));
            ++first;
            break;
        case 'f':
            db.Names.push_back(db.make<NameType>("float"));
            ++first;
            break;
        case 'd':
            db.Names.push_back(db.make<NameType>("double"));
            ++first;
            break;
        case 'e':
            db.Names.push_back(db.make<NameType>("long double"));
            ++first;
            break;
        case 'g':
            db.Names.push_back(db.make<NameType>("__float128"));
            ++first;
            break;
        case 'z':
            db.Names.push_back(db.make<NameType>("..."));
            ++first;
            break;
        case 'u':
            {
                const char*t = parse_source_name(first+1, last, db);
                if (t != first+1)
                    first = t;
            }
            break;
        case 'D':
            if (first+1 != last)
            {
                switch (first[1])
                {
                case 'd':
                    db.Names.push_back(db.make<NameType>("decimal64"));
                    first += 2;
                    break;
                case 'e':
                    db.Names.push_back(db.make<NameType>("decimal128"));
                    first += 2;
                    break;
                case 'f':
                    db.Names.push_back(db.make<NameType>("decimal32"));
                    first += 2;
                    break;
                case 'h':
                    db.Names.push_back(db.make<NameType>("decimal16"));
                    first += 2;
                    break;
                case 'i':
                    db.Names.push_back(db.make<NameType>("char32_t"));
                    first += 2;
                    break;
                case 's':
                    db.Names.push_back(db.make<NameType>("char16_t"));
                    first += 2;
                    break;
                case 'a':
                    db.Names.push_back(db.make<NameType>("auto"));
                    first += 2;
                    break;
                case 'c':
                    db.Names.push_back(db.make<NameType>("decltype(auto)"));
                    first += 2;
                    break;
                case 'n':
                    db.Names.push_back(db.make<NameType>("std::nullptr_t"));
                    first += 2;
                    break;
                }
            }
            break;
        }
    }
    return first;
}

// <CV-Qualifiers> ::= [r] [V] [K]

const char*
parse_cv_qualifiers(const char* first, const char* last, Qualifiers& cv)
{
    cv = QualNone;
    if (first != last)
    {
        if (*first == 'r')
        {
            addQualifiers(cv, QualRestrict);
            ++first;
        }
        if (*first == 'V')
        {
            addQualifiers(cv, QualVolatile);
            ++first;
        }
        if (*first == 'K')
        {
            addQualifiers(cv, QualConst);
            ++first;
        }
    }
    return first;
}

// <template-param> ::= T_    # first template parameter
//                  ::= T <parameter-2 non-negative number> _

const char*
parse_template_param(const char* first, const char* last, Db& db)
{
    if (last - first >= 2)
    {
        if (*first == 'T')
        {
            if (first[1] == '_')
            {
                if (!db.TemplateParams.empty())
                {
                    for (Node *t : db.TemplateParams.nthSubstitution(0))
                        db.Names.push_back(t);
                    first += 2;
                }
                else
                {
                    db.Names.push_back(db.make<NameType>("T_"));
                    first += 2;
                    db.FixForwardReferences = true;
                }
            }
            else if (isdigit(first[1]))
            {
                const char* t = first+1;
                size_t sub = static_cast<size_t>(*t - '0');
                for (++t; t != last && isdigit(*t); ++t)
                {
                    sub *= 10;
                    sub += static_cast<size_t>(*t - '0');
                }
                if (t == last || *t != '_')
                    return first;
                ++sub;
                if (sub < db.TemplateParams.size())
                {
                    for (Node *temp : db.TemplateParams.nthSubstitution(sub))
                        db.Names.push_back(temp);
                    first = t+1;
                }
                else
                {
                    db.Names.push_back(
                        db.make<NameType>(StringView(first, t + 1)));
                    first = t+1;
                    db.FixForwardReferences = true;
                }
            }
        }
    }
    return first;
}

// cc <type> <expression>                               # const_cast<type> (expression)

const char*
parse_const_cast_expr(const char* first, const char* last, Db& db)
{
    if (last - first >= 3 && first[0] == 'c' && first[1] == 'c')
    {
        const char* t = parse_type(first+2, last, db);
        if (t != first+2)
        {
            const char* t1 = parse_expression(t, last, db);
            if (t1 != t)
            {
                if (db.Names.size() < 2)
                    return first;
                auto from_expr = db.Names.back();
                db.Names.pop_back();
                if (db.Names.empty())
                    return first;
                db.Names.back() = db.make<CastExpr>(
                    "const_cast", db.Names.back(), from_expr);
                first = t1;
            }
        }
    }
    return first;
}

// dc <type> <expression>                               # dynamic_cast<type> (expression)

const char*
parse_dynamic_cast_expr(const char* first, const char* last, Db& db)
{
    if (last - first >= 3 && first[0] == 'd' && first[1] == 'c')
    {
        const char* t = parse_type(first+2, last, db);
        if (t != first+2)
        {
            const char* t1 = parse_expression(t, last, db);
            if (t1 != t)
            {
                if (db.Names.size() < 2)
                    return first;
                auto from_expr = db.Names.back();
                db.Names.pop_back();
                if (db.Names.empty())
                    return first;
                db.Names.back() = db.make<CastExpr>(
                    "dynamic_cast", db.Names.back(), from_expr);
                first = t1;
            }
        }
    }
    return first;
}

// rc <type> <expression>                               # reinterpret_cast<type> (expression)

const char*
parse_reinterpret_cast_expr(const char* first, const char* last, Db& db)
{
    if (last - first >= 3 && first[0] == 'r' && first[1] == 'c')
    {
        const char* t = parse_type(first+2, last, db);
        if (t != first+2)
        {
            const char* t1 = parse_expression(t, last, db);
            if (t1 != t)
            {
                if (db.Names.size() < 2)
                    return first;
                auto from_expr = db.Names.back();
                db.Names.pop_back();
                if (db.Names.empty())
                    return first;
                db.Names.back() = db.make<CastExpr>(
                    "reinterpret_cast", db.Names.back(), from_expr);
                first = t1;
            }
        }
    }
    return first;
}

// sc <type> <expression>                               # static_cast<type> (expression)

const char*
parse_static_cast_expr(const char* first, const char* last, Db& db)
{
    if (last - first >= 3 && first[0] == 's' && first[1] == 'c')
    {
        const char* t = parse_type(first+2, last, db);
        if (t != first+2)
        {
            const char* t1 = parse_expression(t, last, db);
            if (t1 != t)
            {
                if (db.Names.size() < 2)
                    return first;
                auto from_expr = db.Names.back();
                db.Names.pop_back();
                db.Names.back() = db.make<CastExpr>(
                    "static_cast", db.Names.back(), from_expr);
                first = t1;
            }
        }
    }
    return first;
}

// sp <expression>                                  # pack expansion

const char*
parse_pack_expansion(const char* first, const char* last, Db& db)
{
    if (last - first >= 3 && first[0] == 's' && first[1] == 'p')
    {
        const char* t = parse_expression(first+2, last, db);
        if (t != first+2)
            first = t;
    }
    return first;
}

// st <type>                                            # sizeof (a type)

const char*
parse_sizeof_type_expr(const char* first, const char* last, Db& db)
{
    if (last - first >= 3 && first[0] == 's' && first[1] == 't')
    {
        const char* t = parse_type(first+2, last, db);
        if (t != first+2)
        {
            if (db.Names.empty())
                return first;
            db.Names.back() = db.make<EnclosingExpr>(
                "sizeof (", db.Names.back(), ")");
            first = t;
        }
    }
    return first;
}

// sz <expr>                                            # sizeof (a expression)

const char*
parse_sizeof_expr_expr(const char* first, const char* last, Db& db)
{
    if (last - first >= 3 && first[0] == 's' && first[1] == 'z')
    {
        const char* t = parse_expression(first+2, last, db);
        if (t != first+2)
        {
            if (db.Names.empty())
                return first;
            db.Names.back() = db.make<EnclosingExpr>(
                "sizeof (", db.Names.back(), ")");
            first = t;
        }
    }
    return first;
}

// sZ <template-param>                                  # size of a parameter pack

const char*
parse_sizeof_param_pack_expr(const char* first, const char* last, Db& db)
{
    if (last - first >= 3 && first[0] == 's' && first[1] == 'Z' && first[2] == 'T')
    {
        size_t k0 = db.Names.size();
        const char* t = parse_template_param(first+2, last, db);
        size_t k1 = db.Names.size();
        if (t != first+2 && k0 <= k1)
        {
            Node* sizeof_expr = db.make<SizeofParamPackExpr>(
                db.popTrailingNodeArray(k0));
            db.Names.push_back(sizeof_expr);
            first = t;
        }
    }
    return first;
}

// <function-param> ::= fp <top-level CV-Qualifiers> _                                     # L == 0, first parameter
//                  ::= fp <top-level CV-Qualifiers> <parameter-2 non-negative number> _   # L == 0, second and later parameters
//                  ::= fL <L-1 non-negative number> p <top-level CV-Qualifiers> _         # L > 0, first parameter
//                  ::= fL <L-1 non-negative number> p <top-level CV-Qualifiers> <parameter-2 non-negative number> _   # L > 0, second and later parameters

const char*
parse_function_param(const char* first, const char* last, Db& db)
{
    if (last - first >= 3 && *first == 'f')
    {
        if (first[1] == 'p')
        {
            Qualifiers cv;
            const char* t = parse_cv_qualifiers(first+2, last, cv);
            const char* t1 = parse_number(t, last);
            if (t1 != last && *t1 == '_')
            {
                db.Names.push_back(
                    db.make<FunctionParam>(StringView(t, t1)));
                first = t1+1;
            }
        }
        else if (first[1] == 'L')
        {
            Qualifiers cv;
            const char* t0 = parse_number(first+2, last);
            if (t0 != last && *t0 == 'p')
            {
                ++t0;
                const char* t = parse_cv_qualifiers(t0, last, cv);
                const char* t1 = parse_number(t, last);
                if (t1 != last && *t1 == '_')
                {
                    db.Names.push_back(
                        db.make<FunctionParam>(StringView(t, t1)));
                    first = t1+1;
                }
            }
        }
    }
    return first;
}

// sZ <function-param>                                  # size of a function parameter pack

const char*
parse_sizeof_function_param_pack_expr(const char* first, const char* last, Db& db)
{
    if (last - first >= 3 && first[0] == 's' && first[1] == 'Z' && first[2] == 'f')
    {
        const char* t = parse_function_param(first+2, last, db);
        if (t != first+2)
        {
            if (db.Names.empty())
                return first;
            db.Names.back() = db.make<EnclosingExpr>(
                "sizeof...(", db.Names.back(), ")");
            first = t;
        }
    }
    return first;
}

// te <expression>                                      # typeid (expression)
// ti <type>                                            # typeid (type)

const char*
parse_typeid_expr(const char* first, const char* last, Db& db)
{
    if (last - first >= 3 && first[0] == 't' && (first[1] == 'e' || first[1] == 'i'))
    {
        const char* t;
        if (first[1] == 'e')
            t = parse_expression(first+2, last, db);
        else
            t = parse_type(first+2, last, db);
        if (t != first+2)
        {
            if (db.Names.empty())
                return first;
            db.Names.back() = db.make<EnclosingExpr>(
                "typeid(", db.Names.back(), ")");
            first = t;
        }
    }
    return first;
}

// tw <expression>                                      # throw expression

const char*
parse_throw_expr(const char* first, const char* last, Db& db)
{
    if (last - first >= 3 && first[0] == 't' && first[1] == 'w')
    {
        const char* t = parse_expression(first+2, last, db);
        if (t != first+2)
        {
            if (db.Names.empty())
                return first;
            db.Names.back() = db.make<ThrowExpr>(db.Names.back());
            first = t;
        }
    }
    return first;
}

// ds <expression> <expression>                         # expr.*expr

const char*
parse_dot_star_expr(const char* first, const char* last, Db& db)
{
    if (last - first >= 3 && first[0] == 'd' && first[1] == 's')
    {
        const char* t = parse_expression(first+2, last, db);
        if (t != first+2)
        {
            const char* t1 = parse_expression(t, last, db);
            if (t1 != t)
            {
                if (db.Names.size() < 2)
                    return first;
                auto rhs_expr = db.Names.back();
                db.Names.pop_back();
                db.Names.back() = db.make<MemberExpr>(
                    db.Names.back(), ".*", rhs_expr);
                first = t1;
            }
        }
    }
    return first;
}

// <simple-id> ::= <source-name> [ <template-args> ]

const char*
parse_simple_id(const char* first, const char* last, Db& db)
{
    if (first != last)
    {
        const char* t = parse_source_name(first, last, db);
        if (t != first)
        {
            const char* t1 = parse_template_args(t, last, db);
            if (t1 != t)
            {
                if (db.Names.size() < 2)
                    return first;
                auto args = db.Names.back();
                db.Names.pop_back();
                db.Names.back() =
                    db.make<NameWithTemplateArgs>(db.Names.back(), args);
            }
            first = t1;
        }
        else
            first = t;
    }
    return first;
}

// <unresolved-type> ::= <template-param>
//                   ::= <decltype>
//                   ::= <substitution>

const char*
parse_unresolved_type(const char* first, const char* last, Db& db)
{
    if (first != last)
    {
        const char* t = first;
        switch (*first)
        {
        case 'T':
          {
            size_t k0 = db.Names.size();
            t = parse_template_param(first, last, db);
            size_t k1 = db.Names.size();
            if (t != first && k1 == k0 + 1)
            {
                db.Subs.pushSubstitution(db.Names.back());
                first = t;
            }
            else
            {
                for (; k1 != k0; --k1)
                    db.Names.pop_back();
            }
            break;
          }
        case 'D':
            t = parse_decltype(first, last, db);
            if (t != first)
            {
                if (db.Names.empty())
                    return first;
                db.Subs.pushSubstitution(db.Names.back());
                first = t;
            }
            break;
        case 'S':
            t = parse_substitution(first, last, db);
            if (t != first)
                first = t;
            else
            {
                if (last - first > 2 && first[1] == 't')
                {
                    t = parse_unqualified_name(first+2, last, db);
                    if (t != first+2)
                    {
                        if (db.Names.empty())
                            return first;
                        db.Names.back() =
                            db.make<StdQualifiedName>(db.Names.back());
                        db.Subs.pushSubstitution(db.Names.back());
                        first = t;
                    }
                }
            }
            break;
       }
    }
    return first;
}

// <destructor-name> ::= <unresolved-type>                               # e.g., ~T or ~decltype(f())
//                   ::= <simple-id>                                     # e.g., ~A<2*N>

const char*
parse_destructor_name(const char* first, const char* last, Db& db)
{
    if (first != last)
    {
        const char* t = parse_unresolved_type(first, last, db);
        if (t == first)
            t = parse_simple_id(first, last, db);
        if (t != first)
        {
            if (db.Names.empty())
                return first;
            db.Names.back() = db.make<DtorName>(db.Names.back());
            first = t;
        }
    }
    return first;
}

// <base-unresolved-name> ::= <simple-id>                                # unresolved name
//          extension     ::= <operator-name>                            # unresolved operator-function-id
//          extension     ::= <operator-name> <template-args>            # unresolved operator template-id
//                        ::= on <operator-name>                         # unresolved operator-function-id
//                        ::= on <operator-name> <template-args>         # unresolved operator template-id
//                        ::= dn <destructor-name>                       # destructor or pseudo-destructor;
//                                                                         # e.g. ~X or ~X<N-1>

const char*
parse_base_unresolved_name(const char* first, const char* last, Db& db)
{
    if (last - first >= 2)
    {
        if ((first[0] == 'o' || first[0] == 'd') && first[1] == 'n')
        {
            if (first[0] == 'o')
            {
                const char* t = parse_operator_name(first+2, last, db);
                if (t != first+2)
                {
                    first = parse_template_args(t, last, db);
                    if (first != t)
                    {
                        if (db.Names.size() < 2)
                            return first;
                        auto args = db.Names.back();
                        db.Names.pop_back();
                        db.Names.back() =
                            db.make<NameWithTemplateArgs>(
                                db.Names.back(), args);
                    }
                }
            }
            else
            {
                const char* t = parse_destructor_name(first+2, last, db);
                if (t != first+2)
                    first = t;
            }
        }
        else
        {
            const char* t = parse_simple_id(first, last, db);
            if (t == first)
            {
                t = parse_operator_name(first, last, db);
                if (t != first)
                {
                    first = parse_template_args(t, last, db);
                    if (first != t)
                    {
                        if (db.Names.size() < 2)
                            return first;
                        auto args = db.Names.back();
                        db.Names.pop_back();
                        db.Names.back() =
                            db.make<NameWithTemplateArgs>(
                                db.Names.back(), args);
                    }
                }
            }
            else
                first = t;
        }
    }
    return first;
}

// <unresolved-qualifier-level> ::= <simple-id>

const char*
parse_unresolved_qualifier_level(const char* first, const char* last, Db& db)
{
    return parse_simple_id(first, last, db);
}

// <unresolved-name>
//  extension        ::= srN <unresolved-type> [<template-args>] <unresolved-qualifier-level>* E <base-unresolved-name>
//                   ::= [gs] <base-unresolved-name>                     # x or (with "gs") ::x
//                   ::= [gs] sr <unresolved-qualifier-level>+ E <base-unresolved-name>  
//                                                                       # A::x, N::y, A<T>::z; "gs" means leading "::"
//                   ::= sr <unresolved-type> <base-unresolved-name>     # T::x / decltype(p)::x
//  extension        ::= sr <unresolved-type> <template-args> <base-unresolved-name>
//                                                                       # T::N::x /decltype(p)::N::x
//  (ignored)        ::= srN <unresolved-type>  <unresolved-qualifier-level>+ E <base-unresolved-name>

const char*
parse_unresolved_name(const char* first, const char* last, Db& db)
{
    if (last - first > 2)
    {
        const char* t = first;
        bool global = false;
        if (t[0] == 'g' && t[1] == 's')
        {
            global = true;
            t += 2;
        }
        const char* t2 = parse_base_unresolved_name(t, last, db);
        if (t2 != t)
        {
            if (global)
            {
                if (db.Names.empty())
                    return first;
                db.Names.back() =
                    db.make<GlobalQualifiedName>(db.Names.back());
            }
            first = t2;
        }
        else if (last - t > 2 && t[0] == 's' && t[1] == 'r')
        {
            if (t[2] == 'N')
            {
                t += 3;
                const char* t1 = parse_unresolved_type(t, last, db);
                if (t1 == t || t1 == last)
                    return first;
                t = t1;
                t1 = parse_template_args(t, last, db);
                if (t1 != t)
                {
                    if (db.Names.size() < 2)
                        return first;
                    auto args = db.Names.back();
                    db.Names.pop_back();
                    db.Names.back() = db.make<NameWithTemplateArgs>(
                        db.Names.back(), args);
                    t = t1;
                    if (t == last)
                    {
                        db.Names.pop_back();
                        return first;
                    }
                }
                while (*t != 'E')
                {
                    t1 = parse_unresolved_qualifier_level(t, last, db);
                    if (t1 == t || t1 == last || db.Names.size() < 2)
                        return first;
                    auto s = db.Names.back();
                    db.Names.pop_back();
                    db.Names.back() =
                        db.make<QualifiedName>(db.Names.back(), s);
                    t = t1;
                }
                ++t;
                t1 = parse_base_unresolved_name(t, last, db);
                if (t1 == t)
                {
                    if (!db.Names.empty())
                        db.Names.pop_back();
                    return first;
                }
                if (db.Names.size() < 2)
                    return first;
                auto s = db.Names.back();
                db.Names.pop_back();
                db.Names.back() =
                    db.make<QualifiedName>(db.Names.back(), s);
                first = t1;
            }
            else
            {
                t += 2;
                const char* t1 = parse_unresolved_type(t, last, db);
                if (t1 != t)
                {
                    t = t1;
                    t1 = parse_template_args(t, last, db);
                    if (t1 != t)
                    {
                        if (db.Names.size() < 2)
                            return first;
                        auto args = db.Names.back();
                        db.Names.pop_back();
                        db.Names.back() =
                            db.make<NameWithTemplateArgs>(
                                db.Names.back(), args);
                        t = t1;
                    }
                    t1 = parse_base_unresolved_name(t, last, db);
                    if (t1 == t)
                    {
                        if (!db.Names.empty())
                            db.Names.pop_back();
                        return first;
                    }
                    if (db.Names.size() < 2)
                        return first;
                    auto s = db.Names.back();
                    db.Names.pop_back();
                    db.Names.back() =
                        db.make<QualifiedName>(db.Names.back(), s);
                    first = t1;
                }
                else
                {
                    t1 = parse_unresolved_qualifier_level(t, last, db);
                    if (t1 == t || t1 == last)
                        return first;
                    t = t1;
                    if (global)
                    {
                        if (db.Names.empty())
                            return first;
                        db.Names.back() =
                            db.make<GlobalQualifiedName>(
                                db.Names.back());
                    }
                    while (*t != 'E')
                    {
                        t1 = parse_unresolved_qualifier_level(t, last, db);
                        if (t1 == t || t1 == last || db.Names.size() < 2)
                            return first;
                        auto s = db.Names.back();
                        db.Names.pop_back();
                        db.Names.back() = db.make<QualifiedName>(
                            db.Names.back(), s);
                        t = t1;
                    }
                    ++t;
                    t1 = parse_base_unresolved_name(t, last, db);
                    if (t1 == t)
                    {
                        if (!db.Names.empty())
                            db.Names.pop_back();
                        return first;
                    }
                    if (db.Names.size() < 2)
                        return first;
                    auto s = db.Names.back();
                    db.Names.pop_back();
                    db.Names.back() =
                        db.make<QualifiedName>(db.Names.back(), s);
                    first = t1;
                }
            }
        }
    }
    return first;
}

// dt <expression> <unresolved-name>                    # expr.name

const char*
parse_dot_expr(const char* first, const char* last, Db& db)
{
    if (last - first >= 3 && first[0] == 'd' && first[1] == 't')
    {
        const char* t = parse_expression(first+2, last, db);
        if (t != first+2)
        {
            const char* t1 = parse_unresolved_name(t, last, db);
            if (t1 != t)
            {
                if (db.Names.size() < 2)
                    return first;
                auto name = db.Names.back();
                db.Names.pop_back();
                if (db.Names.empty())
                    return first;
                db.Names.back() = db.make<MemberExpr>(db.Names.back(), ".", name);
                first = t1;
            }
        }
    }
    return first;
}

// cl <expression>+ E                                   # call

const char*
parse_call_expr(const char* first, const char* last, Db& db)
{
    if (last - first >= 4 && first[0] == 'c' && first[1] == 'l')
    {
        const char* t = parse_expression(first+2, last, db);
        if (t == last || t == first + 2 || db.Names.empty())
            return first;
        Node* callee = db.Names.back();
        db.Names.pop_back();
        size_t args_begin = db.Names.size();
        while (*t != 'E')
        {
            const char* t1 = parse_expression(t, last, db);
            if (t1 == last || t1 == t)
                return first;
            t = t1;
        }
        if (db.Names.size() < args_begin)
            return first;
        ++t;
        CallExpr* the_call = db.make<CallExpr>(
            callee, db.popTrailingNodeArray(args_begin));
        db.Names.push_back(the_call);
        first = t;
    }
    return first;
}

// [gs] nw <expression>* _ <type> E                     # new (expr-list) type
// [gs] nw <expression>* _ <type> <initializer>         # new (expr-list) type (init)
// [gs] na <expression>* _ <type> E                     # new[] (expr-list) type
// [gs] na <expression>* _ <type> <initializer>         # new[] (expr-list) type (init)
// <initializer> ::= pi <expression>* E                 # parenthesized initialization

const char*
parse_new_expr(const char* first, const char* last, Db& db)
{
    if (last - first >= 4)
    {
        const char* t = first;
        bool parsed_gs = false;
        if (t[0] == 'g' && t[1] == 's')
        {
            t += 2;
            parsed_gs = true;
        }
        if (t[0] == 'n' && (t[1] == 'w' || t[1] == 'a'))
        {
            bool is_array = t[1] == 'a';
            t += 2;
            if (t == last)
                return first;
            size_t first_expr_in_list = db.Names.size();
            NodeArray ExprList, init_list;
            while (*t != '_')
            {
                const char* t1 = parse_expression(t, last, db);
                if (t1 == t || t1 == last)
                    return first;
                t = t1;
            }
            if (first_expr_in_list > db.Names.size())
                return first;
            ExprList = db.popTrailingNodeArray(first_expr_in_list);
            ++t;
            const char* t1 = parse_type(t, last, db);
            if (t1 == t || t1 == last)
                return first;
            t = t1;
            bool has_init = false;
            if (last - t >= 3 && t[0] == 'p' && t[1] == 'i')
            {
                t += 2;
                has_init = true;
                size_t init_list_begin = db.Names.size();
                while (*t != 'E')
                {
                    t1 = parse_expression(t, last, db);
                    if (t1 == t || t1 == last)
                        return first;
                    t = t1;
                }
                if (init_list_begin > db.Names.size())
                    return first;
                init_list = db.popTrailingNodeArray(init_list_begin);
            }
            if (*t != 'E' || db.Names.empty())
                return first;
            auto type = db.Names.back();
            db.Names.pop_back();
            db.Names.push_back(
                db.make<NewExpr>(ExprList, type, init_list,
                                  parsed_gs, is_array));
            first = t+1;
        }
    }
    return first;
}

// cv <type> <expression>                               # conversion with one argument
// cv <type> _ <expression>* E                          # conversion with a different number of arguments

const char*
parse_conversion_expr(const char* first, const char* last, Db& db)
{
    if (last - first >= 3 && first[0] == 'c' && first[1] == 'v')
    {
        bool TryToParseTemplateArgs = db.TryToParseTemplateArgs;
        db.TryToParseTemplateArgs = false;
        size_t type_begin = db.Names.size();
        const char* t = parse_type(first+2, last, db);
        db.TryToParseTemplateArgs = TryToParseTemplateArgs;
        if (t != first+2 && t != last)
        {
            size_t expr_list_begin = db.Names.size();
            if (*t != '_')
            {
                const char* t1 = parse_expression(t, last, db);
                if (t1 == t)
                    return first;
                t = t1;
            }
            else
            {
                ++t;
                if (t == last)
                    return first;
                if (*t != 'E')
                {
                    while (*t != 'E')
                    {
                        const char* t1 = parse_expression(t, last, db);
                        if (t1 == t || t1 == last)
                            return first;
                        t = t1;
                    }
                }
                ++t;
            }
            if (db.Names.size() < expr_list_begin ||
                type_begin > expr_list_begin)
                return first;
            NodeArray expressions = db.makeNodeArray(
                db.Names.begin() + (long)expr_list_begin, db.Names.end());
            NodeArray types = db.makeNodeArray(
                db.Names.begin() + (long)type_begin,
                db.Names.begin() + (long)expr_list_begin);
            auto* conv_expr = db.make<ConversionExpr>(
                types, expressions);
            db.Names.dropBack(type_begin);
            db.Names.push_back(conv_expr);
            first = t;
        }
    }
    return first;
}

// pt <expression> <expression>                    # expr->name

const char*
parse_arrow_expr(const char* first, const char* last, Db& db)
{
    if (last - first >= 3 && first[0] == 'p' && first[1] == 't')
    {
        const char* t = parse_expression(first+2, last, db);
        if (t != first+2)
        {
            const char* t1 = parse_expression(t, last, db);
            if (t1 != t)
            {
                if (db.Names.size() < 2)
                    return first;
                auto tmp = db.Names.back();
                db.Names.pop_back();
                db.Names.back() = db.make<MemberExpr>(
                    db.Names.back(), "->", tmp);
                first = t1;
            }
        }
    }
    return first;
}

//  <ref-qualifier> ::= R                   # & ref-qualifier
//  <ref-qualifier> ::= O                   # && ref-qualifier

// <function-type> ::= F [Y] <bare-function-type> [<ref-qualifier>] E

const char*
parse_function_type(const char* first, const char* last, Db& db)
{
    if (first != last && *first == 'F')
    {
        const char* t = first+1;
        if (t != last)
        {
            if (*t == 'Y')
            {
                /* extern "C" */
                if (++t == last)
                    return first;
            }
            const char* t1 = parse_type(t, last, db);
            if (t1 != t && !db.Names.empty())
            {
                Node* ret_type = db.Names.back();
                db.Names.pop_back();
                size_t params_begin = db.Names.size();
                t = t1;
                FunctionRefQual RefQuals = FrefQualNone;
                while (true)
                {
                    if (t == last)
                    {
                        if (!db.Names.empty())
                          db.Names.pop_back();
                        return first;
                    }
                    if (*t == 'E')
                    {
                        ++t;
                        break;
                    }
                    if (*t == 'v')
                    {
                        ++t;
                        continue;
                    }
                    if (*t == 'R' && t+1 != last && t[1] == 'E')
                    {
                        RefQuals = FrefQualLValue;
                        ++t;
                        continue;
                    }
                    if (*t == 'O' && t+1 != last && t[1] == 'E')
                    {
                        RefQuals = FrefQualRValue;
                        ++t;
                        continue;
                    }
                    size_t k0 = db.Names.size();
                    t1 = parse_type(t, last, db);
                    size_t k1 = db.Names.size();
                    if (t1 == t || t1 == last || k1 < k0)
                        return first;
                    t = t1;
                }
                if (db.Names.empty() || params_begin > db.Names.size())
                    return first;
                Node* fty = db.make<FunctionType>(
                    ret_type, db.popTrailingNodeArray(params_begin));
                if (RefQuals)
                    fty = db.make<FunctionRefQualType>(fty, RefQuals);
                db.Names.push_back(fty);
                first = t;
            }
        }
    }
    return first;
}

// <pointer-to-member-type> ::= M <class type> <member type>

const char*
parse_pointer_to_member_type(const char* first, const char* last, Db& db)
{
    if (first != last && *first == 'M')
    {
        const char* t = parse_type(first+1, last, db);
        if (t != first+1)
        {
            const char* t2 = parse_type(t, last, db);
            if (t2 != t)
            {
                if (db.Names.size() < 2)
                    return first;
                auto func = std::move(db.Names.back());
                db.Names.pop_back();
                auto ClassType = std::move(db.Names.back());
                db.Names.back() =
                    db.make<PointerToMemberType>(ClassType, func);
                first = t2;
            }
        }
    }
    return first;
}

// <array-type> ::= A <positive dimension number> _ <element type>
//              ::= A [<dimension expression>] _ <element type>

const char*
parse_array_type(const char* first, const char* last, Db& db)
{
    if (first != last && *first == 'A' && first+1 != last)
    {
        if (first[1] == '_')
        {
            const char* t = parse_type(first+2, last, db);
            if (t != first+2)
            {
                if (db.Names.empty())
                    return first;
                db.Names.back() = db.make<ArrayType>(db.Names.back());
                first = t;
            }
        }
        else if ('1' <= first[1] && first[1] <= '9')
        {
            const char* t = parse_number(first+1, last);
            if (t != last && *t == '_')
            {
                const char* t2 = parse_type(t+1, last, db);
                if (t2 != t+1)
                {
                    if (db.Names.empty())
                        return first;
                    db.Names.back() =
                        db.make<ArrayType>(db.Names.back(),
                                            StringView(first + 1, t));
                    first = t2;
                }
            }
        }
        else
        {
            const char* t = parse_expression(first+1, last, db);
            if (t != first+1 && t != last && *t == '_')
            {
                const char* t2 = parse_type(++t, last, db);
                if (t2 != t)
                {
                    if (db.Names.size() < 2)
                        return first;
                    auto base_type = std::move(db.Names.back());
                    db.Names.pop_back();
                    auto dimension_expr = std::move(db.Names.back());
                    db.Names.back() =
                        db.make<ArrayType>(base_type, dimension_expr);
                    first = t2;
                }
            }
        }
    }
    return first;
}

// <decltype>  ::= Dt <expression> E  # decltype of an id-expression or class member access (C++0x)
//             ::= DT <expression> E  # decltype of an expression (C++0x)

const char*
parse_decltype(const char* first, const char* last, Db& db)
{
    if (last - first >= 4 && first[0] == 'D')
    {
        switch (first[1])
        {
        case 't':
        case 'T':
            {
                const char* t = parse_expression(first+2, last, db);
                if (t != first+2 && t != last && *t == 'E')
                {
                    if (db.Names.empty())
                        return first;
                    db.Names.back() = db.make<EnclosingExpr>(
                        "decltype(", db.Names.back(), ")");
                    first = t+1;
                }
            }
            break;
        }
    }
    return first;
}

// extension:
// <vector-type>           ::= Dv <positive dimension number> _
//                                    <extended element type>
//                         ::= Dv [<dimension expression>] _ <element type>
// <extended element type> ::= <element type>
//                         ::= p # AltiVec vector pixel

const char*
parse_vector_type(const char* first, const char* last, Db& db)
{
    if (last - first > 3 && first[0] == 'D' && first[1] == 'v')
    {
        if ('1' <= first[2] && first[2] <= '9')
        {
            const char* t = parse_number(first+2, last);
            if (t == last || *t != '_')
                return first;
            const char* num = first + 2;
            size_t sz = static_cast<size_t>(t - num);
            if (++t != last)
            {
                if (*t != 'p')
                {
                    const char* t1 = parse_type(t, last, db);
                    if (t1 != t)
                    {
                        if (db.Names.empty())
                            return first;
                        db.Names.back() =
                            db.make<VectorType>(db.Names.back(),
                                                 StringView(num, num + sz));
                        first = t1;
                    }
                }
                else
                {
                    ++t;
                    db.Names.push_back(
                        db.make<VectorType>(StringView(num, num + sz)));
                    first = t;
                }
            }
        }
        else
        {
            Node* num = nullptr;
            const char* t1 = first+2;
            if (*t1 != '_')
            {
                const char* t = parse_expression(t1, last, db);
                if (t != t1)
                {
                    if (db.Names.empty())
                        return first;
                    num = db.Names.back();
                    db.Names.pop_back();
                    t1 = t;
                }
            }
            if (t1 != last && *t1 == '_' && ++t1 != last)
            {
                const char* t = parse_type(t1, last, db);
                if (t != t1)
                {
                    if (db.Names.empty())
                        return first;
                    if (num)
                        db.Names.back() =
                            db.make<VectorType>(db.Names.back(), num);
                    else
                        db.Names.back() =
                            db.make<VectorType>(db.Names.back(), StringView());
                    first = t;
                } else if (num)
                    db.Names.push_back(num);
            }
        }
    }
    return first;
}

// <type> ::= <builtin-type>
//        ::= <function-type>
//        ::= <class-enum-type>
//        ::= <array-type>
//        ::= <pointer-to-member-type>
//        ::= <template-param>
//        ::= <template-template-param> <template-args>
//        ::= <decltype>
//        ::= <substitution>
//        ::= <CV-Qualifiers> <type>
//        ::= P <type>        # pointer-to
//        ::= R <type>        # reference-to
//        ::= O <type>        # rvalue reference-to (C++0x)
//        ::= C <type>        # complex pair (C 2000)
//        ::= G <type>        # imaginary (C 2000)
//        ::= Dp <type>       # pack expansion (C++0x)
//        ::= U <source-name> <type>  # vendor extended type qualifier
// extension := U <objc-name> <objc-type>  # objc-type<identifier>
// extension := <vector-type> # <vector-type> starts with Dv

// <objc-name> ::= <k0 number> objcproto <k1 number> <identifier>  # k0 = 9 + <number of digits in k1> + k1
// <objc-type> := <source-name>  # PU<11+>objcproto 11objc_object<source-name> 11objc_object -> id<source-name>

const char*
parse_type(const char* first, const char* last, Db& db)
{
    if (first != last)
    {
        switch (*first)
        {
            case 'r':
            case 'V':
            case 'K':
              {
                Qualifiers cv = QualNone;
                const char* t = parse_cv_qualifiers(first, last, cv);
                if (t != first)
                {
                    bool is_function = *t == 'F';
                    size_t k0 = db.Names.size();
                    const char* t1 = parse_type(t, last, db);
                    size_t k1 = db.Names.size();
                    if (t1 != t)
                    {
                        if (is_function)
                            db.Subs.popPack();
                        db.Subs.pushPack();
                        for (size_t k = k0; k < k1; ++k)
                        {
                            if (cv) {
                                if (is_function)
                                    db.Names[k] = db.make<FunctionQualType>(
                                        db.Names[k], cv);
                                else
                                    db.Names[k] =
                                        db.make<QualType>(db.Names[k], cv);
                            }
                            db.Subs.pushSubstitutionIntoPack(db.Names[k]);
                        }
                        first = t1;
                    }
                }
              }
                break;
            default:
              {
                const char* t = parse_builtin_type(first, last, db);
                if (t != first)
                {
                    first = t;
                }
                else
                {
                    switch (*first)
                    {
                    case 'A':
                        t = parse_array_type(first, last, db);
                        if (t != first)
                        {
                            if (db.Names.empty())
                                return first;
                            first = t;
                            db.Subs.pushSubstitution(db.Names.back());
                        }
                        break;
                    case 'C':
                        t = parse_type(first+1, last, db);
                        if (t != first+1)
                        {
                            if (db.Names.empty())
                                return first;
                            db.Names.back() = db.make<PostfixQualifiedType>(
                                db.Names.back(), " complex");
                            first = t;
                            db.Subs.pushSubstitution(db.Names.back());
                        }
                        break;
                    case 'F':
                        t = parse_function_type(first, last, db);
                        if (t != first)
                        {
                            if (db.Names.empty())
                                return first;
                            first = t;
                            db.Subs.pushSubstitution(db.Names.back());
                        }
                        break;
                    case 'G':
                        t = parse_type(first+1, last, db);
                        if (t != first+1)
                        {
                            if (db.Names.empty())
                                return first;
                            db.Names.back() = db.make<PostfixQualifiedType>(
                                db.Names.back(), " imaginary");
                            first = t;
                            db.Subs.pushSubstitution(db.Names.back());
                        }
                        break;
                    case 'M':
                        t = parse_pointer_to_member_type(first, last, db);
                        if (t != first)
                        {
                            if (db.Names.empty())
                                return first;
                            first = t;
                            db.Subs.pushSubstitution(db.Names.back());
                        }
                        break;
                    case 'O':
                      {
                        size_t k0 = db.Names.size();
                        t = parse_type(first+1, last, db);
                        size_t k1 = db.Names.size();
                        if (t != first+1)
                        {
                            db.Subs.pushPack();
                            for (size_t k = k0; k < k1; ++k)
                            {
                                db.Names[k] =
                                    db.make<RValueReferenceType>(db.Names[k]);
                                db.Subs.pushSubstitutionIntoPack(db.Names[k]);
                            }
                            first = t;
                        }
                        break;
                      }
                    case 'P':
                      {
                        size_t k0 = db.Names.size();
                        t = parse_type(first+1, last, db);
                        size_t k1 = db.Names.size();
                        if (t != first+1)
                        {
                            db.Subs.pushPack();
                            for (size_t k = k0; k < k1; ++k)
                            {
                                db.Names[k] = db.make<PointerType>(db.Names[k]);
                                db.Subs.pushSubstitutionIntoPack(db.Names[k]);
                            }
                            first = t;
                        }
                        break;
                      }
                    case 'R':
                      {
                        size_t k0 = db.Names.size();
                        t = parse_type(first+1, last, db);
                        size_t k1 = db.Names.size();
                        if (t != first+1)
                        {
                            db.Subs.pushPack();
                            for (size_t k = k0; k < k1; ++k)
                            {
                                db.Names[k] =
                                    db.make<LValueReferenceType>(db.Names[k]);
                                db.Subs.pushSubstitutionIntoPack(db.Names[k]);
                            }
                            first = t;
                        }
                        break;
                      }
                    case 'T':
                      {
                        size_t k0 = db.Names.size();
                        t = parse_template_param(first, last, db);
                        size_t k1 = db.Names.size();
                        if (t != first)
                        {
                            db.Subs.pushPack();
                            for (size_t k = k0; k < k1; ++k)
                                db.Subs.pushSubstitutionIntoPack(db.Names[k]);
                            if (db.TryToParseTemplateArgs && k1 == k0+1)
                            {
                                const char* t1 = parse_template_args(t, last, db);
                                if (t1 != t)
                                {
                                    auto args = db.Names.back();
                                    db.Names.pop_back();
                                    db.Names.back() = db.make<
                                        NameWithTemplateArgs>(
                                        db.Names.back(), args);
                                    db.Subs.pushSubstitution(db.Names.back());
                                    t = t1;
                                }
                            }
                            first = t;
                        }
                        break;
                      }
                    case 'U':
                        if (first+1 != last)
                        {
                            t = parse_source_name(first+1, last, db);
                            if (t != first+1)
                            {
                                const char* t2 = parse_type(t, last, db);
                                if (t2 != t)
                                {
                                    if (db.Names.size() < 2)
                                        return first;
                                    auto type = db.Names.back();
                                    db.Names.pop_back();
                                    if (db.Names.back()->K != Node::KNameType ||
                                        !static_cast<NameType*>(db.Names.back())->getName().startsWith("objcproto"))
                                    {
                                        db.Names.back() = db.make<VendorExtQualType>(type, db.Names.back());
                                    }
                                    else
                                    {
                                        auto* proto = static_cast<NameType*>(db.Names.back());
                                        db.Names.pop_back();
                                        t = parse_source_name(proto->getName().begin() + 9, proto->getName().end(), db);
                                        if (t != proto->getName().begin() + 9)
                                        {
                                            db.Names.back() = db.make<ObjCProtoName>(type, db.Names.back());
                                        }
                                        else
                                        {
                                            db.Names.push_back(db.make<VendorExtQualType>(type, proto));
                                        }
                                    }
                                    db.Subs.pushSubstitution(db.Names.back());
                                    first = t2;
                                }
                            }
                        }
                        break;
                    case 'S':
                        if (first+1 != last && first[1] == 't')
                        {
                            t = parse_name(first, last, db);
                            if (t != first)
                            {
                                if (db.Names.empty())
                                    return first;
                                db.Subs.pushSubstitution(db.Names.back());
                                first = t;
                            }
                        }
                        else
                        {
                            t = parse_substitution(first, last, db);
                            if (t != first)
                            {
                                first = t;
                                // Parsed a substitution.  If the substitution is a
                                //  <template-param> it might be followed by <template-args>.
                                if (db.TryToParseTemplateArgs)
                                {
                                    t = parse_template_args(first, last, db);
                                    if (t != first)
                                    {
                                        if (db.Names.size() < 2)
                                            return first;
                                        auto template_args = db.Names.back();
                                        db.Names.pop_back();
                                        db.Names.back() = db.make<
                                          NameWithTemplateArgs>(
                                              db.Names.back(), template_args);
                                        // Need to create substitution for <template-template-param> <template-args>
                                        db.Subs.pushSubstitution(db.Names.back());
                                        first = t;
                                    }
                                }
                            }
                        }
                        break;
                    case 'D':
                        if (first+1 != last)
                        {
                            switch (first[1])
                            {
                            case 'p':
                              {
                                size_t k0 = db.Names.size();
                                t = parse_type(first+2, last, db);
                                size_t k1 = db.Names.size();
                                if (t != first+2)
                                {
                                    db.Subs.pushPack();
                                    for (size_t k = k0; k < k1; ++k)
                                        db.Subs.pushSubstitutionIntoPack(db.Names[k]);
                                    first = t;
                                    return first;
                                }
                                break;
                              }
                            case 't':
                            case 'T':
                                t = parse_decltype(first, last, db);
                                if (t != first)
                                {
                                    if (db.Names.empty())
                                        return first;
                                    db.Subs.pushSubstitution(db.Names.back());
                                    first = t;
                                    return first;
                                }
                                break;
                            case 'v':
                                t = parse_vector_type(first, last, db);
                                if (t != first)
                                {
                                    if (db.Names.empty())
                                        return first;
                                    db.Subs.pushSubstitution(db.Names.back());
                                    first = t;
                                    return first;
                                }
                                break;
                            }
                        }
                        _LIBCPP_FALLTHROUGH();
                    default:
                        // must check for builtin-types before class-enum-types to avoid
                        // ambiguities with operator-names
                        t = parse_builtin_type(first, last, db);
                        if (t != first)
                        {
                            first = t;
                        }
                        else
                        {
                            t = parse_name(first, last, db);
                            if (t != first)
                            {
                                if (db.Names.empty())
                                    return first;
                                db.Subs.pushSubstitution(db.Names.back());
                                first = t;
                            }
                        }
                        break;
                    }
              }
                break;
            }
        }
    }
    return first;
}

//   <operator-name>
//                   ::= aa    # &&            
//                   ::= ad    # & (unary)     
//                   ::= an    # &             
//                   ::= aN    # &=            
//                   ::= aS    # =             
//                   ::= cl    # ()            
//                   ::= cm    # ,             
//                   ::= co    # ~             
//                   ::= cv <type>    # (cast)        
//                   ::= da    # delete[]      
//                   ::= de    # * (unary)     
//                   ::= dl    # delete        
//                   ::= dv    # /             
//                   ::= dV    # /=            
//                   ::= eo    # ^             
//                   ::= eO    # ^=            
//                   ::= eq    # ==            
//                   ::= ge    # >=            
//                   ::= gt    # >             
//                   ::= ix    # []            
//                   ::= le    # <=            
//                   ::= li <source-name>  # operator ""
//                   ::= ls    # <<            
//                   ::= lS    # <<=           
//                   ::= lt    # <             
//                   ::= mi    # -             
//                   ::= mI    # -=            
//                   ::= ml    # *             
//                   ::= mL    # *=            
//                   ::= mm    # -- (postfix in <expression> context)           
//                   ::= na    # new[]
//                   ::= ne    # !=            
//                   ::= ng    # - (unary)     
//                   ::= nt    # !             
//                   ::= nw    # new           
//                   ::= oo    # ||            
//                   ::= or    # |             
//                   ::= oR    # |=            
//                   ::= pm    # ->*           
//                   ::= pl    # +             
//                   ::= pL    # +=            
//                   ::= pp    # ++ (postfix in <expression> context)
//                   ::= ps    # + (unary)
//                   ::= pt    # ->            
//                   ::= qu    # ?             
//                   ::= rm    # %             
//                   ::= rM    # %=            
//                   ::= rs    # >>            
//                   ::= rS    # >>=           
//                   ::= v <digit> <source-name>        # vendor extended operator
//   extension       ::= <operator-name> <abi-tag-seq>
const char*
parse_operator_name(const char* first, const char* last, Db& db)
{
    const char* original_first = first;
    if (last - first >= 2)
    {
        switch (first[0])
        {
        case 'a':
            switch (first[1])
            {
            case 'a':
                db.Names.push_back(db.make<NameType>("operator&&"));
                first += 2;
                break;
            case 'd':
            case 'n':
                db.Names.push_back(db.make<NameType>("operator&"));
                first += 2;
                break;
            case 'N':
                db.Names.push_back(db.make<NameType>("operator&="));
                first += 2;
                break;
            case 'S':
                db.Names.push_back(db.make<NameType>("operator="));
                first += 2;
                break;
            }
            break;
        case 'c':
            switch (first[1])
            {
            case 'l':
                db.Names.push_back(db.make<NameType>("operator()"));
                first += 2;
                break;
            case 'm':
                db.Names.push_back(db.make<NameType>("operator,"));
                first += 2;
                break;
            case 'o':
                db.Names.push_back(db.make<NameType>("operator~"));
                first += 2;
                break;
            case 'v':
                {
                    bool TryToParseTemplateArgs = db.TryToParseTemplateArgs;
                    db.TryToParseTemplateArgs = false;
                    const char* t = parse_type(first+2, last, db);
                    db.TryToParseTemplateArgs = TryToParseTemplateArgs;
                    if (t != first+2)
                    {
                        if (db.Names.empty())
                            return first;
                        db.Names.back() =
                            db.make<ConversionOperatorType>(db.Names.back());
                        db.ParsedCtorDtorCV = true;
                        first = t;
                    }
                }
                break;
            }
            break;
        case 'd':
            switch (first[1])
            {
            case 'a':
                db.Names.push_back(db.make<NameType>("operator delete[]"));
                first += 2;
                break;
            case 'e':
                db.Names.push_back(db.make<NameType>("operator*"));
                first += 2;
                break;
            case 'l':
                db.Names.push_back(db.make<NameType>("operator delete"));
                first += 2;
                break;
            case 'v':
                db.Names.push_back(db.make<NameType>("operator/"));
                first += 2;
                break;
            case 'V':
                db.Names.push_back(db.make<NameType>("operator/="));
                first += 2;
                break;
            }
            break;
        case 'e':
            switch (first[1])
            {
            case 'o':
                db.Names.push_back(db.make<NameType>("operator^"));
                first += 2;
                break;
            case 'O':
                db.Names.push_back(db.make<NameType>("operator^="));
                first += 2;
                break;
            case 'q':
                db.Names.push_back(db.make<NameType>("operator=="));
                first += 2;
                break;
            }
            break;
        case 'g':
            switch (first[1])
            {
            case 'e':
                db.Names.push_back(db.make<NameType>("operator>="));
                first += 2;
                break;
            case 't':
                db.Names.push_back(db.make<NameType>("operator>"));
                first += 2;
                break;
            }
            break;
        case 'i':
            if (first[1] == 'x')
            {
                db.Names.push_back(db.make<NameType>("operator[]"));
                first += 2;
            }
            break;
        case 'l':
            switch (first[1])
            {
            case 'e':
                db.Names.push_back(db.make<NameType>("operator<="));
                first += 2;
                break;
            case 'i':
                {
                    const char* t = parse_source_name(first+2, last, db);
                    if (t != first+2)
                    {
                        if (db.Names.empty())
                            return first;
                        db.Names.back() =
                            db.make<LiteralOperator>(db.Names.back());
                        first = t;
                    }
                }
                break;
            case 's':
                db.Names.push_back(db.make<NameType>("operator<<"));
                first += 2;
                break;
            case 'S':
                db.Names.push_back(db.make<NameType>("operator<<="));
                first += 2;
                break;
            case 't':
                db.Names.push_back(db.make<NameType>("operator<"));
                first += 2;
                break;
            }
            break;
        case 'm':
            switch (first[1])
            {
            case 'i':
                db.Names.push_back(db.make<NameType>("operator-"));
                first += 2;
                break;
            case 'I':
                db.Names.push_back(db.make<NameType>("operator-="));
                first += 2;
                break;
            case 'l':
                db.Names.push_back(db.make<NameType>("operator*"));
                first += 2;
                break;
            case 'L':
                db.Names.push_back(db.make<NameType>("operator*="));
                first += 2;
                break;
            case 'm':
                db.Names.push_back(db.make<NameType>("operator--"));
                first += 2;
                break;
            }
            break;
        case 'n':
            switch (first[1])
            {
            case 'a':
                db.Names.push_back(db.make<NameType>("operator new[]"));
                first += 2;
                break;
            case 'e':
                db.Names.push_back(db.make<NameType>("operator!="));
                first += 2;
                break;
            case 'g':
                db.Names.push_back(db.make<NameType>("operator-"));
                first += 2;
                break;
            case 't':
                db.Names.push_back(db.make<NameType>("operator!"));
                first += 2;
                break;
            case 'w':
                db.Names.push_back(db.make<NameType>("operator new"));
                first += 2;
                break;
            }
            break;
        case 'o':
            switch (first[1])
            {
            case 'o':
                db.Names.push_back(db.make<NameType>("operator||"));
                first += 2;
                break;
            case 'r':
                db.Names.push_back(db.make<NameType>("operator|"));
                first += 2;
                break;
            case 'R':
                db.Names.push_back(db.make<NameType>("operator|="));
                first += 2;
                break;
            }
            break;
        case 'p':
            switch (first[1])
            {
            case 'm':
                db.Names.push_back(db.make<NameType>("operator->*"));
                first += 2;
                break;
            case 'l':
                db.Names.push_back(db.make<NameType>("operator+"));
                first += 2;
                break;
            case 'L':
                db.Names.push_back(db.make<NameType>("operator+="));
                first += 2;
                break;
            case 'p':
                db.Names.push_back(db.make<NameType>("operator++"));
                first += 2;
                break;
            case 's':
                db.Names.push_back(db.make<NameType>("operator+"));
                first += 2;
                break;
            case 't':
                db.Names.push_back(db.make<NameType>("operator->"));
                first += 2;
                break;
            }
            break;
        case 'q':
            if (first[1] == 'u')
            {
                db.Names.push_back(db.make<NameType>("operator?"));
                first += 2;
            }
            break;
        case 'r':
            switch (first[1])
            {
            case 'm':
                db.Names.push_back(db.make<NameType>("operator%"));
                first += 2;
                break;
            case 'M':
                db.Names.push_back(db.make<NameType>("operator%="));
                first += 2;
                break;
            case 's':
                db.Names.push_back(db.make<NameType>("operator>>"));
                first += 2;
                break;
            case 'S':
                db.Names.push_back(db.make<NameType>("operator>>="));
                first += 2;
                break;
            }
            break;
        case 'v':
            if (std::isdigit(first[1]))
            {
                const char* t = parse_source_name(first+2, last, db);
                if (t != first+2)
                {
                    if (db.Names.empty())
                        return first;
                    db.Names.back() =
                        db.make<ConversionOperatorType>(db.Names.back());
                    first = t;
                }
            }
            break;
        }
    }

    if (original_first != first)
        first = parse_abi_tag_seq(first, last, db);

    return first;
}

const char*
parse_integer_literal(const char* first, const char* last, StringView lit, Db& db)
{
    const char* t = parse_number(first, last);
    if (t != first && t != last && *t == 'E')
    {
        db.Names.push_back(
            db.make<IntegerExpr>(lit, StringView(first, t)));
        first = t+1;
    }
    return first;
}

// <expr-primary> ::= L <type> <value number> E                          # integer literal
//                ::= L <type> <value float> E                           # floating literal
//                ::= L <string type> E                                  # string literal
//                ::= L <nullptr type> E                                 # nullptr literal (i.e., "LDnE")
//                ::= L <type> <real-part float> _ <imag-part float> E   # complex floating point literal (C 2000)
//                ::= L <mangled-name> E                                 # external name

const char*
parse_expr_primary(const char* first, const char* last, Db& db)
{
    if (last - first >= 4 && *first == 'L')
    {
        switch (first[1])
        {
        case 'w':
            {
            const char* t = parse_integer_literal(first+2, last, "wchar_t", db);
            if (t != first+2)
                first = t;
            }
            break;
        case 'b':
            if (first[3] == 'E')
            {
                switch (first[2])
                {
                case '0':
                    db.Names.push_back(db.make<BoolExpr>(0));
                    first += 4;
                    break;
                case '1':
                    db.Names.push_back(db.make<BoolExpr>(1));
                    first += 4;
                    break;
                }
            }
            break;
        case 'c':
            {
            const char* t = parse_integer_literal(first+2, last, "char", db);
            if (t != first+2)
                first = t;
            }
            break;
        case 'a':
            {
            const char* t = parse_integer_literal(first+2, last, "signed char", db);
            if (t != first+2)
                first = t;
            }
            break;
        case 'h':
            {
            const char* t = parse_integer_literal(first+2, last, "unsigned char", db);
            if (t != first+2)
                first = t;
            }
            break;
        case 's':
            {
            const char* t = parse_integer_literal(first+2, last, "short", db);
            if (t != first+2)
                first = t;
            }
            break;
        case 't':
            {
            const char* t = parse_integer_literal(first+2, last, "unsigned short", db);
            if (t != first+2)
                first = t;
            }
            break;
        case 'i':
            {
            const char* t = parse_integer_literal(first+2, last, "", db);
            if (t != first+2)
                first = t;
            }
            break;
        case 'j':
            {
            const char* t = parse_integer_literal(first+2, last, "u", db);
            if (t != first+2)
                first = t;
            }
            break;
        case 'l':
            {
            const char* t = parse_integer_literal(first+2, last, "l", db);
            if (t != first+2)
                first = t;
            }
            break;
        case 'm':
            {
            const char* t = parse_integer_literal(first+2, last, "ul", db);
            if (t != first+2)
                first = t;
            }
            break;
        case 'x':
            {
            const char* t = parse_integer_literal(first+2, last, "ll", db);
            if (t != first+2)
                first = t;
            }
            break;
        case 'y':
            {
            const char* t = parse_integer_literal(first+2, last, "ull", db);
            if (t != first+2)
                first = t;
            }
            break;
        case 'n':
            {
            const char* t = parse_integer_literal(first+2, last, "__int128", db);
            if (t != first+2)
                first = t;
            }
            break;
        case 'o':
            {
            const char* t = parse_integer_literal(first+2, last, "unsigned __int128", db);
            if (t != first+2)
                first = t;
            }
            break;
        case 'f':
            {
            const char* t = parse_floating_number<float>(first+2, last, db);
            if (t != first+2)
                first = t;
            }
            break;
        case 'd':
            {
            const char* t = parse_floating_number<double>(first+2, last, db);
            if (t != first+2)
                first = t;
            }
            break;
         case 'e':
            {
            const char* t = parse_floating_number<long double>(first+2, last, db);
            if (t != first+2)
                first = t;
            }
            break;
        case '_':
            if (first[2] == 'Z')
            {
                const char* t = parse_encoding(first+3, last, db);
                if (t != first+3 && t != last && *t == 'E')
                    first = t+1;
            }
            break;
        case 'T':
            // Invalid mangled name per
            //   http://sourcerytools.com/pipermail/cxx-abi-dev/2011-August/002422.html
            break;
        default:
            {
                // might be named type
                const char* t = parse_type(first+1, last, db);
                if (t != first+1 && t != last)
                {
                    if (*t != 'E')
                    {
                        const char* n = t;
                        for (; n != last && isdigit(*n); ++n)
                            ;
                        if (n != t && n != last && *n == 'E')
                        {
                            if (db.Names.empty())
                                return first;
                            db.Names.back() = db.make<IntegerCastExpr>(
                                db.Names.back(), StringView(t, n));
                            first = n+1;
                            break;
                        }
                    }
                    else
                    {
                        first = t+1;
                        break;
                    }
                }
            }
        }
    }
    return first;
}

Node* maybe_change_special_sub_name(Node* inp, Db& db)
{
    if (inp->K != Node::KSpecialSubstitution)
        return inp;
    auto Kind = static_cast<SpecialSubstitution*>(inp)->SSK;
    switch (Kind)
    {
    case SpecialSubKind::string:
    case SpecialSubKind::istream:
    case SpecialSubKind::ostream:
    case SpecialSubKind::iostream:
        return db.make<ExpandedSpecialSubstitution>(Kind);
    default:
        break;
    }
    return inp;
}

// <ctor-dtor-name> ::= C1    # complete object constructor
//                  ::= C2    # base object constructor
//                  ::= C3    # complete object allocating constructor
//   extension      ::= C5    # ?
//                  ::= D0    # deleting destructor
//                  ::= D1    # complete object destructor
//                  ::= D2    # base object destructor
//   extension      ::= D5    # ?
//   extension      ::= <ctor-dtor-name> <abi-tag-seq>
const char*
parse_ctor_dtor_name(const char* first, const char* last, Db& db)
{
    if (last-first >= 2 && !db.Names.empty())
    {
        switch (first[0])
        {
        case 'C':
            switch (first[1])
            {
            case '1':
            case '2':
            case '3':
            case '5':
                if (db.Names.empty())
                    return first;
                db.Names.back() =
                    maybe_change_special_sub_name(db.Names.back(), db);
                db.Names.push_back(
                    db.make<CtorDtorName>(db.Names.back(), false));
                first += 2;
                first = parse_abi_tag_seq(first, last, db);
                db.ParsedCtorDtorCV = true;
                break;
            }
            break;
        case 'D':
            switch (first[1])
            {
            case '0':
            case '1':
            case '2':
            case '5':
                if (db.Names.empty())
                    return first;
                db.Names.push_back(
                    db.make<CtorDtorName>(db.Names.back(), true));
                first += 2;
                first = parse_abi_tag_seq(first, last, db);
                db.ParsedCtorDtorCV = true;
                break;
            }
            break;
        }
    }
    return first;
}

// <unnamed-type-name> ::= Ut [<nonnegative number>] _ [<abi-tag-seq>]
//                     ::= <closure-type-name>
// 
// <closure-type-name> ::= Ul <lambda-sig> E [ <nonnegative number> ] _ 
// 
// <lambda-sig> ::= <parameter type>+  # Parameter types or "v" if the lambda has no parameters
const char*
parse_unnamed_type_name(const char* first, const char* last, Db& db)
{
    if (last - first > 2 && first[0] == 'U')
    {
        char type = first[1];
        switch (type)
        {
        case 't':
          {
            const char* t0 = first+2;
            if (t0 == last)
                return first;
            StringView count;
            if (std::isdigit(*t0))
            {
                const char* t1 = t0 + 1;
                while (t1 != last && std::isdigit(*t1))
                    ++t1;
                count = StringView(t0, t1);
                t0 = t1;
            }
            if (t0 == last || *t0 != '_')
                return first;
            db.Names.push_back(db.make<UnnamedTypeName>(count));
            first = t0 + 1;
            first = parse_abi_tag_seq(first, last, db);
          }
            break;
        case 'l':
          {
            size_t begin_pos = db.Names.size();
            const char* t0 = first+2;
            NodeArray lambda_params;
            if (first[2] == 'v')
            {
                ++t0;
            }
            else
            {
                while (true)
                {
                    const char* t1 = parse_type(t0, last, db);
                    if (t1 == t0)
                        break;
                    t0 = t1;
                }
                if (db.Names.size() < begin_pos)
                    return first;
                lambda_params = db.popTrailingNodeArray(begin_pos);
            }
            if (t0 == last || *t0 != 'E')
                return first;
            ++t0;
            if (t0 == last)
                return first;
            StringView count;
            if (std::isdigit(*t0))
            {
                const char* t1 = t0 + 1;
                while (t1 != last && std::isdigit(*t1))
                    ++t1;
                count = StringView(t0, t1);
                t0 = t1;
            }
            if (t0 == last || *t0 != '_')
                return first;
            db.Names.push_back(db.make<LambdaTypeName>(lambda_params, count));
            first = t0 + 1;
          }
            break;
        }
    }
    return first;
}

// <unqualified-name> ::= <operator-name>
//                    ::= <ctor-dtor-name>
//                    ::= <source-name>   
//                    ::= <unnamed-type-name>

const char*
parse_unqualified_name(const char* first, const char* last, Db& db)
{
    if (first != last)
    {
        const char* t;
        switch (*first)
        {
        case 'C':
        case 'D':
            t = parse_ctor_dtor_name(first, last, db);
            if (t != first)
                first = t;
            break;
        case 'U':
            t = parse_unnamed_type_name(first, last, db);
            if (t != first)
                first = t;
            break;
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
            t = parse_source_name(first, last, db);
            if (t != first)
                first = t;
            break;
        default:
            t = parse_operator_name(first, last, db);
            if (t != first)
                first = t;
            break;
        };
    }
    return first;
}

// <unscoped-name> ::= <unqualified-name>
//                 ::= St <unqualified-name>   # ::std::
// extension       ::= StL<unqualified-name>

const char*
parse_unscoped_name(const char* first, const char* last, Db& db)
{
    if (last - first >= 2)
    {
        const char* t0 = first;
        bool St = false;
        if (first[0] == 'S' && first[1] == 't')
        {
            t0 += 2;
            St = true;
            if (t0 != last && *t0 == 'L')
                ++t0;
        }
        const char* t1 = parse_unqualified_name(t0, last, db);
        if (t1 != t0)
        {
            if (St)
            {
                if (db.Names.empty())
                    return first;
                db.Names.back() =
                    db.make<StdQualifiedName>(db.Names.back());
            }
            first = t1;
        }
    }
    return first;
}

// at <type>                                            # alignof (a type)

const char*
parse_alignof_type(const char* first, const char* last, Db& db)
{
    if (last - first >= 3 && first[0] == 'a' && first[1] == 't')
    {
        const char* t = parse_type(first+2, last, db);
        if (t != first+2)
        {
            if (db.Names.empty())
                return first;
            db.Names.back() =
                db.make<EnclosingExpr>("alignof (", db.Names.back(), ")");
            first = t;
        }
    }
    return first;
}

// az <expression>                                            # alignof (a expression)

const char*
parse_alignof_expr(const char* first, const char* last, Db& db)
{
    if (last - first >= 3 && first[0] == 'a' && first[1] == 'z')
    {
        const char* t = parse_expression(first+2, last, db);
        if (t != first+2)
        {
            if (db.Names.empty())
                return first;
            db.Names.back() =
                db.make<EnclosingExpr>("alignof (", db.Names.back(), ")");
            first = t;
        }
    }
    return first;
}

const char*
parse_noexcept_expression(const char* first, const char* last, Db& db)
{
    const char* t1 = parse_expression(first, last, db);
    if (t1 != first)
    {
        if (db.Names.empty())
            return first;
        db.Names.back() =
            db.make<EnclosingExpr>("noexcept (", db.Names.back(), ")");
        first = t1;
    }
    return first;
}

const char*
parse_prefix_expression(const char* first, const char* last, StringView op, Db& db)
{
    const char* t1 = parse_expression(first, last, db);
    if (t1 != first)
    {
        if (db.Names.empty())
            return first;
        db.Names.back() = db.make<PrefixExpr>(op, db.Names.back());
        first = t1;
    }
    return first;
}

const char*
parse_binary_expression(const char* first, const char* last, StringView op, Db& db)
{
    const char* t1 = parse_expression(first, last, db);
    if (t1 != first)
    {
        const char* t2 = parse_expression(t1, last, db);
        if (t2 != t1)
        {
            if (db.Names.size() < 2)
                return first;
            auto op2 = db.Names.back();
            db.Names.pop_back();
            auto op1 = db.Names.back();
            db.Names.back() = db.make<BinaryExpr>(op1, op, op2);
            first = t2;
        }
    }
    return first;
}

// <expression> ::= <unary operator-name> <expression>
//              ::= <binary operator-name> <expression> <expression>
//              ::= <ternary operator-name> <expression> <expression> <expression>
//              ::= cl <expression>+ E                                   # call
//              ::= cv <type> <expression>                               # conversion with one argument
//              ::= cv <type> _ <expression>* E                          # conversion with a different number of arguments
//              ::= [gs] nw <expression>* _ <type> E                     # new (expr-list) type
//              ::= [gs] nw <expression>* _ <type> <initializer>         # new (expr-list) type (init)
//              ::= [gs] na <expression>* _ <type> E                     # new[] (expr-list) type
//              ::= [gs] na <expression>* _ <type> <initializer>         # new[] (expr-list) type (init)
//              ::= [gs] dl <expression>                                 # delete expression
//              ::= [gs] da <expression>                                 # delete[] expression
//              ::= pp_ <expression>                                     # prefix ++
//              ::= mm_ <expression>                                     # prefix --
//              ::= ti <type>                                            # typeid (type)
//              ::= te <expression>                                      # typeid (expression)
//              ::= dc <type> <expression>                               # dynamic_cast<type> (expression)
//              ::= sc <type> <expression>                               # static_cast<type> (expression)
//              ::= cc <type> <expression>                               # const_cast<type> (expression)
//              ::= rc <type> <expression>                               # reinterpret_cast<type> (expression)
//              ::= st <type>                                            # sizeof (a type)
//              ::= sz <expression>                                      # sizeof (an expression)
//              ::= at <type>                                            # alignof (a type)
//              ::= az <expression>                                      # alignof (an expression)
//              ::= nx <expression>                                      # noexcept (expression)
//              ::= <template-param>
//              ::= <function-param>
//              ::= dt <expression> <unresolved-name>                    # expr.name
//              ::= pt <expression> <unresolved-name>                    # expr->name
//              ::= ds <expression> <expression>                         # expr.*expr
//              ::= sZ <template-param>                                  # size of a parameter pack
//              ::= sZ <function-param>                                  # size of a function parameter pack
//              ::= sp <expression>                                      # pack expansion
//              ::= tw <expression>                                      # throw expression
//              ::= tr                                                   # throw with no operand (rethrow)
//              ::= <unresolved-name>                                    # f(p), N::f(p), ::f(p),
//                                                                       # freestanding dependent name (e.g., T::x),
//                                                                       # objectless nonstatic member reference
//              ::= <expr-primary>

const char*
parse_expression(const char* first, const char* last, Db& db)
{
    if (last - first >= 2)
    {
        const char* t = first;
        bool parsed_gs = false;
        if (last - first >= 4 && t[0] == 'g' && t[1] == 's')
        {
            t += 2;
            parsed_gs = true;
        }
        switch (*t)
        {
        case 'L':
            first = parse_expr_primary(first, last, db);
            break;
        case 'T':
            first = parse_template_param(first, last, db);
            break;
        case 'f':
            first = parse_function_param(first, last, db);
            break;
        case 'a':
            switch (t[1])
            {
            case 'a':
                t = parse_binary_expression(first+2, last, "&&", db);
                if (t != first+2)
                    first = t;
                break;
            case 'd':
                t = parse_prefix_expression(first+2, last, "&", db);
                if (t != first+2)
                    first = t;
                break;
            case 'n':
                t = parse_binary_expression(first+2, last, "&", db);
                if (t != first+2)
                    first = t;
                break;
            case 'N':
                t = parse_binary_expression(first+2, last, "&=", db);
                if (t != first+2)
                    first = t;
                break;
            case 'S':
                t = parse_binary_expression(first+2, last, "=", db);
                if (t != first+2)
                    first = t;
                break;
            case 't':
                first = parse_alignof_type(first, last, db);
                break;
            case 'z':
                first = parse_alignof_expr(first, last, db);
                break;
            }
            break;
        case 'c':
            switch (t[1])
            {
            case 'c':
                first = parse_const_cast_expr(first, last, db);
                break;
            case 'l':
                first = parse_call_expr(first, last, db);
                break;
            case 'm':
                t = parse_binary_expression(first+2, last, ",", db);
                if (t != first+2)
                    first = t;
                break;
            case 'o':
                t = parse_prefix_expression(first+2, last, "~", db);
                if (t != first+2)
                    first = t;
                break;
            case 'v':
                first = parse_conversion_expr(first, last, db);
                break;
            }
            break;
        case 'd':
            switch (t[1])
            {
            case 'a':
                {
                    const char* t1 = parse_expression(t+2, last, db);
                    if (t1 != t+2)
                    {
                        if (db.Names.empty())
                            return first;
                        db.Names.back() = db.make<DeleteExpr>(
                            db.Names.back(), parsed_gs, /*is_array=*/true);
                        first = t1;
                    }
                }
                break;
            case 'c':
                first = parse_dynamic_cast_expr(first, last, db);
                break;
            case 'e':
                t = parse_prefix_expression(first+2, last, "*", db);
                if (t != first+2)
                    first = t;
                break;
            case 'l':
                {
                    const char* t1 = parse_expression(t+2, last, db);
                    if (t1 != t+2)
                    {
                        if (db.Names.empty())
                            return first;
                        db.Names.back() = db.make<DeleteExpr>(
                            db.Names.back(), parsed_gs, /*is_array=*/false);
                        first = t1;
                    }
                }
                break;
            case 'n':
                return parse_unresolved_name(first, last, db);
            case 's':
                first = parse_dot_star_expr(first, last, db);
                break;
            case 't':
                first = parse_dot_expr(first, last, db);
                break;
            case 'v':
                t = parse_binary_expression(first+2, last, "/", db);
                if (t != first+2)
                    first = t;
                break;
            case 'V':
                t = parse_binary_expression(first+2, last, "/=", db);
                if (t != first+2)
                    first = t;
                break;
            }
            break;
        case 'e':
            switch (t[1])
            {
            case 'o':
                t = parse_binary_expression(first+2, last, "^", db);
                if (t != first+2)
                    first = t;
                break;
            case 'O':
                t = parse_binary_expression(first+2, last, "^=", db);
                if (t != first+2)
                    first = t;
                break;
            case 'q':
                t = parse_binary_expression(first+2, last, "==", db);
                if (t != first+2)
                    first = t;
                break;
            }
            break;
        case 'g':
            switch (t[1])
            {
            case 'e':
                t = parse_binary_expression(first+2, last, ">=", db);
                if (t != first+2)
                    first = t;
                break;
            case 't':
                t = parse_binary_expression(first+2, last, ">", db);
                if (t != first+2)
                    first = t;
                break;
            }
            break;
        case 'i':
            if (t[1] == 'x')
            {
                const char* t1 = parse_expression(first+2, last, db);
                if (t1 != first+2)
                {
                    const char* t2 = parse_expression(t1, last, db);
                    if (t2 != t1)
                    {
                        if (db.Names.size() < 2)
                            return first;
                        auto op2 = db.Names.back();
                        db.Names.pop_back();
                        auto op1 = db.Names.back();
                        db.Names.back() =
                            db.make<ArraySubscriptExpr>(op1, op2);
                        first = t2;
                    }
                    else if (!db.Names.empty())
                        db.Names.pop_back();
                }
            }
            break;
        case 'l':
            switch (t[1])
            {
            case 'e':
                t = parse_binary_expression(first+2, last, "<=", db);
                if (t != first+2)
                    first = t;
                break;
            case 's':
                t = parse_binary_expression(first+2, last, "<<", db);
                if (t != first+2)
                    first = t;
                break;
            case 'S':
                t = parse_binary_expression(first+2, last, "<<=", db);
                if (t != first+2)
                    first = t;
                break;
            case 't':
                t = parse_binary_expression(first+2, last, "<", db);
                if (t != first+2)
                    first = t;
                break;
            }
            break;
        case 'm':
            switch (t[1])
            {
            case 'i':
                t = parse_binary_expression(first+2, last, "-", db);
                if (t != first+2)
                    first = t;
                break;
            case 'I':
                t = parse_binary_expression(first+2, last, "-=", db);
                if (t != first+2)
                    first = t;
                break;
            case 'l':
                t = parse_binary_expression(first+2, last, "*", db);
                if (t != first+2)
                    first = t;
                break;
            case 'L':
                t = parse_binary_expression(first+2, last, "*=", db);
                if (t != first+2)
                    first = t;
                break;
            case 'm':
                if (first+2 != last && first[2] == '_')
                {
                    t = parse_prefix_expression(first+3, last, "--", db);
                    if (t != first+3)
                        first = t;
                }
                else
                {
                    const char* t1 = parse_expression(first+2, last, db);
                    if (t1 != first+2)
                    {
                        if (db.Names.empty())
                            return first;
                        db.Names.back() =
                            db.make<PostfixExpr>(db.Names.back(), "--");
                        first = t1;
                    }
                }
                break;
            }
            break;
        case 'n':
            switch (t[1])
            {
            case 'a':
            case 'w':
                first = parse_new_expr(first, last, db);
                break;
            case 'e':
                t = parse_binary_expression(first+2, last, "!=", db);
                if (t != first+2)
                    first = t;
                break;
            case 'g':
                t = parse_prefix_expression(first+2, last, "-", db);
                if (t != first+2)
                    first = t;
                break;
            case 't':
                t = parse_prefix_expression(first+2, last, "!", db);
                if (t != first+2)
                    first = t;
                break;
            case 'x':
                t = parse_noexcept_expression(first+2, last, db);
                if (t != first+2)
                    first = t;
                break;
            }
            break;
        case 'o':
            switch (t[1])
            {
            case 'n':
                return parse_unresolved_name(first, last, db);
            case 'o':
                t = parse_binary_expression(first+2, last, "||", db);
                if (t != first+2)
                    first = t;
                break;
            case 'r':
                t = parse_binary_expression(first+2, last, "|", db);
                if (t != first+2)
                    first = t;
                break;
            case 'R':
                t = parse_binary_expression(first+2, last, "|=", db);
                if (t != first+2)
                    first = t;
                break;
            }
            break;
        case 'p':
            switch (t[1])
            {
            case 'm':
                t = parse_binary_expression(first+2, last, "->*", db);
                if (t != first+2)
                    first = t;
                break;
            case 'l':
                t = parse_binary_expression(first+2, last, "+", db);
                if (t != first+2)
                    first = t;
                break;
            case 'L':
                t = parse_binary_expression(first+2, last, "+=", db);
                if (t != first+2)
                    first = t;
                break;
            case 'p':
                if (first+2 != last && first[2] == '_')
                {
                    t = parse_prefix_expression(first+3, last, "++", db);
                    if (t != first+3)
                        first = t;
                }
                else
                {
                    const char* t1 = parse_expression(first+2, last, db);
                    if (t1 != first+2)
                    {
                        if (db.Names.empty())
                            return first;
                        db.Names.back() =
                            db.make<PostfixExpr>(db.Names.back(), "++");
                        first = t1;
                    }
                }
                break;
            case 's':
                t = parse_prefix_expression(first+2, last, "+", db);
                if (t != first+2)
                    first = t;
                break;
            case 't':
                first = parse_arrow_expr(first, last, db);
                break;
            }
            break;
        case 'q':
            if (t[1] == 'u')
            {
                const char* t1 = parse_expression(first+2, last, db);
                if (t1 != first+2)
                {
                    const char* t2 = parse_expression(t1, last, db);
                    if (t2 != t1)
                    {
                        const char* t3 = parse_expression(t2, last, db);
                        if (t3 != t2)
                        {
                            if (db.Names.size() < 3)
                                return first;
                            auto op3 = db.Names.back();
                            db.Names.pop_back();
                            auto op2 = db.Names.back();
                            db.Names.pop_back();
                            auto op1 = db.Names.back();
                            db.Names.back() =
                                db.make<ConditionalExpr>(op1, op2, op3);
                            first = t3;
                        }
                        else
                        {
                            if (db.Names.size() < 2)
                              return first;
                            db.Names.pop_back();
                            db.Names.pop_back();
                        }
                    }
                    else if (!db.Names.empty())
                        db.Names.pop_back();
                }
            }
            break;
        case 'r':
            switch (t[1])
            {
            case 'c':
                first = parse_reinterpret_cast_expr(first, last, db);
                break;
            case 'm':
                t = parse_binary_expression(first+2, last, "%", db);
                if (t != first+2)
                    first = t;
                break;
            case 'M':
                t = parse_binary_expression(first+2, last, "%=", db);
                if (t != first+2)
                    first = t;
                break;
            case 's':
                t = parse_binary_expression(first+2, last, ">>", db);
                if (t != first+2)
                    first = t;
                break;
            case 'S':
                t = parse_binary_expression(first+2, last, ">>=", db);
                if (t != first+2)
                    first = t;
                break;
            }
            break;
        case 's':
            switch (t[1])
            {
            case 'c':
                first = parse_static_cast_expr(first, last, db);
                break;
            case 'p':
                first = parse_pack_expansion(first, last, db);
                break;
            case 'r':
                return parse_unresolved_name(first, last, db);
            case 't':
                first = parse_sizeof_type_expr(first, last, db);
                break;
            case 'z':
                first = parse_sizeof_expr_expr(first, last, db);
                break;
            case 'Z':
                if (last - t >= 3)
                {
                    switch (t[2])
                    {
                    case 'T':
                        first = parse_sizeof_param_pack_expr(first, last, db);
                        break;
                    case 'f':
                        first = parse_sizeof_function_param_pack_expr(first, last, db);
                        break;
                    }
                }
                break;
            }
            break;
        case 't':
            switch (t[1])
            {
            case 'e':
            case 'i':
                first = parse_typeid_expr(first, last, db);
                break;
            case 'r':
                db.Names.push_back(db.make<NameType>("throw"));
                first += 2;
                break;
            case 'w':
                first = parse_throw_expr(first, last, db);
                break;
            }
            break;
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
            return parse_unresolved_name(first, last, db);
        }
    }
    return first;
}

// <template-arg> ::= <type>                                             # type or template
//                ::= X <expression> E                                   # expression
//                ::= <expr-primary>                                     # simple expressions
//                ::= J <template-arg>* E                                # argument pack
//                ::= LZ <encoding> E                                    # extension

const char*
parse_template_arg(const char* first, const char* last, Db& db)
{
    if (first != last)
    {
        const char* t;
        switch (*first)
        {
        case 'X':
            t = parse_expression(first+1, last, db);
            if (t != first+1)
            {
                if (t != last && *t == 'E')
                    first = t+1;
            }
            break;
        case 'J':
            t = first+1;
            if (t == last)
                return first;
            while (*t != 'E')
            {
                const char* t1 = parse_template_arg(t, last, db);
                if (t1 == t)
                    return first;
                t = t1;
            }
            first = t+1;
            break;
        case 'L':
            // <expr-primary> or LZ <encoding> E
            if (first+1 != last && first[1] == 'Z')
            {
                t = parse_encoding(first+2, last, db);
                if (t != first+2 && t != last && *t == 'E')
                    first = t+1;
            }
            else
                first = parse_expr_primary(first, last, db);
            break;
        default:
            // <type>
            first = parse_type(first, last, db);
            break;
        }
    }
    return first;
}

// <template-args> ::= I <template-arg>* E
//     extension, the abi says <template-arg>+

const char*
parse_template_args(const char* first, const char* last, Db& db)
{
    if (last - first >= 2 && *first == 'I')
    {
        if (db.TagTemplates)
            db.TemplateParams.clear();
        const char* t = first+1;
        size_t begin_idx = db.Names.size();
        while (*t != 'E')
        {
            if (db.TagTemplates)
            {
                auto TmpParams = std::move(db.TemplateParams);
                size_t k0 = db.Names.size();
                const char* t1 = parse_template_arg(t, last, db);
                size_t k1 = db.Names.size();
                db.TemplateParams = std::move(TmpParams);

                if (t1 == t || t1 == last || k0 > k1)
                    return first;
                db.TemplateParams.pushPack();
                for (size_t k = k0; k < k1; ++k)
                    db.TemplateParams.pushSubstitutionIntoPack(db.Names[k]);
                t = t1;
                continue;
            }
            size_t k0 = db.Names.size();
            const char* t1 = parse_template_arg(t, last, db);
            size_t k1 = db.Names.size();
            if (t1 == t || t1 == last || k0 > k1)
              return first;
            t = t1;
        }
        if (begin_idx > db.Names.size())
            return first;
        first = t + 1;
        TemplateParams* tp = db.make<TemplateParams>(
            db.popTrailingNodeArray(begin_idx));
        db.Names.push_back(tp);
    }
    return first;
}

// <nested-name> ::= N [<CV-Qualifiers>] [<ref-qualifier>] <prefix> <unqualified-name> E
//               ::= N [<CV-Qualifiers>] [<ref-qualifier>] <template-prefix> <template-args> E
// 
// <prefix> ::= <prefix> <unqualified-name>
//          ::= <template-prefix> <template-args>
//          ::= <template-param>
//          ::= <decltype>
//          ::= # empty
//          ::= <substitution>
//          ::= <prefix> <data-member-prefix>
//  extension ::= L
// 
// <template-prefix> ::= <prefix> <template unqualified-name>
//                   ::= <template-param>
//                   ::= <substitution>

const char*
parse_nested_name(const char* first, const char* last, Db& db,
                  bool* ends_with_template_args)
{
    if (first != last && *first == 'N')
    {
        Qualifiers cv;
        const char* t0 = parse_cv_qualifiers(first+1, last, cv);
        if (t0 == last)
            return first;
        db.RefQuals = FrefQualNone;
        if (*t0 == 'R')
        {
            db.RefQuals = FrefQualLValue;
            ++t0;
        }
        else if (*t0 == 'O')
        {
            db.RefQuals = FrefQualRValue;
            ++t0;
        }
        db.Names.push_back(db.make<EmptyName>());
        if (last - t0 >= 2 && t0[0] == 'S' && t0[1] == 't')
        {
            t0 += 2;
            db.Names.back() = db.make<NameType>("std");
        }
        if (t0 == last)
            return first;
        bool pop_subs = false;
        bool component_ends_with_template_args = false;
        while (*t0 != 'E')
        {
            component_ends_with_template_args = false;
            const char* t1;
            switch (*t0)
            {
            case 'S':
                if (t0 + 1 != last && t0[1] == 't')
                    goto do_parse_unqualified_name;
                t1 = parse_substitution(t0, last, db);
                if (t1 != t0 && t1 != last)
                {
                    if (db.Names.size() < 2)
                        return first;
                    auto name = db.Names.back();
                    db.Names.pop_back();
                    if (db.Names.back()->K != Node::KEmptyName)
                    {
                        db.Names.back() = db.make<QualifiedName>(
                            db.Names.back(), name);
                        db.Subs.pushSubstitution(db.Names.back());
                    }
                    else
                        db.Names.back() = name;
                    pop_subs = true;
                    t0 = t1;
                }
                else
                    return first;
                break;
            case 'T':
                t1 = parse_template_param(t0, last, db);
                if (t1 != t0 && t1 != last)
                {
                    if (db.Names.size() < 2)
                        return first;
                    auto name = db.Names.back();
                    db.Names.pop_back();
                    if (db.Names.back()->K != Node::KEmptyName)
                        db.Names.back() =
                            db.make<QualifiedName>(db.Names.back(), name);
                    else
                        db.Names.back() = name;
                    db.Subs.pushSubstitution(db.Names.back());
                    pop_subs = true;
                    t0 = t1;
                }
                else
                    return first;
                break;
            case 'D':
                if (t0 + 1 != last && t0[1] != 't' && t0[1] != 'T')
                    goto do_parse_unqualified_name;
                t1 = parse_decltype(t0, last, db);
                if (t1 != t0 && t1 != last)
                {
                    if (db.Names.size() < 2)
                        return first;
                    auto name = db.Names.back();
                    db.Names.pop_back();
                    if (db.Names.back()->K != Node::KEmptyName)
                        db.Names.back() =
                            db.make<QualifiedName>(db.Names.back(), name);
                    else
                        db.Names.back() = name;
                    db.Subs.pushSubstitution(db.Names.back());
                    pop_subs = true;
                    t0 = t1;
                }
                else
                    return first;
                break;
            case 'I':
                t1 = parse_template_args(t0, last, db);
                if (t1 != t0 && t1 != last)
                {
                    if (db.Names.size() < 2)
                        return first;
                    auto name = db.Names.back();
                    db.Names.pop_back();
                    db.Names.back() = db.make<NameWithTemplateArgs>(
                        db.Names.back(), name);
                    db.Subs.pushSubstitution(db.Names.back());
                    t0 = t1;
                    component_ends_with_template_args = true;
                }
                else
                    return first;
                break;
            case 'L':
                if (++t0 == last)
                    return first;
                break;
            default:
            do_parse_unqualified_name:
                t1 = parse_unqualified_name(t0, last, db);
                if (t1 != t0 && t1 != last)
                {
                    if (db.Names.size() < 2)
                        return first;
                    auto name = db.Names.back();
                    db.Names.pop_back();
                    if (db.Names.back()->K != Node::KEmptyName)
                        db.Names.back() =
                            db.make<QualifiedName>(db.Names.back(), name);
                    else
                        db.Names.back() = name;
                    db.Subs.pushSubstitution(db.Names.back());
                    pop_subs = true;
                    t0 = t1;
                }
                else
                    return first;
            }
        }
        first = t0 + 1;
        db.CV = cv;
        if (pop_subs && !db.Subs.empty())
            db.Subs.popPack();
        if (ends_with_template_args)
            *ends_with_template_args = component_ends_with_template_args;
    }
    return first;
}

// <discriminator> := _ <non-negative number>      # when number < 10
//                 := __ <non-negative number> _   # when number >= 10
//  extension      := decimal-digit+               # at the end of string

const char*
parse_discriminator(const char* first, const char* last)
{
    // parse but ignore discriminator
    if (first != last)
    {
        if (*first == '_')
        {
            const char* t1 = first+1;
            if (t1 != last)
            {
                if (std::isdigit(*t1))
                    first = t1+1;
                else if (*t1 == '_')
                {
                    for (++t1; t1 != last && std::isdigit(*t1); ++t1)
                        ;
                    if (t1 != last && *t1 == '_')
                        first = t1 + 1;
                }
            }
        }
        else if (std::isdigit(*first))
        {
            const char* t1 = first+1;
            for (; t1 != last && std::isdigit(*t1); ++t1)
                ;
            if (t1 == last)
                first = last;
        }
    }
    return first;
}

// <local-name> := Z <function encoding> E <entity name> [<discriminator>]
//              := Z <function encoding> E s [<discriminator>]
//              := Z <function encoding> Ed [ <parameter number> ] _ <entity name>

const char*
parse_local_name(const char* first, const char* last, Db& db,
                 bool* ends_with_template_args)
{
    if (first != last && *first == 'Z')
    {
        const char* t = parse_encoding(first+1, last, db);
        if (t != first+1 && t != last && *t == 'E' && ++t != last)
        {
            switch (*t)
            {
            case 's':
                first = parse_discriminator(t+1, last);
                if (db.Names.empty())
                    return first;
                db.Names.back() = db.make<QualifiedName>(
                    db.Names.back(), db.make<NameType>("string literal"));
                break;
            case 'd':
                if (++t != last)
                {
                    const char* t1 = parse_number(t, last);
                    if (t1 != last && *t1 == '_')
                    {
                        t = t1 + 1;
                        t1 = parse_name(t, last, db,
                                        ends_with_template_args);
                        if (t1 != t)
                        {
                            if (db.Names.size() < 2)
                                return first;
                            auto name = db.Names.back();
                            db.Names.pop_back();
                            if (db.Names.empty())
                                return first;
                            db.Names.back() =
                                db.make<QualifiedName>(db.Names.back(), name);
                            first = t1;
                        }
                        else if (!db.Names.empty())
                            db.Names.pop_back();
                    }
                }
                break;
            default:
                {
                    const char* t1 = parse_name(t, last, db,
                                                ends_with_template_args);
                    if (t1 != t)
                    {
                        // parse but ignore discriminator
                        first = parse_discriminator(t1, last);
                        if (db.Names.size() < 2)
                            return first;
                        auto name = db.Names.back();
                        db.Names.pop_back();
                        if (db.Names.empty())
                            return first;
                        db.Names.back() =
                            db.make<QualifiedName>(db.Names.back(), name);
                    }
                    else if (!db.Names.empty())
                        db.Names.pop_back();
                }
                break;
            }
        }
    }
    return first;
}

// <name> ::= <nested-name> // N
//        ::= <local-name> # See Scope Encoding below  // Z
//        ::= <unscoped-template-name> <template-args>
//        ::= <unscoped-name>

// <unscoped-template-name> ::= <unscoped-name>
//                          ::= <substitution>

const char*
parse_name(const char* first, const char* last, Db& db,
           bool* ends_with_template_args)
{
    if (last - first >= 2)
    {
        const char* t0 = first;
        // extension: ignore L here
        if (*t0 == 'L')
            ++t0;
        switch (*t0)
        {
        case 'N':
          {
            const char* t1 = parse_nested_name(t0, last, db,
                                               ends_with_template_args);
            if (t1 != t0)
                first = t1;
            break;
          }
        case 'Z':
          {
            const char* t1 = parse_local_name(t0, last, db,
                                              ends_with_template_args);
            if (t1 != t0)
                first = t1;
            break;
          }
        default:
          {
            const char* t1 = parse_unscoped_name(t0, last, db);
            if (t1 != t0)
            {
                if (t1 != last && *t1 == 'I')  // <unscoped-template-name> <template-args>
                {
                    if (db.Names.empty())
                        return first;
                    db.Subs.pushSubstitution(db.Names.back());
                    t0 = t1;
                    t1 = parse_template_args(t0, last, db);
                    if (t1 != t0)
                    {
                        if (db.Names.size() < 2)
                            return first;
                        auto tmp = db.Names.back();
                        db.Names.pop_back();
                        if (db.Names.empty())
                            return first;
                        db.Names.back() =
                            db.make<NameWithTemplateArgs>(
                                db.Names.back(), tmp);
                        first = t1;
                        if (ends_with_template_args)
                            *ends_with_template_args = true;
                    }
                }
                else   // <unscoped-name>
                    first = t1;
            }
            else
            {   // try <substitution> <template-args>
                t1 = parse_substitution(t0, last, db);
                if (t1 != t0 && t1 != last && *t1 == 'I')
                {
                    t0 = t1;
                    t1 = parse_template_args(t0, last, db);
                    if (t1 != t0)
                    {
                        if (db.Names.size() < 2)
                            return first;
                        auto tmp = db.Names.back();
                        db.Names.pop_back();
                        if (db.Names.empty())
                            return first;
                        db.Names.back() =
                            db.make<NameWithTemplateArgs>(
                                db.Names.back(), tmp);
                        first = t1;
                        if (ends_with_template_args)
                            *ends_with_template_args = true;
                    }
                }
            }
            break;
          }
        }
    }
    return first;
}

// <call-offset> ::= h <nv-offset> _
//               ::= v <v-offset> _
// 
// <nv-offset> ::= <offset number>
//               # non-virtual base override
// 
// <v-offset>  ::= <offset number> _ <virtual offset number>
//               # virtual base override, with vcall offset

const char*
parse_call_offset(const char* first, const char* last)
{
    if (first != last)
    {
        switch (*first)
        {
        case 'h':
            {
            const char* t = parse_number(first + 1, last);
            if (t != first + 1 && t != last && *t == '_')
                first = t + 1;
            }
            break;
        case 'v':
            {
            const char* t = parse_number(first + 1, last);
            if (t != first + 1 && t != last && *t == '_')
            {
                const char* t2 = parse_number(++t, last);
                if (t2 != t && t2 != last && *t2 == '_')
                    first = t2 + 1;
            }
            }
            break;
        }
    }
    return first;
}

// <special-name> ::= TV <type>    # virtual table
//                ::= TT <type>    # VTT structure (construction vtable index)
//                ::= TI <type>    # typeinfo structure
//                ::= TS <type>    # typeinfo name (null-terminated byte string)
//                ::= Tc <call-offset> <call-offset> <base encoding>
//                    # base is the nominal target function of thunk
//                    # first call-offset is 'this' adjustment
//                    # second call-offset is result adjustment
//                ::= T <call-offset> <base encoding>
//                    # base is the nominal target function of thunk
//                ::= GV <object name> # Guard variable for one-time initialization
//                                     # No <type>
//                ::= TW <object name> # Thread-local wrapper
//                ::= TH <object name> # Thread-local initialization
//      extension ::= TC <first type> <number> _ <second type> # construction vtable for second-in-first
//      extension ::= GR <object name> # reference temporary for object

const char*
parse_special_name(const char* first, const char* last, Db& db)
{
    if (last - first > 2)
    {
        const char* t;
        switch (*first)
        {
        case 'T':
            switch (first[1])
            {
            case 'V':
                // TV <type>    # virtual table
                t = parse_type(first+2, last, db);
                if (t != first+2)
                {
                    if (db.Names.empty())
                        return first;
                    db.Names.back() =
                        db.make<SpecialName>("vtable for ", db.Names.back());
                    first = t;
                }
                break;
            case 'T':
                // TT <type>    # VTT structure (construction vtable index)
                t = parse_type(first+2, last, db);
                if (t != first+2)
                {
                    if (db.Names.empty())
                        return first;
                    db.Names.back() =
                        db.make<SpecialName>("VTT for ", db.Names.back());
                    first = t;
                }
                break;
            case 'I':
                // TI <type>    # typeinfo structure
                t = parse_type(first+2, last, db);
                if (t != first+2)
                {
                    if (db.Names.empty())
                        return first;
                    db.Names.back() =
                        db.make<SpecialName>("typeinfo for ", db.Names.back());
                    first = t;
                }
                break;
            case 'S':
                // TS <type>    # typeinfo name (null-terminated byte string)
                t = parse_type(first+2, last, db);
                if (t != first+2)
                {
                    if (db.Names.empty())
                        return first;
                    db.Names.back() =
                        db.make<SpecialName>("typeinfo name for ", db.Names.back());
                    first = t;
                }
                break;
            case 'c':
                // Tc <call-offset> <call-offset> <base encoding>
              {
                const char* t0 = parse_call_offset(first+2, last);
                if (t0 == first+2)
                    break;
                const char* t1 = parse_call_offset(t0, last);
                if (t1 == t0)
                    break;
                t = parse_encoding(t1, last, db);
                if (t != t1)
                {
                    if (db.Names.empty())
                        return first;
                    db.Names.back() =
                        db.make<SpecialName>("covariant return thunk to ",
                                              db.Names.back());
                    first = t;
                }
              }
                break;
            case 'C':
                // extension ::= TC <first type> <number> _ <second type> # construction vtable for second-in-first
                t = parse_type(first+2, last, db);
                if (t != first+2)
                {
                    const char* t0 = parse_number(t, last);
                    if (t0 != t && t0 != last && *t0 == '_')
                    {
                        const char* t1 = parse_type(++t0, last, db);
                        if (t1 != t0)
                        {
                            if (db.Names.size() < 2)
                                return first;
                            auto left = db.Names.back();
                            db.Names.pop_back();
                            if (db.Names.empty())
                                return first;
                            db.Names.back() = db.make<CtorVtableSpecialName>(
                                left, db.Names.back());
                            first = t1;
                        }
                    }
                }
                break;
            case 'W':
                // TW <object name> # Thread-local wrapper
                t = parse_name(first + 2, last, db);
                if (t != first + 2) 
                {
                    if (db.Names.empty())
                        return first;
                    db.Names.back() =
                        db.make<SpecialName>("thread-local wrapper routine for ",
                                              db.Names.back());
                    first = t;
                }
                break;
            case 'H':
                //TH <object name> # Thread-local initialization
                t = parse_name(first + 2, last, db);
                if (t != first + 2) 
                {
                    if (db.Names.empty())
                        return first;
                    db.Names.back() = db.make<SpecialName>(
                        "thread-local initialization routine for ", db.Names.back());
                    first = t;
                }
                break;
            default:
                // T <call-offset> <base encoding>
                {
                const char* t0 = parse_call_offset(first+1, last);
                if (t0 == first+1)
                    break;
                t = parse_encoding(t0, last, db);
                if (t != t0)
                {
                    if (db.Names.empty())
                        return first;
                    if (first[1] == 'v')
                    {
                        db.Names.back() =
                            db.make<SpecialName>("virtual thunk to ",
                                                  db.Names.back());
                        first = t;
                    }
                    else
                    {
                        db.Names.back() =
                            db.make<SpecialName>("non-virtual thunk to ",
                                                  db.Names.back());
                        first = t;
                    }
                }
                }
                break;
            }
            break;
        case 'G':
            switch (first[1])
            {
            case 'V':
                // GV <object name> # Guard variable for one-time initialization
                t = parse_name(first+2, last, db);
                if (t != first+2)
                {
                    if (db.Names.empty())
                        return first;
                    db.Names.back() =
                        db.make<SpecialName>("guard variable for ", db.Names.back());
                    first = t;
                }
                break;
            case 'R':
                // extension ::= GR <object name> # reference temporary for object
                t = parse_name(first+2, last, db);
                if (t != first+2)
                {
                    if (db.Names.empty())
                        return first;
                    db.Names.back() =
                        db.make<SpecialName>("reference temporary for ",
                                              db.Names.back());
                    first = t;
                }
                break;
            }
            break;
        }
    }
    return first;
}

template <class T>
class save_value
{
    T& restore_;
    T original_value_;
public:
    save_value(T& restore)
        : restore_(restore),
          original_value_(restore)
        {}

    ~save_value()
    {
        restore_ = std::move(original_value_);
    }

    save_value(const save_value&) = delete;
    save_value& operator=(const save_value&) = delete;
};

// <encoding> ::= <function name> <bare-function-type>
//            ::= <data name>
//            ::= <special-name>

const char*
parse_encoding(const char* first, const char* last, Db& db)
{
    if (first != last)
    {
        save_value<decltype(db.EncodingDepth)> su(db.EncodingDepth);
        ++db.EncodingDepth;
        save_value<decltype(db.TagTemplates)> sb(db.TagTemplates);
        if (db.EncodingDepth > 1)
            db.TagTemplates = true;
        save_value<decltype(db.ParsedCtorDtorCV)> sp(db.ParsedCtorDtorCV);
        db.ParsedCtorDtorCV = false;
        switch (*first)
        {
        case 'G':
        case 'T':
            first = parse_special_name(first, last, db);
            break;
        default:
          {
            bool ends_with_template_args = false;
            const char* t = parse_name(first, last, db,
                                       &ends_with_template_args);
            if (db.Names.empty())
                return first;
            Qualifiers cv = db.CV;
            FunctionRefQual ref = db.RefQuals;
            if (t != first)
            {
                if (t != last && *t != 'E' && *t != '.')
                {
                    save_value<bool> sb2(db.TagTemplates);
                    db.TagTemplates = false;
                    const char* t2;
                    if (db.Names.empty())
                        return first;
                    if (!db.Names.back())
                        return first;
                    Node* return_type = nullptr;
                    if (!db.ParsedCtorDtorCV && ends_with_template_args)
                    {
                        t2 = parse_type(t, last, db);
                        if (t2 == t)
                            return first;
                        if (db.Names.size() < 1)
                            return first;
                        return_type = db.Names.back();
                        db.Names.pop_back();
                        t = t2;
                    }

                    Node* result = nullptr;

                    if (t != last && *t == 'v')
                    {
                        ++t;
                        if (db.Names.empty())
                            return first;
                        Node* name = db.Names.back();
                        db.Names.pop_back();
                        result = db.make<TopLevelFunctionDecl>(
                            return_type, name, NodeArray());
                    }
                    else
                    {
                        size_t params_begin = db.Names.size();
                        while (true)
                        {
                            t2 = parse_type(t, last, db);
                            if (t2 == t)
                                break;
                            t = t2;
                        }
                        if (db.Names.size() < params_begin)
                            return first;
                        NodeArray params =
                            db.popTrailingNodeArray(params_begin);
                        if (db.Names.empty())
                            return first;
                        Node* name = db.Names.back();
                        db.Names.pop_back();
                        result = db.make<TopLevelFunctionDecl>(
                            return_type, name, params);
                    }
                    if (ref != FrefQualNone)
                        result = db.make<FunctionRefQualType>(result, ref);
                    if (cv != QualNone)
                        result = db.make<FunctionQualType>(result, cv);
                    db.Names.push_back(result);
                    first = t;
                }
                else
                    first = t;
            }
            break;
          }
        }
    }
    return first;
}

// _block_invoke
// _block_invoke<decimal-digit>+
// _block_invoke_<decimal-digit>+

const char*
parse_block_invoke(const char* first, const char* last, Db& db)
{
    if (last - first >= 13)
    {
        // FIXME: strcmp?
        const char test[] = "_block_invoke";
        const char* t = first;
        for (int i = 0; i < 13; ++i, ++t)
        {
            if (*t != test[i])
                return first;
        }
        if (t != last)
        {
            if (*t == '_')
            {
                // must have at least 1 decimal digit
                if (++t == last || !std::isdigit(*t))
                    return first;
                ++t;
            }
            // parse zero or more digits
            while (t != last && isdigit(*t))
                ++t;
        }
        if (db.Names.empty())
            return first;
        db.Names.back() =
            db.make<SpecialName>("invocation function for block in ",
                                  db.Names.back());
        first = t;
    }
    return first;
}

// extension
// <dot-suffix> := .<anything and everything>

const char*
parse_dot_suffix(const char* first, const char* last, Db& db)
{
    if (first != last && *first == '.')
    {
        if (db.Names.empty())
            return first;
        db.Names.back() =
            db.make<DotSuffix>(db.Names.back(), StringView(first, last));
        first = last;
    }
    return first;
}

// <block-involcaton-function> ___Z<encoding>_block_invoke
// <block-involcaton-function> ___Z<encoding>_block_invoke<decimal-digit>+
// <block-involcaton-function> ___Z<encoding>_block_invoke_<decimal-digit>+
// <mangled-name> ::= _Z<encoding>
//                ::= <type>

void
demangle(const char* first, const char* last, Db& db, int& status)
{
    if (first >= last)
    {
        status = invalid_mangled_name;
        return;
    }
    if (*first == '_')
    {
        if (last - first >= 4)
        {
            if (first[1] == 'Z')
            {
                const char* t = parse_encoding(first+2, last, db);
                if (t != first+2 && t != last && *t == '.')
                    t = parse_dot_suffix(t, last, db);
                if (t != last)
                    status = invalid_mangled_name;
            }
            else if (first[1] == '_' && first[2] == '_' && first[3] == 'Z')
            {
                const char* t = parse_encoding(first+4, last, db);
                if (t != first+4 && t != last)
                {
                    const char* t1 = parse_block_invoke(t, last, db);
                    if (t1 != last)
                        status = invalid_mangled_name;
                }
                else
                    status = invalid_mangled_name;
            }
            else
                status = invalid_mangled_name;
        }
        else
            status = invalid_mangled_name;
    }
    else
    {
        const char* t = parse_type(first, last, db);
        if (t != last)
            status = invalid_mangled_name;
    }
    if (status == success && db.Names.empty())
        status = invalid_mangled_name;
}

}  // unnamed namespace

extern "C" _LIBCXXABI_FUNC_VIS char *
__cxa_demangle(const char *mangled_name, char *buf, size_t *n, int *status) {
    if (mangled_name == nullptr || (buf != nullptr && n == nullptr))
    {
        if (status)
            *status = invalid_args;
        return nullptr;
    }

    size_t internal_size = buf != nullptr ? *n : 0;
    Db db;
    int internal_status = success;
    size_t len = std::strlen(mangled_name);
    demangle(mangled_name, mangled_name + len, db,
             internal_status);

    if (internal_status == success && db.FixForwardReferences &&
        !db.TemplateParams.empty())
    {
        db.FixForwardReferences = false;
        db.TagTemplates = false;
        db.Names.clear();
        db.Subs.clear();
        demangle(mangled_name, mangled_name + len, db, internal_status);
        if (db.FixForwardReferences)
            internal_status = invalid_mangled_name;
    }

    if (internal_status == success)
    {
        if (!buf)
        {
            internal_size = 1024;
            buf = static_cast<char*>(std::malloc(internal_size));
        }

        if (buf)
        {
            OutputStream s(buf, internal_size);
            db.Names.back()->print(s);
            s += '\0';
            if (n) *n = s.getCurrentPosition();
            buf = s.getBuffer();
        }
        else
            internal_status = memory_alloc_failure;
    }
    else
        buf = nullptr;
    if (status)
        *status = internal_status;
    return buf;
}

}  // __cxxabiv1
