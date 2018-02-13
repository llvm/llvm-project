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
//   - C++ modules TS
//   - All C++14 and C++17 features

#define _LIBCPP_NO_EXCEPTIONS

#include "__cxxabi_config.h"

#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>
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

#ifndef NDEBUG
#if __has_attribute(noinline) && __has_attribute(used)
#define DUMP_METHOD __attribute__((noinline,used))
#else
#define DUMP_METHOD
#endif
#endif

namespace {

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
  bool empty() const { return First == Last; }
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

  /// If a ParameterPackExpansion (or similar type) is encountered, the offset
  /// into the pack that we're currently printing.
  unsigned CurrentPackIndex = std::numeric_limits<unsigned>::max();

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

  size_t getCurrentPosition() const { return CurrentPosition; };

  char back() const {
    return CurrentPosition ? Buffer[CurrentPosition - 1] : '\0';
  }

  bool empty() const { return CurrentPosition == 0; }

  char *getBuffer() { return Buffer; }
  char *getBufferEnd() { return Buffer + CurrentPosition - 1; }
  size_t getBufferCapacity() { return BufferCapacity; }
};

template <class T>
class SwapAndRestore {
  T &Restore;
  T OriginalValue;
public:
  SwapAndRestore(T& Restore_, T NewVal)
      : Restore(Restore_), OriginalValue(Restore) {
    Restore = std::move(NewVal);
  }
  ~SwapAndRestore() { Restore = std::move(OriginalValue); }

  SwapAndRestore(const SwapAndRestore &) = delete;
  SwapAndRestore &operator=(const SwapAndRestore &) = delete;
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
    KElaboratedTypeSpefType,
    KNameType,
    KAbiTagAttr,
    KObjCProtoName,
    KPointerType,
    KLValueReferenceType,
    KRValueReferenceType,
    KPointerToMemberType,
    KArrayType,
    KFunctionType,
    KFunctionEncoding,
    KFunctionQualType,
    KFunctionRefQualType,
    KLiteralOperator,
    KSpecialName,
    KCtorVtableSpecialName,
    KQualifiedName,
    KEmptyName,
    KVectorType,
    KParameterPack,
    KTemplateArgumentPack,
    KParameterPackExpansion,
    KTemplateArgs,
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
    KBracedExpr,
    KBracedRangeExpr,
  };

  static constexpr unsigned NoParameterPack =
    std::numeric_limits<unsigned>::max();
  unsigned ParameterPackSize = NoParameterPack;

  Kind K;

  /// Three-way bool to track a cached value. Unknown is possible if this node
  /// has an unexpanded parameter pack below it that may affect this cache.
  enum class Cache : unsigned char { Yes, No, Unknown, };

  /// Tracks if this node has a component on its right side, in which case we
  /// need to call printRight.
  Cache RHSComponentCache;

  /// Track if this node is a (possibly qualified) array type. This can affect
  /// how we format the output string.
  Cache ArrayCache;

  /// Track if this node is a (possibly qualified) function type. This can
  /// affect how we format the output string.
  Cache FunctionCache;

  Node(Kind K_, unsigned ParameterPackSize_ = NoParameterPack,
       Cache RHSComponentCache_ = Cache::No, Cache ArrayCache_ = Cache::No,
       Cache FunctionCache_ = Cache::No)
      : ParameterPackSize(ParameterPackSize_), K(K_),
        RHSComponentCache(RHSComponentCache_), ArrayCache(ArrayCache_),
        FunctionCache(FunctionCache_) {}

  bool containsUnexpandedParameterPack() const {
    return ParameterPackSize != NoParameterPack;
  }

  bool hasRHSComponent(OutputStream &S) const {
    if (RHSComponentCache != Cache::Unknown)
      return RHSComponentCache == Cache::Yes;
    return hasRHSComponentSlow(S);
  }

  bool hasArray(OutputStream &S) const {
    if (ArrayCache != Cache::Unknown)
      return ArrayCache == Cache::Yes;
    return hasArraySlow(S);
  }

  bool hasFunction(OutputStream &S) const {
    if (FunctionCache != Cache::Unknown)
      return FunctionCache == Cache::Yes;
    return hasFunctionSlow(S);
  }

  Kind getKind() const { return K; }

  virtual bool hasRHSComponentSlow(OutputStream &) const { return false; }
  virtual bool hasArraySlow(OutputStream &) const { return false; }
  virtual bool hasFunctionSlow(OutputStream &) const { return false; }

  /// If this node is a pack expansion that expands to 0 elements. This can have
  /// an effect on how we should format the output.
  bool isEmptyPackExpansion() const;

  void print(OutputStream &S) const {
    printLeft(S);
    if (RHSComponentCache != Cache::No)
      printRight(S);
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

#ifndef NDEBUG
  DUMP_METHOD void dump() const {
    char *Buffer = static_cast<char*>(std::malloc(1024));
    OutputStream S(Buffer, 1024);
    print(S);
    S += '\0';
    printf("Symbol dump for %p: %s\n", (const void*)this, S.getBuffer());
    std::free(S.getBuffer());
  }
#endif
};

class NodeArray {
  Node **Elements;
  size_t NumElements;

public:
  NodeArray() : Elements(nullptr), NumElements(0) {}
  NodeArray(Node **Elements_, size_t NumElements_)
      : Elements(Elements_), NumElements(NumElements_) {}

  bool empty() const { return NumElements == 0; }
  size_t size() const { return NumElements; }

  Node **begin() const { return Elements; }
  Node **end() const { return Elements + NumElements; }

  Node *operator[](size_t Idx) const { return Elements[Idx]; }

  void printWithComma(OutputStream &S) const {
    bool FirstElement = true;
    for (size_t Idx = 0; Idx != NumElements; ++Idx) {
      if (Elements[Idx]->isEmptyPackExpansion())
        continue;
      if (!FirstElement)
        S += ", ";
      FirstElement = false;
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
  const Node *Ty;
  StringView Ext;

public:
  VendorExtQualType(Node *Ty_, StringView Ext_)
      : Node(KVendorExtQualType, Ty_->ParameterPackSize),
        Ty(Ty_), Ext(Ext_) {}

  void printLeft(OutputStream &S) const override {
    Ty->print(S);
    S += " ";
    S += Ext;
  }
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
      : Node(KQualType, Child_->ParameterPackSize, Child_->RHSComponentCache,
             Child_->ArrayCache, Child_->FunctionCache),
        Quals(Quals_), Child(Child_) {}

  bool hasRHSComponentSlow(OutputStream &S) const override {
    return Child->hasRHSComponent(S);
  }
  bool hasArraySlow(OutputStream &S) const override {
    return Child->hasArray(S);
  }
  bool hasFunctionSlow(OutputStream &S) const override {
    return Child->hasFunction(S);
  }

  void printLeft(OutputStream &S) const override {
    Child->printLeft(S);
    printQuals(S);
  }

  void printRight(OutputStream &S) const override { Child->printRight(S); }
};

class ConversionOperatorType final : public Node {
  const Node *Ty;

public:
  ConversionOperatorType(Node *Ty_)
      : Node(KConversionOperatorType, Ty_->ParameterPackSize), Ty(Ty_) {}

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
      : Node(KPostfixQualifiedType, Ty_->ParameterPackSize),
        Ty(Ty_), Postfix(Postfix_) {}

  void printLeft(OutputStream &s) const override {
    Ty->printLeft(s);
    s += Postfix;
  }
};

class NameType final : public Node {
  const StringView Name;

public:
  NameType(StringView Name_) : Node(KNameType), Name(Name_) {}

  StringView getName() const { return Name; }
  StringView getBaseName() const override { return Name; }

  void printLeft(OutputStream &s) const override { s += Name; }
};

class ElaboratedTypeSpefType : public Node {
  StringView Kind;
  Node *Child;
public:
  ElaboratedTypeSpefType(StringView Kind_, Node *Child_)
      : Node(KElaboratedTypeSpefType), Kind(Kind_), Child(Child_) {
    ParameterPackSize = Child->ParameterPackSize;
  }

  void printLeft(OutputStream &S) const override {
    S += Kind;
    S += ' ';
    Child->print(S);
  }
};

class AbiTagAttr final : public Node {
  const Node* Base;
  StringView Tag;
public:
  AbiTagAttr(const Node* Base_, StringView Tag_)
      : Node(KAbiTagAttr, Base_->ParameterPackSize, Base_->RHSComponentCache,
             Base_->ArrayCache, Base_->FunctionCache),
        Base(Base_), Tag(Tag_) {}

  void printLeft(OutputStream &S) const override {
    Base->printLeft(S);
    S += "[abi:";
    S += Tag;
    S += "]";
  }
};

class ObjCProtoName : public Node {
  Node *Ty;
  StringView Protocol;

  friend class PointerType;

public:
  ObjCProtoName(Node *Ty_, StringView Protocol_)
      : Node(KObjCProtoName), Ty(Ty_), Protocol(Protocol_) {}

  bool isObjCObject() const {
    return Ty->getKind() == KNameType &&
           static_cast<NameType *>(Ty)->getName() == "objc_object";
  }

  void printLeft(OutputStream &S) const override {
    Ty->print(S);
    S += "<";
    S += Protocol;
    S += ">";
  }
};

class PointerType final : public Node {
  const Node *Pointee;

public:
  PointerType(Node *Pointee_)
      : Node(KPointerType, Pointee_->ParameterPackSize,
             Pointee_->RHSComponentCache),
        Pointee(Pointee_) {}

  bool hasRHSComponentSlow(OutputStream &S) const override {
    return Pointee->hasRHSComponent(S);
  }

  void printLeft(OutputStream &s) const override {
    // We rewrite objc_object<SomeProtocol>* into id<SomeProtocol>.
    if (Pointee->getKind() != KObjCProtoName ||
        !static_cast<const ObjCProtoName *>(Pointee)->isObjCObject()) {
      Pointee->printLeft(s);
      if (Pointee->hasArray(s))
        s += " ";
      if (Pointee->hasArray(s) || Pointee->hasFunction(s))
        s += "(";
      s += "*";
    } else {
      const auto *objcProto = static_cast<const ObjCProtoName *>(Pointee);
      s += "id<";
      s += objcProto->Protocol;
      s += ">";
    }
  }

  void printRight(OutputStream &s) const override {
    if (Pointee->getKind() != KObjCProtoName ||
        !static_cast<const ObjCProtoName *>(Pointee)->isObjCObject()) {
      if (Pointee->hasArray(s) || Pointee->hasFunction(s))
        s += ")";
      Pointee->printRight(s);
    }
  }
};

class LValueReferenceType final : public Node {
  const Node *Pointee;

public:
  LValueReferenceType(Node *Pointee_)
      : Node(KLValueReferenceType, Pointee_->ParameterPackSize,
             Pointee_->RHSComponentCache),
        Pointee(Pointee_) {}

  bool hasRHSComponentSlow(OutputStream &S) const override {
    return Pointee->hasRHSComponent(S);
  }

  void printLeft(OutputStream &s) const override {
    Pointee->printLeft(s);
    if (Pointee->hasArray(s))
      s += " ";
    if (Pointee->hasArray(s) || Pointee->hasFunction(s))
      s += "(&";
    else
      s += "&";
  }
  void printRight(OutputStream &s) const override {
    if (Pointee->hasArray(s) || Pointee->hasFunction(s))
      s += ")";
    Pointee->printRight(s);
  }
};

class RValueReferenceType final : public Node {
  const Node *Pointee;

public:
  RValueReferenceType(Node *Pointee_)
      : Node(KRValueReferenceType, Pointee_->ParameterPackSize,
             Pointee_->RHSComponentCache),
        Pointee(Pointee_) {}

  bool hasRHSComponentSlow(OutputStream &S) const override {
    return Pointee->hasRHSComponent(S);
  }

  void printLeft(OutputStream &s) const override {
    Pointee->printLeft(s);
    if (Pointee->hasArray(s))
      s += " ";
    if (Pointee->hasArray(s) || Pointee->hasFunction(s))
      s += "(&&";
    else
      s += "&&";
  }

  void printRight(OutputStream &s) const override {
    if (Pointee->hasArray(s) || Pointee->hasFunction(s))
      s += ")";
    Pointee->printRight(s);
  }
};

class PointerToMemberType final : public Node {
  const Node *ClassType;
  const Node *MemberType;

public:
  PointerToMemberType(Node *ClassType_, Node *MemberType_)
      : Node(KPointerToMemberType,
             std::min(MemberType_->ParameterPackSize,
                      ClassType_->ParameterPackSize),
             MemberType_->RHSComponentCache),
        ClassType(ClassType_), MemberType(MemberType_) {}

  bool hasRHSComponentSlow(OutputStream &S) const override {
    return MemberType->hasRHSComponent(S);
  }

  void printLeft(OutputStream &s) const override {
    MemberType->printLeft(s);
    if (MemberType->hasArray(s) || MemberType->hasFunction(s))
      s += "(";
    else
      s += " ";
    ClassType->print(s);
    s += "::*";
  }

  void printRight(OutputStream &s) const override {
    if (MemberType->hasArray(s) || MemberType->hasFunction(s))
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
      : Node(KArrayType, Base_->ParameterPackSize,
             /*RHSComponentCache=*/Cache::Yes,
             /*ArrayCache=*/Cache::Yes),
        Base(Base_), Dimension(Dimension_) {
    if (Dimension.isNode())
      ParameterPackSize =
          std::min(ParameterPackSize, Dimension.asNode()->ParameterPackSize);
  }

  // Incomplete array type.
  ArrayType(Node *Base_)
      : Node(KArrayType, Base_->ParameterPackSize,
             /*RHSComponentCache=*/Cache::Yes,
             /*ArrayCache=*/Cache::Yes),
        Base(Base_) {}

  bool hasRHSComponentSlow(OutputStream &) const override { return true; }
  bool hasArraySlow(OutputStream &) const override { return true; }

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
      : Node(KFunctionType, Ret_->ParameterPackSize,
             /*RHSComponentCache=*/Cache::Yes, /*ArrayCache=*/Cache::No,
             /*FunctionCache=*/Cache::Yes),
        Ret(Ret_), Params(Params_) {
    for (Node *P : Params)
      ParameterPackSize = std::min(ParameterPackSize, P->ParameterPackSize);
  }

  bool hasRHSComponentSlow(OutputStream &) const override { return true; }
  bool hasFunctionSlow(OutputStream &) const override { return true; }

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
    Params.printWithComma(S);
    S += ")";
    Ret->printRight(S);
  }
};

class FunctionEncoding final : public Node {
  const Node *Ret;
  const Node *Name;
  NodeArray Params;

public:
  FunctionEncoding(Node *Ret_, Node *Name_, NodeArray Params_)
      : Node(KFunctionEncoding, NoParameterPack,
             /*RHSComponentCache=*/Cache::Yes, /*ArrayCache=*/Cache::No,
             /*FunctionCache=*/Cache::Yes),
        Ret(Ret_), Name(Name_), Params(Params_) {
    for (Node *P : Params)
      ParameterPackSize = std::min(ParameterPackSize, P->ParameterPackSize);
    if (Ret)
      ParameterPackSize = std::min(ParameterPackSize, Ret->ParameterPackSize);
  }

  bool hasRHSComponentSlow(OutputStream &) const override { return true; }
  bool hasFunctionSlow(OutputStream &) const override { return true; }

  Node *getName() { return const_cast<Node *>(Name); }

  void printLeft(OutputStream &S) const override {
    if (Ret) {
      Ret->printLeft(S);
      if (!Ret->hasRHSComponent(S))
        S += " ";
    }
    Name->print(S);
  }

  void printRight(OutputStream &S) const override {
    S += "(";
    Params.printWithComma(S);
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
      : Node(KFunctionRefQualType, Fn_->ParameterPackSize,
             /*RHSComponentCache=*/Cache::Yes, /*ArrayCache=*/Cache::No,
             /*FunctionCache=*/Cache::Yes),
        Fn(Fn_), Quals(Quals_) {}

  bool hasFunctionSlow(OutputStream &) const override { return true; }
  bool hasRHSComponentSlow(OutputStream &) const override { return true; }

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
      : QualType(Child_, Quals_) {
    K = KFunctionQualType;
  }

  void printLeft(OutputStream &S) const override { Child->printLeft(S); }

  void printRight(OutputStream &S) const override {
    if (Child->getKind() == KFunctionRefQualType) {
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
  LiteralOperator(Node *OpName_)
      : Node(KLiteralOperator, OpName_->ParameterPackSize), OpName(OpName_) {}

  void printLeft(OutputStream &S) const override {
    S += "operator\"\" ";
    OpName->print(S);
  }
};

class SpecialName final : public Node {
  const StringView Special;
  const Node *Child;

public:
  SpecialName(StringView Special_, Node* Child_)
      : Node(KSpecialName, Child_->ParameterPackSize), Special(Special_),
        Child(Child_) {}

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
      : Node(KCtorVtableSpecialName, std::min(FirstType_->ParameterPackSize,
                                              SecondType_->ParameterPackSize)),
        FirstType(FirstType_), SecondType(SecondType_) {}

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

public:
  QualifiedName(Node* Qualifier_, Node* Name_)
      : Node(KQualifiedName,
             std::min(Qualifier_->ParameterPackSize, Name_->ParameterPackSize)),
        Qualifier(Qualifier_), Name(Name_) {}

  StringView getBaseName() const override { return Name->getBaseName(); }

  void printLeft(OutputStream &S) const override {
    Qualifier->print(S);
    S += "::";
    Name->print(S);
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
        IsPixel(true) {
    if (Dimension.isNode())
      ParameterPackSize = Dimension.asNode()->ParameterPackSize;
  }
  VectorType(Node *BaseType_, NodeOrString Dimension_)
      : Node(KVectorType, BaseType_->ParameterPackSize), BaseType(BaseType_),
        Dimension(Dimension_), IsPixel(false) {
    if (Dimension.isNode())
      ParameterPackSize =
          std::min(ParameterPackSize, Dimension.asNode()->ParameterPackSize);
  }

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

/// An unexpanded parameter pack (either in the expression or type context). If
/// this AST is correct, this node will have a ParameterPackExpansion node above
/// it.
///
/// This node is created when some <template-args> are found that apply to an
/// <encoding>, and is stored in the TemplateParams table. In order for this to
/// appear in the final AST, it has to referenced via a <template-param> (ie,
/// T_).
class ParameterPack final : public Node {
  NodeArray Data;
public:
  ParameterPack(NodeArray Data_)
      : Node(KParameterPack, static_cast<unsigned>(Data_.size())), Data(Data_) {
    ArrayCache = FunctionCache = RHSComponentCache = Cache::Unknown;
    if (std::all_of(Data.begin(), Data.end(), [](Node* P) {
          return P->ArrayCache == Cache::No;
        }))
      ArrayCache = Cache::No;
    if (std::all_of(Data.begin(), Data.end(), [](Node* P) {
          return P->FunctionCache == Cache::No;
        }))
      FunctionCache = Cache::No;
    if (std::all_of(Data.begin(), Data.end(), [](Node* P) {
          return P->RHSComponentCache == Cache::No;
        }))
      RHSComponentCache = Cache::No;
  }

  bool hasRHSComponentSlow(OutputStream &S) const override {
    size_t Idx = S.CurrentPackIndex;
    return Idx < Data.size() && Data[Idx]->hasRHSComponent(S);
  }
  bool hasArraySlow(OutputStream &S) const override {
    size_t Idx = S.CurrentPackIndex;
    return Idx < Data.size() && Data[Idx]->hasArray(S);
  }
  bool hasFunctionSlow(OutputStream &S) const override {
    size_t Idx = S.CurrentPackIndex;
    return Idx < Data.size() && Data[Idx]->hasFunction(S);
  }

  void printLeft(OutputStream &S) const override {
    size_t Idx = S.CurrentPackIndex;
    if (Idx < Data.size())
      Data[Idx]->printLeft(S);
  }
  void printRight(OutputStream &S) const override {
    size_t Idx = S.CurrentPackIndex;
    if (Idx < Data.size())
      Data[Idx]->printRight(S);
  }
};

/// A variadic template argument. This node represents an occurance of
/// J<something>E in some <template-args>. It isn't itself unexpanded, unless
/// one of it's Elements is. The parser inserts a ParameterPack into the
/// TemplateParams table if the <template-args> this pack belongs to apply to an
/// <encoding>.
class TemplateArgumentPack final : public Node {
  NodeArray Elements;
public:
  TemplateArgumentPack(NodeArray Elements_)
      : Node(KTemplateArgumentPack), Elements(Elements_) {
    for (Node *E : Elements)
      ParameterPackSize = std::min(E->ParameterPackSize, ParameterPackSize);
  }

  NodeArray getElements() const { return Elements; }

  void printLeft(OutputStream &S) const override {
    Elements.printWithComma(S);
  }
};

/// A pack expansion. Below this node, there are some unexpanded ParameterPacks
/// which each have Child->ParameterPackSize elements.
class ParameterPackExpansion final : public Node {
  const Node *Child;

public:
  ParameterPackExpansion(Node* Child_)
      : Node(KParameterPackExpansion), Child(Child_) {}

  const Node *getChild() const { return Child; }

  void printLeft(OutputStream &S) const override {
    unsigned PackSize = Child->ParameterPackSize;
    if (PackSize == NoParameterPack) {
      Child->print(S);
      S += "...";
      return;
    }

    SwapAndRestore<unsigned> SavePackIndex(S.CurrentPackIndex, 0);
    for (unsigned I = 0; I != PackSize; ++I) {
      if (I != 0)
        S += ", ";
      S.CurrentPackIndex = I;
      Child->print(S);
    }
  }
};

inline bool Node::isEmptyPackExpansion() const {
  if (getKind() == KParameterPackExpansion) {
    auto *AsPack = static_cast<const ParameterPackExpansion *>(this);
    return AsPack->getChild()->isEmptyPackExpansion();
  }
  if (getKind() == KTemplateArgumentPack) {
    auto *AsTemplateArg = static_cast<const TemplateArgumentPack *>(this);
    for (Node *E : AsTemplateArg->getElements())
      if (!E->isEmptyPackExpansion())
        return false;
    return true;
  }
  return ParameterPackSize == 0;
}

class TemplateArgs final : public Node {
  NodeArray Params;

public:
  TemplateArgs(NodeArray Params_) : Node(KTemplateArgs), Params(Params_) {
    for (Node *P : Params)
      ParameterPackSize = std::min(ParameterPackSize, P->ParameterPackSize);
  }

  NodeArray getParams() { return Params; }

  void printLeft(OutputStream &S) const override {
    S += "<";
    bool FirstElement = true;
    for (size_t Idx = 0, E = Params.size(); Idx != E; ++Idx) {
      if (Params[Idx]->isEmptyPackExpansion())
        continue;
      if (!FirstElement)
        S += ", ";
      FirstElement = false;
      Params[Idx]->print(S);
    }
    if (S.back() == '>')
      S += " ";
    S += ">";
  }
};

class NameWithTemplateArgs final : public Node {
  // name<template_args>
  Node *Name;
  Node *TemplateArgs;

public:
  NameWithTemplateArgs(Node *Name_, Node *TemplateArgs_)
      : Node(KNameWithTemplateArgs, std::min(Name_->ParameterPackSize,
                                             TemplateArgs_->ParameterPackSize)),
        Name(Name_), TemplateArgs(TemplateArgs_) {}

  StringView getBaseName() const override { return Name->getBaseName(); }

  void printLeft(OutputStream &S) const override {
    Name->print(S);
    TemplateArgs->print(S);
  }
};

class GlobalQualifiedName final : public Node {
  Node *Child;

public:
  GlobalQualifiedName(Node* Child_)
      : Node(KGlobalQualifiedName, Child_->ParameterPackSize), Child(Child_) {}

  StringView getBaseName() const override { return Child->getBaseName(); }

  void printLeft(OutputStream &S) const override {
    S += "::";
    Child->print(S);
  }
};

class StdQualifiedName final : public Node {
  Node *Child;

public:
  StdQualifiedName(Node *Child_)
      : Node(KStdQualifiedName, Child_->ParameterPackSize), Child(Child_) {}

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
      : Node(KCtorDtorName, Basename_->ParameterPackSize),
        Basename(Basename_), IsDtor(IsDtor_) {}

  void printLeft(OutputStream &S) const override {
    if (IsDtor)
      S += "~";
    S += Basename->getBaseName();
  }
};

class DtorName : public Node {
  const Node *Base;

public:
  DtorName(Node *Base_) : Node(KDtorName), Base(Base_) {
    ParameterPackSize = Base->ParameterPackSize;
  }

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
      : Node(KLambdaTypeName), Params(Params_), Count(Count_) {
    for (Node *P : Params)
      ParameterPackSize = std::min(ParameterPackSize, P->ParameterPackSize);
  }

  void printLeft(OutputStream &S) const override {
    S += "\'lambda";
    S += Count;
    S += "\'(";
    Params.printWithComma(S);
    S += ")";
  }
};

// -- Expression Nodes --

struct Expr : public Node {
  Expr(Kind K = KExpr) : Node(K) {}
};

class BinaryExpr : public Expr {
  const Node *LHS;
  const StringView InfixOperator;
  const Node *RHS;

public:
  BinaryExpr(Node *LHS_, StringView InfixOperator_, Node *RHS_)
      : LHS(LHS_), InfixOperator(InfixOperator_), RHS(RHS_) {
    ParameterPackSize =
      std::min(LHS->ParameterPackSize, RHS->ParameterPackSize);
  }

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
  ArraySubscriptExpr(Node *Op1_, Node *Op2_) : Op1(Op1_), Op2(Op2_) {
    ParameterPackSize =
      std::min(Op1->ParameterPackSize, Op2->ParameterPackSize);
  }

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
      : Child(Child_), Operand(Operand_) {
    ParameterPackSize = Child->ParameterPackSize;
  }

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
      : Cond(Cond_), Then(Then_), Else(Else_) {
    ParameterPackSize =
        std::min(Cond->ParameterPackSize,
                 std::min(Then->ParameterPackSize, Else->ParameterPackSize));
  }

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
      : LHS(LHS_), Kind(Kind_), RHS(RHS_) {
    ParameterPackSize =
      std::min(LHS->ParameterPackSize, RHS->ParameterPackSize);
  }

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
      : Prefix(Prefix_), Infix(Infix_), Postfix(Postfix_) {
    ParameterPackSize = Infix->ParameterPackSize;
  }

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
      : CastKind(CastKind_), To(To_), From(From_) {
    ParameterPackSize =
      std::min(To->ParameterPackSize, From->ParameterPackSize);
  }

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
  Node *Pack;

public:
  SizeofParamPackExpr(Node *Pack_) : Pack(Pack_) {}

  void printLeft(OutputStream &S) const override {
    S += "sizeof...(";
    ParameterPackExpansion PPE(Pack);
    PPE.printLeft(S);
    S += ")";
  }
};

class CallExpr : public Expr {
  const Node *Callee;
  NodeArray Args;

public:
  CallExpr(Node *Callee_, NodeArray Args_) : Callee(Callee_), Args(Args_) {
    for (Node *P : Args)
      ParameterPackSize = std::min(ParameterPackSize, P->ParameterPackSize);
    ParameterPackSize = std::min(ParameterPackSize, Callee->ParameterPackSize);
  }

  void printLeft(OutputStream &S) const override {
    Callee->print(S);
    S += "(";
    Args.printWithComma(S);
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
      : ExprList(ExprList_), Type(Type_), InitList(InitList_),
        IsGlobal(IsGlobal_), IsArray(IsArray_) {
    for (Node *E : ExprList)
      ParameterPackSize = std::min(ParameterPackSize, E->ParameterPackSize);
    for (Node *I : InitList)
      ParameterPackSize = std::min(ParameterPackSize, I->ParameterPackSize);
    if (Type)
      ParameterPackSize = std::min(ParameterPackSize, Type->ParameterPackSize);
  }

  void printLeft(OutputStream &S) const override {
    if (IsGlobal)
      S += "::operator ";
    S += "new";
    if (IsArray)
      S += "[]";
    S += ' ';
    if (!ExprList.empty()) {
      S += "(";
      ExprList.printWithComma(S);
      S += ")";
    }
    Type->print(S);
    if (!InitList.empty()) {
      S += "(";
      InitList.printWithComma(S);
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
      : Op(Op_), IsGlobal(IsGlobal_), IsArray(IsArray_) {
    ParameterPackSize = Op->ParameterPackSize;
  }

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
  PrefixExpr(StringView Prefix_, Node *Child_) : Prefix(Prefix_), Child(Child_) {
    ParameterPackSize = Child->ParameterPackSize;
  }

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

class ConversionExpr : public Expr {
  const Node *Type;
  NodeArray Expressions;

public:
  ConversionExpr(const Node *Type_, NodeArray Expressions_)
      : Type(Type_), Expressions(Expressions_) {
    for (Node *E : Expressions)
      ParameterPackSize = std::min(ParameterPackSize, E->ParameterPackSize);
    ParameterPackSize = std::min(ParameterPackSize, Type->ParameterPackSize);
  }

  void printLeft(OutputStream &S) const override {
    S += "(";
    Type->print(S);
    S += ")(";
    Expressions.printWithComma(S);
    S += ")";
  }
};

class InitListExpr : public Expr {
  Node *Ty;
  NodeArray Inits;
public:
  InitListExpr(Node *Ty_, NodeArray Inits_)
      : Ty(Ty_), Inits(Inits_) {
    if (Ty)
      ParameterPackSize = Ty->ParameterPackSize;
    for (Node *I : Inits)
      ParameterPackSize = std::min(I->ParameterPackSize, ParameterPackSize);
  }

  void printLeft(OutputStream &S) const override {
    if (Ty)
      Ty->print(S);
    S += '{';
    Inits.printWithComma(S);
    S += '}';
  }
};

class BracedExpr : public Expr {
  Node *Elem;
  Node *Init;
  bool IsArray;
public:
  BracedExpr(Node *Elem_, Node *Init_, bool IsArray_)
      : Expr(KBracedExpr), Elem(Elem_), Init(Init_), IsArray(IsArray_) {}

  void printLeft(OutputStream &S) const override {
    if (IsArray) {
      S += '[';
      Elem->print(S);
      S += ']';
    } else {
      S += '.';
      Elem->print(S);
    }
    if (Init->getKind() != KBracedExpr && Init->getKind() != KBracedRangeExpr)
      S += " = ";
    Init->print(S);
  }
};

class BracedRangeExpr : public Expr {
  Node *First;
  Node *Last;
  Node *Init;
public:
  BracedRangeExpr(Node *First_, Node *Last_, Node *Init_)
      : Expr(KBracedRangeExpr), First(First_), Last(Last_), Init(Init_) {}

  void printLeft(OutputStream &S) const override {
    S += '[';
    First->print(S);
    S += " ... ";
    Last->print(S);
    S += ']';
    if (Init->getKind() != KBracedExpr && Init->getKind() != KBracedRangeExpr)
      S += " = ";
    Init->print(S);
  }
};

class ThrowExpr : public Expr {
  const Node *Op;

public:
  ThrowExpr(Node *Op_) : Op(Op_) {
    ParameterPackSize = Op->ParameterPackSize;
  }

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
  IntegerCastExpr(Node *Ty_, StringView Integer_) : Ty(Ty_), Integer(Integer_) {
    ParameterPackSize = Ty->ParameterPackSize;
  }

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

struct Db {
  const char *First;
  const char *Last;

  // Name stack, this is used by the parser to hold temporary names that were
  // parsed. The parser colapses multiple names into new nodes to construct
  // the AST. Once the parser is finished, names.size() == 1.
  PODSmallVector<Node *, 32> Names;

  // Substitution table. Itanium supports name substitutions as a means of
  // compression. The string "S42_" refers to the 44nd entry (base-36) in this
  // table.
  PODSmallVector<Node *, 32> Subs;

  // Template parameter table. Like the above, but referenced like "T42_".
  // This has a smaller size compared to Subs and Names because it can be
  // stored on the stack.
  PODSmallVector<Node *, 8> TemplateParams;

  Qualifiers CV = QualNone;
  FunctionRefQual RefQuals = FrefQualNone;
  unsigned EncodingDepth = 0;
  bool ParsedCtorDtorCV = false;
  bool TagTemplates = true;
  bool FixForwardReferences = false;
  bool TryToParseTemplateArgs = true;

  BumpPointerAllocator ASTAllocator;

  template <class T, class... Args> T *make(Args &&... args) {
    return new (ASTAllocator.allocate(sizeof(T)))
        T(std::forward<Args>(args)...);
  }

  template <class It> NodeArray makeNodeArray(It begin, It end) {
    size_t sz = static_cast<size_t>(end - begin);
    void *mem = ASTAllocator.allocate(sizeof(Node *) * sz);
    Node **data = new (mem) Node *[sz];
    std::copy(begin, end, data);
    return NodeArray(data, sz);
  }

  NodeArray popTrailingNodeArray(size_t FromPosition) {
    assert(FromPosition <= Names.size());
    NodeArray res =
        makeNodeArray(Names.begin() + (long)FromPosition, Names.end());
    Names.dropBack(FromPosition);
    return res;
  }

  bool consumeIf(StringView S) {
    if (StringView(First, Last).startsWith(S)) {
      First += S.size();
      return true;
    }
    return false;
  }

  bool consumeIf(char C) {
    if (First != Last && *First == C) {
      ++First;
      return true;
    }
    return false;
  }

  char consume() { return First != Last ? *First++ : '\0'; }

  char look(unsigned Lookahead = 0) {
    if (static_cast<size_t>(Last - First) <= Lookahead)
      return '\0';
    return First[Lookahead];
  }

  size_t numLeft() const { return static_cast<size_t>(Last - First); }

  StringView parseNumber(bool AllowNegative = false);
  Qualifiers parseCVQualifiers();
  bool parsePositiveInteger(size_t *Out);
  StringView parseBareSourceName();

  /// Parse the <expr> production.
  Node *parseExpr();
  Node *parsePrefixExpr(StringView Kind);
  Node *parseBinaryExpr(StringView Kind);
  Node *parseIntegerLiteral(StringView Lit);
  Node *parseExprPrimary();
  template <class Float> Node *parseFloatingLiteral();
  Node *parseFunctionParam();
  Node *parseNewExpr();
  Node *parseConversionExpr();
  Node *parseBracedExpr();

  /// Parse the <type> production.
  Node *parseType();
  Node *parseFunctionType();
  Node *parseVectorType();
  Node *parseDecltype();
  Node *parseArrayType();
  Node *parsePointerToMemberType();
  Node *parseClassEnumType();
  Node *parseQualifiedType(bool &AppliesToFunction);

  // FIXME: remove this when all the parse_* functions have been rewritten.
  template <const char *(*parse_fn)(const char *, const char *, Db &)>
  Node *legacyParse() {
    size_t BeforeType = Names.size();
    const char *OrigFirst = First;
    const char *T = parse_fn(First, Last, *this);
    if (T == OrigFirst || BeforeType + 1 != Names.size())
      return nullptr;
    First = T;
    Node *R = Names.back();
    Names.pop_back();
    return R;
  }
  template <const char *(*parse_fn)(const char *, const char *, Db &, bool *)>
  Node *legacyParse() {
    size_t BeforeType = Names.size();
    const char *OrigFirst = First;
    const char *T = parse_fn(First, Last, *this, nullptr);
    if (T == OrigFirst || BeforeType + 1 != Names.size())
      return nullptr;
    First = T;
    Node *R = Names.back();
    Names.pop_back();
    return R;
  }
};

const char *parse_expression(const char *first, const char *last, Db &db) {
  db.First = first;
  db.Last = last;
  Node *R = db.parseExpr();
  if (R == nullptr)
    return first;
  db.Names.push_back(R);
  return db.First;
}

const char *parse_expr_primary(const char *first, const char *last, Db &db) {
  db.First = first;
  db.Last = last;
  Node *R = db.parseExprPrimary();
  if (R == nullptr)
    return first;
  db.Names.push_back(R);
  return db.First;
}

const char *parse_type(const char *first, const char *last, Db &db) {
  db.First = first;
  db.Last = last;
  Node *R = db.parseType();
  if (R == nullptr)
    return first;
  db.Names.push_back(R);
  return db.First;
}

const char *parse_decltype(const char *first, const char *last, Db &db) {
  db.First = first;
  db.Last = last;
  Node *R = db.parseDecltype();
  if (R == nullptr)
    return first;
  db.Names.push_back(R);
  return db.First;
}

const char *parse_type(const char *first, const char *last, Db &db);
const char *parse_encoding(const char *first, const char *last, Db &db);
const char *parse_name(const char *first, const char *last, Db &db,
                       bool *ends_with_template_args = 0);
const char *parse_template_args(const char *first, const char *last, Db &db);
const char *parse_template_param(const char *, const char *, Db &);
const char *parse_operator_name(const char *first, const char *last, Db &db);
const char *parse_source_name(const char *, const char *, Db &);
const char *parse_unqualified_name(const char *first, const char *last, Db &db);
const char *parse_decltype(const char *first, const char *last, Db &db);
const char *parse_unresolved_name(const char *, const char *, Db &);
const char *parse_substitution(const char *, const char *, Db &);

// <number> ::= [n] <non-negative decimal integer>
StringView Db::parseNumber(bool AllowNegative) {
  const char *Tmp = First;
  if (AllowNegative)
    consumeIf('n');
  if (numLeft() == 0 || !std::isdigit(*First))
    return StringView();
  while (numLeft() != 0 && std::isdigit(*First))
    ++First;
  return StringView(Tmp, First);
}

// <positive length number> ::= [0-9]*
bool Db::parsePositiveInteger(size_t *Out) {
  *Out = 0;
  if (look() < '0' || look() > '9')
    return true;
  while (look() >= '0' && look() <= '9') {
    *Out *= 10;
    *Out += static_cast<size_t>(consume() - '0');
  }
  return false;
}

StringView Db::parseBareSourceName() {
  size_t Int = 0;
  if (parsePositiveInteger(&Int) || numLeft() < Int)
    return StringView();
  StringView R(First, First + Int);
  First += Int;
  return R;
}

// <function-type> ::= F [Y] <bare-function-type> [<ref-qualifier>] E
//
//  <ref-qualifier> ::= R                   # & ref-qualifier
//  <ref-qualifier> ::= O                   # && ref-qualifier
Node *Db::parseFunctionType() {
  if (!consumeIf('F'))
    return nullptr;
  consumeIf('Y'); // extern "C"
  Node *ReturnType = parseType();
  if (ReturnType == nullptr)
    return nullptr;

  FunctionRefQual ReferenceQualifier = FrefQualNone;
  size_t ParamsBegin = Names.size();
  while (true) {
    if (consumeIf('E'))
      break;
    if (consumeIf('v'))
      continue;
    if (consumeIf("RE")) {
      ReferenceQualifier = FrefQualLValue;
      break;
    }
    if (consumeIf("OE")) {
      ReferenceQualifier = FrefQualRValue;
      break;
    }
    Node *T = parseType();
    if (T == nullptr)
      return nullptr;
    Names.push_back(T);
  }

  NodeArray Params = popTrailingNodeArray(ParamsBegin);
  Node *Fn = make<FunctionType>(ReturnType, Params);
  if (ReferenceQualifier != FrefQualNone)
    Fn = make<FunctionRefQualType>(Fn, ReferenceQualifier);
  return Fn;
}

// extension:
// <vector-type>           ::= Dv <positive dimension number> _ <extended element type>
//                         ::= Dv [<dimension expression>] _ <element type>
// <extended element type> ::= <element type>
//                         ::= p # AltiVec vector pixel
Node *Db::parseVectorType() {
  if (!consumeIf("Dv"))
    return nullptr;
  if (look() >= '1' && look() <= '9') {
    StringView DimensionNumber = parseNumber();
    if (!consumeIf('_'))
      return nullptr;
    if (consumeIf('p'))
      return make<VectorType>(DimensionNumber);
    Node *ElemType = parseType();
    if (ElemType == nullptr)
      return nullptr;
    return make<VectorType>(ElemType, DimensionNumber);
  }

  if (!consumeIf('_')) {
    Node *DimExpr = parseExpr();
    if (!DimExpr)
      return nullptr;
    if (!consumeIf('_'))
      return nullptr;
    Node *ElemType = parseType();
    if (!ElemType)
      return nullptr;
    return make<VectorType>(ElemType, DimExpr);
  }
  Node *ElemType = parseType();
  if (!ElemType)
    return nullptr;
  return make<VectorType>(ElemType, StringView());
}

// <decltype>  ::= Dt <expression> E  # decltype of an id-expression or class member access (C++0x)
//             ::= DT <expression> E  # decltype of an expression (C++0x)
Node *Db::parseDecltype() {
  if (!consumeIf('D'))
    return nullptr;
  if (!consumeIf('t') && !consumeIf('T'))
    return nullptr;
  Node *E = parseExpr();
  if (E == nullptr)
    return nullptr;
  if (!consumeIf('E'))
    return nullptr;
  return make<EnclosingExpr>("decltype(", E, ")");
}

// <array-type> ::= A <positive dimension number> _ <element type>
//              ::= A [<dimension expression>] _ <element type>
Node *Db::parseArrayType() {
  if (!consumeIf('A'))
    return nullptr;

  if (std::isdigit(look())) {
    StringView Dimension = parseNumber();
    if (!consumeIf('_'))
      return nullptr;
    Node *Ty = parseType();
    if (Ty == nullptr)
      return nullptr;
    return make<ArrayType>(Ty, Dimension);
  }

  if (!consumeIf('_')) {
    Node *DimExpr = parseExpr();
    if (DimExpr == nullptr)
      return nullptr;
    if (!consumeIf('_'))
      return nullptr;
    Node *ElementType = parseType();
    if (ElementType == nullptr)
      return nullptr;
    return make<ArrayType>(ElementType, DimExpr);
  }

  Node *Ty = parseType();
  if (Ty == nullptr)
    return nullptr;
  return make<ArrayType>(Ty);
}

// <pointer-to-member-type> ::= M <class type> <member type>
Node *Db::parsePointerToMemberType() {
  if (!consumeIf('M'))
    return nullptr;
  Node *ClassType = parseType();
  if (ClassType == nullptr)
    return nullptr;
  Node *MemberType = parseType();
  if (MemberType == nullptr)
    return nullptr;
  return make<PointerToMemberType>(ClassType, MemberType);
}

// <class-enum-type> ::= <name>     # non-dependent type name, dependent type name, or dependent typename-specifier
//                   ::= Ts <name>  # dependent elaborated type specifier using 'struct' or 'class'
//                   ::= Tu <name>  # dependent elaborated type specifier using 'union'
//                   ::= Te <name>  # dependent elaborated type specifier using 'enum'
Node *Db::parseClassEnumType() {
  StringView ElabSpef;
  if (consumeIf("Ts"))
    ElabSpef = "struct";
  else if (consumeIf("Tu"))
    ElabSpef = "union";
  else if (consumeIf("Te"))
    ElabSpef = "enum";

  Node *Name = legacyParse<parse_name>();
  if (Name == nullptr)
    return nullptr;

  if (!ElabSpef.empty())
    return make<ElaboratedTypeSpefType>(ElabSpef, Name);

  return Name;
}

// <qualified-type>     ::= <qualifiers> <type>
// <qualifiers> ::= <extended-qualifier>* <CV-qualifiers>
// <extended-qualifier> ::= U <source-name> [<template-args>] # vendor extended type qualifier
Node *Db::parseQualifiedType(bool &AppliesToFunction) {
  if (consumeIf('U')) {
    StringView Qual = parseBareSourceName();
    if (Qual.empty())
      return nullptr;

    // FIXME parse the optional <template-args> here!

    // extension            ::= U <objc-name> <objc-type>  # objc-type<identifier>
    if (Qual.startsWith("objcproto")) {
      StringView ProtoSourceName = Qual.dropFront(std::strlen("objcproto"));
      StringView Proto;
      {
        SwapAndRestore<const char *> SaveFirst(First, ProtoSourceName.begin()),
                                     SaveLast(Last, ProtoSourceName.end());
        Proto = parseBareSourceName();
      }
      if (Proto.empty())
        return nullptr;
      Node *Child = parseQualifiedType(AppliesToFunction);
      if (Child == nullptr)
        return nullptr;
      return make<ObjCProtoName>(Child, Proto);
    }

    Node *Child = parseQualifiedType(AppliesToFunction);
    if (Child == nullptr)
      return nullptr;
    return make<VendorExtQualType>(Child, Qual);
  }

  Qualifiers Quals = parseCVQualifiers();
  AppliesToFunction = look() == 'F';
  Node *Ty = parseType();
  if (Ty == nullptr)
    return nullptr;
  if (Quals != QualNone) {
    return AppliesToFunction ?
      make<FunctionQualType>(Ty, Quals) : make<QualType>(Ty, Quals);
  }
  return Ty;
}

// <type>      ::= <builtin-type>
//             ::= <qualified-type>
//             ::= <function-type>
//             ::= <class-enum-type>
//             ::= <array-type>
//             ::= <pointer-to-member-type>
//             ::= <template-param>
//             ::= <template-template-param> <template-args>
//             ::= <decltype>
//             ::= P <type>        # pointer
//             ::= R <type>        # l-value reference
//             ::= O <type>        # r-value reference (C++11)
//             ::= C <type>        # complex pair (C99)
//             ::= G <type>        # imaginary (C99)
//             ::= <substitution>  # See Compression below
// extension   ::= U <objc-name> <objc-type>  # objc-type<identifier>
// extension   ::= <vector-type> # <vector-type> starts with Dv
//
// <objc-name> ::= <k0 number> objcproto <k1 number> <identifier>  # k0 = 9 + <number of digits in k1> + k1
// <objc-type> ::= <source-name>  # PU<11+>objcproto 11objc_object<source-name> 11objc_object -> id<source-name>
Node *Db::parseType() {
  Node *Result = nullptr;

  switch (look()) {
  //             ::= <qualified-type>
  case 'r':
  case 'V':
  case 'K':
  case 'U': {
    bool AppliesToFunction = false;
    Result = parseQualifiedType(AppliesToFunction);

    // Itanium C++ ABI 5.1.5.3:
    //   For the purposes of substitution, the CV-qualifiers and ref-qualifier
    //   of a function type are an indivisible part of the type.
    if (AppliesToFunction)
      return Result;
    break;
  }
  // <builtin-type> ::= v    # void
  case 'v':
    ++First;
    return make<NameType>("void");
  //                ::= w    # wchar_t
  case 'w':
    ++First;
    return make<NameType>("wchar_t");
  //                ::= b    # bool
  case 'b':
    ++First;
    return make<NameType>("bool");
  //                ::= c    # char
  case 'c':
    ++First;
    return make<NameType>("char");
  //                ::= a    # signed char
  case 'a':
    ++First;
    return make<NameType>("signed char");
  //                ::= h    # unsigned char
  case 'h':
    ++First;
    return make<NameType>("unsigned char");
  //                ::= s    # short
  case 's':
    ++First;
    return make<NameType>("short");
  //                ::= t    # unsigned short
  case 't':
    ++First;
    return make<NameType>("unsigned short");
  //                ::= i    # int
  case 'i':
    ++First;
    return make<NameType>("int");
  //                ::= j    # unsigned int
  case 'j':
    ++First;
    return make<NameType>("unsigned int");
  //                ::= l    # long
  case 'l':
    ++First;
    return make<NameType>("long");
  //                ::= m    # unsigned long
  case 'm':
    ++First;
    return make<NameType>("unsigned long");
  //                ::= x    # long long, __int64
  case 'x':
    ++First;
    return make<NameType>("long long");
  //                ::= y    # unsigned long long, __int64
  case 'y':
    ++First;
    return make<NameType>("unsigned long long");
  //                ::= n    # __int128
  case 'n':
    ++First;
    return make<NameType>("__int128");
  //                ::= o    # unsigned __int128
  case 'o':
    ++First;
    return make<NameType>("unsigned __int128");
  //                ::= f    # float
  case 'f':
    ++First;
    return make<NameType>("float");
  //                ::= d    # double
  case 'd':
    ++First;
    return make<NameType>("double");
  //                ::= e    # long double, __float80
  case 'e':
    ++First;
    return make<NameType>("long double");
  //                ::= g    # __float128
  case 'g':
    ++First;
    return make<NameType>("__float128");
  //                ::= z    # ellipsis
  case 'z':
    ++First;
    return make<NameType>("...");

  // <builtin-type> ::= u <source-name>    # vendor extended type
  case 'u': {
    ++First;
    StringView Res = parseBareSourceName();
    if (Res.empty())
      return nullptr;
    return make<NameType>(Res);
  }
  case 'D':
    switch (look(1)) {
    //                ::= Dd   # IEEE 754r decimal floating point (64 bits)
    case 'd':
      First += 2;
      return make<NameType>("decimal64");
    //                ::= De   # IEEE 754r decimal floating point (128 bits)
    case 'e':
      First += 2;
      return make<NameType>("decimal128");
    //                ::= Df   # IEEE 754r decimal floating point (32 bits)
    case 'f':
      First += 2;
      return make<NameType>("decimal32");
    //                ::= Dh   # IEEE 754r half-precision floating point (16 bits)
    case 'h':
      First += 2;
      return make<NameType>("decimal16");
    //                ::= Di   # char32_t
    case 'i':
      First += 2;
      return make<NameType>("char32_t");
    //                ::= Ds   # char16_t
    case 's':
      First += 2;
      return make<NameType>("char16_t");
    //                ::= Da   # auto (in dependent new-expressions)
    case 'a':
      First += 2;
      return make<NameType>("auto");
    //                ::= Dc   # decltype(auto)
    case 'c':
      First += 2;
      return make<NameType>("decltype(auto)");
    //                ::= Dn   # std::nullptr_t (i.e., decltype(nullptr))
    case 'n':
      First += 2;
      return make<NameType>("std::nullptr_t");

    //             ::= <decltype>
    case 't':
    case 'T': {
      Result = parseDecltype();
      break;
    }
    // extension   ::= <vector-type> # <vector-type> starts with Dv
    case 'v': {
      Result = parseVectorType();
      break;
    }
    //           ::= Dp <type>       # pack expansion (C++0x)
    case 'p': {
      First += 2;
      Node *Child = parseType();
      if (!Child)
        return nullptr;
      Result = make<ParameterPackExpansion>(Child);
      break;
    }
    }
    break;
  //             ::= <function-type>
  case 'F': {
    Result = parseFunctionType();
    break;
  }
  //             ::= <array-type>
  case 'A': {
    Result = parseArrayType();
    break;
  }
  //             ::= <pointer-to-member-type>
  case 'M': {
    Result = parsePointerToMemberType();
    break;
  }
  //             ::= <template-param>
  case 'T': {
    // This could be an elaborate type specifier on a <class-enum-type>.
    if (look(1) == 's' || look(1) == 'u' || look(1) == 'e') {
      Result = parseClassEnumType();
      break;
    }

    Result = legacyParse<parse_template_param>();
    if (Result == nullptr)
      return nullptr;

    // Result could be either of:
    //   <type>        ::= <template-param>
    //   <type>        ::= <template-template-param> <template-args>
    //
    //   <template-template-param> ::= <template-param>
    //                             ::= <substitution>
    //
    // If this is followed by some <template-args>, and we're permitted to
    // parse them, take the second production.

    if (TryToParseTemplateArgs && look() == 'I') {
      Node *TA = legacyParse<parse_template_args>();
      if (TA == nullptr)
        return nullptr;
      Result = make<NameWithTemplateArgs>(Result, TA);
    }
    break;
  }
  //             ::= P <type>        # pointer
  case 'P': {
    ++First;
    Node *Ptr = parseType();
    if (Ptr == nullptr)
      return nullptr;
    Result = make<PointerType>(Ptr);
    break;
  }
  //             ::= R <type>        # l-value reference
  case 'R': {
    ++First;
    Node *Ref = parseType();
    if (Ref == nullptr)
      return nullptr;
    Result = make<LValueReferenceType>(Ref);
    break;
  }
  //             ::= O <type>        # r-value reference (C++11)
  case 'O': {
    ++First;
    Node *Ref = parseType();
    if (Ref == nullptr)
      return nullptr;
    Result = make<RValueReferenceType>(Ref);
    break;
  }
  //             ::= C <type>        # complex pair (C99)
  case 'C': {
    ++First;
    Node *P = parseType();
    if (P == nullptr)
      return nullptr;
    Result = make<PostfixQualifiedType>(P, " complex");
    break;
  }
  //             ::= G <type>        # imaginary (C99)
  case 'G': {
    ++First;
    Node *P = parseType();
    if (P == nullptr)
      return P;
    Result = make<PostfixQualifiedType>(P, " imaginary");
    break;
  }
  //             ::= <substitution>  # See Compression below
  case 'S': {
    if (look(1) && look(1) != 't') {
      Node *Sub = legacyParse<parse_substitution>();
      if (Sub == nullptr)
        return nullptr;

      // Sub could be either of:
      //   <type>        ::= <substitution>
      //   <type>        ::= <template-template-param> <template-args>
      //
      //   <template-template-param> ::= <template-param>
      //                             ::= <substitution>
      //
      // If this is followed by some <template-args>, and we're permitted to
      // parse them, take the second production.

      if (TryToParseTemplateArgs && look() == 'I') {
        Node *TA = legacyParse<parse_template_args>();
        if (TA == nullptr)
          return nullptr;
        Result = make<NameWithTemplateArgs>(Sub, TA);
        break;
      }

      // If all we parsed was a substitution, don't re-insert into the
      // substitution table.
      return Sub;
    }
    _LIBCPP_FALLTHROUGH();
  }
  //        ::= <class-enum-type>
  default: {
    Result = parseClassEnumType();
    break;
  }
  }

  // If we parsed a type, insert it into the substitution table. Note that all
  // <builtin-type>s and <substitution>s have already bailed out, because they
  // don't get substitutions.
  if (Result != nullptr)
    Subs.push_back(Result);
  return Result;
}

Node *Db::parsePrefixExpr(StringView Kind) {
  Node *E = parseExpr();
  if (E == nullptr)
    return nullptr;
  return make<PrefixExpr>(Kind, E);
}

Node *Db::parseBinaryExpr(StringView Kind) {
  Node *LHS = parseExpr();
  if (LHS == nullptr)
    return nullptr;
  Node *RHS = parseExpr();
  if (RHS == nullptr)
    return nullptr;
  return make<BinaryExpr>(LHS, Kind, RHS);
}

Node *Db::parseIntegerLiteral(StringView Lit) {
  StringView Tmp = parseNumber(true);
  if (!Tmp.empty() && consumeIf('E'))
    return make<IntegerExpr>(Lit, Tmp);
  return nullptr;
}

Qualifiers Db::parseCVQualifiers() {
  Qualifiers CVR = QualNone;
  if (consumeIf('r'))
    addQualifiers(CVR, QualRestrict);
  if (consumeIf('V'))
    addQualifiers(CVR, QualVolatile);
  if (consumeIf('K'))
    addQualifiers(CVR, QualConst);
  return CVR;
}

// <function-param> ::= fp <top-level CV-Qualifiers> _                                     # L == 0, first parameter
//                  ::= fp <top-level CV-Qualifiers> <parameter-2 non-negative number> _   # L == 0, second and later parameters
//                  ::= fL <L-1 non-negative number> p <top-level CV-Qualifiers> _         # L > 0, first parameter
//                  ::= fL <L-1 non-negative number> p <top-level CV-Qualifiers> <parameter-2 non-negative number> _   # L > 0, second and later parameters
Node *Db::parseFunctionParam() {
  if (consumeIf("fp")) {
    parseCVQualifiers();
    StringView Num = parseNumber();
    if (!consumeIf('_'))
      return nullptr;
    return make<FunctionParam>(Num);
  }
  if (consumeIf("fL")) {
    if (parseNumber().empty())
      return nullptr;
    if (!consumeIf('p'))
      return nullptr;
    parseCVQualifiers();
    StringView Num = parseNumber();
    if (!consumeIf('_'))
      return nullptr;
    return make<FunctionParam>(Num);
  }
  return nullptr;
}

// [gs] nw <expression>* _ <type> E                     # new (expr-list) type
// [gs] nw <expression>* _ <type> <initializer>         # new (expr-list) type (init)
// [gs] na <expression>* _ <type> E                     # new[] (expr-list) type
// [gs] na <expression>* _ <type> <initializer>         # new[] (expr-list) type (init)
// <initializer> ::= pi <expression>* E                 # parenthesized initialization
Node *Db::parseNewExpr() {
  bool Global = consumeIf("gs");
  bool IsArray = look(1) == 'a';
  if (!consumeIf("nw") && !consumeIf("na"))
    return nullptr;
  size_t Exprs = Names.size();
  while (!consumeIf('_')) {
    Node *Ex = parseExpr();
    if (Ex == nullptr)
      return nullptr;
    Names.push_back(Ex);
  }
  NodeArray ExprList = popTrailingNodeArray(Exprs);
  Node *Ty = parseType();
  if (Ty == nullptr)
    return Ty;
  if (consumeIf("pi")) {
    size_t InitsBegin = Names.size();
    while (!consumeIf('E')) {
      Node *Init = parseExpr();
      if (Init == nullptr)
        return Init;
      Names.push_back(Init);
    }
    NodeArray Inits = popTrailingNodeArray(InitsBegin);
    return make<NewExpr>(ExprList, Ty, Inits, Global, IsArray);
  } else if (!consumeIf('E'))
    return nullptr;
  return make<NewExpr>(ExprList, Ty, NodeArray(), Global, IsArray);
}

// cv <type> <expression>                               # conversion with one argument
// cv <type> _ <expression>* E                          # conversion with a different number of arguments
Node *Db::parseConversionExpr() {
  if (!consumeIf("cv"))
    return nullptr;
  Node *Ty;
  {
    SwapAndRestore<bool> SaveTemp(TryToParseTemplateArgs, false);
    Ty = parseType();
  }

  if (Ty == nullptr)
    return nullptr;

  if (consumeIf('_')) {
    size_t ExprsBegin = Names.size();
    while (!consumeIf('E')) {
      Node *E = parseExpr();
      if (E == nullptr)
        return E;
      Names.push_back(E);
    }
    NodeArray Exprs = popTrailingNodeArray(ExprsBegin);
    return make<ConversionExpr>(Ty, Exprs);
  }

  Node *E[1] = {parseExpr()};
  if (E[0] == nullptr)
    return nullptr;
  return make<ConversionExpr>(Ty, makeNodeArray(E, E + 1));
}

// <expr-primary> ::= L <type> <value number> E                          # integer literal
//                ::= L <type> <value float> E                           # floating literal
//                ::= L <string type> E                                  # string literal
//                ::= L <nullptr type> E                                 # nullptr literal (i.e., "LDnE")
// FIXME:         ::= L <type> <real-part float> _ <imag-part float> E   # complex floating point literal (C 2000)
//                ::= L <mangled-name> E                                 # external name
Node *Db::parseExprPrimary() {
  if (!consumeIf('L'))
    return nullptr;
  switch (look()) {
  case 'w':
    ++First;
    return parseIntegerLiteral("wchar_t");
  case 'b':
    if (consumeIf("b0E"))
      return make<BoolExpr>(0);
    if (consumeIf("b1E"))
      return make<BoolExpr>(1);
    return nullptr;
  case 'c':
    ++First;
    return parseIntegerLiteral("char");
  case 'a':
    ++First;
    return parseIntegerLiteral("signed char");
  case 'h':
    ++First;
    return parseIntegerLiteral("unsigned char");
  case 's':
    ++First;
    return parseIntegerLiteral("short");
  case 't':
    ++First;
    return parseIntegerLiteral("unsigned short");
  case 'i':
    ++First;
    return parseIntegerLiteral("");
  case 'j':
    ++First;
    return parseIntegerLiteral("u");
  case 'l':
    ++First;
    return parseIntegerLiteral("l");
  case 'm':
    ++First;
    return parseIntegerLiteral("ul");
  case 'x':
    ++First;
    return parseIntegerLiteral("ll");
  case 'y':
    ++First;
    return parseIntegerLiteral("ull");
  case 'n':
    ++First;
    return parseIntegerLiteral("__int128");
  case 'o':
    ++First;
    return parseIntegerLiteral("unsigned __int128");
  case 'f':
    ++First;
    return parseFloatingLiteral<float>();
  case 'd':
    ++First;
    return parseFloatingLiteral<double>();
  case 'e':
    ++First;
    return parseFloatingLiteral<long double>();
  case '_':
    if (consumeIf("_Z")) {
      Node *R = legacyParse<parse_encoding>();
      if (R != nullptr && consumeIf('E'))
        return R;
    }
    return nullptr;
  case 'T':
    // Invalid mangled name per
    //   http://sourcerytools.com/pipermail/cxx-abi-dev/2011-August/002422.html
    return nullptr;
  default: {
    // might be named type
    Node *T = parseType();
    if (T == nullptr)
      return nullptr;
    StringView N = parseNumber();
    if (!N.empty()) {
      if (!consumeIf('E'))
        return nullptr;
      return make<IntegerCastExpr>(T, N);
    }
    if (consumeIf('E'))
      return T;
    return nullptr;
  }
  }
}

// <braced-expression> ::= <expression>
//                     ::= di <field source-name> <braced-expression>    # .name = expr
//                     ::= dx <index expression> <braced-expression>     # [expr] = expr
//                     ::= dX <range begin expression> <range end expression> <braced-expression>
Node *Db::parseBracedExpr() {
  if (look() == 'd') {
    switch (look(1)) {
    case 'i': {
      First += 2;
      Node *Field = legacyParse<parse_source_name>();
      if (Field == nullptr)
        return nullptr;
      Node *Init = parseBracedExpr();
      if (Init == nullptr)
        return nullptr;
      return make<BracedExpr>(Field, Init, /*isArray=*/false);
    }
    case 'x': {
      First += 2;
      Node *Index = parseExpr();
      if (Index == nullptr)
        return nullptr;
      Node *Init = parseBracedExpr();
      if (Init == nullptr)
        return nullptr;
      return make<BracedExpr>(Index, Init, /*isArray=*/true);
    }
    case 'X': {
      First += 2;
      Node *RangeBegin = parseExpr();
      if (RangeBegin == nullptr)
        return nullptr;
      Node *RangeEnd = parseExpr();
      if (RangeEnd == nullptr)
        return nullptr;
      Node *Init = parseBracedExpr();
      if (Init == nullptr)
        return nullptr;
      return make<BracedRangeExpr>(RangeBegin, RangeEnd, Init);
    }
    }
  }
  return parseExpr();
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
//              ::= fL <binary-operator-name> <expression> <expression>
//              ::= fR <binary-operator-name> <expression> <expression>
//              ::= fl <binary-operator-name> <expression>
//              ::= fr <binary-operator-name> <expression>
//              ::= <expr-primary>
Node *Db::parseExpr() {
  bool Global = consumeIf("gs");
  if (numLeft() < 2)
    return nullptr;

  switch (*First) {
  case 'L':
    return parseExprPrimary();
  case 'T':
    return legacyParse<parse_template_param>();
  case 'f':
    return parseFunctionParam();
  case 'a':
    switch (First[1]) {
    case 'a':
      First += 2;
      return parseBinaryExpr("&&");
    case 'd':
      First += 2;
      return parsePrefixExpr("&");
    case 'n':
      First += 2;
      return parseBinaryExpr("&");
    case 'N':
      First += 2;
      return parseBinaryExpr("&=");
    case 'S':
      First += 2;
      return parseBinaryExpr("=");
    case 't': {
      First += 2;
      Node *Ty = parseType();
      if (Ty == nullptr)
        return nullptr;
      return make<EnclosingExpr>("alignof (", Ty, ")");
    }
    case 'z': {
      First += 2;
      Node *Ty = parseExpr();
      if (Ty == nullptr)
        return nullptr;
      return make<EnclosingExpr>("alignof (", Ty, ")");
    }
    }
    return nullptr;
  case 'c':
    switch (First[1]) {
    // cc <type> <expression>                               # const_cast<type>(expression)
    case 'c': {
      First += 2;
      Node *Ty = parseType();
      if (Ty == nullptr)
        return Ty;
      Node *Ex = parseExpr();
      if (Ex == nullptr)
        return Ex;
      return make<CastExpr>("const_cast", Ty, Ex);
    }
    // cl <expression>+ E                                   # call
    case 'l': {
      First += 2;
      Node *Callee = parseExpr();
      if (Callee == nullptr)
        return Callee;
      size_t ExprsBegin = Names.size();
      while (!consumeIf('E')) {
        Node *E = parseExpr();
        if (E == nullptr)
          return E;
        Names.push_back(E);
      }
      return make<CallExpr>(Callee, popTrailingNodeArray(ExprsBegin));
    }
    case 'm':
      First += 2;
      return parseBinaryExpr(",");
    case 'o':
      First += 2;
      return parsePrefixExpr("~");
    case 'v':
      return parseConversionExpr();
    }
    return nullptr;
  case 'd':
    switch (First[1]) {
    case 'a': {
      First += 2;
      Node *Ex = parseExpr();
      if (Ex == nullptr)
        return Ex;
      return make<DeleteExpr>(Ex, Global, /*is_array=*/true);
    }
    case 'c': {
      First += 2;
      Node *T = parseType();
      if (T == nullptr)
        return T;
      Node *Ex = parseExpr();
      if (Ex == nullptr)
        return Ex;
      return make<CastExpr>("dynamic_cast", T, Ex);
    }
    case 'e':
      First += 2;
      return parsePrefixExpr("*");
    case 'l': {
      First += 2;
      Node *E = parseExpr();
      if (E == nullptr)
        return E;
      return make<DeleteExpr>(E, Global, /*is_array=*/false);
    }
    case 'n':
      return legacyParse<parse_unresolved_name>();
    case 's': {
      First += 2;
      Node *LHS = parseExpr();
      if (LHS == nullptr)
        return nullptr;
      Node *RHS = parseExpr();
      if (RHS == nullptr)
        return nullptr;
      return make<MemberExpr>(LHS, ".*", RHS);
    }
    case 't': {
      First += 2;
      Node *LHS = parseExpr();
      if (LHS == nullptr)
        return LHS;
      Node *RHS = parseExpr();
      if (RHS == nullptr)
        return nullptr;
      return make<MemberExpr>(LHS, ".", RHS);
    }
    case 'v':
      First += 2;
      return parseBinaryExpr("/");
    case 'V':
      First += 2;
      return parseBinaryExpr("/=");
    }
    return nullptr;
  case 'e':
    switch (First[1]) {
    case 'o':
      First += 2;
      return parseBinaryExpr("^");
    case 'O':
      First += 2;
      return parseBinaryExpr("^=");
    case 'q':
      First += 2;
      return parseBinaryExpr("==");
    }
    return nullptr;
  case 'g':
    switch (First[1]) {
    case 'e':
      First += 2;
      return parseBinaryExpr(">=");
    case 't':
      First += 2;
      return parseBinaryExpr(">");
    }
    return nullptr;
  case 'i':
    switch (First[1]) {
    case 'x': {
      First += 2;
      Node *Base = parseExpr();
      if (Base == nullptr)
        return nullptr;
      Node *Index = parseExpr();
      if (Index == nullptr)
        return Index;
      return make<ArraySubscriptExpr>(Base, Index);
    }
    case 'l': {
      First += 2;
      size_t InitsBegin = Names.size();
      while (!consumeIf('E')) {
        Node *E = parseBracedExpr();
        if (E == nullptr)
          return nullptr;
        Names.push_back(E);
      }
      return make<InitListExpr>(nullptr, popTrailingNodeArray(InitsBegin));
    }
    }
    return nullptr;
  case 'l':
    switch (First[1]) {
    case 'e':
      First += 2;
      return parseBinaryExpr("<=");
    case 's':
      First += 2;
      return parseBinaryExpr("<<");
    case 'S':
      First += 2;
      return parseBinaryExpr("<<=");
    case 't':
      First += 2;
      return parseBinaryExpr("<");
    }
    return nullptr;
  case 'm':
    switch (First[1]) {
    case 'i':
      First += 2;
      return parseBinaryExpr("-");
    case 'I':
      First += 2;
      return parseBinaryExpr("-=");
    case 'l':
      First += 2;
      return parseBinaryExpr("*");
    case 'L':
      First += 2;
      return parseBinaryExpr("*=");
    case 'm':
      First += 2;
      if (consumeIf('_'))
        return parsePrefixExpr("--");
      Node *Ex = parseExpr();
      if (Ex == nullptr)
        return nullptr;
      return make<PostfixExpr>(Ex, "--");
    }
    return nullptr;
  case 'n':
    switch (First[1]) {
    case 'a':
    case 'w':
      return parseNewExpr();
    case 'e':
      First += 2;
      return parseBinaryExpr("!=");
    case 'g':
      First += 2;
      return parsePrefixExpr("-");
    case 't':
      First += 2;
      return parsePrefixExpr("!");
    case 'x':
      First += 2;
      Node *Ex = parseExpr();
      if (Ex == nullptr)
        return Ex;
      return make<EnclosingExpr>("noexcept (", Ex, ")");
    }
    return nullptr;
  case 'o':
    switch (First[1]) {
    case 'n':
      return legacyParse<parse_unresolved_name>();
    case 'o':
      First += 2;
      return parseBinaryExpr("||");
    case 'r':
      First += 2;
      return parseBinaryExpr("|");
    case 'R':
      First += 2;
      return parseBinaryExpr("|=");
    }
    return nullptr;
  case 'p':
    switch (First[1]) {
    case 'm':
      First += 2;
      return parseBinaryExpr("->*");
    case 'l':
      First += 2;
      return parseBinaryExpr("+");
    case 'L':
      First += 2;
      return parseBinaryExpr("+=");
    case 'p': {
      First += 2;
      if (consumeIf('_'))
        return parsePrefixExpr("++");
      Node *Ex = parseExpr();
      if (Ex == nullptr)
        return Ex;
      return make<PostfixExpr>(Ex, "++");
    }
    case 's':
      First += 2;
      return parsePrefixExpr("+");
    case 't': {
      First += 2;
      Node *L = parseExpr();
      if (L == nullptr)
        return nullptr;
      Node *R = parseExpr();
      if (R == nullptr)
        return nullptr;
      return make<MemberExpr>(L, "->", R);
    }
    }
    return nullptr;
  case 'q':
    if (First[1] == 'u') {
      First += 2;
      Node *Cond = parseExpr();
      if (Cond == nullptr)
        return nullptr;
      Node *LHS = parseExpr();
      if (LHS == nullptr)
        return nullptr;
      Node *RHS = parseExpr();
      if (RHS == nullptr)
        return nullptr;
      return make<ConditionalExpr>(Cond, LHS, RHS);
    }
    return nullptr;
  case 'r':
    switch (First[1]) {
    case 'c': {
      First += 2;
      Node *T = parseType();
      if (T == nullptr)
        return T;
      Node *Ex = parseExpr();
      if (Ex == nullptr)
        return Ex;
      return make<CastExpr>("reinterpret_cast", T, Ex);
    }
    case 'm':
      First += 2;
      return parseBinaryExpr("%");
    case 'M':
      First += 2;
      return parseBinaryExpr("%=");
    case 's':
      First += 2;
      return parseBinaryExpr(">>");
    case 'S':
      First += 2;
      return parseBinaryExpr(">>=");
    }
    return nullptr;
  case 's':
    switch (First[1]) {
    case 'c': {
      First += 2;
      Node *T = parseType();
      if (T == nullptr)
        return T;
      Node *Ex = parseExpr();
      if (Ex == nullptr)
        return Ex;
      return make<CastExpr>("static_cast", T, Ex);
    }
    case 'p': {
      First += 2;
      Node *Child = parseExpr();
      if (Child == nullptr)
        return nullptr;
      return make<ParameterPackExpansion>(Child);
    }
    case 'r':
      return legacyParse<parse_unresolved_name>();
    case 't': {
      First += 2;
      Node *Ty = parseType();
      if (Ty == nullptr)
        return Ty;
      return make<EnclosingExpr>("sizeof (", Ty, ")");
    }
    case 'z': {
      First += 2;
      Node *Ex = parseExpr();
      if (Ex == nullptr)
        return Ex;
      return make<EnclosingExpr>("sizeof (", Ex, ")");
    }
    case 'Z':
      First += 2;
      if (look() == 'T') {
        Node *R = legacyParse<parse_template_param>();
        if (R == nullptr)
          return nullptr;
        return make<SizeofParamPackExpr>(R);
      } else if (look() == 'f') {
        Node *FP = parseFunctionParam();
        if (FP == nullptr)
          return nullptr;
        return make<EnclosingExpr>("sizeof...", FP, ")");
      }
      return nullptr;
    }
    return nullptr;
  case 't':
    switch (First[1]) {
    case 'e': {
      First += 2;
      Node *Ex = parseExpr();
      if (Ex == nullptr)
        return Ex;
      return make<EnclosingExpr>("typeid (", Ex, ")");
    }
    case 'i': {
      First += 2;
      Node *Ty = parseType();
      if (Ty == nullptr)
        return Ty;
      return make<EnclosingExpr>("typeid (", Ty, ")");
    }
    case 'l': {
      First += 2;
      Node *Ty = parseType();
      if (Ty == nullptr)
        return nullptr;
      size_t InitsBegin = Names.size();
      while (!consumeIf('E')) {
        Node *E = parseBracedExpr();
        if (E == nullptr)
          return nullptr;
        Names.push_back(E);
      }
      return make<InitListExpr>(Ty, popTrailingNodeArray(InitsBegin));
    }
    case 'r':
      First += 2;
      return make<NameType>("throw");
    case 'w': {
      First += 2;
      Node *Ex = parseExpr();
      if (Ex == nullptr)
        return nullptr;
      return make<ThrowExpr>(Ex);
    }
    }
    return nullptr;
  case '1':
  case '2':
  case '3':
  case '4':
  case '5':
  case '6':
  case '7':
  case '8':
  case '9':
    return legacyParse<parse_unresolved_name>();
  }
  return nullptr;
}

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
    static constexpr const char *spec = "%LaL";
};

constexpr const char *FloatData<long double>::spec;

template <class Float> Node *Db::parseFloatingLiteral() {
  const size_t N = FloatData<Float>::mangled_size;
  if (numLeft() <= N)
    return nullptr;
  StringView Data(First, First + N);
  for (char C : Data)
    if (!std::isxdigit(C))
      return nullptr;
  First += N;
  if (!consumeIf('E'))
    return nullptr;
  return make<FloatExpr<Float>>(Data);
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
                    db.Names.push_back(db.Subs[0]);
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
                        db.Names.push_back(db.Subs[sub]);
                        first = t+1;
                    }
                }
                break;
            }
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
                    db.Names.push_back(db.TemplateParams[0]);
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
                    db.Names.push_back(db.TemplateParams[sub]);
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
                db.Subs.push_back(db.Names.back());
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
                db.Subs.push_back(db.Names.back());
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
                        db.Subs.push_back(db.Names.back());
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
        case 'J': {
            t = first+1;
            if (t == last)
                return first;
            size_t ArgsBegin = db.Names.size();
            while (*t != 'E')
            {
                const char* t1 = parse_template_arg(t, last, db);
                if (t1 == t)
                    return first;
                t = t1;
            }
            NodeArray Args = db.popTrailingNodeArray(ArgsBegin);
            db.Names.push_back(db.make<TemplateArgumentPack>(Args));
            first = t+1;
            break;
        }
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
                if (t1 == t || t1 == last || k0 + 1 != k1)
                    return first;
                Node *TableEntry = db.Names.back();
                if (TableEntry->getKind() == Node::KTemplateArgumentPack)
                  TableEntry = db.make<ParameterPack>(
                      static_cast<TemplateArgumentPack*>(TableEntry)
                          ->getElements());
                db.TemplateParams.push_back(TableEntry);
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
        auto *tp = db.make<TemplateArgs>(
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
                        db.Subs.push_back(db.Names.back());
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
                    db.Subs.push_back(db.Names.back());
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
                    db.Subs.push_back(db.Names.back());
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
                    db.Subs.push_back(db.Names.back());
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
                    db.Subs.push_back(db.Names.back());
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
            db.Subs.pop_back();
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
                    db.Subs.push_back(db.Names.back());
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
                        result = db.make<FunctionEncoding>(
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
                        result = db.make<FunctionEncoding>(
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

enum {
    unknown_error = -4,
    invalid_args = -3,
    invalid_mangled_name,
    memory_alloc_failure,
    success
};

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


namespace __cxxabiv1 {
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

    if (internal_status == success &&
        db.Names.back()->containsUnexpandedParameterPack())
        internal_status = invalid_mangled_name;

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
