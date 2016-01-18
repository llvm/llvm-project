//===-- TemplateBase.h - Core classes for C++ templates ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file provides definitions which are common for all kinds of
//  template representation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_TEMPLATEBASE_H
#define LLVM_CLANG_AST_TEMPLATEBASE_H

#include "clang/AST/TemplateName.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {
  class FoldingSetNodeID;
}

namespace clang {

class DiagnosticBuilder;
class Expr;
struct PrintingPolicy;
class TypeSourceInfo;
class ValueDecl;

/// \brief Represents a template argument.
class TemplateArgument {
public:
  /// \brief The kind of template argument we're storing.
  enum ArgKind {
    /// \brief Represents an empty template argument, e.g., one that has not
    /// been deduced.
    Null = 0,
    /// The template argument is a type.
    Type,
    /// The template argument is a declaration that was provided for a pointer,
    /// reference, or pointer to member non-type template parameter.
    Declaration,
    /// The template argument is a null pointer or null pointer to member that
    /// was provided for a non-type template parameter.
    NullPtr,
    /// The template argument is an integral value stored in an llvm::APSInt
    /// that was provided for an integral non-type template parameter.
    Integral,
    /// The template argument is a template name that was provided for a
    /// template template parameter.
    Template,
    /// The template argument is a pack expansion of a template name that was
    /// provided for a template template parameter.
    TemplateExpansion,
    /// The template argument is an expression, and we've not resolved it to one
    /// of the other forms yet, either because it's dependent or because we're
    /// representing a non-canonical template argument (for instance, in a
    /// TemplateSpecializationType). Also used to represent a non-dependent
    /// __uuidof expression (a Microsoft extension).
    Expression,
    /// The template argument is actually a parameter pack. Arguments are stored
    /// in the Args struct.
    Pack
  };

private:
  /// \brief The kind of template argument we're storing.

  struct DA {
    unsigned Kind;
    void *QT;
    ValueDecl *D;
  };
  struct I {
    unsigned Kind;
    // We store a decomposed APSInt with the data allocated by ASTContext if
    // BitWidth > 64. The memory may be shared between multiple
    // TemplateArgument instances.
    unsigned BitWidth : 31;
    unsigned IsUnsigned : 1;
    union {
      uint64_t VAL;          ///< Used to store the <= 64 bits integer value.
      const uint64_t *pVal;  ///< Used to store the >64 bits integer value.
    };
    void *Type;
  };
  struct A {
    unsigned Kind;
    unsigned NumArgs;
    const TemplateArgument *Args;
  };
  struct TA {
    unsigned Kind;
    unsigned NumExpansions;
    void *Name;
  };
  struct TV {
    unsigned Kind;
    uintptr_t V;
  };
  union {
    struct DA DeclArg;
    struct I Integer;
    struct A Args;
    struct TA TemplateArg;
    struct TV TypeOrValue;
  };

  TemplateArgument(TemplateName, bool) = delete;
  
public:
  /// \brief Construct an empty, invalid template argument.
  TemplateArgument() {
    TypeOrValue.Kind = Null;
    TypeOrValue.V = 0;
  }

  /// \brief Construct a template type argument.
  TemplateArgument(QualType T, bool isNullPtr = false) {
    TypeOrValue.Kind = isNullPtr ? NullPtr : Type;
    TypeOrValue.V = reinterpret_cast<uintptr_t>(T.getAsOpaquePtr());
  }

  /// \brief Construct a template argument that refers to a
  /// declaration, which is either an external declaration or a
  /// template declaration.
  TemplateArgument(ValueDecl *D, QualType QT) {
    assert(D && "Expected decl");
    DeclArg.Kind = Declaration;
    DeclArg.QT = QT.getAsOpaquePtr();
    DeclArg.D = D;
  }

  /// \brief Construct an integral constant template argument. The memory to
  /// store the value is allocated with Ctx.
  TemplateArgument(ASTContext &Ctx, const llvm::APSInt &Value, QualType Type);

  /// \brief Construct an integral constant template argument with the same
  /// value as Other but a different type.
  TemplateArgument(const TemplateArgument &Other, QualType Type) {
    Integer = Other.Integer;
    Integer.Type = Type.getAsOpaquePtr();
  }

  /// \brief Construct a template argument that is a template.
  ///
  /// This form of template argument is generally used for template template
  /// parameters. However, the template name could be a dependent template
  /// name that ends up being instantiated to a function template whose address
  /// is taken.
  ///
  /// \param Name The template name.
  TemplateArgument(TemplateName Name) {
    TemplateArg.Kind = Template;
    TemplateArg.Name = Name.getAsVoidPointer();
    TemplateArg.NumExpansions = 0;
  }

  /// \brief Construct a template argument that is a template pack expansion.
  ///
  /// This form of template argument is generally used for template template
  /// parameters. However, the template name could be a dependent template
  /// name that ends up being instantiated to a function template whose address
  /// is taken.
  ///
  /// \param Name The template name.
  ///
  /// \param NumExpansions The number of expansions that will be generated by
  /// instantiating
  TemplateArgument(TemplateName Name, Optional<unsigned> NumExpansions) {
    TemplateArg.Kind = TemplateExpansion;
    TemplateArg.Name = Name.getAsVoidPointer();
    if (NumExpansions)
      TemplateArg.NumExpansions = *NumExpansions + 1;
    else
      TemplateArg.NumExpansions = 0;
  }

  /// \brief Construct a template argument that is an expression.
  ///
  /// This form of template argument only occurs in template argument
  /// lists used for dependent types and for expression; it will not
  /// occur in a non-dependent, canonical template argument list.
  TemplateArgument(Expr *E) {
    TypeOrValue.Kind = Expression;
    TypeOrValue.V = reinterpret_cast<uintptr_t>(E);
  }

  /// \brief Construct a template argument that is a template argument pack.
  ///
  /// We assume that storage for the template arguments provided
  /// outlives the TemplateArgument itself.
  explicit TemplateArgument(ArrayRef<TemplateArgument> Args) {
    this->Args.Kind = Pack;
    this->Args.Args = Args.data();
    this->Args.NumArgs = Args.size();
  }

  static TemplateArgument getEmptyPack() { return TemplateArgument(None); }

  /// \brief Create a new template argument pack by copying the given set of
  /// template arguments.
  static TemplateArgument CreatePackCopy(ASTContext &Context,
                                         ArrayRef<TemplateArgument> Args);

  /// \brief Return the kind of stored template argument.
  ArgKind getKind() const { return (ArgKind)TypeOrValue.Kind; }

  /// \brief Determine whether this template argument has no value.
  bool isNull() const { return getKind() == Null; }

  /// \brief Whether this template argument is dependent on a template
  /// parameter such that its result can change from one instantiation to
  /// another.
  bool isDependent() const;

  /// \brief Whether this template argument is dependent on a template
  /// parameter.
  bool isInstantiationDependent() const;

  /// \brief Whether this template argument contains an unexpanded
  /// parameter pack.
  bool containsUnexpandedParameterPack() const;

  /// \brief Determine whether this template argument is a pack expansion.
  bool isPackExpansion() const;
  
  /// \brief Retrieve the type for a type template argument.
  QualType getAsType() const {
    assert(getKind() == Type && "Unexpected kind");
    return QualType::getFromOpaquePtr(reinterpret_cast<void*>(TypeOrValue.V));
  }

  /// \brief Retrieve the declaration for a declaration non-type
  /// template argument.
  ValueDecl *getAsDecl() const {
    assert(getKind() == Declaration && "Unexpected kind");
    return DeclArg.D;
  }

  QualType getParamTypeForDecl() const {
    assert(getKind() == Declaration && "Unexpected kind");
    return QualType::getFromOpaquePtr(DeclArg.QT);
  }

  /// \brief Retrieve the type for null non-type template argument.
  QualType getNullPtrType() const {
    assert(getKind() == NullPtr && "Unexpected kind");
    return QualType::getFromOpaquePtr(reinterpret_cast<void*>(TypeOrValue.V));
  }

  /// \brief Retrieve the template name for a template name argument.
  TemplateName getAsTemplate() const {
    assert(getKind() == Template && "Unexpected kind");
    return TemplateName::getFromVoidPointer(TemplateArg.Name);
  }

  /// \brief Retrieve the template argument as a template name; if the argument
  /// is a pack expansion, return the pattern as a template name.
  TemplateName getAsTemplateOrTemplatePattern() const {
    assert((getKind() == Template || getKind() == TemplateExpansion) &&
           "Unexpected kind");
    
    return TemplateName::getFromVoidPointer(TemplateArg.Name);
  }

  /// \brief Retrieve the number of expansions that a template template argument
  /// expansion will produce, if known.
  Optional<unsigned> getNumTemplateExpansions() const;
  
  /// \brief Retrieve the template argument as an integral value.
  // FIXME: Provide a way to read the integral data without copying the value.
  llvm::APSInt getAsIntegral() const {
    assert(getKind() == Integral && "Unexpected kind");
    using namespace llvm;
    if (Integer.BitWidth <= 64)
      return APSInt(APInt(Integer.BitWidth, Integer.VAL), Integer.IsUnsigned);

    unsigned NumWords = APInt::getNumWords(Integer.BitWidth);
    return APSInt(APInt(Integer.BitWidth, makeArrayRef(Integer.pVal, NumWords)),
                  Integer.IsUnsigned);
  }

  /// \brief Retrieve the type of the integral value.
  QualType getIntegralType() const {
    assert(getKind() == Integral && "Unexpected kind");
    return QualType::getFromOpaquePtr(Integer.Type);
  }

  void setIntegralType(QualType T) {
    assert(getKind() == Integral && "Unexpected kind");
    Integer.Type = T.getAsOpaquePtr();
  }

  /// \brief Retrieve the template argument as an expression.
  Expr *getAsExpr() const {
    assert(getKind() == Expression && "Unexpected kind");
    return reinterpret_cast<Expr *>(TypeOrValue.V);
  }

  /// \brief Iterator that traverses the elements of a template argument pack.
  typedef const TemplateArgument * pack_iterator;

  /// \brief Iterator referencing the first argument of a template argument
  /// pack.
  pack_iterator pack_begin() const {
    assert(getKind() == Pack);
    return Args.Args;
  }

  /// \brief Iterator referencing one past the last argument of a template
  /// argument pack.
  pack_iterator pack_end() const {
    assert(getKind() == Pack);
    return Args.Args + Args.NumArgs;
  }

  /// \brief Iterator range referencing all of the elements of a template
  /// argument pack.
  llvm::iterator_range<pack_iterator> pack_elements() const {
    return llvm::make_range(pack_begin(), pack_end());
  }

  /// \brief The number of template arguments in the given template argument
  /// pack.
  unsigned pack_size() const {
    assert(getKind() == Pack);
    return Args.NumArgs;
  }

  /// \brief Return the array of arguments in this template argument pack.
  ArrayRef<TemplateArgument> getPackAsArray() const {
    assert(getKind() == Pack);
    return llvm::makeArrayRef(Args.Args, Args.NumArgs);
  }

  /// \brief Determines whether two template arguments are superficially the
  /// same.
  bool structurallyEquals(const TemplateArgument &Other) const;

  /// \brief When the template argument is a pack expansion, returns
  /// the pattern of the pack expansion.
  TemplateArgument getPackExpansionPattern() const;

  /// \brief Print this template argument to the given output stream.
  void print(const PrintingPolicy &Policy, raw_ostream &Out) const;
             
  /// \brief Used to insert TemplateArguments into FoldingSets.
  void Profile(llvm::FoldingSetNodeID &ID, const ASTContext &Context) const;
};

/// Location information for a TemplateArgument.
struct TemplateArgumentLocInfo {
private:

  struct T {
    // FIXME: We'd like to just use the qualifier in the TemplateName,
    // but template arguments get canonicalized too quickly.
    NestedNameSpecifier *Qualifier;
    void *QualifierLocData;
    unsigned TemplateNameLoc;
    unsigned EllipsisLoc;
  };

  union {
    struct T Template;
    Expr *Expression;
    TypeSourceInfo *Declarator;
  };

public:
  TemplateArgumentLocInfo();
  
  TemplateArgumentLocInfo(TypeSourceInfo *TInfo) : Declarator(TInfo) {}
  
  TemplateArgumentLocInfo(Expr *E) : Expression(E) {}
  
  TemplateArgumentLocInfo(NestedNameSpecifierLoc QualifierLoc,
                          SourceLocation TemplateNameLoc,
                          SourceLocation EllipsisLoc)
  {
    Template.Qualifier = QualifierLoc.getNestedNameSpecifier();
    Template.QualifierLocData = QualifierLoc.getOpaqueData();
    Template.TemplateNameLoc = TemplateNameLoc.getRawEncoding();
    Template.EllipsisLoc = EllipsisLoc.getRawEncoding();
  }

  TypeSourceInfo *getAsTypeSourceInfo() const {
    return Declarator;
  }

  Expr *getAsExpr() const {
    return Expression;
  }

  NestedNameSpecifierLoc getTemplateQualifierLoc() const {
    return NestedNameSpecifierLoc(Template.Qualifier, 
                                  Template.QualifierLocData);
  }
  
  SourceLocation getTemplateNameLoc() const {
    return SourceLocation::getFromRawEncoding(Template.TemplateNameLoc);
  }
  
  SourceLocation getTemplateEllipsisLoc() const {
    return SourceLocation::getFromRawEncoding(Template.EllipsisLoc);
  }
};

/// Location wrapper for a TemplateArgument.  TemplateArgument is to
/// TemplateArgumentLoc as Type is to TypeLoc.
class TemplateArgumentLoc {
  TemplateArgument Argument;
  TemplateArgumentLocInfo LocInfo;

public:
  TemplateArgumentLoc() {}

  TemplateArgumentLoc(const TemplateArgument &Argument,
                      TemplateArgumentLocInfo Opaque)
    : Argument(Argument), LocInfo(Opaque) {
  }

  TemplateArgumentLoc(const TemplateArgument &Argument, TypeSourceInfo *TInfo)
    : Argument(Argument), LocInfo(TInfo) {
    assert(Argument.getKind() == TemplateArgument::Type);
  }

  TemplateArgumentLoc(const TemplateArgument &Argument, Expr *E)
    : Argument(Argument), LocInfo(E) {
    assert(Argument.getKind() == TemplateArgument::Expression);
  }

  TemplateArgumentLoc(const TemplateArgument &Argument, 
                      NestedNameSpecifierLoc QualifierLoc,
                      SourceLocation TemplateNameLoc,
                      SourceLocation EllipsisLoc = SourceLocation())
    : Argument(Argument), LocInfo(QualifierLoc, TemplateNameLoc, EllipsisLoc) {
    assert(Argument.getKind() == TemplateArgument::Template ||
           Argument.getKind() == TemplateArgument::TemplateExpansion);
  }
  
  /// \brief - Fetches the primary location of the argument.
  SourceLocation getLocation() const {
    if (Argument.getKind() == TemplateArgument::Template ||
        Argument.getKind() == TemplateArgument::TemplateExpansion)
      return getTemplateNameLoc();
    
    return getSourceRange().getBegin();
  }

  /// \brief - Fetches the full source range of the argument.
  SourceRange getSourceRange() const LLVM_READONLY;

  const TemplateArgument &getArgument() const {
    return Argument;
  }

  TemplateArgumentLocInfo getLocInfo() const {
    return LocInfo;
  }

  TypeSourceInfo *getTypeSourceInfo() const {
    assert(Argument.getKind() == TemplateArgument::Type);
    return LocInfo.getAsTypeSourceInfo();
  }

  Expr *getSourceExpression() const {
    assert(Argument.getKind() == TemplateArgument::Expression);
    return LocInfo.getAsExpr();
  }

  Expr *getSourceDeclExpression() const {
    assert(Argument.getKind() == TemplateArgument::Declaration);
    return LocInfo.getAsExpr();
  }

  Expr *getSourceNullPtrExpression() const {
    assert(Argument.getKind() == TemplateArgument::NullPtr);
    return LocInfo.getAsExpr();
  }

  Expr *getSourceIntegralExpression() const {
    assert(Argument.getKind() == TemplateArgument::Integral);
    return LocInfo.getAsExpr();
  }

  NestedNameSpecifierLoc getTemplateQualifierLoc() const {
    assert(Argument.getKind() == TemplateArgument::Template ||
           Argument.getKind() == TemplateArgument::TemplateExpansion);
    return LocInfo.getTemplateQualifierLoc();
  }
  
  SourceLocation getTemplateNameLoc() const {
    assert(Argument.getKind() == TemplateArgument::Template ||
           Argument.getKind() == TemplateArgument::TemplateExpansion);
    return LocInfo.getTemplateNameLoc();
  }  
  
  SourceLocation getTemplateEllipsisLoc() const {
    assert(Argument.getKind() == TemplateArgument::TemplateExpansion);
    return LocInfo.getTemplateEllipsisLoc();
  }
};

/// A convenient class for passing around template argument
/// information.  Designed to be passed by reference.
class TemplateArgumentListInfo {
  SmallVector<TemplateArgumentLoc, 8> Arguments;
  SourceLocation LAngleLoc;
  SourceLocation RAngleLoc;

  // This can leak if used in an AST node, use ASTTemplateArgumentListInfo
  // instead.
  void *operator new(size_t bytes, ASTContext &C) = delete;

public:
  TemplateArgumentListInfo() {}

  TemplateArgumentListInfo(SourceLocation LAngleLoc,
                           SourceLocation RAngleLoc)
    : LAngleLoc(LAngleLoc), RAngleLoc(RAngleLoc) {}

  SourceLocation getLAngleLoc() const { return LAngleLoc; }
  SourceLocation getRAngleLoc() const { return RAngleLoc; }

  void setLAngleLoc(SourceLocation Loc) { LAngleLoc = Loc; }
  void setRAngleLoc(SourceLocation Loc) { RAngleLoc = Loc; }

  unsigned size() const { return Arguments.size(); }

  const TemplateArgumentLoc *getArgumentArray() const {
    return Arguments.data();
  }

  llvm::ArrayRef<TemplateArgumentLoc> arguments() const {
    return Arguments;
  }

  const TemplateArgumentLoc &operator[](unsigned I) const {
    return Arguments[I];
  }

  TemplateArgumentLoc &operator[](unsigned I) {
    return Arguments[I];
  }

  void addArgument(const TemplateArgumentLoc &Loc) {
    Arguments.push_back(Loc);
  }
};

/// \brief Represents an explicit template argument list in C++, e.g.,
/// the "<int>" in "sort<int>".
/// This is safe to be used inside an AST node, in contrast with
/// TemplateArgumentListInfo.
struct ASTTemplateArgumentListInfo {
  /// \brief The source location of the left angle bracket ('<').
  SourceLocation LAngleLoc;
  
  /// \brief The source location of the right angle bracket ('>').
  SourceLocation RAngleLoc;
  
  union {
    /// \brief The number of template arguments in TemplateArgs.
    /// The actual template arguments (if any) are stored after the
    /// ExplicitTemplateArgumentList structure.
    unsigned NumTemplateArgs;

    /// Force ASTTemplateArgumentListInfo to the right alignment
    /// for the following array of TemplateArgumentLocs.
    llvm::AlignedCharArray<
        llvm::AlignOf<TemplateArgumentLoc>::Alignment, 1> Aligner;
  };

  /// \brief Retrieve the template arguments
  TemplateArgumentLoc *getTemplateArgs() {
    return reinterpret_cast<TemplateArgumentLoc *> (this + 1);
  }
  
  /// \brief Retrieve the template arguments
  const TemplateArgumentLoc *getTemplateArgs() const {
    return reinterpret_cast<const TemplateArgumentLoc *> (this + 1);
  }

  const TemplateArgumentLoc &operator[](unsigned I) const {
    return getTemplateArgs()[I];
  }

  static const ASTTemplateArgumentListInfo *Create(ASTContext &C,
                                          const TemplateArgumentListInfo &List);

  void initializeFrom(const TemplateArgumentListInfo &List);
  void initializeFrom(const TemplateArgumentListInfo &List,
                      bool &Dependent, bool &InstantiationDependent,
                      bool &ContainsUnexpandedParameterPack);
  void copyInto(TemplateArgumentListInfo &List) const;
  static std::size_t sizeFor(unsigned NumTemplateArgs);
};

/// \brief Extends ASTTemplateArgumentListInfo with the source location
/// information for the template keyword; this is used as part of the
/// representation of qualified identifiers, such as S<T>::template apply<T>.
struct ASTTemplateKWAndArgsInfo : public ASTTemplateArgumentListInfo {
  typedef ASTTemplateArgumentListInfo Base;

  // NOTE: the source location of the (optional) template keyword is
  // stored after all template arguments.

  /// \brief Get the source location of the template keyword.
  SourceLocation getTemplateKeywordLoc() const {
    return *reinterpret_cast<const SourceLocation*>
      (getTemplateArgs() + NumTemplateArgs);
  }

  /// \brief Sets the source location of the template keyword.
  void setTemplateKeywordLoc(SourceLocation TemplateKWLoc) {
    *reinterpret_cast<SourceLocation*>
      (getTemplateArgs() + NumTemplateArgs) = TemplateKWLoc;
  }

  static const ASTTemplateKWAndArgsInfo*
  Create(ASTContext &C, SourceLocation TemplateKWLoc,
         const TemplateArgumentListInfo &List);

  void initializeFrom(SourceLocation TemplateKWLoc,
                      const TemplateArgumentListInfo &List);
  void initializeFrom(SourceLocation TemplateKWLoc,
                      const TemplateArgumentListInfo &List,
                      bool &Dependent, bool &InstantiationDependent,
                      bool &ContainsUnexpandedParameterPack);
  void initializeFrom(SourceLocation TemplateKWLoc);

  static std::size_t sizeFor(unsigned NumTemplateArgs);
};

const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
                                    const TemplateArgument &Arg);

inline TemplateSpecializationType::iterator
    TemplateSpecializationType::end() const {
  return getArgs() + getNumArgs();
}

inline DependentTemplateSpecializationType::iterator
    DependentTemplateSpecializationType::end() const {
  return getArgs() + getNumArgs();
}

inline const TemplateArgument &
    TemplateSpecializationType::getArg(unsigned Idx) const {
  assert(Idx < getNumArgs() && "Template argument out of range");
  return getArgs()[Idx];
}

inline const TemplateArgument &
    DependentTemplateSpecializationType::getArg(unsigned Idx) const {
  assert(Idx < getNumArgs() && "Template argument out of range");
  return getArgs()[Idx];
}
  
} // end namespace clang

#endif
