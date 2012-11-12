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

#include "clang/AST/Type.h"
#include "clang/AST/TemplateName.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallVector.h"
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

/// \brief Represents a template argument within a class template
/// specialization.
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
    /// The template argument is a value- or type-dependent expression
    /// stored in an Expr*.
    Expression,
    /// The template argument is actually a parameter pack. Arguments are stored
    /// in the Args struct.
    Pack
  };

private:
  /// \brief The kind of template argument we're storing.
  unsigned Kind;

  union {
    uintptr_t TypeOrValue;
    struct {
      ValueDecl *D;
      bool ForRefParam;
    } DeclArg;
    struct {
      // We store a decomposed APSInt with the data allocated by ASTContext if
      // BitWidth > 64. The memory may be shared between multiple
      // TemplateArgument instances.
      union {
        uint64_t VAL;          ///< Used to store the <= 64 bits integer value.
        const uint64_t *pVal;  ///< Used to store the >64 bits integer value.
      };
      unsigned BitWidth : 31;
      unsigned IsUnsigned : 1;
      void *Type;
    } Integer;
    struct {
      const TemplateArgument *Args;
      unsigned NumArgs;
    } Args;
    struct {
      void *Name;
      unsigned NumExpansions;
    } TemplateArg;
  };

  TemplateArgument(TemplateName, bool); // DO NOT USE
  
public:
  /// \brief Construct an empty, invalid template argument.
  TemplateArgument() : Kind(Null), TypeOrValue(0) { }

  /// \brief Construct a template type argument.
  TemplateArgument(QualType T, bool isNullPtr = false)
    : Kind(isNullPtr ? NullPtr : Type) {
    TypeOrValue = reinterpret_cast<uintptr_t>(T.getAsOpaquePtr());
  }

  /// \brief Construct a template argument that refers to a
  /// declaration, which is either an external declaration or a
  /// template declaration.
  TemplateArgument(ValueDecl *D, bool ForRefParam) : Kind(Declaration) {
    assert(D && "Expected decl");
    DeclArg.D = D;
    DeclArg.ForRefParam = ForRefParam;
  }

  /// \brief Construct an integral constant template argument. The memory to
  /// store the value is allocated with Ctx.
  TemplateArgument(ASTContext &Ctx, const llvm::APSInt &Value, QualType Type);

  /// \brief Construct an integral constant template argument with the same
  /// value as Other but a different type.
  TemplateArgument(const TemplateArgument &Other, QualType Type)
    : Kind(Integral) {
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
  TemplateArgument(TemplateName Name) : Kind(Template) 
  {
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
  TemplateArgument(TemplateName Name, llvm::Optional<unsigned> NumExpansions)
    : Kind(TemplateExpansion) 
  {
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
  TemplateArgument(Expr *E) : Kind(Expression) {
    TypeOrValue = reinterpret_cast<uintptr_t>(E);
  }

  /// \brief Construct a template argument that is a template argument pack.
  ///
  /// We assume that storage for the template arguments provided
  /// outlives the TemplateArgument itself.
  TemplateArgument(const TemplateArgument *Args, unsigned NumArgs) : Kind(Pack){
    this->Args.Args = Args;
    this->Args.NumArgs = NumArgs;
  }

  static TemplateArgument getEmptyPack() {
    return TemplateArgument((TemplateArgument*)0, 0);
  }

  /// \brief Create a new template argument pack by copying the given set of
  /// template arguments.
  static TemplateArgument CreatePackCopy(ASTContext &Context,
                                         const TemplateArgument *Args,
                                         unsigned NumArgs);
  
  /// \brief Return the kind of stored template argument.
  ArgKind getKind() const { return (ArgKind)Kind; }

  /// \brief Determine whether this template argument has no value.
  bool isNull() const { return Kind == Null; }

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
    assert(Kind == Type && "Unexpected kind");
    return QualType::getFromOpaquePtr(reinterpret_cast<void*>(TypeOrValue));
  }

  /// \brief Retrieve the declaration for a declaration non-type
  /// template argument.
  ValueDecl *getAsDecl() const {
    assert(Kind == Declaration && "Unexpected kind");
    return DeclArg.D;
  }

  /// \brief Retrieve whether a declaration is binding to a
  /// reference parameter in a declaration non-type template argument.
  bool isDeclForReferenceParam() const {
    assert(Kind == Declaration && "Unexpected kind");
    return DeclArg.ForRefParam;
  }

  /// \brief Retrieve the type for null non-type template argument.
  QualType getNullPtrType() const {
    assert(Kind == NullPtr && "Unexpected kind");
    return QualType::getFromOpaquePtr(reinterpret_cast<void*>(TypeOrValue));
  }

  /// \brief Retrieve the template name for a template name argument.
  TemplateName getAsTemplate() const {
    assert(Kind == Template && "Unexpected kind");
    return TemplateName::getFromVoidPointer(TemplateArg.Name);
  }

  /// \brief Retrieve the template argument as a template name; if the argument
  /// is a pack expansion, return the pattern as a template name.
  TemplateName getAsTemplateOrTemplatePattern() const {
    assert((Kind == Template || Kind == TemplateExpansion) &&
           "Unexpected kind");
    
    return TemplateName::getFromVoidPointer(TemplateArg.Name);
  }

  /// \brief Retrieve the number of expansions that a template template argument
  /// expansion will produce, if known.
  llvm::Optional<unsigned> getNumTemplateExpansions() const;
  
  /// \brief Retrieve the template argument as an integral value.
  // FIXME: Provide a way to read the integral data without copying the value.
  llvm::APSInt getAsIntegral() const {
    assert(Kind == Integral && "Unexpected kind");
    using namespace llvm;
    if (Integer.BitWidth <= 64)
      return APSInt(APInt(Integer.BitWidth, Integer.VAL), Integer.IsUnsigned);

    unsigned NumWords = APInt::getNumWords(Integer.BitWidth);
    return APSInt(APInt(Integer.BitWidth, makeArrayRef(Integer.pVal, NumWords)),
                  Integer.IsUnsigned);
  }

  /// \brief Retrieve the type of the integral value.
  QualType getIntegralType() const {
    assert(Kind == Integral && "Unexpected kind");
    return QualType::getFromOpaquePtr(Integer.Type);
  }

  void setIntegralType(QualType T) {
    assert(Kind == Integral && "Unexpected kind");
    Integer.Type = T.getAsOpaquePtr();
  }

  /// \brief Retrieve the template argument as an expression.
  Expr *getAsExpr() const {
    assert(Kind == Expression && "Unexpected kind");
    return reinterpret_cast<Expr *>(TypeOrValue);
  }

  /// \brief Iterator that traverses the elements of a template argument pack.
  typedef const TemplateArgument * pack_iterator;

  /// \brief Iterator referencing the first argument of a template argument
  /// pack.
  pack_iterator pack_begin() const {
    assert(Kind == Pack);
    return Args.Args;
  }

  /// \brief Iterator referencing one past the last argument of a template
  /// argument pack.
  pack_iterator pack_end() const {
    assert(Kind == Pack);
    return Args.Args + Args.NumArgs;
  }

  /// \brief The number of template arguments in the given template argument
  /// pack.
  unsigned pack_size() const {
    assert(Kind == Pack);
    return Args.NumArgs;
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
  union {
    Expr *Expression;
    TypeSourceInfo *Declarator;
    struct {
      // FIXME: We'd like to just use the qualifier in the TemplateName,
      // but template arguments get canonicalized too quickly.
      NestedNameSpecifier *Qualifier;
      void *QualifierLocData;
      unsigned TemplateNameLoc;
      unsigned EllipsisLoc;
    } Template;
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
  
  /// \brief When the template argument is a pack expansion, returns 
  /// the pattern of the pack expansion.
  ///
  /// \param Ellipsis Will be set to the location of the ellipsis.
  ///
  /// \param NumExpansions Will be set to the number of expansions that will
  /// be generated from this pack expansion, if known a priori.
  TemplateArgumentLoc getPackExpansionPattern(SourceLocation &Ellipsis,
                                        llvm::Optional<unsigned> &NumExpansions,
                                              ASTContext &Context) const;
};

/// A convenient class for passing around template argument
/// information.  Designed to be passed by reference.
class TemplateArgumentListInfo {
  SmallVector<TemplateArgumentLoc, 8> Arguments;
  SourceLocation LAngleLoc;
  SourceLocation RAngleLoc;

  // This can leak if used in an AST node, use ASTTemplateArgumentListInfo
  // instead.
  void* operator new(size_t bytes, ASTContext& C);

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

  const TemplateArgumentLoc &operator[](unsigned I) const {
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
    void *Aligner;
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
