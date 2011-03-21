//===--- Sema.h - Semantic Analysis & AST Building --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Sema class, which performs semantic analysis and
// builds ASTs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMA_H
#define LLVM_CLANG_SEMA_SEMA_H

#include "clang/Sema/Ownership.h"
#include "clang/Sema/AnalysisBasedWarnings.h"
#include "clang/Sema/IdentifierResolver.h"
#include "clang/Sema/ObjCMethodList.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/AST/Expr.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/TemplateKinds.h"
#include "clang/Basic/TypeTraits.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include <deque>
#include <string>

namespace llvm {
  class APSInt;
  template <typename ValueT> struct DenseMapInfo;
  template <typename ValueT, typename ValueInfoT> class DenseSet;
}

namespace clang {
  class ADLResult;
  class ASTConsumer;
  class ASTContext;
  class ArrayType;
  class AttributeList;
  class BlockDecl;
  class CXXBasePath;
  class CXXBasePaths;
  typedef llvm::SmallVector<CXXBaseSpecifier*, 4> CXXCastPath;
  class CXXConstructorDecl;
  class CXXConversionDecl;
  class CXXDestructorDecl;
  class CXXFieldCollector;
  class CXXMemberCallExpr;
  class CXXMethodDecl;
  class CXXScopeSpec;
  class CXXTemporary;
  class CXXTryStmt;
  class CallExpr;
  class ClassTemplateDecl;
  class ClassTemplatePartialSpecializationDecl;
  class ClassTemplateSpecializationDecl;
  class CodeCompleteConsumer;
  class CodeCompletionAllocator;
  class CodeCompletionResult;
  class Decl;
  class DeclAccessPair;
  class DeclContext;
  class DeclRefExpr;
  class DeclaratorDecl;
  class DeducedTemplateArgument;
  class DependentDiagnostic;
  class DesignatedInitExpr;
  class Designation;
  class EnumConstantDecl;
  class Expr;
  class ExtVectorType;
  class ExternalSemaSource;
  class FormatAttr;
  class FriendDecl;
  class FunctionDecl;
  class FunctionProtoType;
  class FunctionTemplateDecl;
  class ImplicitConversionSequence;
  class InitListExpr;
  class InitializationKind;
  class InitializationSequence;
  class InitializedEntity;
  class IntegerLiteral;
  class LabelStmt;
  class LangOptions;
  class LocalInstantiationScope;
  class LookupResult;
  class MacroInfo;
  class MultiLevelTemplateArgumentList;
  class NamedDecl;
  class NonNullAttr;
  class ObjCCategoryDecl;
  class ObjCCategoryImplDecl;
  class ObjCCompatibleAliasDecl;
  class ObjCContainerDecl;
  class ObjCImplDecl;
  class ObjCImplementationDecl;
  class ObjCInterfaceDecl;
  class ObjCIvarDecl;
  template <class T> class ObjCList;
  class ObjCMethodDecl;
  class ObjCPropertyDecl;
  class ObjCProtocolDecl;
  class OverloadCandidateSet;
  class ParenListExpr;
  class ParmVarDecl;
  class Preprocessor;
  class PseudoDestructorTypeStorage;
  class QualType;
  class StandardConversionSequence;
  class Stmt;
  class StringLiteral;
  class SwitchStmt;
  class TargetAttributesSema;
  class TemplateArgument;
  class TemplateArgumentList;
  class TemplateArgumentLoc;
  class TemplateDecl;
  class TemplateParameterList;
  class TemplatePartialOrderingContext;
  class TemplateTemplateParmDecl;
  class Token;
  class TypedefDecl;
  class TypeLoc;
  class UnqualifiedId;
  class UnresolvedLookupExpr;
  class UnresolvedMemberExpr;
  class UnresolvedSetImpl;
  class UnresolvedSetIterator;
  class UsingDecl;
  class UsingShadowDecl;
  class ValueDecl;
  class VarDecl;
  class VisibilityAttr;
  class VisibleDeclConsumer;
  class IndirectFieldDecl;
  
namespace sema {
  class AccessedEntity;
  class BlockScopeInfo;
  class DelayedDiagnostic;
  class FunctionScopeInfo;
  class TemplateDeductionInfo;
}

/// \brief Holds a QualType and a TypeSourceInfo* that came out of a declarator
/// parsing.
///
/// LocInfoType is a "transient" type, only needed for passing to/from Parser
/// and Sema, when we want to preserve type source info for a parsed type.
/// It will not participate in the type system semantics in any way.
class LocInfoType : public Type {
  enum {
    // The last number that can fit in Type's TC.
    // Avoids conflict with an existing Type class.
    LocInfo = Type::TypeLast + 1
  };

  TypeSourceInfo *DeclInfo;

  LocInfoType(QualType ty, TypeSourceInfo *TInfo)
    : Type((TypeClass)LocInfo, ty, ty->isDependentType(), 
           ty->isVariablyModifiedType(),
           ty->containsUnexpandedParameterPack()), 
      DeclInfo(TInfo) {
    assert(getTypeClass() == (TypeClass)LocInfo && "LocInfo didn't fit in TC?");
  }
  friend class Sema;

public:
  QualType getType() const { return getCanonicalTypeInternal(); }
  TypeSourceInfo *getTypeSourceInfo() const { return DeclInfo; }

  void getAsStringInternal(std::string &Str,
                                   const PrintingPolicy &Policy) const;

  static bool classof(const Type *T) {
    return T->getTypeClass() == (TypeClass)LocInfo;
  }
  static bool classof(const LocInfoType *) { return true; }
};

// FIXME: No way to easily map from TemplateTypeParmTypes to
// TemplateTypeParmDecls, so we have this horrible PointerUnion.
typedef std::pair<llvm::PointerUnion<const TemplateTypeParmType*, NamedDecl*>,
                  SourceLocation> UnexpandedParameterPack;

/// Sema - This implements semantic analysis and AST building for C.
class Sema {
  Sema(const Sema&);           // DO NOT IMPLEMENT
  void operator=(const Sema&); // DO NOT IMPLEMENT
  mutable const TargetAttributesSema* TheTargetAttributesSema;
public:
  typedef OpaquePtr<DeclGroupRef> DeclGroupPtrTy;
  typedef OpaquePtr<TemplateName> TemplateTy;
  typedef OpaquePtr<QualType> TypeTy;
  typedef Attr AttrTy;
  typedef CXXBaseSpecifier BaseTy;
  typedef CXXCtorInitializer MemInitTy;
  typedef Expr ExprTy;
  typedef Stmt StmtTy;
  typedef TemplateParameterList TemplateParamsTy;
  typedef NestedNameSpecifier CXXScopeTy;

  OpenCLOptions OpenCLFeatures;
  FPOptions FPFeatures;

  const LangOptions &LangOpts;
  Preprocessor &PP;
  ASTContext &Context;
  ASTConsumer &Consumer;
  Diagnostic &Diags;
  SourceManager &SourceMgr;

  /// \brief Source of additional semantic information.
  ExternalSemaSource *ExternalSource;

  /// \brief Code-completion consumer.
  CodeCompleteConsumer *CodeCompleter;

  /// CurContext - This is the current declaration context of parsing.
  DeclContext *CurContext;

  /// VAListTagName - The declaration name corresponding to __va_list_tag.
  /// This is used as part of a hack to omit that class from ADL results.
  DeclarationName VAListTagName;

  /// PackContext - Manages the stack for #pragma pack. An alignment
  /// of 0 indicates default alignment.
  void *PackContext; // Really a "PragmaPackStack*"

  /// VisContext - Manages the stack for #pragma GCC visibility.
  void *VisContext; // Really a "PragmaVisStack*"

  /// \brief Stack containing information about each of the nested
  /// function, block, and method scopes that are currently active.
  ///
  /// This array is never empty.  Clients should ignore the first
  /// element, which is used to cache a single FunctionScopeInfo
  /// that's used to parse every top-level function.
  llvm::SmallVector<sema::FunctionScopeInfo *, 4> FunctionScopes;

  /// ExprTemporaries - This is the stack of temporaries that are created by
  /// the current full expression.
  llvm::SmallVector<CXXTemporary*, 8> ExprTemporaries;

  /// ExtVectorDecls - This is a list all the extended vector types. This allows
  /// us to associate a raw vector type with one of the ext_vector type names.
  /// This is only necessary for issuing pretty diagnostics.
  llvm::SmallVector<TypedefDecl*, 24> ExtVectorDecls;

  /// FieldCollector - Collects CXXFieldDecls during parsing of C++ classes.
  llvm::OwningPtr<CXXFieldCollector> FieldCollector;

  typedef llvm::SmallPtrSet<const CXXRecordDecl*, 8> RecordDeclSetTy;

  /// PureVirtualClassDiagSet - a set of class declarations which we have
  /// emitted a list of pure virtual functions. Used to prevent emitting the
  /// same list more than once.
  llvm::OwningPtr<RecordDeclSetTy> PureVirtualClassDiagSet;

  /// ParsingInitForAutoVars - a set of declarations with auto types for which
  /// we are currently parsing the initializer.
  llvm::SmallPtrSet<const Decl*, 4> ParsingInitForAutoVars;

  /// \brief A mapping from external names to the most recent
  /// locally-scoped external declaration with that name.
  ///
  /// This map contains external declarations introduced in local
  /// scoped, e.g.,
  ///
  /// \code
  /// void f() {
  ///   void foo(int, int);
  /// }
  /// \endcode
  ///
  /// Here, the name "foo" will be associated with the declaration on
  /// "foo" within f. This name is not visible outside of
  /// "f". However, we still find it in two cases:
  ///
  ///   - If we are declaring another external with the name "foo", we
  ///     can find "foo" as a previous declaration, so that the types
  ///     of this external declaration can be checked for
  ///     compatibility.
  ///
  ///   - If we would implicitly declare "foo" (e.g., due to a call to
  ///     "foo" in C when no prototype or definition is visible), then
  ///     we find this declaration of "foo" and complain that it is
  ///     not visible.
  llvm::DenseMap<DeclarationName, NamedDecl *> LocallyScopedExternalDecls;

  /// \brief All the tentative definitions encountered in the TU.
  llvm::SmallVector<VarDecl *, 2> TentativeDefinitions;

  /// \brief The set of file scoped decls seen so far that have not been used
  /// and must warn if not used. Only contains the first declaration.
  llvm::SmallVector<const DeclaratorDecl*, 4> UnusedFileScopedDecls;

  class DelayedDiagnostics;

  class ParsingDeclState {
    unsigned SavedStackSize;
    friend class Sema::DelayedDiagnostics;
  };

  class ProcessingContextState {
    unsigned SavedParsingDepth;
    unsigned SavedActiveStackBase;
    friend class Sema::DelayedDiagnostics;
  };

  /// A class which encapsulates the logic for delaying diagnostics
  /// during parsing and other processing.
  class DelayedDiagnostics {
    /// \brief The stack of diagnostics that were delayed due to being
    /// produced during the parsing of a declaration.
    sema::DelayedDiagnostic *Stack;

    /// \brief The number of objects on the delayed-diagnostics stack.
    unsigned StackSize;

    /// \brief The current capacity of the delayed-diagnostics stack.
    unsigned StackCapacity;

    /// \brief The index of the first "active" delayed diagnostic in
    /// the stack.  When parsing class definitions, we ignore active
    /// delayed diagnostics from the surrounding context.
    unsigned ActiveStackBase;

    /// \brief The depth of the declarations we're currently parsing.
    /// This gets saved and reset whenever we enter a class definition.
    unsigned ParsingDepth;

  public:
    DelayedDiagnostics() : Stack(0), StackSize(0), StackCapacity(0),
      ActiveStackBase(0), ParsingDepth(0) {}

    ~DelayedDiagnostics() {
      delete[] reinterpret_cast<char*>(Stack);
    }

    /// Adds a delayed diagnostic.
    void add(const sema::DelayedDiagnostic &diag);

    /// Determines whether diagnostics should be delayed.
    bool shouldDelayDiagnostics() { return ParsingDepth > 0; }

    /// Observe that we've started parsing a declaration.  Access and
    /// deprecation diagnostics will be delayed; when the declaration
    /// is completed, all active delayed diagnostics will be evaluated
    /// in its context, and then active diagnostics stack will be
    /// popped down to the saved depth.
    ParsingDeclState pushParsingDecl() {
      ParsingDepth++;

      ParsingDeclState state;
      state.SavedStackSize = StackSize;
      return state;
    }

    /// Observe that we're completed parsing a declaration.
    static void popParsingDecl(Sema &S, ParsingDeclState state, Decl *decl);

    /// Observe that we've started processing a different context, the
    /// contents of which are semantically separate from the
    /// declarations it may lexically appear in.  This sets aside the
    /// current stack of active diagnostics and starts afresh.
    ProcessingContextState pushContext() {
      assert(StackSize >= ActiveStackBase);

      ProcessingContextState state;
      state.SavedParsingDepth = ParsingDepth;
      state.SavedActiveStackBase = ActiveStackBase;

      ActiveStackBase = StackSize;
      ParsingDepth = 0;

      return state;
    }

    /// Observe that we've stopped processing a context.  This
    /// restores the previous stack of active diagnostics.
    void popContext(ProcessingContextState state) {
      assert(ActiveStackBase == StackSize);
      assert(ParsingDepth == 0);
      ActiveStackBase = state.SavedActiveStackBase;
      ParsingDepth = state.SavedParsingDepth;
    }  
  } DelayedDiagnostics;

  /// A RAII object to temporarily push a declaration context.
  class ContextRAII {
  private:
    Sema &S;
    DeclContext *SavedContext;
    ProcessingContextState SavedContextState;
    
  public:
    ContextRAII(Sema &S, DeclContext *ContextToPush)
      : S(S), SavedContext(S.CurContext), 
        SavedContextState(S.DelayedDiagnostics.pushContext()) 
    {
      assert(ContextToPush && "pushing null context");
      S.CurContext = ContextToPush;
    }

    void pop() {
      if (!SavedContext) return;
      S.CurContext = SavedContext;
      S.DelayedDiagnostics.popContext(SavedContextState);
      SavedContext = 0;
    }

    ~ContextRAII() {
      pop();
    }
  };

  /// WeakUndeclaredIdentifiers - Identifiers contained in
  /// #pragma weak before declared. rare. may alias another
  /// identifier, declared or undeclared
  class WeakInfo {
    IdentifierInfo *alias;  // alias (optional)
    SourceLocation loc;     // for diagnostics
    bool used;              // identifier later declared?
  public:
    WeakInfo()
      : alias(0), loc(SourceLocation()), used(false) {}
    WeakInfo(IdentifierInfo *Alias, SourceLocation Loc)
      : alias(Alias), loc(Loc), used(false) {}
    inline IdentifierInfo * getAlias() const { return alias; }
    inline SourceLocation getLocation() const { return loc; }
    void setUsed(bool Used=true) { used = Used; }
    inline bool getUsed() { return used; }
    bool operator==(WeakInfo RHS) const {
      return alias == RHS.getAlias() && loc == RHS.getLocation();
    }
    bool operator!=(WeakInfo RHS) const { return !(*this == RHS); }
  };
  llvm::DenseMap<IdentifierInfo*,WeakInfo> WeakUndeclaredIdentifiers;

  /// WeakTopLevelDecl - Translation-unit scoped declarations generated by
  /// #pragma weak during processing of other Decls.
  /// I couldn't figure out a clean way to generate these in-line, so
  /// we store them here and handle separately -- which is a hack.
  /// It would be best to refactor this.
  llvm::SmallVector<Decl*,2> WeakTopLevelDecl;

  IdentifierResolver IdResolver;

  /// Translation Unit Scope - useful to Objective-C actions that need
  /// to lookup file scope declarations in the "ordinary" C decl namespace.
  /// For example, user-defined classes, built-in "id" type, etc.
  Scope *TUScope;

  /// \brief The C++ "std" namespace, where the standard library resides.
  LazyDeclPtr StdNamespace;

  /// \brief The C++ "std::bad_alloc" class, which is defined by the C++
  /// standard library.
  LazyDeclPtr StdBadAlloc;

  /// \brief The C++ "type_info" declaration, which is defined in <typeinfo>.
  RecordDecl *CXXTypeInfoDecl;
  
  /// \brief The MSVC "_GUID" struct, which is defined in MSVC header files.
  RecordDecl *MSVCGuidDecl;

  /// A flag to remember whether the implicit forms of operator new and delete
  /// have been declared.
  bool GlobalNewDeleteDeclared;

  /// \brief The set of declarations that have been referenced within
  /// a potentially evaluated expression.
  typedef llvm::SmallVector<std::pair<SourceLocation, Decl *>, 10>
    PotentiallyReferencedDecls;

  /// \brief A set of diagnostics that may be emitted.
  typedef llvm::SmallVector<std::pair<SourceLocation, PartialDiagnostic>, 10>
    PotentiallyEmittedDiagnostics;

  /// \brief Describes how the expressions currently being parsed are
  /// evaluated at run-time, if at all.
  enum ExpressionEvaluationContext {
    /// \brief The current expression and its subexpressions occur within an
    /// unevaluated operand (C++0x [expr]p8), such as a constant expression
    /// or the subexpression of \c sizeof, where the type or the value of the
    /// expression may be significant but no code will be generated to evaluate
    /// the value of the expression at run time.
    Unevaluated,

    /// \brief The current expression is potentially evaluated at run time,
    /// which means that code may be generated to evaluate the value of the
    /// expression at run time.
    PotentiallyEvaluated,

    /// \brief The current expression may be potentially evaluated or it may
    /// be unevaluated, but it is impossible to tell from the lexical context.
    /// This evaluation context is used primary for the operand of the C++
    /// \c typeid expression, whose argument is potentially evaluated only when
    /// it is an lvalue of polymorphic class type (C++ [basic.def.odr]p2).
    PotentiallyPotentiallyEvaluated,
    
    /// \brief The current expression is potentially evaluated, but any
    /// declarations referenced inside that expression are only used if
    /// in fact the current expression is used.
    ///
    /// This value is used when parsing default function arguments, for which
    /// we would like to provide diagnostics (e.g., passing non-POD arguments
    /// through varargs) but do not want to mark declarations as "referenced"
    /// until the default argument is used.
    PotentiallyEvaluatedIfUsed
  };

  /// \brief Data structure used to record current or nested
  /// expression evaluation contexts.
  struct ExpressionEvaluationContextRecord {
    /// \brief The expression evaluation context.
    ExpressionEvaluationContext Context;

    /// \brief The number of temporaries that were active when we
    /// entered this expression evaluation context.
    unsigned NumTemporaries;

    /// \brief The set of declarations referenced within a
    /// potentially potentially-evaluated context.
    ///
    /// When leaving a potentially potentially-evaluated context, each
    /// of these elements will be as referenced if the corresponding
    /// potentially potentially evaluated expression is potentially
    /// evaluated.
    PotentiallyReferencedDecls *PotentiallyReferenced;

    /// \brief The set of diagnostics to emit should this potentially
    /// potentially-evaluated context become evaluated.
    PotentiallyEmittedDiagnostics *PotentiallyDiagnosed;

    ExpressionEvaluationContextRecord(ExpressionEvaluationContext Context,
                                      unsigned NumTemporaries)
      : Context(Context), NumTemporaries(NumTemporaries),
        PotentiallyReferenced(0), PotentiallyDiagnosed(0) { }

    void addReferencedDecl(SourceLocation Loc, Decl *Decl) {
      if (!PotentiallyReferenced)
        PotentiallyReferenced = new PotentiallyReferencedDecls;
      PotentiallyReferenced->push_back(std::make_pair(Loc, Decl));
    }

    void addDiagnostic(SourceLocation Loc, const PartialDiagnostic &PD) {
      if (!PotentiallyDiagnosed)
        PotentiallyDiagnosed = new PotentiallyEmittedDiagnostics;
      PotentiallyDiagnosed->push_back(std::make_pair(Loc, PD));
    }

    void Destroy() {
      delete PotentiallyReferenced;
      delete PotentiallyDiagnosed;
      PotentiallyReferenced = 0;
      PotentiallyDiagnosed = 0;
    }
  };

  /// A stack of expression evaluation contexts.
  llvm::SmallVector<ExpressionEvaluationContextRecord, 8> ExprEvalContexts;

  /// \brief Whether the code handled by Sema should be considered a
  /// complete translation unit or not.
  ///
  /// When true (which is generally the case), Sema will perform
  /// end-of-translation-unit semantic tasks (such as creating
  /// initializers for tentative definitions in C) once parsing has
  /// completed. This flag will be false when building PCH files,
  /// since a PCH file is by definition not a complete translation
  /// unit.
  bool CompleteTranslationUnit;

  llvm::BumpPtrAllocator BumpAlloc;

  /// \brief The number of SFINAE diagnostics that have been trapped.
  unsigned NumSFINAEErrors;

  typedef llvm::DenseMap<ParmVarDecl *, llvm::SmallVector<ParmVarDecl *, 1> >
    UnparsedDefaultArgInstantiationsMap;
  
  /// \brief A mapping from parameters with unparsed default arguments to the
  /// set of instantiations of each parameter.
  ///
  /// This mapping is a temporary data structure used when parsing
  /// nested class templates or nested classes of class templates,
  /// where we might end up instantiating an inner class before the
  /// default arguments of its methods have been parsed. 
  UnparsedDefaultArgInstantiationsMap UnparsedDefaultArgInstantiations;
  
  // Contains the locations of the beginning of unparsed default
  // argument locations.
  llvm::DenseMap<ParmVarDecl *,SourceLocation> UnparsedDefaultArgLocs;

  /// UndefinedInternals - all the used, undefined objects with
  /// internal linkage in this translation unit.
  llvm::DenseMap<NamedDecl*, SourceLocation> UndefinedInternals;

  typedef std::pair<ObjCMethodList, ObjCMethodList> GlobalMethods;
  typedef llvm::DenseMap<Selector, GlobalMethods> GlobalMethodPool;

  /// Method Pool - allows efficient lookup when typechecking messages to "id".
  /// We need to maintain a list, since selectors can have differing signatures
  /// across classes. In Cocoa, this happens to be extremely uncommon (only 1%
  /// of selectors are "overloaded").
  GlobalMethodPool MethodPool;

  /// Method selectors used in a @selector expression. Used for implementation 
  /// of -Wselector.
  llvm::DenseMap<Selector, SourceLocation> ReferencedSelectors;

  GlobalMethodPool::iterator ReadMethodPool(Selector Sel);

  /// Private Helper predicate to check for 'self'.
  bool isSelfExpr(Expr *RExpr);
public:
  Sema(Preprocessor &pp, ASTContext &ctxt, ASTConsumer &consumer,
       bool CompleteTranslationUnit = true,
       CodeCompleteConsumer *CompletionConsumer = 0);
  ~Sema();
  
  /// \brief Perform initialization that occurs after the parser has been
  /// initialized but before it parses anything.
  void Initialize();
  
  const LangOptions &getLangOptions() const { return LangOpts; }
  OpenCLOptions &getOpenCLOptions() { return OpenCLFeatures; }
  FPOptions     &getFPOptions() { return FPFeatures; }

  Diagnostic &getDiagnostics() const { return Diags; }
  SourceManager &getSourceManager() const { return SourceMgr; }
  const TargetAttributesSema &getTargetAttributesSema() const;
  Preprocessor &getPreprocessor() const { return PP; }
  ASTContext &getASTContext() const { return Context; }
  ASTConsumer &getASTConsumer() const { return Consumer; }
  
  /// \brief Helper class that creates diagnostics with optional
  /// template instantiation stacks.
  ///
  /// This class provides a wrapper around the basic DiagnosticBuilder
  /// class that emits diagnostics. SemaDiagnosticBuilder is
  /// responsible for emitting the diagnostic (as DiagnosticBuilder
  /// does) and, if the diagnostic comes from inside a template
  /// instantiation, printing the template instantiation stack as
  /// well.
  class SemaDiagnosticBuilder : public DiagnosticBuilder {
    Sema &SemaRef;
    unsigned DiagID;

  public:
    SemaDiagnosticBuilder(DiagnosticBuilder &DB, Sema &SemaRef, unsigned DiagID)
      : DiagnosticBuilder(DB), SemaRef(SemaRef), DiagID(DiagID) { }

    explicit SemaDiagnosticBuilder(Sema &SemaRef)
      : DiagnosticBuilder(DiagnosticBuilder::Suppress), SemaRef(SemaRef) { }

    ~SemaDiagnosticBuilder();
  };

  /// \brief Emit a diagnostic.
  SemaDiagnosticBuilder Diag(SourceLocation Loc, unsigned DiagID);

  /// \brief Emit a partial diagnostic.
  SemaDiagnosticBuilder Diag(SourceLocation Loc, const PartialDiagnostic& PD);

  /// \brief Build a partial diagnostic.
  PartialDiagnostic PDiag(unsigned DiagID = 0); // in SemaInternal.h

  ExprResult Owned(Expr* E) { return E; }
  ExprResult Owned(ExprResult R) { return R; }
  StmtResult Owned(Stmt* S) { return S; }

  void ActOnEndOfTranslationUnit();

  Scope *getScopeForContext(DeclContext *Ctx);

  void PushFunctionScope();
  void PushBlockScope(Scope *BlockScope, BlockDecl *Block);
  void PopFunctionOrBlockScope(const sema::AnalysisBasedWarnings::Policy *WP =0,
                               const Decl *D = 0, const BlockExpr *blkExpr = 0);

  sema::FunctionScopeInfo *getCurFunction() const {
    return FunctionScopes.back();
  }

  bool hasAnyErrorsInThisFunction() const;

  /// \brief Retrieve the current block, if any.
  sema::BlockScopeInfo *getCurBlock();

  /// WeakTopLevelDeclDecls - access to #pragma weak-generated Decls
  llvm::SmallVector<Decl*,2> &WeakTopLevelDecls() { return WeakTopLevelDecl; }

  //===--------------------------------------------------------------------===//
  // Type Analysis / Processing: SemaType.cpp.
  //

  QualType adjustParameterType(QualType T);
  QualType BuildQualifiedType(QualType T, SourceLocation Loc, Qualifiers Qs);
  QualType BuildQualifiedType(QualType T, SourceLocation Loc, unsigned CVR) {
    return BuildQualifiedType(T, Loc, Qualifiers::fromCVRMask(CVR));
  }
  QualType BuildPointerType(QualType T,
                            SourceLocation Loc, DeclarationName Entity);
  QualType BuildReferenceType(QualType T, bool LValueRef,
                              SourceLocation Loc, DeclarationName Entity);
  QualType BuildArrayType(QualType T, ArrayType::ArraySizeModifier ASM,
                          Expr *ArraySize, unsigned Quals,
                          SourceRange Brackets, DeclarationName Entity);
  QualType BuildExtVectorType(QualType T, Expr *ArraySize,
                              SourceLocation AttrLoc);
  QualType BuildFunctionType(QualType T,
                             QualType *ParamTypes, unsigned NumParamTypes,
                             bool Variadic, unsigned Quals, 
                             RefQualifierKind RefQualifier,
                             SourceLocation Loc, DeclarationName Entity,
                             FunctionType::ExtInfo Info);
  QualType BuildMemberPointerType(QualType T, QualType Class,
                                  SourceLocation Loc,
                                  DeclarationName Entity);
  QualType BuildBlockPointerType(QualType T,
                                 SourceLocation Loc, DeclarationName Entity);
  QualType BuildParenType(QualType T);

  TypeSourceInfo *GetTypeForDeclarator(Declarator &D, Scope *S,
                                       TagDecl **OwnedDecl = 0,
                                       bool AllowAutoInTypeName = false);
  TypeSourceInfo *GetTypeSourceInfoForDeclarator(Declarator &D, QualType T,
                                               TypeSourceInfo *ReturnTypeInfo);
  /// \brief Package the given type and TSI into a ParsedType.
  ParsedType CreateParsedType(QualType T, TypeSourceInfo *TInfo);
  DeclarationNameInfo GetNameForDeclarator(Declarator &D);
  DeclarationNameInfo GetNameFromUnqualifiedId(const UnqualifiedId &Name);
  static QualType GetTypeFromParser(ParsedType Ty, TypeSourceInfo **TInfo = 0);
  bool CheckSpecifiedExceptionType(QualType T, const SourceRange &Range);
  bool CheckDistantExceptionSpec(QualType T);
  bool CheckEquivalentExceptionSpec(FunctionDecl *Old, FunctionDecl *New);
  bool CheckEquivalentExceptionSpec(
      const FunctionProtoType *Old, SourceLocation OldLoc,
      const FunctionProtoType *New, SourceLocation NewLoc);
  bool CheckEquivalentExceptionSpec(
      const PartialDiagnostic &DiagID, const PartialDiagnostic & NoteID,
      const FunctionProtoType *Old, SourceLocation OldLoc,
      const FunctionProtoType *New, SourceLocation NewLoc,
      bool *MissingExceptionSpecification = 0,
      bool *MissingEmptyExceptionSpecification = 0);
  bool CheckExceptionSpecSubset(
      const PartialDiagnostic &DiagID, const PartialDiagnostic & NoteID,
      const FunctionProtoType *Superset, SourceLocation SuperLoc,
      const FunctionProtoType *Subset, SourceLocation SubLoc);
  bool CheckParamExceptionSpec(const PartialDiagnostic & NoteID,
      const FunctionProtoType *Target, SourceLocation TargetLoc,
      const FunctionProtoType *Source, SourceLocation SourceLoc);

  TypeResult ActOnTypeName(Scope *S, Declarator &D);

  bool RequireCompleteType(SourceLocation Loc, QualType T,
                           const PartialDiagnostic &PD,
                           std::pair<SourceLocation, PartialDiagnostic> Note);
  bool RequireCompleteType(SourceLocation Loc, QualType T,
                           const PartialDiagnostic &PD);
  bool RequireCompleteType(SourceLocation Loc, QualType T,
                           unsigned DiagID);

  QualType getElaboratedType(ElaboratedTypeKeyword Keyword,
                             const CXXScopeSpec &SS, QualType T);

  QualType BuildTypeofExprType(Expr *E, SourceLocation Loc);
  QualType BuildDecltypeType(Expr *E, SourceLocation Loc);

  //===--------------------------------------------------------------------===//
  // Symbol table / Decl tracking callbacks: SemaDecl.cpp.
  //

  DeclGroupPtrTy ConvertDeclToDeclGroup(Decl *Ptr);

  void DiagnoseUseOfUnimplementedSelectors();

  ParsedType getTypeName(IdentifierInfo &II, SourceLocation NameLoc,
                         Scope *S, CXXScopeSpec *SS = 0,
                         bool isClassName = false,
                         bool HasTrailingDot = false,
                         ParsedType ObjectType = ParsedType(),
                         bool WantNontrivialTypeSourceInfo = false);
  TypeSpecifierType isTagName(IdentifierInfo &II, Scope *S);
  bool DiagnoseUnknownTypeName(const IdentifierInfo &II,
                               SourceLocation IILoc,
                               Scope *S,
                               CXXScopeSpec *SS,
                               ParsedType &SuggestedType);

  Decl *ActOnDeclarator(Scope *S, Declarator &D);

  Decl *HandleDeclarator(Scope *S, Declarator &D,
                         MultiTemplateParamsArg TemplateParameterLists,
                         bool IsFunctionDefinition);
  void RegisterLocallyScopedExternCDecl(NamedDecl *ND,
                                        const LookupResult &Previous,
                                        Scope *S);
  void DiagnoseFunctionSpecifiers(Declarator& D);
  void CheckShadow(Scope *S, VarDecl *D, const LookupResult& R);
  void CheckShadow(Scope *S, VarDecl *D);
  void CheckCastAlign(Expr *Op, QualType T, SourceRange TRange);
  NamedDecl* ActOnTypedefDeclarator(Scope* S, Declarator& D, DeclContext* DC,
                                    QualType R, TypeSourceInfo *TInfo,
                                    LookupResult &Previous, bool &Redeclaration);
  NamedDecl* ActOnVariableDeclarator(Scope* S, Declarator& D, DeclContext* DC,
                                     QualType R, TypeSourceInfo *TInfo,
                                     LookupResult &Previous,
                                     MultiTemplateParamsArg TemplateParamLists,
                                     bool &Redeclaration);
  void CheckVariableDeclaration(VarDecl *NewVD, LookupResult &Previous,
                                bool &Redeclaration);
  void CheckCompleteVariableDeclaration(VarDecl *var);
  NamedDecl* ActOnFunctionDeclarator(Scope* S, Declarator& D, DeclContext* DC,
                                     QualType R, TypeSourceInfo *TInfo,
                                     LookupResult &Previous,
                                     MultiTemplateParamsArg TemplateParamLists,
                                     bool IsFunctionDefinition,
                                     bool &Redeclaration);
  bool AddOverriddenMethods(CXXRecordDecl *DC, CXXMethodDecl *MD);
  void DiagnoseHiddenVirtualMethods(CXXRecordDecl *DC, CXXMethodDecl *MD);
  void CheckFunctionDeclaration(Scope *S,
                                FunctionDecl *NewFD, LookupResult &Previous,
                                bool IsExplicitSpecialization,
                                bool &Redeclaration);
  void CheckMain(FunctionDecl *FD);
  Decl *ActOnParamDeclarator(Scope *S, Declarator &D);
  ParmVarDecl *BuildParmVarDeclForTypedef(DeclContext *DC,
                                          SourceLocation Loc,
                                          QualType T);
  ParmVarDecl *CheckParameter(DeclContext *DC,
                              TypeSourceInfo *TSInfo, QualType T,
                              IdentifierInfo *Name,
                              SourceLocation NameLoc,
                              StorageClass SC,
                              StorageClass SCAsWritten);
  void ActOnParamDefaultArgument(Decl *param,
                                 SourceLocation EqualLoc,
                                 Expr *defarg);
  void ActOnParamUnparsedDefaultArgument(Decl *param,
                                         SourceLocation EqualLoc,
                                         SourceLocation ArgLoc);
  void ActOnParamDefaultArgumentError(Decl *param);
  bool SetParamDefaultArgument(ParmVarDecl *Param, Expr *DefaultArg,
                               SourceLocation EqualLoc);

  void AddInitializerToDecl(Decl *dcl, Expr *init, bool DirectInit,
                            bool TypeMayContainAuto);
  void ActOnUninitializedDecl(Decl *dcl, bool TypeMayContainAuto);
  void ActOnInitializerError(Decl *Dcl);
  void SetDeclDeleted(Decl *dcl, SourceLocation DelLoc);
  void FinalizeDeclaration(Decl *D);
  DeclGroupPtrTy FinalizeDeclaratorGroup(Scope *S, const DeclSpec &DS,
                                         Decl **Group,
                                         unsigned NumDecls);
  DeclGroupPtrTy BuildDeclaratorGroup(Decl **Group, unsigned NumDecls,
                                      bool TypeMayContainAuto = true);
  void ActOnFinishKNRParamDeclarations(Scope *S, Declarator &D,
                                       SourceLocation LocAfterDecls);
  Decl *ActOnStartOfFunctionDef(Scope *S, Declarator &D);
  Decl *ActOnStartOfFunctionDef(Scope *S, Decl *D);
  void ActOnStartOfObjCMethodDef(Scope *S, Decl *D);

  Decl *ActOnFinishFunctionBody(Decl *Decl, Stmt *Body);
  Decl *ActOnFinishFunctionBody(Decl *Decl, Stmt *Body, bool IsInstantiation);

  /// \brief Diagnose any unused parameters in the given sequence of
  /// ParmVarDecl pointers.
  void DiagnoseUnusedParameters(ParmVarDecl * const *Begin,
                                ParmVarDecl * const *End);

  /// \brief Diagnose whether the size of parameters or return value of a
  /// function or obj-c method definition is pass-by-value and larger than a
  /// specified threshold.
  void DiagnoseSizeOfParametersAndReturnValue(ParmVarDecl * const *Begin,
                                              ParmVarDecl * const *End,
                                              QualType ReturnTy,
                                              NamedDecl *D);

  void DiagnoseInvalidJumps(Stmt *Body);
  Decl *ActOnFileScopeAsmDecl(Expr *expr,
                              SourceLocation AsmLoc,
                              SourceLocation RParenLoc);

  /// Scope actions.
  void ActOnPopScope(SourceLocation Loc, Scope *S);
  void ActOnTranslationUnitScope(Scope *S);

  /// ParsedFreeStandingDeclSpec - This method is invoked when a declspec with
  /// no declarator (e.g. "struct foo;") is parsed.
  Decl *ParsedFreeStandingDeclSpec(Scope *S, AccessSpecifier AS,
                                   DeclSpec &DS);
  
  StmtResult ActOnVlaStmt(const DeclSpec &DS);

  Decl *BuildAnonymousStructOrUnion(Scope *S, DeclSpec &DS,
                                    AccessSpecifier AS,
                                    RecordDecl *Record);

  Decl *BuildMicrosoftCAnonymousStruct(Scope *S, DeclSpec &DS, 
                                       RecordDecl *Record);

  bool isAcceptableTagRedeclaration(const TagDecl *Previous,
                                    TagTypeKind NewTag,
                                    SourceLocation NewTagLoc,
                                    const IdentifierInfo &Name);

  enum TagUseKind {
    TUK_Reference,   // Reference to a tag:  'struct foo *X;'
    TUK_Declaration, // Fwd decl of a tag:   'struct foo;'
    TUK_Definition,  // Definition of a tag: 'struct foo { int X; } Y;'
    TUK_Friend       // Friend declaration:  'friend struct foo;'
  };

  Decl *ActOnTag(Scope *S, unsigned TagSpec, TagUseKind TUK,
                 SourceLocation KWLoc, CXXScopeSpec &SS,
                 IdentifierInfo *Name, SourceLocation NameLoc,
                 AttributeList *Attr, AccessSpecifier AS,
                 MultiTemplateParamsArg TemplateParameterLists,
                 bool &OwnedDecl, bool &IsDependent, bool ScopedEnum,
                 bool ScopedEnumUsesClassTag, TypeResult UnderlyingType);

  Decl *ActOnTemplatedFriendTag(Scope *S, SourceLocation FriendLoc,
                                unsigned TagSpec, SourceLocation TagLoc,
                                CXXScopeSpec &SS,
                                IdentifierInfo *Name, SourceLocation NameLoc,
                                AttributeList *Attr,
                                MultiTemplateParamsArg TempParamLists);

  TypeResult ActOnDependentTag(Scope *S,
                               unsigned TagSpec,
                               TagUseKind TUK,
                               const CXXScopeSpec &SS,
                               IdentifierInfo *Name,
                               SourceLocation TagLoc,
                               SourceLocation NameLoc);

  void ActOnDefs(Scope *S, Decl *TagD, SourceLocation DeclStart,
                 IdentifierInfo *ClassName,
                 llvm::SmallVectorImpl<Decl *> &Decls);
  Decl *ActOnField(Scope *S, Decl *TagD, SourceLocation DeclStart,
                   Declarator &D, Expr *BitfieldWidth);

  FieldDecl *HandleField(Scope *S, RecordDecl *TagD, SourceLocation DeclStart,
                         Declarator &D, Expr *BitfieldWidth,
                         AccessSpecifier AS);

  FieldDecl *CheckFieldDecl(DeclarationName Name, QualType T,
                            TypeSourceInfo *TInfo,
                            RecordDecl *Record, SourceLocation Loc,
                            bool Mutable, Expr *BitfieldWidth,
                            SourceLocation TSSL,
                            AccessSpecifier AS, NamedDecl *PrevDecl,
                            Declarator *D = 0);

  enum CXXSpecialMember {
    CXXInvalid = -1,
    CXXConstructor = 0,
    CXXCopyConstructor = 1,
    CXXCopyAssignment = 2,
    CXXDestructor = 3
  };
  bool CheckNontrivialField(FieldDecl *FD);
  void DiagnoseNontrivial(const RecordType* Record, CXXSpecialMember mem);
  CXXSpecialMember getSpecialMember(const CXXMethodDecl *MD);
  void ActOnLastBitfield(SourceLocation DeclStart, Decl *IntfDecl, 
                         llvm::SmallVectorImpl<Decl *> &AllIvarDecls);
  Decl *ActOnIvar(Scope *S, SourceLocation DeclStart, Decl *IntfDecl,
                  Declarator &D, Expr *BitfieldWidth,
                  tok::ObjCKeywordKind visibility);

  // This is used for both record definitions and ObjC interface declarations.
  void ActOnFields(Scope* S, SourceLocation RecLoc, Decl *TagDecl,
                   Decl **Fields, unsigned NumFields,
                   SourceLocation LBrac, SourceLocation RBrac,
                   AttributeList *AttrList);

  /// ActOnTagStartDefinition - Invoked when we have entered the
  /// scope of a tag's definition (e.g., for an enumeration, class,
  /// struct, or union).
  void ActOnTagStartDefinition(Scope *S, Decl *TagDecl);

  /// ActOnStartCXXMemberDeclarations - Invoked when we have parsed a
  /// C++ record definition's base-specifiers clause and are starting its
  /// member declarations.
  void ActOnStartCXXMemberDeclarations(Scope *S, Decl *TagDecl,
                                       ClassVirtSpecifiers &CVS,
                                       SourceLocation LBraceLoc);

  /// ActOnTagFinishDefinition - Invoked once we have finished parsing
  /// the definition of a tag (enumeration, class, struct, or union).
  void ActOnTagFinishDefinition(Scope *S, Decl *TagDecl,
                                SourceLocation RBraceLoc);

  /// ActOnTagDefinitionError - Invoked when there was an unrecoverable
  /// error parsing the definition of a tag.
  void ActOnTagDefinitionError(Scope *S, Decl *TagDecl);

  EnumConstantDecl *CheckEnumConstant(EnumDecl *Enum,
                                      EnumConstantDecl *LastEnumConst,
                                      SourceLocation IdLoc,
                                      IdentifierInfo *Id,
                                      Expr *val);

  Decl *ActOnEnumConstant(Scope *S, Decl *EnumDecl, Decl *LastEnumConstant,
                          SourceLocation IdLoc, IdentifierInfo *Id,
                          AttributeList *Attrs,
                          SourceLocation EqualLoc, Expr *Val);
  void ActOnEnumBody(SourceLocation EnumLoc, SourceLocation LBraceLoc,
                     SourceLocation RBraceLoc, Decl *EnumDecl,
                     Decl **Elements, unsigned NumElements,
                     Scope *S, AttributeList *Attr);

  DeclContext *getContainingDC(DeclContext *DC);

  /// Set the current declaration context until it gets popped.
  void PushDeclContext(Scope *S, DeclContext *DC);
  void PopDeclContext();

  /// EnterDeclaratorContext - Used when we must lookup names in the context
  /// of a declarator's nested name specifier.
  void EnterDeclaratorContext(Scope *S, DeclContext *DC);
  void ExitDeclaratorContext(Scope *S);

  DeclContext *getFunctionLevelDeclContext();

  /// getCurFunctionDecl - If inside of a function body, this returns a pointer
  /// to the function decl for the function being parsed.  If we're currently
  /// in a 'block', this returns the containing context.
  FunctionDecl *getCurFunctionDecl();

  /// getCurMethodDecl - If inside of a method body, this returns a pointer to
  /// the method decl for the method being parsed.  If we're currently
  /// in a 'block', this returns the containing context.
  ObjCMethodDecl *getCurMethodDecl();

  /// getCurFunctionOrMethodDecl - Return the Decl for the current ObjC method
  /// or C function we're in, otherwise return null.  If we're currently
  /// in a 'block', this returns the containing context.
  NamedDecl *getCurFunctionOrMethodDecl();

  /// Add this decl to the scope shadowed decl chains.
  void PushOnScopeChains(NamedDecl *D, Scope *S, bool AddToContext = true);

  /// isDeclInScope - If 'Ctx' is a function/method, isDeclInScope returns true
  /// if 'D' is in Scope 'S', otherwise 'S' is ignored and isDeclInScope returns
  /// true if 'D' belongs to the given declaration context.
  ///
  /// \param ExplicitInstantiationOrSpecialization When true, we are checking
  /// whether the declaration is in scope for the purposes of explicit template
  /// instantiation or specialization. The default is false.
  bool isDeclInScope(NamedDecl *&D, DeclContext *Ctx, Scope *S = 0,
                     bool ExplicitInstantiationOrSpecialization = false);

  /// Finds the scope corresponding to the given decl context, if it
  /// happens to be an enclosing scope.  Otherwise return NULL.
  static Scope *getScopeForDeclContext(Scope *S, DeclContext *DC);

  /// Subroutines of ActOnDeclarator().
  TypedefDecl *ParseTypedefDecl(Scope *S, Declarator &D, QualType T,
                                TypeSourceInfo *TInfo);
  void MergeTypeDefDecl(TypedefDecl *New, LookupResult &OldDecls);
  bool MergeFunctionDecl(FunctionDecl *New, Decl *Old);
  bool MergeCompatibleFunctionDecls(FunctionDecl *New, FunctionDecl *Old);
  void mergeObjCMethodDecls(ObjCMethodDecl *New, const ObjCMethodDecl *Old);
  void MergeVarDeclTypes(VarDecl *New, VarDecl *Old);
  void MergeVarDecl(VarDecl *New, LookupResult &OldDecls);
  bool MergeCXXFunctionDecl(FunctionDecl *New, FunctionDecl *Old);

  // AssignmentAction - This is used by all the assignment diagnostic functions
  // to represent what is actually causing the operation
  enum AssignmentAction {
    AA_Assigning,
    AA_Passing,
    AA_Returning,
    AA_Converting,
    AA_Initializing,
    AA_Sending,
    AA_Casting
  };

  /// C++ Overloading.
  enum OverloadKind {
    /// This is a legitimate overload: the existing declarations are
    /// functions or function templates with different signatures.
    Ovl_Overload,

    /// This is not an overload because the signature exactly matches
    /// an existing declaration.
    Ovl_Match,

    /// This is not an overload because the lookup results contain a
    /// non-function.
    Ovl_NonFunction
  };
  OverloadKind CheckOverload(Scope *S,
                             FunctionDecl *New, 
                             const LookupResult &OldDecls,
                             NamedDecl *&OldDecl,
                             bool IsForUsingDecl);
  bool IsOverload(FunctionDecl *New, FunctionDecl *Old, bool IsForUsingDecl);

  bool TryImplicitConversion(InitializationSequence &Sequence,
                             const InitializedEntity &Entity,
                             Expr *From,
                             bool SuppressUserConversions,
                             bool AllowExplicit,
                             bool InOverloadResolution,
                             bool CStyle);

  bool IsIntegralPromotion(Expr *From, QualType FromType, QualType ToType);
  bool IsFloatingPointPromotion(QualType FromType, QualType ToType);
  bool IsComplexPromotion(QualType FromType, QualType ToType);
  bool IsPointerConversion(Expr *From, QualType FromType, QualType ToType,
                           bool InOverloadResolution,
                           QualType& ConvertedType, bool &IncompatibleObjC);
  bool isObjCPointerConversion(QualType FromType, QualType ToType,
                               QualType& ConvertedType, bool &IncompatibleObjC);
  bool IsBlockPointerConversion(QualType FromType, QualType ToType,
                                QualType& ConvertedType);
  bool FunctionArgTypesAreEqual(const FunctionProtoType *OldType, 
                                const FunctionProtoType *NewType);
  
  bool CheckPointerConversion(Expr *From, QualType ToType,
                              CastKind &Kind,
                              CXXCastPath& BasePath,
                              bool IgnoreBaseAccess);
  bool IsMemberPointerConversion(Expr *From, QualType FromType, QualType ToType,
                                 bool InOverloadResolution,
                                 QualType &ConvertedType);
  bool CheckMemberPointerConversion(Expr *From, QualType ToType,
                                    CastKind &Kind,
                                    CXXCastPath &BasePath,
                                    bool IgnoreBaseAccess);
  bool IsQualificationConversion(QualType FromType, QualType ToType,
                                 bool CStyle);
  bool DiagnoseMultipleUserDefinedConversion(Expr *From, QualType ToType);


  ExprResult PerformMoveOrCopyInitialization(const InitializedEntity &Entity,
                                             const VarDecl *NRVOCandidate,
                                             QualType ResultType,
                                             Expr *Value);
  
  ExprResult PerformCopyInitialization(const InitializedEntity &Entity,
                                       SourceLocation EqualLoc,
                                       ExprResult Init);
  bool PerformObjectArgumentInitialization(Expr *&From,
                                           NestedNameSpecifier *Qualifier,
                                           NamedDecl *FoundDecl,
                                           CXXMethodDecl *Method);

  bool PerformContextuallyConvertToBool(Expr *&From);
  bool PerformContextuallyConvertToObjCId(Expr *&From);

  ExprResult 
  ConvertToIntegralOrEnumerationType(SourceLocation Loc, Expr *FromE,
                                     const PartialDiagnostic &NotIntDiag,
                                     const PartialDiagnostic &IncompleteDiag,
                                     const PartialDiagnostic &ExplicitConvDiag,
                                     const PartialDiagnostic &ExplicitConvNote,
                                     const PartialDiagnostic &AmbigDiag,
                                     const PartialDiagnostic &AmbigNote,
                                     const PartialDiagnostic &ConvDiag);
  
  bool PerformObjectMemberConversion(Expr *&From,
                                     NestedNameSpecifier *Qualifier,
                                     NamedDecl *FoundDecl,
                                     NamedDecl *Member);

  // Members have to be NamespaceDecl* or TranslationUnitDecl*.
  // TODO: make this is a typesafe union.
  typedef llvm::SmallPtrSet<DeclContext   *, 16> AssociatedNamespaceSet;
  typedef llvm::SmallPtrSet<CXXRecordDecl *, 16> AssociatedClassSet;

  void AddOverloadCandidate(NamedDecl *Function,
                            DeclAccessPair FoundDecl,
                            Expr **Args, unsigned NumArgs,
                            OverloadCandidateSet &CandidateSet);

  void AddOverloadCandidate(FunctionDecl *Function,
                            DeclAccessPair FoundDecl,
                            Expr **Args, unsigned NumArgs,
                            OverloadCandidateSet& CandidateSet,
                            bool SuppressUserConversions = false,
                            bool PartialOverloading = false);
  void AddFunctionCandidates(const UnresolvedSetImpl &Functions,
                             Expr **Args, unsigned NumArgs,
                             OverloadCandidateSet& CandidateSet,
                             bool SuppressUserConversions = false);
  void AddMethodCandidate(DeclAccessPair FoundDecl,
                          QualType ObjectType,
                          Expr::Classification ObjectClassification,
                          Expr **Args, unsigned NumArgs,
                          OverloadCandidateSet& CandidateSet,
                          bool SuppressUserConversion = false);
  void AddMethodCandidate(CXXMethodDecl *Method,
                          DeclAccessPair FoundDecl,
                          CXXRecordDecl *ActingContext, QualType ObjectType,
                          Expr::Classification ObjectClassification,
                          Expr **Args, unsigned NumArgs,
                          OverloadCandidateSet& CandidateSet,
                          bool SuppressUserConversions = false);
  void AddMethodTemplateCandidate(FunctionTemplateDecl *MethodTmpl,
                                  DeclAccessPair FoundDecl,
                                  CXXRecordDecl *ActingContext,
                                 TemplateArgumentListInfo *ExplicitTemplateArgs,
                                  QualType ObjectType,
                                  Expr::Classification ObjectClassification,
                                  Expr **Args, unsigned NumArgs,
                                  OverloadCandidateSet& CandidateSet,
                                  bool SuppressUserConversions = false);
  void AddTemplateOverloadCandidate(FunctionTemplateDecl *FunctionTemplate,
                                    DeclAccessPair FoundDecl,
                                 TemplateArgumentListInfo *ExplicitTemplateArgs,
                                    Expr **Args, unsigned NumArgs,
                                    OverloadCandidateSet& CandidateSet,
                                    bool SuppressUserConversions = false);
  void AddConversionCandidate(CXXConversionDecl *Conversion,
                              DeclAccessPair FoundDecl,
                              CXXRecordDecl *ActingContext,
                              Expr *From, QualType ToType,
                              OverloadCandidateSet& CandidateSet);
  void AddTemplateConversionCandidate(FunctionTemplateDecl *FunctionTemplate,
                                      DeclAccessPair FoundDecl,
                                      CXXRecordDecl *ActingContext,
                                      Expr *From, QualType ToType,
                                      OverloadCandidateSet &CandidateSet);
  void AddSurrogateCandidate(CXXConversionDecl *Conversion,
                             DeclAccessPair FoundDecl,
                             CXXRecordDecl *ActingContext,
                             const FunctionProtoType *Proto,
                             Expr *Object, Expr **Args, unsigned NumArgs,
                             OverloadCandidateSet& CandidateSet);
  void AddMemberOperatorCandidates(OverloadedOperatorKind Op,
                                   SourceLocation OpLoc,
                                   Expr **Args, unsigned NumArgs,
                                   OverloadCandidateSet& CandidateSet,
                                   SourceRange OpRange = SourceRange());
  void AddBuiltinCandidate(QualType ResultTy, QualType *ParamTys,
                           Expr **Args, unsigned NumArgs,
                           OverloadCandidateSet& CandidateSet,
                           bool IsAssignmentOperator = false,
                           unsigned NumContextualBoolArguments = 0);
  void AddBuiltinOperatorCandidates(OverloadedOperatorKind Op,
                                    SourceLocation OpLoc,
                                    Expr **Args, unsigned NumArgs,
                                    OverloadCandidateSet& CandidateSet);
  void AddArgumentDependentLookupCandidates(DeclarationName Name,
                                            bool Operator,
                                            Expr **Args, unsigned NumArgs,
                                TemplateArgumentListInfo *ExplicitTemplateArgs,
                                            OverloadCandidateSet& CandidateSet,
                                            bool PartialOverloading = false);

  // Emit as a 'note' the specific overload candidate
  void NoteOverloadCandidate(FunctionDecl *Fn);
   
  // Emit as a series of 'note's all template and non-templates
  // identified by the expression Expr
  void NoteAllOverloadCandidates(Expr* E);
  
  // [PossiblyAFunctionType]  -->   [Return]
  // NonFunctionType --> NonFunctionType
  // R (A) --> R(A)
  // R (*)(A) --> R (A)
  // R (&)(A) --> R (A)
  // R (S::*)(A) --> R (A)
  QualType ExtractUnqualifiedFunctionType(QualType PossiblyAFunctionType);

  FunctionDecl *ResolveAddressOfOverloadedFunction(Expr *AddressOfExpr, QualType TargetType,
                                                   bool Complain,
                                                   DeclAccessPair &Found);

  FunctionDecl *ResolveSingleFunctionTemplateSpecialization(Expr *From,
                                                   bool Complain = false,
                                                   DeclAccessPair* Found = 0);

  Expr *FixOverloadedFunctionReference(Expr *E,
                                       DeclAccessPair FoundDecl,
                                       FunctionDecl *Fn);
  ExprResult FixOverloadedFunctionReference(ExprResult,
                                            DeclAccessPair FoundDecl,
                                            FunctionDecl *Fn);

  void AddOverloadedCallCandidates(UnresolvedLookupExpr *ULE,
                                   Expr **Args, unsigned NumArgs,
                                   OverloadCandidateSet &CandidateSet,
                                   bool PartialOverloading = false);

  ExprResult BuildOverloadedCallExpr(Scope *S, Expr *Fn,
                                     UnresolvedLookupExpr *ULE,
                                     SourceLocation LParenLoc,
                                     Expr **Args, unsigned NumArgs,
                                     SourceLocation RParenLoc,
                                     Expr *ExecConfig);

  ExprResult CreateOverloadedUnaryOp(SourceLocation OpLoc,
                                     unsigned Opc,
                                     const UnresolvedSetImpl &Fns,
                                     Expr *input);

  ExprResult CreateOverloadedBinOp(SourceLocation OpLoc,
                                   unsigned Opc,
                                   const UnresolvedSetImpl &Fns,
                                   Expr *LHS, Expr *RHS);

  ExprResult CreateOverloadedArraySubscriptExpr(SourceLocation LLoc,
                                                SourceLocation RLoc,
                                                Expr *Base,Expr *Idx);

  ExprResult
  BuildCallToMemberFunction(Scope *S, Expr *MemExpr,
                            SourceLocation LParenLoc, Expr **Args,
                            unsigned NumArgs, SourceLocation RParenLoc);
  ExprResult
  BuildCallToObjectOfClassType(Scope *S, Expr *Object, SourceLocation LParenLoc,
                               Expr **Args, unsigned NumArgs,
                               SourceLocation RParenLoc);

  ExprResult BuildOverloadedArrowExpr(Scope *S, Expr *Base,
                                      SourceLocation OpLoc);

  /// CheckCallReturnType - Checks that a call expression's return type is
  /// complete. Returns true on failure. The location passed in is the location
  /// that best represents the call.
  bool CheckCallReturnType(QualType ReturnType, SourceLocation Loc,
                           CallExpr *CE, FunctionDecl *FD);

  /// Helpers for dealing with blocks and functions.
  bool CheckParmsForFunctionDef(ParmVarDecl **Param, ParmVarDecl **ParamEnd,
                                bool CheckParameterNames);
  void CheckCXXDefaultArguments(FunctionDecl *FD);
  void CheckExtraCXXDefaultArguments(Declarator &D);
  Scope *getNonFieldDeclScope(Scope *S);

  /// \name Name lookup
  ///
  /// These routines provide name lookup that is used during semantic
  /// analysis to resolve the various kinds of names (identifiers,
  /// overloaded operator names, constructor names, etc.) into zero or
  /// more declarations within a particular scope. The major entry
  /// points are LookupName, which performs unqualified name lookup,
  /// and LookupQualifiedName, which performs qualified name lookup.
  ///
  /// All name lookup is performed based on some specific criteria,
  /// which specify what names will be visible to name lookup and how
  /// far name lookup should work. These criteria are important both
  /// for capturing language semantics (certain lookups will ignore
  /// certain names, for example) and for performance, since name
  /// lookup is often a bottleneck in the compilation of C++. Name
  /// lookup criteria is specified via the LookupCriteria enumeration.
  ///
  /// The results of name lookup can vary based on the kind of name
  /// lookup performed, the current language, and the translation
  /// unit. In C, for example, name lookup will either return nothing
  /// (no entity found) or a single declaration. In C++, name lookup
  /// can additionally refer to a set of overloaded functions or
  /// result in an ambiguity. All of the possible results of name
  /// lookup are captured by the LookupResult class, which provides
  /// the ability to distinguish among them.
  //@{

  /// @brief Describes the kind of name lookup to perform.
  enum LookupNameKind {
    /// Ordinary name lookup, which finds ordinary names (functions,
    /// variables, typedefs, etc.) in C and most kinds of names
    /// (functions, variables, members, types, etc.) in C++.
    LookupOrdinaryName = 0,
    /// Tag name lookup, which finds the names of enums, classes,
    /// structs, and unions.
    LookupTagName,
    /// Label name lookup.
    LookupLabel,
    /// Member name lookup, which finds the names of
    /// class/struct/union members.
    LookupMemberName,
    /// Look up of an operator name (e.g., operator+) for use with
    /// operator overloading. This lookup is similar to ordinary name
    /// lookup, but will ignore any declarations that are class members.
    LookupOperatorName,
    /// Look up of a name that precedes the '::' scope resolution
    /// operator in C++. This lookup completely ignores operator, object,
    /// function, and enumerator names (C++ [basic.lookup.qual]p1).
    LookupNestedNameSpecifierName,
    /// Look up a namespace name within a C++ using directive or
    /// namespace alias definition, ignoring non-namespace names (C++
    /// [basic.lookup.udir]p1).
    LookupNamespaceName,
    /// Look up all declarations in a scope with the given name,
    /// including resolved using declarations.  This is appropriate
    /// for checking redeclarations for a using declaration.
    LookupUsingDeclName,
    /// Look up an ordinary name that is going to be redeclared as a
    /// name with linkage. This lookup ignores any declarations that
    /// are outside of the current scope unless they have linkage. See
    /// C99 6.2.2p4-5 and C++ [basic.link]p6.
    LookupRedeclarationWithLinkage,
    /// Look up the name of an Objective-C protocol.
    LookupObjCProtocolName,
    /// \brief Look up any declaration with any name.
    LookupAnyName
  };

  /// \brief Specifies whether (or how) name lookup is being performed for a
  /// redeclaration (vs. a reference).
  enum RedeclarationKind {
    /// \brief The lookup is a reference to this name that is not for the
    /// purpose of redeclaring the name.
    NotForRedeclaration = 0,
    /// \brief The lookup results will be used for redeclaration of a name,
    /// if an entity by that name already exists.
    ForRedeclaration
  };

private:
  bool CppLookupName(LookupResult &R, Scope *S);

public:
  /// \brief Look up a name, looking for a single declaration.  Return
  /// null if the results were absent, ambiguous, or overloaded.
  ///
  /// It is preferable to use the elaborated form and explicitly handle
  /// ambiguity and overloaded.
  NamedDecl *LookupSingleName(Scope *S, DeclarationName Name,
                              SourceLocation Loc,
                              LookupNameKind NameKind,
                              RedeclarationKind Redecl
                                = NotForRedeclaration);
  bool LookupName(LookupResult &R, Scope *S,
                  bool AllowBuiltinCreation = false);
  bool LookupQualifiedName(LookupResult &R, DeclContext *LookupCtx,
                           bool InUnqualifiedLookup = false);
  bool LookupParsedName(LookupResult &R, Scope *S, CXXScopeSpec *SS,
                        bool AllowBuiltinCreation = false,
                        bool EnteringContext = false);
  ObjCProtocolDecl *LookupProtocol(IdentifierInfo *II, SourceLocation IdLoc);

  void LookupOverloadedOperatorName(OverloadedOperatorKind Op, Scope *S,
                                    QualType T1, QualType T2,
                                    UnresolvedSetImpl &Functions);

  LabelDecl *LookupOrCreateLabel(IdentifierInfo *II, SourceLocation IdentLoc,
                                 SourceLocation GnuLabelLoc = SourceLocation());

  DeclContextLookupResult LookupConstructors(CXXRecordDecl *Class);
  CXXDestructorDecl *LookupDestructor(CXXRecordDecl *Class);

  void ArgumentDependentLookup(DeclarationName Name, bool Operator,
                               Expr **Args, unsigned NumArgs,
                               ADLResult &Functions);

  void LookupVisibleDecls(Scope *S, LookupNameKind Kind,
                          VisibleDeclConsumer &Consumer,
                          bool IncludeGlobalScope = true);
  void LookupVisibleDecls(DeclContext *Ctx, LookupNameKind Kind,
                          VisibleDeclConsumer &Consumer,
                          bool IncludeGlobalScope = true);
  
  /// \brief The context in which typo-correction occurs.
  ///
  /// The typo-correction context affects which keywords (if any) are
  /// considered when trying to correct for typos.
  enum CorrectTypoContext {
    /// \brief An unknown context, where any keyword might be valid.
    CTC_Unknown,
    /// \brief A context where no keywords are used (e.g. we expect an actual
    /// name).
    CTC_NoKeywords,
    /// \brief A context where we're correcting a type name.
    CTC_Type,
    /// \brief An expression context.
    CTC_Expression,
    /// \brief A type cast, or anything else that can be followed by a '<'.
    CTC_CXXCasts,
    /// \brief A member lookup context.
    CTC_MemberLookup,
    /// \brief An Objective-C ivar lookup context (e.g., self->ivar).
    CTC_ObjCIvarLookup,
    /// \brief An Objective-C property lookup context (e.g., self.prop).
    CTC_ObjCPropertyLookup,
    /// \brief The receiver of an Objective-C message send within an
    /// Objective-C method where 'super' is a valid keyword.
    CTC_ObjCMessageReceiver
  };

  DeclarationName CorrectTypo(LookupResult &R, Scope *S, CXXScopeSpec *SS,
                              DeclContext *MemberContext = 0,
                              bool EnteringContext = false,
                              CorrectTypoContext CTC = CTC_Unknown,
                              const ObjCObjectPointerType *OPT = 0);

  void FindAssociatedClassesAndNamespaces(Expr **Args, unsigned NumArgs,
                                   AssociatedNamespaceSet &AssociatedNamespaces,
                                   AssociatedClassSet &AssociatedClasses);

  bool DiagnoseAmbiguousLookup(LookupResult &Result);
  //@}

  ObjCInterfaceDecl *getObjCInterfaceDecl(IdentifierInfo *&Id,
                                          SourceLocation IdLoc,
                                          bool TypoCorrection = false);
  NamedDecl *LazilyCreateBuiltin(IdentifierInfo *II, unsigned ID,
                                 Scope *S, bool ForRedeclaration,
                                 SourceLocation Loc);
  NamedDecl *ImplicitlyDefineFunction(SourceLocation Loc, IdentifierInfo &II,
                                      Scope *S);
  void AddKnownFunctionAttributes(FunctionDecl *FD);

  // More parsing and symbol table subroutines.

  // Decl attributes - this routine is the top level dispatcher.
  void ProcessDeclAttributes(Scope *S, Decl *D, const Declarator &PD,
                           bool NonInheritable = true, bool Inheritable = true);
  void ProcessDeclAttributeList(Scope *S, Decl *D, const AttributeList *AL,
                           bool NonInheritable = true, bool Inheritable = true);

  bool CheckRegparmAttr(const AttributeList &attr, unsigned &value);
  bool CheckCallingConvAttr(const AttributeList &attr, CallingConv &CC);
  bool CheckNoReturnAttr(const AttributeList &attr);

  void WarnUndefinedMethod(SourceLocation ImpLoc, ObjCMethodDecl *method,
                           bool &IncompleteImpl, unsigned DiagID);
  void WarnConflictingTypedMethods(ObjCMethodDecl *ImpMethod,
                                   ObjCMethodDecl *MethodDecl,
                                   bool IsProtocolMethodDecl);

  bool isPropertyReadonly(ObjCPropertyDecl *PropertyDecl,
                          ObjCInterfaceDecl *IDecl);

  typedef llvm::DenseSet<Selector, llvm::DenseMapInfo<Selector> > SelectorSet;

  /// CheckProtocolMethodDefs - This routine checks unimplemented
  /// methods declared in protocol, and those referenced by it.
  /// \param IDecl - Used for checking for methods which may have been
  /// inherited.
  void CheckProtocolMethodDefs(SourceLocation ImpLoc,
                               ObjCProtocolDecl *PDecl,
                               bool& IncompleteImpl,
                               const SelectorSet &InsMap,
                               const SelectorSet &ClsMap,
                               ObjCContainerDecl *CDecl);

  /// CheckImplementationIvars - This routine checks if the instance variables
  /// listed in the implelementation match those listed in the interface.
  void CheckImplementationIvars(ObjCImplementationDecl *ImpDecl,
                                ObjCIvarDecl **Fields, unsigned nIvars,
                                SourceLocation Loc);

  /// \brief Determine whether we can synthesize a provisional ivar for the
  /// given name.
  ObjCPropertyDecl *canSynthesizeProvisionalIvar(IdentifierInfo *II);

  /// \brief Determine whether we can synthesize a provisional ivar for the
  /// given property.
  bool canSynthesizeProvisionalIvar(ObjCPropertyDecl *Property);

  /// ImplMethodsVsClassMethods - This is main routine to warn if any method
  /// remains unimplemented in the class or category @implementation.
  void ImplMethodsVsClassMethods(Scope *S, ObjCImplDecl* IMPDecl,
                                 ObjCContainerDecl* IDecl,
                                 bool IncompleteImpl = false);

  /// DiagnoseUnimplementedProperties - This routine warns on those properties
  /// which must be implemented by this implementation.
  void DiagnoseUnimplementedProperties(Scope *S, ObjCImplDecl* IMPDecl,
                                       ObjCContainerDecl *CDecl,
                                       const SelectorSet &InsMap);

  /// DefaultSynthesizeProperties - This routine default synthesizes all 
  /// properties which must be synthesized in class's @implementation.
  void DefaultSynthesizeProperties (Scope *S, ObjCImplDecl* IMPDecl,
                                    ObjCInterfaceDecl *IDecl);
  
  /// CollectImmediateProperties - This routine collects all properties in
  /// the class and its conforming protocols; but not those it its super class.
  void CollectImmediateProperties(ObjCContainerDecl *CDecl,
            llvm::DenseMap<IdentifierInfo *, ObjCPropertyDecl*>& PropMap,
            llvm::DenseMap<IdentifierInfo *, ObjCPropertyDecl*>& SuperPropMap);
  

  /// LookupPropertyDecl - Looks up a property in the current class and all
  /// its protocols.
  ObjCPropertyDecl *LookupPropertyDecl(const ObjCContainerDecl *CDecl,
                                       IdentifierInfo *II);

  /// Called by ActOnProperty to handle @property declarations in
  ////  class extensions.
  Decl *HandlePropertyInClassExtension(Scope *S,
                                       ObjCCategoryDecl *CDecl,
                                       SourceLocation AtLoc,
                                       FieldDeclarator &FD,
                                       Selector GetterSel,
                                       Selector SetterSel,
                                       const bool isAssign,
                                       const bool isReadWrite,
                                       const unsigned Attributes,
                                       bool *isOverridingProperty,
                                       TypeSourceInfo *T,
                                       tok::ObjCKeywordKind MethodImplKind);

  /// Called by ActOnProperty and HandlePropertyInClassExtension to
  ///  handle creating the ObjcPropertyDecl for a category or @interface.
  ObjCPropertyDecl *CreatePropertyDecl(Scope *S,
                                       ObjCContainerDecl *CDecl,
                                       SourceLocation AtLoc,
                                       FieldDeclarator &FD,
                                       Selector GetterSel,
                                       Selector SetterSel,
                                       const bool isAssign,
                                       const bool isReadWrite,
                                       const unsigned Attributes,
                                       TypeSourceInfo *T,
                                       tok::ObjCKeywordKind MethodImplKind,
                                       DeclContext *lexicalDC = 0);

  /// AtomicPropertySetterGetterRules - This routine enforces the rule (via
  /// warning) when atomic property has one but not the other user-declared
  /// setter or getter.
  void AtomicPropertySetterGetterRules(ObjCImplDecl* IMPDecl,
                                       ObjCContainerDecl* IDecl);

  void DiagnoseDuplicateIvars(ObjCInterfaceDecl *ID, ObjCInterfaceDecl *SID);

  /// MatchTwoMethodDeclarations - Checks if two methods' type match and returns
  /// true, or false, accordingly.
  bool MatchTwoMethodDeclarations(const ObjCMethodDecl *Method,
                                  const ObjCMethodDecl *PrevMethod,
                                  bool matchBasedOnSizeAndAlignment = false,
                                  bool matchBasedOnStrictEqulity = false);

  /// MatchAllMethodDeclarations - Check methods declaraed in interface or
  /// or protocol against those declared in their implementations.
  void MatchAllMethodDeclarations(const SelectorSet &InsMap,
                                  const SelectorSet &ClsMap,
                                  SelectorSet &InsMapSeen,
                                  SelectorSet &ClsMapSeen,
                                  ObjCImplDecl* IMPDecl,
                                  ObjCContainerDecl* IDecl,
                                  bool &IncompleteImpl,
                                  bool ImmediateClass);

private:
  /// AddMethodToGlobalPool - Add an instance or factory method to the global
  /// pool. See descriptoin of AddInstanceMethodToGlobalPool.
  void AddMethodToGlobalPool(ObjCMethodDecl *Method, bool impl, bool instance);

  /// LookupMethodInGlobalPool - Returns the instance or factory method and
  /// optionally warns if there are multiple signatures.
  ObjCMethodDecl *LookupMethodInGlobalPool(Selector Sel, SourceRange R,
                                           bool receiverIdOrClass,
                                           bool warn, bool instance);

public:
  /// AddInstanceMethodToGlobalPool - All instance methods in a translation
  /// unit are added to a global pool. This allows us to efficiently associate
  /// a selector with a method declaraation for purposes of typechecking
  /// messages sent to "id" (where the class of the object is unknown).
  void AddInstanceMethodToGlobalPool(ObjCMethodDecl *Method, bool impl=false) {
    AddMethodToGlobalPool(Method, impl, /*instance*/true);
  }

  /// AddFactoryMethodToGlobalPool - Same as above, but for factory methods.
  void AddFactoryMethodToGlobalPool(ObjCMethodDecl *Method, bool impl=false) {
    AddMethodToGlobalPool(Method, impl, /*instance*/false);
  }

  /// LookupInstanceMethodInGlobalPool - Returns the method and warns if
  /// there are multiple signatures.
  ObjCMethodDecl *LookupInstanceMethodInGlobalPool(Selector Sel, SourceRange R,
                                                   bool receiverIdOrClass=false,
                                                   bool warn=true) {
    return LookupMethodInGlobalPool(Sel, R, receiverIdOrClass, 
                                    warn, /*instance*/true);
  }

  /// LookupFactoryMethodInGlobalPool - Returns the method and warns if
  /// there are multiple signatures.
  ObjCMethodDecl *LookupFactoryMethodInGlobalPool(Selector Sel, SourceRange R,
                                                  bool receiverIdOrClass=false,
                                                  bool warn=true) {
    return LookupMethodInGlobalPool(Sel, R, receiverIdOrClass,
                                    warn, /*instance*/false);
  }

  /// LookupImplementedMethodInGlobalPool - Returns the method which has an
  /// implementation.
  ObjCMethodDecl *LookupImplementedMethodInGlobalPool(Selector Sel);

  /// CollectIvarsToConstructOrDestruct - Collect those ivars which require
  /// initialization.
  void CollectIvarsToConstructOrDestruct(ObjCInterfaceDecl *OI,
                                  llvm::SmallVectorImpl<ObjCIvarDecl*> &Ivars);
  //===--------------------------------------------------------------------===//
  // Statement Parsing Callbacks: SemaStmt.cpp.
public:
  class FullExprArg {
  public:
    FullExprArg(Sema &actions) : E(0) { }
                
    // FIXME: The const_cast here is ugly. RValue references would make this
    // much nicer (or we could duplicate a bunch of the move semantics
    // emulation code from Ownership.h).
    FullExprArg(const FullExprArg& Other) : E(Other.E) {}

    ExprResult release() {
      return move(E);
    }

    Expr *get() const { return E; }

    Expr *operator->() {
      return E;
    }

  private:
    // FIXME: No need to make the entire Sema class a friend when it's just
    // Sema::MakeFullExpr that needs access to the constructor below.
    friend class Sema;

    explicit FullExprArg(Expr *expr) : E(expr) {}

    Expr *E;
  };

  FullExprArg MakeFullExpr(Expr *Arg) {
    return FullExprArg(ActOnFinishFullExpr(Arg).release());
  }

  StmtResult ActOnExprStmt(FullExprArg Expr);

  StmtResult ActOnNullStmt(SourceLocation SemiLoc,
                           bool LeadingEmptyMacro = false);
  StmtResult ActOnCompoundStmt(SourceLocation L, SourceLocation R,
                                       MultiStmtArg Elts,
                                       bool isStmtExpr);
  StmtResult ActOnDeclStmt(DeclGroupPtrTy Decl,
                                   SourceLocation StartLoc,
                                   SourceLocation EndLoc);
  void ActOnForEachDeclStmt(DeclGroupPtrTy Decl);
  StmtResult ActOnForEachLValueExpr(Expr *E);
  StmtResult ActOnCaseStmt(SourceLocation CaseLoc, Expr *LHSVal,
                                   SourceLocation DotDotDotLoc, Expr *RHSVal,
                                   SourceLocation ColonLoc);
  void ActOnCaseStmtBody(Stmt *CaseStmt, Stmt *SubStmt);

  StmtResult ActOnDefaultStmt(SourceLocation DefaultLoc,
                                      SourceLocation ColonLoc,
                                      Stmt *SubStmt, Scope *CurScope);
  StmtResult ActOnLabelStmt(SourceLocation IdentLoc, LabelDecl *TheDecl,
                            SourceLocation ColonLoc, Stmt *SubStmt);
    
  StmtResult ActOnIfStmt(SourceLocation IfLoc,
                         FullExprArg CondVal, Decl *CondVar,
                         Stmt *ThenVal,
                         SourceLocation ElseLoc, Stmt *ElseVal);
  StmtResult ActOnStartOfSwitchStmt(SourceLocation SwitchLoc,
                                            Expr *Cond,
                                            Decl *CondVar);
  StmtResult ActOnFinishSwitchStmt(SourceLocation SwitchLoc,
                                           Stmt *Switch, Stmt *Body);
  StmtResult ActOnWhileStmt(SourceLocation WhileLoc,
                            FullExprArg Cond,
                            Decl *CondVar, Stmt *Body);
  StmtResult ActOnDoStmt(SourceLocation DoLoc, Stmt *Body,
                                 SourceLocation WhileLoc,
                                 SourceLocation CondLParen, Expr *Cond,
                                 SourceLocation CondRParen);

  StmtResult ActOnForStmt(SourceLocation ForLoc,
                          SourceLocation LParenLoc,
                          Stmt *First, FullExprArg Second,
                          Decl *SecondVar,
                          FullExprArg Third,
                          SourceLocation RParenLoc,
                          Stmt *Body);
  StmtResult ActOnObjCForCollectionStmt(SourceLocation ForColLoc,
                                        SourceLocation LParenLoc,
                                        Stmt *First, Expr *Second,
                                        SourceLocation RParenLoc, Stmt *Body);

  StmtResult ActOnGotoStmt(SourceLocation GotoLoc,
                           SourceLocation LabelLoc,
                           LabelDecl *TheDecl);
  StmtResult ActOnIndirectGotoStmt(SourceLocation GotoLoc,
                                   SourceLocation StarLoc,
                                   Expr *DestExp);
  StmtResult ActOnContinueStmt(SourceLocation ContinueLoc, Scope *CurScope);
  StmtResult ActOnBreakStmt(SourceLocation GotoLoc, Scope *CurScope);

  const VarDecl *getCopyElisionCandidate(QualType ReturnType, Expr *E,
                                         bool AllowFunctionParameters);
  
  StmtResult ActOnReturnStmt(SourceLocation ReturnLoc, Expr *RetValExp);
  StmtResult ActOnBlockReturnStmt(SourceLocation ReturnLoc, Expr *RetValExp);

  StmtResult ActOnAsmStmt(SourceLocation AsmLoc,
                          bool IsSimple, bool IsVolatile,
                          unsigned NumOutputs, unsigned NumInputs,
                          IdentifierInfo **Names,
                          MultiExprArg Constraints,
                          MultiExprArg Exprs,
                          Expr *AsmString,
                          MultiExprArg Clobbers,
                          SourceLocation RParenLoc,
                          bool MSAsm = false);


  VarDecl *BuildObjCExceptionDecl(TypeSourceInfo *TInfo, QualType ExceptionType,
                                  IdentifierInfo *Name, SourceLocation NameLoc,
                                  bool Invalid = false);

  Decl *ActOnObjCExceptionDecl(Scope *S, Declarator &D);

  StmtResult ActOnObjCAtCatchStmt(SourceLocation AtLoc, SourceLocation RParen,
                                  Decl *Parm, Stmt *Body);

  StmtResult ActOnObjCAtFinallyStmt(SourceLocation AtLoc, Stmt *Body);

  StmtResult ActOnObjCAtTryStmt(SourceLocation AtLoc, Stmt *Try,
                                MultiStmtArg Catch, Stmt *Finally);

  StmtResult BuildObjCAtThrowStmt(SourceLocation AtLoc, Expr *Throw);
  StmtResult ActOnObjCAtThrowStmt(SourceLocation AtLoc, Expr *Throw,
                                  Scope *CurScope);
  StmtResult ActOnObjCAtSynchronizedStmt(SourceLocation AtLoc,
                                         Expr *SynchExpr,
                                         Stmt *SynchBody);

  VarDecl *BuildExceptionDeclaration(Scope *S, 
                                     TypeSourceInfo *TInfo,
                                     IdentifierInfo *Name,
                                     SourceLocation Loc);
  Decl *ActOnExceptionDeclarator(Scope *S, Declarator &D);

  StmtResult ActOnCXXCatchBlock(SourceLocation CatchLoc,
                                Decl *ExDecl, Stmt *HandlerBlock);
  StmtResult ActOnCXXTryBlock(SourceLocation TryLoc, Stmt *TryBlock,
                              MultiStmtArg Handlers);
  void DiagnoseReturnInConstructorExceptionHandler(CXXTryStmt *TryBlock);

  bool ShouldWarnIfUnusedFileScopedDecl(const DeclaratorDecl *D) const;
  
  /// \brief If it's a file scoped decl that must warn if not used, keep track
  /// of it.
  void MarkUnusedFileScopedDecl(const DeclaratorDecl *D);

  /// DiagnoseUnusedExprResult - If the statement passed in is an expression
  /// whose result is unused, warn.
  void DiagnoseUnusedExprResult(const Stmt *S);
  void DiagnoseUnusedDecl(const NamedDecl *ND);
  
  ParsingDeclState PushParsingDeclaration() {
    return DelayedDiagnostics.pushParsingDecl();
  }
  void PopParsingDeclaration(ParsingDeclState state, Decl *decl) {
    DelayedDiagnostics::popParsingDecl(*this, state, decl);
  }

  typedef ProcessingContextState ParsingClassState;
  ParsingClassState PushParsingClass() {
    return DelayedDiagnostics.pushContext();
  }
  void PopParsingClass(ParsingClassState state) {
    DelayedDiagnostics.popContext(state);
  }

  void EmitDeprecationWarning(NamedDecl *D, llvm::StringRef Message,
                              SourceLocation Loc, bool UnknownObjCClass=false);

  void HandleDelayedDeprecationCheck(sema::DelayedDiagnostic &DD, Decl *Ctx);

  //===--------------------------------------------------------------------===//
  // Expression Parsing Callbacks: SemaExpr.cpp.

  bool DiagnoseUseOfDecl(NamedDecl *D, SourceLocation Loc, 
                         bool UnknownObjCClass=false);
  bool DiagnosePropertyAccessorMismatch(ObjCPropertyDecl *PD,
                                        ObjCMethodDecl *Getter,
                                        SourceLocation Loc);
  void DiagnoseSentinelCalls(NamedDecl *D, SourceLocation Loc,
                             Expr **Args, unsigned NumArgs);

  void PushExpressionEvaluationContext(ExpressionEvaluationContext NewContext);

  void PopExpressionEvaluationContext();

  void MarkDeclarationReferenced(SourceLocation Loc, Decl *D);
  void MarkDeclarationsReferencedInType(SourceLocation Loc, QualType T);
  void MarkDeclarationsReferencedInExpr(Expr *E);
  
  /// \brief Conditionally issue a diagnostic based on the current
  /// evaluation context.
  ///
  /// \param stmt - If stmt is non-null, delay reporting the diagnostic until
  ///  the function body is parsed, and then do a basic reachability analysis to
  ///  determine if the statement is reachable.  If it is unreachable, the
  ///  diagnostic will not be emitted.
  bool DiagRuntimeBehavior(SourceLocation Loc, const Stmt *stmt,
                           const PartialDiagnostic &PD);

  // Primary Expressions.
  SourceRange getExprRange(Expr *E) const;

  ExprResult ActOnIdExpression(Scope *S, CXXScopeSpec &SS, UnqualifiedId &Name,
                               bool HasTrailingLParen, bool IsAddressOfOperand);

  bool DiagnoseEmptyLookup(Scope *S, CXXScopeSpec &SS, LookupResult &R,
                           CorrectTypoContext CTC = CTC_Unknown);

  ExprResult LookupInObjCMethod(LookupResult &R, Scope *S, IdentifierInfo *II,
                                bool AllowBuiltinCreation=false);

  ExprResult ActOnDependentIdExpression(const CXXScopeSpec &SS,
                                        const DeclarationNameInfo &NameInfo,
                                        bool isAddressOfOperand,
                                const TemplateArgumentListInfo *TemplateArgs);

  ExprResult BuildDeclRefExpr(ValueDecl *D, QualType Ty,
                              ExprValueKind VK,
                              SourceLocation Loc,
                              const CXXScopeSpec *SS = 0);
  ExprResult BuildDeclRefExpr(ValueDecl *D, QualType Ty,
                              ExprValueKind VK,
                              const DeclarationNameInfo &NameInfo,
                              const CXXScopeSpec *SS = 0);
  ExprResult
  BuildAnonymousStructUnionMemberReference(const CXXScopeSpec &SS,
                                           SourceLocation nameLoc,
                                           IndirectFieldDecl *indirectField,
                                           Expr *baseObjectExpr = 0,
                                      SourceLocation opLoc = SourceLocation());
  ExprResult BuildPossibleImplicitMemberExpr(const CXXScopeSpec &SS,
                                             LookupResult &R,
                                const TemplateArgumentListInfo *TemplateArgs);
  ExprResult BuildImplicitMemberExpr(const CXXScopeSpec &SS,
                                     LookupResult &R,
                                const TemplateArgumentListInfo *TemplateArgs,
                                     bool IsDefiniteInstance);
  bool UseArgumentDependentLookup(const CXXScopeSpec &SS,
                                  const LookupResult &R,
                                  bool HasTrailingLParen);

  ExprResult BuildQualifiedDeclarationNameExpr(CXXScopeSpec &SS,
                                         const DeclarationNameInfo &NameInfo);
  ExprResult BuildDependentDeclRefExpr(const CXXScopeSpec &SS,
                                const DeclarationNameInfo &NameInfo,
                                const TemplateArgumentListInfo *TemplateArgs);

  ExprResult BuildDeclarationNameExpr(const CXXScopeSpec &SS,
                                      LookupResult &R,
                                      bool ADL);
  ExprResult BuildDeclarationNameExpr(const CXXScopeSpec &SS,
                                      const DeclarationNameInfo &NameInfo,
                                      NamedDecl *D);

  ExprResult ActOnPredefinedExpr(SourceLocation Loc, tok::TokenKind Kind);
  ExprResult ActOnNumericConstant(const Token &);
  ExprResult ActOnCharacterConstant(const Token &);
  ExprResult ActOnParenExpr(SourceLocation L, SourceLocation R, Expr *Val);
  ExprResult ActOnParenOrParenListExpr(SourceLocation L,
                                       SourceLocation R,
                                       MultiExprArg Val,
                                       ParsedType TypeOfCast = ParsedType());

  /// ActOnStringLiteral - The specified tokens were lexed as pasted string
  /// fragments (e.g. "foo" "bar" L"baz").
  ExprResult ActOnStringLiteral(const Token *Toks, unsigned NumToks);

  // Binary/Unary Operators.  'Tok' is the token for the operator.
  ExprResult CreateBuiltinUnaryOp(SourceLocation OpLoc, UnaryOperatorKind Opc,
                                  Expr *InputArg);
  ExprResult BuildUnaryOp(Scope *S, SourceLocation OpLoc,
                          UnaryOperatorKind Opc, Expr *input);
  ExprResult ActOnUnaryOp(Scope *S, SourceLocation OpLoc,
                          tok::TokenKind Op, Expr *Input);

  ExprResult CreateSizeOfAlignOfExpr(TypeSourceInfo *T,
                                     SourceLocation OpLoc,
                                     bool isSizeOf, SourceRange R);
  ExprResult CreateSizeOfAlignOfExpr(Expr *E, SourceLocation OpLoc,
                                     bool isSizeOf, SourceRange R);
  ExprResult
    ActOnSizeOfAlignOfExpr(SourceLocation OpLoc, bool isSizeof, bool isType,
                           void *TyOrEx, const SourceRange &ArgRange);

  ExprResult CheckPlaceholderExpr(Expr *E, SourceLocation Loc);

  bool CheckSizeOfAlignOfOperand(QualType type, SourceLocation OpLoc,
                                 SourceRange R, bool isSizeof);
  ExprResult ActOnSizeofParameterPackExpr(Scope *S,
                                          SourceLocation OpLoc,
                                          IdentifierInfo &Name,
                                          SourceLocation NameLoc,
                                          SourceLocation RParenLoc);
  ExprResult ActOnPostfixUnaryOp(Scope *S, SourceLocation OpLoc,
                                 tok::TokenKind Kind, Expr *Input);

  ExprResult ActOnArraySubscriptExpr(Scope *S, Expr *Base, SourceLocation LLoc,
                                     Expr *Idx, SourceLocation RLoc);
  ExprResult CreateBuiltinArraySubscriptExpr(Expr *Base, SourceLocation LLoc,
                                             Expr *Idx, SourceLocation RLoc);

  ExprResult BuildMemberReferenceExpr(Expr *Base, QualType BaseType,
                                      SourceLocation OpLoc, bool IsArrow,
                                      CXXScopeSpec &SS,
                                      NamedDecl *FirstQualifierInScope,
                                const DeclarationNameInfo &NameInfo,
                                const TemplateArgumentListInfo *TemplateArgs);

  ExprResult BuildMemberReferenceExpr(Expr *Base, QualType BaseType,
                                      SourceLocation OpLoc, bool IsArrow,
                                      const CXXScopeSpec &SS,
                                      NamedDecl *FirstQualifierInScope,
                                      LookupResult &R,
                                 const TemplateArgumentListInfo *TemplateArgs,
                                      bool SuppressQualifierCheck = false);

  ExprResult LookupMemberExpr(LookupResult &R, Expr *&Base,
                              bool &IsArrow, SourceLocation OpLoc,
                              CXXScopeSpec &SS,
                              Decl *ObjCImpDecl,
                              bool HasTemplateArgs);

  bool CheckQualifiedMemberReference(Expr *BaseExpr, QualType BaseType,
                                     const CXXScopeSpec &SS,
                                     const LookupResult &R);

  ExprResult ActOnDependentMemberExpr(Expr *Base, QualType BaseType,
                                      bool IsArrow, SourceLocation OpLoc,
                                      const CXXScopeSpec &SS,
                                      NamedDecl *FirstQualifierInScope,
                               const DeclarationNameInfo &NameInfo,
                               const TemplateArgumentListInfo *TemplateArgs);

  ExprResult ActOnMemberAccessExpr(Scope *S, Expr *Base,
                                   SourceLocation OpLoc,
                                   tok::TokenKind OpKind,
                                   CXXScopeSpec &SS,
                                   UnqualifiedId &Member,
                                   Decl *ObjCImpDecl,
                                   bool HasTrailingLParen);

  void ActOnDefaultCtorInitializers(Decl *CDtorDecl);
  bool ConvertArgumentsForCall(CallExpr *Call, Expr *Fn,
                               FunctionDecl *FDecl,
                               const FunctionProtoType *Proto,
                               Expr **Args, unsigned NumArgs,
                               SourceLocation RParenLoc);

  /// ActOnCallExpr - Handle a call to Fn with the specified array of arguments.
  /// This provides the location of the left/right parens and a list of comma
  /// locations.
  ExprResult ActOnCallExpr(Scope *S, Expr *Fn, SourceLocation LParenLoc,
                           MultiExprArg Args, SourceLocation RParenLoc,
                           Expr *ExecConfig = 0);
  ExprResult BuildResolvedCallExpr(Expr *Fn, NamedDecl *NDecl,
                                   SourceLocation LParenLoc,
                                   Expr **Args, unsigned NumArgs,
                                   SourceLocation RParenLoc,
                                   Expr *ExecConfig = 0);

  ExprResult ActOnCUDAExecConfigExpr(Scope *S, SourceLocation LLLLoc,
                                MultiExprArg ExecConfig, SourceLocation GGGLoc);

  ExprResult ActOnCastExpr(Scope *S, SourceLocation LParenLoc,
                           ParsedType Ty, SourceLocation RParenLoc,
                           Expr *Op);
  ExprResult BuildCStyleCastExpr(SourceLocation LParenLoc,
                                 TypeSourceInfo *Ty,
                                 SourceLocation RParenLoc,
                                 Expr *Op);

  bool TypeIsVectorType(ParsedType Ty) {
    return GetTypeFromParser(Ty)->isVectorType();
  }

  ExprResult MaybeConvertParenListExprToParenExpr(Scope *S, Expr *ME);
  ExprResult ActOnCastOfParenListExpr(Scope *S, SourceLocation LParenLoc,
                                      SourceLocation RParenLoc, Expr *E,
                                      TypeSourceInfo *TInfo);

  ExprResult ActOnCompoundLiteral(SourceLocation LParenLoc,
                                  ParsedType Ty,
                                  SourceLocation RParenLoc,
                                  Expr *Op);

  ExprResult BuildCompoundLiteralExpr(SourceLocation LParenLoc,
                                      TypeSourceInfo *TInfo,
                                      SourceLocation RParenLoc,
                                      Expr *InitExpr);

  ExprResult ActOnInitList(SourceLocation LParenLoc,
                           MultiExprArg InitList,
                           SourceLocation RParenLoc);

  ExprResult ActOnDesignatedInitializer(Designation &Desig,
                                        SourceLocation Loc,
                                        bool GNUSyntax,
                                        ExprResult Init);

  ExprResult ActOnBinOp(Scope *S, SourceLocation TokLoc,
                        tok::TokenKind Kind, Expr *LHS, Expr *RHS);
  ExprResult BuildBinOp(Scope *S, SourceLocation OpLoc,
                        BinaryOperatorKind Opc, Expr *lhs, Expr *rhs);
  ExprResult CreateBuiltinBinOp(SourceLocation TokLoc,
                                BinaryOperatorKind Opc, Expr *lhs, Expr *rhs);

  /// ActOnConditionalOp - Parse a ?: operation.  Note that 'LHS' may be null
  /// in the case of a the GNU conditional expr extension.
  ExprResult ActOnConditionalOp(SourceLocation QuestionLoc,
                                SourceLocation ColonLoc,
                                Expr *Cond, Expr *LHS, Expr *RHS);

  /// ActOnAddrLabel - Parse the GNU address of label extension: "&&foo".
  ExprResult ActOnAddrLabel(SourceLocation OpLoc, SourceLocation LabLoc,
                            LabelDecl *LD);
  
  ExprResult ActOnStmtExpr(SourceLocation LPLoc, Stmt *SubStmt,
                           SourceLocation RPLoc); // "({..})"

  // __builtin_offsetof(type, identifier(.identifier|[expr])*)
  struct OffsetOfComponent {
    SourceLocation LocStart, LocEnd;
    bool isBrackets;  // true if [expr], false if .ident
    union {
      IdentifierInfo *IdentInfo;
      ExprTy *E;
    } U;
  };

  /// __builtin_offsetof(type, a.b[123][456].c)
  ExprResult BuildBuiltinOffsetOf(SourceLocation BuiltinLoc,
                                  TypeSourceInfo *TInfo,
                                  OffsetOfComponent *CompPtr,
                                  unsigned NumComponents,
                                  SourceLocation RParenLoc);
  ExprResult ActOnBuiltinOffsetOf(Scope *S,
                                  SourceLocation BuiltinLoc,
                                  SourceLocation TypeLoc,
                                  ParsedType Arg1,
                                  OffsetOfComponent *CompPtr,
                                  unsigned NumComponents,
                                  SourceLocation RParenLoc);

  // __builtin_choose_expr(constExpr, expr1, expr2)
  ExprResult ActOnChooseExpr(SourceLocation BuiltinLoc,
                             Expr *cond, Expr *expr1,
                             Expr *expr2, SourceLocation RPLoc);

  // __builtin_va_arg(expr, type)
  ExprResult ActOnVAArg(SourceLocation BuiltinLoc,
                        Expr *expr, ParsedType type,
                        SourceLocation RPLoc);
  ExprResult BuildVAArgExpr(SourceLocation BuiltinLoc,
                            Expr *expr, TypeSourceInfo *TInfo,
                            SourceLocation RPLoc);

  // __null
  ExprResult ActOnGNUNullExpr(SourceLocation TokenLoc);

  //===------------------------- "Block" Extension ------------------------===//

  /// ActOnBlockStart - This callback is invoked when a block literal is
  /// started.
  void ActOnBlockStart(SourceLocation CaretLoc, Scope *CurScope);

  /// ActOnBlockArguments - This callback allows processing of block arguments.
  /// If there are no arguments, this is still invoked.
  void ActOnBlockArguments(Declarator &ParamInfo, Scope *CurScope);

  /// ActOnBlockError - If there is an error parsing a block, this callback
  /// is invoked to pop the information about the block from the action impl.
  void ActOnBlockError(SourceLocation CaretLoc, Scope *CurScope);

  /// ActOnBlockStmtExpr - This is called when the body of a block statement
  /// literal was successfully completed.  ^(int x){...}
  ExprResult ActOnBlockStmtExpr(SourceLocation CaretLoc,
                                        Stmt *Body, Scope *CurScope);

  //===---------------------------- C++ Features --------------------------===//

  // Act on C++ namespaces
  Decl *ActOnStartNamespaceDef(Scope *S, SourceLocation InlineLoc,
                               SourceLocation IdentLoc,
                               IdentifierInfo *Ident,
                               SourceLocation LBrace,
                               AttributeList *AttrList);
  void ActOnFinishNamespaceDef(Decl *Dcl, SourceLocation RBrace);

  NamespaceDecl *getStdNamespace() const;
  NamespaceDecl *getOrCreateStdNamespace();

  CXXRecordDecl *getStdBadAlloc() const;

  Decl *ActOnUsingDirective(Scope *CurScope,
                            SourceLocation UsingLoc,
                            SourceLocation NamespcLoc,
                            CXXScopeSpec &SS,
                            SourceLocation IdentLoc,
                            IdentifierInfo *NamespcName,
                            AttributeList *AttrList);

  void PushUsingDirective(Scope *S, UsingDirectiveDecl *UDir);

  Decl *ActOnNamespaceAliasDef(Scope *CurScope,
                               SourceLocation NamespaceLoc,
                               SourceLocation AliasLoc,
                               IdentifierInfo *Alias,
                               CXXScopeSpec &SS,
                               SourceLocation IdentLoc,
                               IdentifierInfo *Ident);

  void HideUsingShadowDecl(Scope *S, UsingShadowDecl *Shadow);
  bool CheckUsingShadowDecl(UsingDecl *UD, NamedDecl *Target,
                            const LookupResult &PreviousDecls);
  UsingShadowDecl *BuildUsingShadowDecl(Scope *S, UsingDecl *UD,
                                        NamedDecl *Target);

  bool CheckUsingDeclRedeclaration(SourceLocation UsingLoc,
                                   bool isTypeName,
                                   const CXXScopeSpec &SS,
                                   SourceLocation NameLoc,
                                   const LookupResult &Previous);
  bool CheckUsingDeclQualifier(SourceLocation UsingLoc,
                               const CXXScopeSpec &SS,
                               SourceLocation NameLoc);

  NamedDecl *BuildUsingDeclaration(Scope *S, AccessSpecifier AS,
                                   SourceLocation UsingLoc,
                                   CXXScopeSpec &SS,
                                   const DeclarationNameInfo &NameInfo,
                                   AttributeList *AttrList,
                                   bool IsInstantiation,
                                   bool IsTypeName,
                                   SourceLocation TypenameLoc);

  bool CheckInheritedConstructorUsingDecl(UsingDecl *UD);

  Decl *ActOnUsingDeclaration(Scope *CurScope,
                              AccessSpecifier AS,
                              bool HasUsingKeyword,
                              SourceLocation UsingLoc,
                              CXXScopeSpec &SS,
                              UnqualifiedId &Name,
                              AttributeList *AttrList,
                              bool IsTypeName,
                              SourceLocation TypenameLoc);

  /// AddCXXDirectInitializerToDecl - This action is called immediately after
  /// ActOnDeclarator, when a C++ direct initializer is present.
  /// e.g: "int x(1);"
  void AddCXXDirectInitializerToDecl(Decl *Dcl,
                                     SourceLocation LParenLoc,
                                     MultiExprArg Exprs,
                                     SourceLocation RParenLoc,
                                     bool TypeMayContainAuto);

  /// InitializeVarWithConstructor - Creates an CXXConstructExpr
  /// and sets it as the initializer for the the passed in VarDecl.
  bool InitializeVarWithConstructor(VarDecl *VD,
                                    CXXConstructorDecl *Constructor,
                                    MultiExprArg Exprs);

  /// BuildCXXConstructExpr - Creates a complete call to a constructor,
  /// including handling of its default argument expressions.
  ///
  /// \param ConstructKind - a CXXConstructExpr::ConstructionKind
  ExprResult
  BuildCXXConstructExpr(SourceLocation ConstructLoc, QualType DeclInitType,
                        CXXConstructorDecl *Constructor, MultiExprArg Exprs,
                        bool RequiresZeroInit, unsigned ConstructKind,
                        SourceRange ParenRange);

  // FIXME: Can re remove this and have the above BuildCXXConstructExpr check if
  // the constructor can be elidable?
  ExprResult
  BuildCXXConstructExpr(SourceLocation ConstructLoc, QualType DeclInitType,
                        CXXConstructorDecl *Constructor, bool Elidable,
                        MultiExprArg Exprs, bool RequiresZeroInit,
                        unsigned ConstructKind,
                        SourceRange ParenRange);

  /// BuildCXXDefaultArgExpr - Creates a CXXDefaultArgExpr, instantiating
  /// the default expr if needed.
  ExprResult BuildCXXDefaultArgExpr(SourceLocation CallLoc,
                                    FunctionDecl *FD,
                                    ParmVarDecl *Param);

  /// FinalizeVarWithDestructor - Prepare for calling destructor on the
  /// constructed variable.
  void FinalizeVarWithDestructor(VarDecl *VD, const RecordType *DeclInitType);

  /// \brief Declare the implicit default constructor for the given class.
  ///
  /// \param ClassDecl The class declaration into which the implicit 
  /// default constructor will be added.
  ///
  /// \returns The implicitly-declared default constructor.
  CXXConstructorDecl *DeclareImplicitDefaultConstructor(
                                                     CXXRecordDecl *ClassDecl);
  
  /// DefineImplicitDefaultConstructor - Checks for feasibility of
  /// defining this constructor as the default constructor.
  void DefineImplicitDefaultConstructor(SourceLocation CurrentLocation,
                                        CXXConstructorDecl *Constructor);

  /// \brief Declare the implicit destructor for the given class.
  ///
  /// \param ClassDecl The class declaration into which the implicit 
  /// destructor will be added.
  ///
  /// \returns The implicitly-declared destructor.
  CXXDestructorDecl *DeclareImplicitDestructor(CXXRecordDecl *ClassDecl);
                                               
  /// DefineImplicitDestructor - Checks for feasibility of
  /// defining this destructor as the default destructor.
  void DefineImplicitDestructor(SourceLocation CurrentLocation,
                                CXXDestructorDecl *Destructor);

  /// \brief Declare all inherited constructors for the given class.
  ///
  /// \param ClassDecl The class declaration into which the inherited
  /// constructors will be added.
  void DeclareInheritedConstructors(CXXRecordDecl *ClassDecl);

  /// \brief Declare the implicit copy constructor for the given class.
  ///
  /// \param S The scope of the class, which may be NULL if this is a 
  /// template instantiation.
  ///
  /// \param ClassDecl The class declaration into which the implicit 
  /// copy constructor will be added.
  ///
  /// \returns The implicitly-declared copy constructor.
  CXXConstructorDecl *DeclareImplicitCopyConstructor(CXXRecordDecl *ClassDecl);
                                                     
  /// DefineImplicitCopyConstructor - Checks for feasibility of
  /// defining this constructor as the copy constructor.
  void DefineImplicitCopyConstructor(SourceLocation CurrentLocation,
                                     CXXConstructorDecl *Constructor,
                                     unsigned TypeQuals);

  /// \brief Declare the implicit copy assignment operator for the given class.
  ///
  /// \param S The scope of the class, which may be NULL if this is a 
  /// template instantiation.
  ///
  /// \param ClassDecl The class declaration into which the implicit 
  /// copy-assignment operator will be added.
  ///
  /// \returns The implicitly-declared copy assignment operator.
  CXXMethodDecl *DeclareImplicitCopyAssignment(CXXRecordDecl *ClassDecl);
  
  /// \brief Defined an implicitly-declared copy assignment operator.
  void DefineImplicitCopyAssignment(SourceLocation CurrentLocation,
                                    CXXMethodDecl *MethodDecl);

  /// \brief Force the declaration of any implicitly-declared members of this
  /// class.
  void ForceDeclarationOfImplicitMembers(CXXRecordDecl *Class);
  
  /// MaybeBindToTemporary - If the passed in expression has a record type with
  /// a non-trivial destructor, this will return CXXBindTemporaryExpr. Otherwise
  /// it simply returns the passed in expression.
  ExprResult MaybeBindToTemporary(Expr *E);

  bool CompleteConstructorCall(CXXConstructorDecl *Constructor,
                               MultiExprArg ArgsPtr,
                               SourceLocation Loc,
                               ASTOwningVector<Expr*> &ConvertedArgs);

  ParsedType getDestructorName(SourceLocation TildeLoc,
                               IdentifierInfo &II, SourceLocation NameLoc,
                               Scope *S, CXXScopeSpec &SS,
                               ParsedType ObjectType,
                               bool EnteringContext);

  /// ActOnCXXNamedCast - Parse {dynamic,static,reinterpret,const}_cast's.
  ExprResult ActOnCXXNamedCast(SourceLocation OpLoc,
                               tok::TokenKind Kind,
                               SourceLocation LAngleBracketLoc,
                               ParsedType Ty,
                               SourceLocation RAngleBracketLoc,
                               SourceLocation LParenLoc,
                               Expr *E,
                               SourceLocation RParenLoc);

  ExprResult BuildCXXNamedCast(SourceLocation OpLoc,
                               tok::TokenKind Kind,
                               TypeSourceInfo *Ty,
                               Expr *E,
                               SourceRange AngleBrackets,
                               SourceRange Parens);

  ExprResult BuildCXXTypeId(QualType TypeInfoType,
                            SourceLocation TypeidLoc,
                            TypeSourceInfo *Operand,
                            SourceLocation RParenLoc);
  ExprResult BuildCXXTypeId(QualType TypeInfoType,
                            SourceLocation TypeidLoc,
                            Expr *Operand,
                            SourceLocation RParenLoc);

  /// ActOnCXXTypeid - Parse typeid( something ).
  ExprResult ActOnCXXTypeid(SourceLocation OpLoc,
                            SourceLocation LParenLoc, bool isType,
                            void *TyOrExpr,
                            SourceLocation RParenLoc);

  ExprResult BuildCXXUuidof(QualType TypeInfoType,
                            SourceLocation TypeidLoc,
                            TypeSourceInfo *Operand,
                            SourceLocation RParenLoc);
  ExprResult BuildCXXUuidof(QualType TypeInfoType,
                            SourceLocation TypeidLoc,
                            Expr *Operand,
                            SourceLocation RParenLoc);

  /// ActOnCXXUuidof - Parse __uuidof( something ).
  ExprResult ActOnCXXUuidof(SourceLocation OpLoc,
                            SourceLocation LParenLoc, bool isType,
                            void *TyOrExpr,
                            SourceLocation RParenLoc);


  //// ActOnCXXThis -  Parse 'this' pointer.
  ExprResult ActOnCXXThis(SourceLocation loc);

  /// tryCaptureCXXThis - Try to capture a 'this' pointer.  Returns a
  /// pointer to an instance method whose 'this' pointer is
  /// capturable, or null if this is not possible.
  CXXMethodDecl *tryCaptureCXXThis();

  /// ActOnCXXBoolLiteral - Parse {true,false} literals.
  ExprResult ActOnCXXBoolLiteral(SourceLocation OpLoc, tok::TokenKind Kind);

  /// ActOnCXXNullPtrLiteral - Parse 'nullptr'.
  ExprResult ActOnCXXNullPtrLiteral(SourceLocation Loc);

  //// ActOnCXXThrow -  Parse throw expressions.
  ExprResult ActOnCXXThrow(SourceLocation OpLoc, Expr *expr);
  bool CheckCXXThrowOperand(SourceLocation ThrowLoc, Expr *&E);

  /// ActOnCXXTypeConstructExpr - Parse construction of a specified type.
  /// Can be interpreted either as function-style casting ("int(x)")
  /// or class type construction ("ClassType(x,y,z)")
  /// or creation of a value-initialized type ("int()").
  ExprResult ActOnCXXTypeConstructExpr(ParsedType TypeRep,
                                       SourceLocation LParenLoc,
                                       MultiExprArg Exprs,
                                       SourceLocation RParenLoc);

  ExprResult BuildCXXTypeConstructExpr(TypeSourceInfo *Type,
                                       SourceLocation LParenLoc,
                                       MultiExprArg Exprs,
                                       SourceLocation RParenLoc);

  /// ActOnCXXNew - Parsed a C++ 'new' expression.
  ExprResult ActOnCXXNew(SourceLocation StartLoc, bool UseGlobal,
                         SourceLocation PlacementLParen,
                         MultiExprArg PlacementArgs,
                         SourceLocation PlacementRParen,
                         SourceRange TypeIdParens, Declarator &D,
                         SourceLocation ConstructorLParen,
                         MultiExprArg ConstructorArgs,
                         SourceLocation ConstructorRParen);
  ExprResult BuildCXXNew(SourceLocation StartLoc, bool UseGlobal,
                         SourceLocation PlacementLParen,
                         MultiExprArg PlacementArgs,
                         SourceLocation PlacementRParen,
                         SourceRange TypeIdParens,
                         QualType AllocType,
                         TypeSourceInfo *AllocTypeInfo,
                         Expr *ArraySize,
                         SourceLocation ConstructorLParen,
                         MultiExprArg ConstructorArgs,
                         SourceLocation ConstructorRParen,
                         bool TypeMayContainAuto = true);

  bool CheckAllocatedType(QualType AllocType, SourceLocation Loc,
                          SourceRange R);
  bool FindAllocationFunctions(SourceLocation StartLoc, SourceRange Range,
                               bool UseGlobal, QualType AllocType, bool IsArray,
                               Expr **PlaceArgs, unsigned NumPlaceArgs,
                               FunctionDecl *&OperatorNew,
                               FunctionDecl *&OperatorDelete);
  bool FindAllocationOverload(SourceLocation StartLoc, SourceRange Range,
                              DeclarationName Name, Expr** Args,
                              unsigned NumArgs, DeclContext *Ctx,
                              bool AllowMissing, FunctionDecl *&Operator);
  void DeclareGlobalNewDelete();
  void DeclareGlobalAllocationFunction(DeclarationName Name, QualType Return,
                                       QualType Argument,
                                       bool addMallocAttr = false);

  bool FindDeallocationFunction(SourceLocation StartLoc, CXXRecordDecl *RD,
                                DeclarationName Name, FunctionDecl* &Operator);

  /// ActOnCXXDelete - Parsed a C++ 'delete' expression
  ExprResult ActOnCXXDelete(SourceLocation StartLoc,
                            bool UseGlobal, bool ArrayForm,
                            Expr *Operand);

  DeclResult ActOnCXXConditionDeclaration(Scope *S, Declarator &D);
  ExprResult CheckConditionVariable(VarDecl *ConditionVar,
                                    SourceLocation StmtLoc,
                                    bool ConvertToBoolean);

  ExprResult ActOnNoexceptExpr(SourceLocation KeyLoc, SourceLocation LParen,
                               Expr *Operand, SourceLocation RParen);
  ExprResult BuildCXXNoexceptExpr(SourceLocation KeyLoc, Expr *Operand,
                                  SourceLocation RParen);

  /// ActOnUnaryTypeTrait - Parsed one of the unary type trait support
  /// pseudo-functions.
  ExprResult ActOnUnaryTypeTrait(UnaryTypeTrait OTT,
                                 SourceLocation KWLoc,
                                 ParsedType Ty,
                                 SourceLocation RParen);

  ExprResult BuildUnaryTypeTrait(UnaryTypeTrait OTT,
                                 SourceLocation KWLoc,
                                 TypeSourceInfo *T,
                                 SourceLocation RParen);

  /// ActOnBinaryTypeTrait - Parsed one of the bianry type trait support
  /// pseudo-functions.
  ExprResult ActOnBinaryTypeTrait(BinaryTypeTrait OTT,
                                  SourceLocation KWLoc,
                                  ParsedType LhsTy,
                                  ParsedType RhsTy,
                                  SourceLocation RParen);

  ExprResult BuildBinaryTypeTrait(BinaryTypeTrait BTT,
                                  SourceLocation KWLoc,
                                  TypeSourceInfo *LhsT,
                                  TypeSourceInfo *RhsT,
                                  SourceLocation RParen);

  ExprResult ActOnStartCXXMemberReference(Scope *S,
                                          Expr *Base,
                                          SourceLocation OpLoc,
                                          tok::TokenKind OpKind,
                                          ParsedType &ObjectType,
                                          bool &MayBePseudoDestructor);

  ExprResult DiagnoseDtorReference(SourceLocation NameLoc, Expr *MemExpr);

  ExprResult BuildPseudoDestructorExpr(Expr *Base,
                                       SourceLocation OpLoc,
                                       tok::TokenKind OpKind,
                                       const CXXScopeSpec &SS,
                                       TypeSourceInfo *ScopeType,
                                       SourceLocation CCLoc,
                                       SourceLocation TildeLoc,
                                     PseudoDestructorTypeStorage DestroyedType,
                                       bool HasTrailingLParen);

  ExprResult ActOnPseudoDestructorExpr(Scope *S, Expr *Base,
                                       SourceLocation OpLoc,
                                       tok::TokenKind OpKind,
                                       CXXScopeSpec &SS,
                                       UnqualifiedId &FirstTypeName,
                                       SourceLocation CCLoc,
                                       SourceLocation TildeLoc,
                                       UnqualifiedId &SecondTypeName,
                                       bool HasTrailingLParen);

  /// MaybeCreateExprWithCleanups - If the current full-expression
  /// requires any cleanups, surround it with a ExprWithCleanups node.
  /// Otherwise, just returns the passed-in expression.
  Expr *MaybeCreateExprWithCleanups(Expr *SubExpr);
  Stmt *MaybeCreateStmtWithCleanups(Stmt *SubStmt);
  ExprResult MaybeCreateExprWithCleanups(ExprResult SubExpr);

  ExprResult ActOnFinishFullExpr(Expr *Expr);
  StmtResult ActOnFinishFullStmt(Stmt *Stmt);

  // Marks SS invalid if it represents an incomplete type.
  bool RequireCompleteDeclContext(CXXScopeSpec &SS, DeclContext *DC);

  DeclContext *computeDeclContext(QualType T);
  DeclContext *computeDeclContext(const CXXScopeSpec &SS,
                                  bool EnteringContext = false);
  bool isDependentScopeSpecifier(const CXXScopeSpec &SS);
  CXXRecordDecl *getCurrentInstantiationOf(NestedNameSpecifier *NNS);
  bool isUnknownSpecialization(const CXXScopeSpec &SS);

  /// \brief The parser has parsed a global nested-name-specifier '::'.
  ///
  /// \param S The scope in which this nested-name-specifier occurs.
  ///
  /// \param CCLoc The location of the '::'.
  ///
  /// \param SS The nested-name-specifier, which will be updated in-place
  /// to reflect the parsed nested-name-specifier.
  ///
  /// \returns true if an error occurred, false otherwise.
  bool ActOnCXXGlobalScopeSpecifier(Scope *S, SourceLocation CCLoc,
                                    CXXScopeSpec &SS);
  
  bool isAcceptableNestedNameSpecifier(NamedDecl *SD);
  NamedDecl *FindFirstQualifierInScope(Scope *S, NestedNameSpecifier *NNS);

  bool isNonTypeNestedNameSpecifier(Scope *S, CXXScopeSpec &SS,
                                    SourceLocation IdLoc,
                                    IdentifierInfo &II,
                                    ParsedType ObjectType);

  bool BuildCXXNestedNameSpecifier(Scope *S,
                                   IdentifierInfo &Identifier,
                                   SourceLocation IdentifierLoc,
                                   SourceLocation CCLoc,
                                   QualType ObjectType,
                                   bool EnteringContext,
                                   CXXScopeSpec &SS,
                                   NamedDecl *ScopeLookupResult,
                                   bool ErrorRecoveryLookup);

  /// \brief The parser has parsed a nested-name-specifier 'identifier::'.
  ///
  /// \param S The scope in which this nested-name-specifier occurs.
  ///
  /// \param Identifier The identifier preceding the '::'.
  ///
  /// \param IdentifierLoc The location of the identifier.
  ///
  /// \param CCLoc The location of the '::'.
  ///
  /// \param ObjectType The type of the object, if we're parsing 
  /// nested-name-specifier in a member access expression.
  ///
  /// \param EnteringContext Whether we're entering the context nominated by
  /// this nested-name-specifier.
  ///
  /// \param SS The nested-name-specifier, which is both an input
  /// parameter (the nested-name-specifier before this type) and an
  /// output parameter (containing the full nested-name-specifier,
  /// including this new type).
  ///
  /// \returns true if an error occurred, false otherwise.
  bool ActOnCXXNestedNameSpecifier(Scope *S,
                                   IdentifierInfo &Identifier,
                                   SourceLocation IdentifierLoc,
                                   SourceLocation CCLoc,
                                   ParsedType ObjectType,
                                   bool EnteringContext,
                                   CXXScopeSpec &SS);

  bool IsInvalidUnlessNestedName(Scope *S, CXXScopeSpec &SS,
                                 IdentifierInfo &Identifier,
                                 SourceLocation IdentifierLoc,
                                 SourceLocation ColonLoc,
                                 ParsedType ObjectType,
                                 bool EnteringContext);

  /// \brief The parser has parsed a nested-name-specifier 
  /// 'template[opt] template-name < template-args >::'.
  ///
  /// \param S The scope in which this nested-name-specifier occurs.
  ///
  /// \param TemplateLoc The location of the 'template' keyword, if any.
  ///
  /// \param SS The nested-name-specifier, which is both an input
  /// parameter (the nested-name-specifier before this type) and an
  /// output parameter (containing the full nested-name-specifier,
  /// including this new type).
  /// 
  /// \param TemplateLoc the location of the 'template' keyword, if any.
  /// \param TemplateName The template name.
  /// \param TemplateNameLoc The location of the template name.
  /// \param LAngleLoc The location of the opening angle bracket  ('<').
  /// \param TemplateArgs The template arguments.
  /// \param RAngleLoc The location of the closing angle bracket  ('>').
  /// \param CCLoc The location of the '::'.
  
  /// \param EnteringContext Whether we're entering the context of the 
  /// nested-name-specifier.
  ///
  ///
  /// \returns true if an error occurred, false otherwise.
  bool ActOnCXXNestedNameSpecifier(Scope *S,
                                   SourceLocation TemplateLoc, 
                                   CXXScopeSpec &SS, 
                                   TemplateTy Template,
                                   SourceLocation TemplateNameLoc,
                                   SourceLocation LAngleLoc,
                                   ASTTemplateArgsPtr TemplateArgs,
                                   SourceLocation RAngleLoc,
                                   SourceLocation CCLoc,
                                   bool EnteringContext);

  /// \brief Given a C++ nested-name-specifier, produce an annotation value
  /// that the parser can use later to reconstruct the given 
  /// nested-name-specifier.
  ///
  /// \param SS A nested-name-specifier.
  ///
  /// \returns A pointer containing all of the information in the 
  /// nested-name-specifier \p SS.
  void *SaveNestedNameSpecifierAnnotation(CXXScopeSpec &SS);
  
  /// \brief Given an annotation pointer for a nested-name-specifier, restore 
  /// the nested-name-specifier structure.
  ///
  /// \param Annotation The annotation pointer, produced by 
  /// \c SaveNestedNameSpecifierAnnotation().
  ///
  /// \param AnnotationRange The source range corresponding to the annotation.
  ///
  /// \param SS The nested-name-specifier that will be updated with the contents
  /// of the annotation pointer.
  void RestoreNestedNameSpecifierAnnotation(void *Annotation, 
                                            SourceRange AnnotationRange,
                                            CXXScopeSpec &SS);
  
  bool ShouldEnterDeclaratorScope(Scope *S, const CXXScopeSpec &SS);

  /// ActOnCXXEnterDeclaratorScope - Called when a C++ scope specifier (global
  /// scope or nested-name-specifier) is parsed, part of a declarator-id.
  /// After this method is called, according to [C++ 3.4.3p3], names should be
  /// looked up in the declarator-id's scope, until the declarator is parsed and
  /// ActOnCXXExitDeclaratorScope is called.
  /// The 'SS' should be a non-empty valid CXXScopeSpec.
  bool ActOnCXXEnterDeclaratorScope(Scope *S, CXXScopeSpec &SS);

  /// ActOnCXXExitDeclaratorScope - Called when a declarator that previously
  /// invoked ActOnCXXEnterDeclaratorScope(), is finished. 'SS' is the same
  /// CXXScopeSpec that was passed to ActOnCXXEnterDeclaratorScope as well.
  /// Used to indicate that names should revert to being looked up in the
  /// defining scope.
  void ActOnCXXExitDeclaratorScope(Scope *S, const CXXScopeSpec &SS);

  /// ActOnCXXEnterDeclInitializer - Invoked when we are about to parse an
  /// initializer for the declaration 'Dcl'.
  /// After this method is called, according to [C++ 3.4.1p13], if 'Dcl' is a
  /// static data member of class X, names should be looked up in the scope of
  /// class X.
  void ActOnCXXEnterDeclInitializer(Scope *S, Decl *Dcl);

  /// ActOnCXXExitDeclInitializer - Invoked after we are finished parsing an
  /// initializer for the declaration 'Dcl'.
  void ActOnCXXExitDeclInitializer(Scope *S, Decl *Dcl);

  // ParseObjCStringLiteral - Parse Objective-C string literals.
  ExprResult ParseObjCStringLiteral(SourceLocation *AtLocs,
                                    Expr **Strings,
                                    unsigned NumStrings);

  Expr *BuildObjCEncodeExpression(SourceLocation AtLoc,
                                  TypeSourceInfo *EncodedTypeInfo,
                                  SourceLocation RParenLoc);
  ExprResult BuildCXXMemberCallExpr(Expr *Exp, NamedDecl *FoundDecl,
                                    CXXMethodDecl *Method);

  ExprResult ParseObjCEncodeExpression(SourceLocation AtLoc,
                                       SourceLocation EncodeLoc,
                                       SourceLocation LParenLoc,
                                       ParsedType Ty,
                                       SourceLocation RParenLoc);

  // ParseObjCSelectorExpression - Build selector expression for @selector
  ExprResult ParseObjCSelectorExpression(Selector Sel,
                                         SourceLocation AtLoc,
                                         SourceLocation SelLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation RParenLoc);

  // ParseObjCProtocolExpression - Build protocol expression for @protocol
  ExprResult ParseObjCProtocolExpression(IdentifierInfo * ProtocolName,
                                         SourceLocation AtLoc,
                                         SourceLocation ProtoLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation RParenLoc);

  //===--------------------------------------------------------------------===//
  // C++ Declarations
  //
  Decl *ActOnStartLinkageSpecification(Scope *S,
                                       SourceLocation ExternLoc,
                                       SourceLocation LangLoc,
                                       llvm::StringRef Lang,
                                       SourceLocation LBraceLoc);
  Decl *ActOnFinishLinkageSpecification(Scope *S,
                                        Decl *LinkageSpec,
                                        SourceLocation RBraceLoc);


  //===--------------------------------------------------------------------===//
  // C++ Classes
  //
  bool isCurrentClassName(const IdentifierInfo &II, Scope *S,
                          const CXXScopeSpec *SS = 0);

  Decl *ActOnAccessSpecifier(AccessSpecifier Access,
                             SourceLocation ASLoc,
                             SourceLocation ColonLoc);

  Decl *ActOnCXXMemberDeclarator(Scope *S, AccessSpecifier AS,
                                 Declarator &D,
                                 MultiTemplateParamsArg TemplateParameterLists,
                                 Expr *BitfieldWidth, const VirtSpecifiers &VS,
                                 Expr *Init, bool IsDefinition,
                                 bool Deleted = false);

  MemInitResult ActOnMemInitializer(Decl *ConstructorD,
                                    Scope *S,
                                    CXXScopeSpec &SS,
                                    IdentifierInfo *MemberOrBase,
                                    ParsedType TemplateTypeTy,
                                    SourceLocation IdLoc,
                                    SourceLocation LParenLoc,
                                    Expr **Args, unsigned NumArgs,
                                    SourceLocation RParenLoc,
                                    SourceLocation EllipsisLoc);

  MemInitResult BuildMemberInitializer(ValueDecl *Member, Expr **Args,
                                       unsigned NumArgs, SourceLocation IdLoc,
                                       SourceLocation LParenLoc,
                                       SourceLocation RParenLoc);

  MemInitResult BuildBaseInitializer(QualType BaseType,
                                     TypeSourceInfo *BaseTInfo,
                                     Expr **Args, unsigned NumArgs,
                                     SourceLocation LParenLoc,
                                     SourceLocation RParenLoc,
                                     CXXRecordDecl *ClassDecl,
                                     SourceLocation EllipsisLoc);

  MemInitResult BuildDelegatingInitializer(TypeSourceInfo *TInfo,
                                           Expr **Args, unsigned NumArgs,
                                           SourceLocation BaseLoc,
                                           SourceLocation RParenLoc,
                                           SourceLocation LParenLoc,
                                           CXXRecordDecl *ClassDecl);

  bool SetCtorInitializers(CXXConstructorDecl *Constructor,
                           CXXCtorInitializer **Initializers,
                           unsigned NumInitializers, bool AnyErrors);
  
  void SetIvarInitializers(ObjCImplementationDecl *ObjCImplementation);
                           

  /// MarkBaseAndMemberDestructorsReferenced - Given a record decl,
  /// mark all the non-trivial destructors of its members and bases as
  /// referenced.
  void MarkBaseAndMemberDestructorsReferenced(SourceLocation Loc,
                                              CXXRecordDecl *Record);

  /// \brief The list of classes whose vtables have been used within
  /// this translation unit, and the source locations at which the
  /// first use occurred.
  typedef std::pair<CXXRecordDecl*, SourceLocation> VTableUse;

  /// \brief The list of vtables that are required but have not yet been
  /// materialized.
  llvm::SmallVector<VTableUse, 16> VTableUses;

  /// \brief The set of classes whose vtables have been used within
  /// this translation unit, and a bit that will be true if the vtable is
  /// required to be emitted (otherwise, it should be emitted only if needed
  /// by code generation).
  llvm::DenseMap<CXXRecordDecl *, bool> VTablesUsed;

  /// \brief A list of all of the dynamic classes in this translation
  /// unit.
  llvm::SmallVector<CXXRecordDecl *, 16> DynamicClasses;

  /// \brief Note that the vtable for the given class was used at the
  /// given location.
  void MarkVTableUsed(SourceLocation Loc, CXXRecordDecl *Class,
                      bool DefinitionRequired = false);

  /// MarkVirtualMembersReferenced - Will mark all members of the given
  /// CXXRecordDecl referenced.
  void MarkVirtualMembersReferenced(SourceLocation Loc,
                                    const CXXRecordDecl *RD);

  /// \brief Define all of the vtables that have been used in this
  /// translation unit and reference any virtual members used by those
  /// vtables.
  ///
  /// \returns true if any work was done, false otherwise.
  bool DefineUsedVTables();

  void AddImplicitlyDeclaredMembersToClass(CXXRecordDecl *ClassDecl);

  void ActOnMemInitializers(Decl *ConstructorDecl,
                            SourceLocation ColonLoc,
                            MemInitTy **MemInits, unsigned NumMemInits,
                            bool AnyErrors);

  void CheckCompletedCXXClass(CXXRecordDecl *Record);
  void ActOnFinishCXXMemberSpecification(Scope* S, SourceLocation RLoc,
                                         Decl *TagDecl,
                                         SourceLocation LBrac,
                                         SourceLocation RBrac,
                                         AttributeList *AttrList);

  void ActOnReenterTemplateScope(Scope *S, Decl *Template);
  void ActOnStartDelayedMemberDeclarations(Scope *S, Decl *Record);
  void ActOnStartDelayedCXXMethodDeclaration(Scope *S, Decl *Method);
  void ActOnDelayedCXXMethodParameter(Scope *S, Decl *Param);
  void ActOnFinishDelayedCXXMethodDeclaration(Scope *S, Decl *Method);
  void ActOnFinishDelayedMemberDeclarations(Scope *S, Decl *Record);

  Decl *ActOnStaticAssertDeclaration(SourceLocation AssertLoc,
                                     Expr *AssertExpr,
                                     Expr *AssertMessageExpr);

  FriendDecl *CheckFriendTypeDecl(SourceLocation FriendLoc,
                                  TypeSourceInfo *TSInfo);
  Decl *ActOnFriendTypeDecl(Scope *S, const DeclSpec &DS,
                                MultiTemplateParamsArg TemplateParams);
  Decl *ActOnFriendFunctionDecl(Scope *S, Declarator &D, bool IsDefinition,
                                    MultiTemplateParamsArg TemplateParams);

  QualType CheckConstructorDeclarator(Declarator &D, QualType R,
                                      StorageClass& SC);
  void CheckConstructor(CXXConstructorDecl *Constructor);
  QualType CheckDestructorDeclarator(Declarator &D, QualType R,
                                     StorageClass& SC);
  bool CheckDestructor(CXXDestructorDecl *Destructor);
  void CheckConversionDeclarator(Declarator &D, QualType &R,
                                 StorageClass& SC);
  Decl *ActOnConversionDeclarator(CXXConversionDecl *Conversion);

  //===--------------------------------------------------------------------===//
  // C++ Derived Classes
  //

  /// ActOnBaseSpecifier - Parsed a base specifier
  CXXBaseSpecifier *CheckBaseSpecifier(CXXRecordDecl *Class,
                                       SourceRange SpecifierRange,
                                       bool Virtual, AccessSpecifier Access,
                                       TypeSourceInfo *TInfo,
                                       SourceLocation EllipsisLoc);

  BaseResult ActOnBaseSpecifier(Decl *classdecl,
                                SourceRange SpecifierRange,
                                bool Virtual, AccessSpecifier Access,
                                ParsedType basetype, 
                                SourceLocation BaseLoc,
                                SourceLocation EllipsisLoc);

  bool AttachBaseSpecifiers(CXXRecordDecl *Class, CXXBaseSpecifier **Bases,
                            unsigned NumBases);
  void ActOnBaseSpecifiers(Decl *ClassDecl, BaseTy **Bases, unsigned NumBases);

  bool IsDerivedFrom(QualType Derived, QualType Base);
  bool IsDerivedFrom(QualType Derived, QualType Base, CXXBasePaths &Paths);

  // FIXME: I don't like this name.
  void BuildBasePathArray(const CXXBasePaths &Paths, CXXCastPath &BasePath);

  bool BasePathInvolvesVirtualBase(const CXXCastPath &BasePath);

  bool CheckDerivedToBaseConversion(QualType Derived, QualType Base,
                                    SourceLocation Loc, SourceRange Range,
                                    CXXCastPath *BasePath = 0,
                                    bool IgnoreAccess = false);
  bool CheckDerivedToBaseConversion(QualType Derived, QualType Base,
                                    unsigned InaccessibleBaseID,
                                    unsigned AmbigiousBaseConvID,
                                    SourceLocation Loc, SourceRange Range,
                                    DeclarationName Name,
                                    CXXCastPath *BasePath);

  std::string getAmbiguousPathsDisplayString(CXXBasePaths &Paths);

  /// CheckOverridingFunctionReturnType - Checks whether the return types are
  /// covariant, according to C++ [class.virtual]p5.
  bool CheckOverridingFunctionReturnType(const CXXMethodDecl *New,
                                         const CXXMethodDecl *Old);

  /// CheckOverridingFunctionExceptionSpec - Checks whether the exception
  /// spec is a subset of base spec.
  bool CheckOverridingFunctionExceptionSpec(const CXXMethodDecl *New,
                                            const CXXMethodDecl *Old);

  bool CheckPureMethod(CXXMethodDecl *Method, SourceRange InitRange);

  /// CheckOverrideControl - Check C++0x override control semantics.
  void CheckOverrideControl(const Decl *D);

  /// CheckForFunctionMarkedFinal - Checks whether a virtual member function
  /// overrides a virtual member function marked 'final', according to
  /// C++0x [class.virtual]p3.
  bool CheckIfOverriddenFunctionIsMarkedFinal(const CXXMethodDecl *New,
                                              const CXXMethodDecl *Old);
  

  //===--------------------------------------------------------------------===//
  // C++ Access Control
  //

  enum AccessResult {
    AR_accessible,
    AR_inaccessible,
    AR_dependent,
    AR_delayed
  };

  bool SetMemberAccessSpecifier(NamedDecl *MemberDecl,
                                NamedDecl *PrevMemberDecl,
                                AccessSpecifier LexicalAS);

  AccessResult CheckUnresolvedMemberAccess(UnresolvedMemberExpr *E,
                                           DeclAccessPair FoundDecl);
  AccessResult CheckUnresolvedLookupAccess(UnresolvedLookupExpr *E,
                                           DeclAccessPair FoundDecl);
  AccessResult CheckAllocationAccess(SourceLocation OperatorLoc,
                                     SourceRange PlacementRange,
                                     CXXRecordDecl *NamingClass,
                                     DeclAccessPair FoundDecl);
  AccessResult CheckConstructorAccess(SourceLocation Loc,
                                      CXXConstructorDecl *D,
                                      const InitializedEntity &Entity,
                                      AccessSpecifier Access,
                                      bool IsCopyBindingRefToTemp = false);
  AccessResult CheckDestructorAccess(SourceLocation Loc,
                                     CXXDestructorDecl *Dtor,
                                     const PartialDiagnostic &PDiag);
  AccessResult CheckDirectMemberAccess(SourceLocation Loc,
                                       NamedDecl *D,
                                       const PartialDiagnostic &PDiag);
  AccessResult CheckMemberOperatorAccess(SourceLocation Loc,
                                         Expr *ObjectExpr,
                                         Expr *ArgExpr,
                                         DeclAccessPair FoundDecl);
  AccessResult CheckAddressOfMemberAccess(Expr *OvlExpr,
                                          DeclAccessPair FoundDecl);
  AccessResult CheckBaseClassAccess(SourceLocation AccessLoc,
                                    QualType Base, QualType Derived,
                                    const CXXBasePath &Path,
                                    unsigned DiagID,
                                    bool ForceCheck = false,
                                    bool ForceUnprivileged = false);
  void CheckLookupAccess(const LookupResult &R);

  void HandleDependentAccessCheck(const DependentDiagnostic &DD,
                         const MultiLevelTemplateArgumentList &TemplateArgs);
  void PerformDependentDiagnostics(const DeclContext *Pattern,
                        const MultiLevelTemplateArgumentList &TemplateArgs);

  void HandleDelayedAccessCheck(sema::DelayedDiagnostic &DD, Decl *Ctx);

  /// A flag to suppress access checking.
  bool SuppressAccessChecking;

  /// \brief When true, access checking violations are treated as SFINAE
  /// failures rather than hard errors.
  bool AccessCheckingSFINAE;
  
  void ActOnStartSuppressingAccessChecks();
  void ActOnStopSuppressingAccessChecks();

  enum AbstractDiagSelID {
    AbstractNone = -1,
    AbstractReturnType,
    AbstractParamType,
    AbstractVariableType,
    AbstractFieldType,
    AbstractArrayType
  };

  bool RequireNonAbstractType(SourceLocation Loc, QualType T,
                              const PartialDiagnostic &PD);
  void DiagnoseAbstractType(const CXXRecordDecl *RD);

  bool RequireNonAbstractType(SourceLocation Loc, QualType T, unsigned DiagID,
                              AbstractDiagSelID SelID = AbstractNone);

  //===--------------------------------------------------------------------===//
  // C++ Overloaded Operators [C++ 13.5]
  //

  bool CheckOverloadedOperatorDeclaration(FunctionDecl *FnDecl);

  bool CheckLiteralOperatorDeclaration(FunctionDecl *FnDecl);

  //===--------------------------------------------------------------------===//
  // C++ Templates [C++ 14]
  //
  void LookupTemplateName(LookupResult &R, Scope *S, CXXScopeSpec &SS,
                          QualType ObjectType, bool EnteringContext,
                          bool &MemberOfUnknownSpecialization);

  TemplateNameKind isTemplateName(Scope *S,
                                          CXXScopeSpec &SS,
                                          bool hasTemplateKeyword,
                                          UnqualifiedId &Name,
                                          ParsedType ObjectType,
                                          bool EnteringContext,
                                          TemplateTy &Template,
                                          bool &MemberOfUnknownSpecialization);

  bool DiagnoseUnknownTemplateName(const IdentifierInfo &II,
                                   SourceLocation IILoc,
                                   Scope *S,
                                   const CXXScopeSpec *SS,
                                   TemplateTy &SuggestedTemplate,
                                   TemplateNameKind &SuggestedKind);

  bool DiagnoseTemplateParameterShadow(SourceLocation Loc, Decl *PrevDecl);
  TemplateDecl *AdjustDeclIfTemplate(Decl *&Decl);

  Decl *ActOnTypeParameter(Scope *S, bool Typename, bool Ellipsis,
                           SourceLocation EllipsisLoc,
                           SourceLocation KeyLoc,
                           IdentifierInfo *ParamName,
                           SourceLocation ParamNameLoc,
                           unsigned Depth, unsigned Position,
                           SourceLocation EqualLoc,
                           ParsedType DefaultArg);

  QualType CheckNonTypeTemplateParameterType(QualType T, SourceLocation Loc);
  Decl *ActOnNonTypeTemplateParameter(Scope *S, Declarator &D,
                                      unsigned Depth,
                                      unsigned Position,
                                      SourceLocation EqualLoc,
                                      Expr *DefaultArg);
  Decl *ActOnTemplateTemplateParameter(Scope *S,
                                       SourceLocation TmpLoc,
                                       TemplateParamsTy *Params,
                                       SourceLocation EllipsisLoc,
                                       IdentifierInfo *ParamName,
                                       SourceLocation ParamNameLoc,
                                       unsigned Depth,
                                       unsigned Position,
                                       SourceLocation EqualLoc,
                                       ParsedTemplateArgument DefaultArg);

  TemplateParamsTy *
  ActOnTemplateParameterList(unsigned Depth,
                             SourceLocation ExportLoc,
                             SourceLocation TemplateLoc,
                             SourceLocation LAngleLoc,
                             Decl **Params, unsigned NumParams,
                             SourceLocation RAngleLoc);

  /// \brief The context in which we are checking a template parameter
  /// list.
  enum TemplateParamListContext {
    TPC_ClassTemplate,
    TPC_FunctionTemplate,
    TPC_ClassTemplateMember,
    TPC_FriendFunctionTemplate,
    TPC_FriendFunctionTemplateDefinition
  };

  bool CheckTemplateParameterList(TemplateParameterList *NewParams,
                                  TemplateParameterList *OldParams,
                                  TemplateParamListContext TPC);
  TemplateParameterList *
  MatchTemplateParametersToScopeSpecifier(SourceLocation DeclStartLoc,
                                          const CXXScopeSpec &SS,
                                          TemplateParameterList **ParamLists,
                                          unsigned NumParamLists,
                                          bool IsFriend,
                                          bool &IsExplicitSpecialization,
                                          bool &Invalid);

  DeclResult CheckClassTemplate(Scope *S, unsigned TagSpec, TagUseKind TUK,
                                SourceLocation KWLoc, CXXScopeSpec &SS,
                                IdentifierInfo *Name, SourceLocation NameLoc,
                                AttributeList *Attr,
                                TemplateParameterList *TemplateParams,
                                AccessSpecifier AS);

  void translateTemplateArguments(const ASTTemplateArgsPtr &In,
                                  TemplateArgumentListInfo &Out);

  void NoteAllFoundTemplates(TemplateName Name);
  
  QualType CheckTemplateIdType(TemplateName Template,
                               SourceLocation TemplateLoc,
                              TemplateArgumentListInfo &TemplateArgs);

  TypeResult
  ActOnTemplateIdType(CXXScopeSpec &SS,
                      TemplateTy Template, SourceLocation TemplateLoc,
                      SourceLocation LAngleLoc,
                      ASTTemplateArgsPtr TemplateArgs,
                      SourceLocation RAngleLoc);

  /// \brief Parsed an elaborated-type-specifier that refers to a template-id,
  /// such as \c class T::template apply<U>.
  ///
  /// \param TUK 
  TypeResult ActOnTagTemplateIdType(TagUseKind TUK,
                                    TypeSpecifierType TagSpec,
                                    SourceLocation TagLoc,
                                    CXXScopeSpec &SS,
                                    TemplateTy TemplateD, 
                                    SourceLocation TemplateLoc,
                                    SourceLocation LAngleLoc,
                                    ASTTemplateArgsPtr TemplateArgsIn,
                                    SourceLocation RAngleLoc);

  
  ExprResult BuildTemplateIdExpr(const CXXScopeSpec &SS,
                                 LookupResult &R,
                                 bool RequiresADL,
                               const TemplateArgumentListInfo &TemplateArgs);
  ExprResult BuildQualifiedTemplateIdExpr(CXXScopeSpec &SS,
                               const DeclarationNameInfo &NameInfo,
                               const TemplateArgumentListInfo &TemplateArgs);

  TemplateNameKind ActOnDependentTemplateName(Scope *S,
                                              SourceLocation TemplateKWLoc,
                                              CXXScopeSpec &SS,
                                              UnqualifiedId &Name,
                                              ParsedType ObjectType,
                                              bool EnteringContext,
                                              TemplateTy &Template);

  DeclResult
  ActOnClassTemplateSpecialization(Scope *S, unsigned TagSpec, TagUseKind TUK,
                                   SourceLocation KWLoc,
                                   CXXScopeSpec &SS,
                                   TemplateTy Template,
                                   SourceLocation TemplateNameLoc,
                                   SourceLocation LAngleLoc,
                                   ASTTemplateArgsPtr TemplateArgs,
                                   SourceLocation RAngleLoc,
                                   AttributeList *Attr,
                                 MultiTemplateParamsArg TemplateParameterLists);

  Decl *ActOnTemplateDeclarator(Scope *S,
                                MultiTemplateParamsArg TemplateParameterLists,
                                Declarator &D);

  Decl *ActOnStartOfFunctionTemplateDef(Scope *FnBodyScope,
                                  MultiTemplateParamsArg TemplateParameterLists,
                                        Declarator &D);

  bool
  CheckSpecializationInstantiationRedecl(SourceLocation NewLoc,
                                         TemplateSpecializationKind NewTSK,
                                         NamedDecl *PrevDecl,
                                         TemplateSpecializationKind PrevTSK,
                                         SourceLocation PrevPtOfInstantiation,
                                         bool &SuppressNew);

  bool CheckDependentFunctionTemplateSpecialization(FunctionDecl *FD,
                    const TemplateArgumentListInfo &ExplicitTemplateArgs,
                                                    LookupResult &Previous);

  bool CheckFunctionTemplateSpecialization(FunctionDecl *FD,
                         TemplateArgumentListInfo *ExplicitTemplateArgs,
                                           LookupResult &Previous);
  bool CheckMemberSpecialization(NamedDecl *Member, LookupResult &Previous);

  DeclResult
  ActOnExplicitInstantiation(Scope *S,
                             SourceLocation ExternLoc,
                             SourceLocation TemplateLoc,
                             unsigned TagSpec,
                             SourceLocation KWLoc,
                             const CXXScopeSpec &SS,
                             TemplateTy Template,
                             SourceLocation TemplateNameLoc,
                             SourceLocation LAngleLoc,
                             ASTTemplateArgsPtr TemplateArgs,
                             SourceLocation RAngleLoc,
                             AttributeList *Attr);

  DeclResult
  ActOnExplicitInstantiation(Scope *S,
                             SourceLocation ExternLoc,
                             SourceLocation TemplateLoc,
                             unsigned TagSpec,
                             SourceLocation KWLoc,
                             CXXScopeSpec &SS,
                             IdentifierInfo *Name,
                             SourceLocation NameLoc,
                             AttributeList *Attr);

  DeclResult ActOnExplicitInstantiation(Scope *S,
                                        SourceLocation ExternLoc,
                                        SourceLocation TemplateLoc,
                                        Declarator &D);

  TemplateArgumentLoc
  SubstDefaultTemplateArgumentIfAvailable(TemplateDecl *Template,
                                          SourceLocation TemplateLoc,
                                          SourceLocation RAngleLoc,
                                          Decl *Param,
                          llvm::SmallVectorImpl<TemplateArgument> &Converted);

  /// \brief Specifies the context in which a particular template
  /// argument is being checked.
  enum CheckTemplateArgumentKind {
    /// \brief The template argument was specified in the code or was
    /// instantiated with some deduced template arguments.
    CTAK_Specified,

    /// \brief The template argument was deduced via template argument
    /// deduction.
    CTAK_Deduced,

    /// \brief The template argument was deduced from an array bound
    /// via template argument deduction.
    CTAK_DeducedFromArrayBound
  };

  bool CheckTemplateArgument(NamedDecl *Param,
                             const TemplateArgumentLoc &Arg,
                             NamedDecl *Template,
                             SourceLocation TemplateLoc,
                             SourceLocation RAngleLoc,
                             unsigned ArgumentPackIndex,
                           llvm::SmallVectorImpl<TemplateArgument> &Converted,
                             CheckTemplateArgumentKind CTAK = CTAK_Specified);

  /// \brief Check that the given template arguments can be be provided to
  /// the given template, converting the arguments along the way.
  ///
  /// \param Template The template to which the template arguments are being
  /// provided.
  ///
  /// \param TemplateLoc The location of the template name in the source.
  ///
  /// \param TemplateArgs The list of template arguments. If the template is
  /// a template template parameter, this function may extend the set of
  /// template arguments to also include substituted, defaulted template
  /// arguments.
  ///
  /// \param PartialTemplateArgs True if the list of template arguments is
  /// intentionally partial, e.g., because we're checking just the initial
  /// set of template arguments.
  ///
  /// \param Converted Will receive the converted, canonicalized template
  /// arguments.
  ///
  /// \returns True if an error occurred, false otherwise.
  bool CheckTemplateArgumentList(TemplateDecl *Template,
                                 SourceLocation TemplateLoc,
                                 TemplateArgumentListInfo &TemplateArgs,
                                 bool PartialTemplateArgs,
                           llvm::SmallVectorImpl<TemplateArgument> &Converted);

  bool CheckTemplateTypeArgument(TemplateTypeParmDecl *Param,
                                 const TemplateArgumentLoc &Arg,
                           llvm::SmallVectorImpl<TemplateArgument> &Converted);

  bool CheckTemplateArgument(TemplateTypeParmDecl *Param,
                             TypeSourceInfo *Arg);
  bool CheckTemplateArgumentPointerToMember(Expr *Arg,
                                            TemplateArgument &Converted);
  bool CheckTemplateArgument(NonTypeTemplateParmDecl *Param,
                             QualType InstantiatedParamType, Expr *&Arg,
                             TemplateArgument &Converted,
                             CheckTemplateArgumentKind CTAK = CTAK_Specified);
  bool CheckTemplateArgument(TemplateTemplateParmDecl *Param,
                             const TemplateArgumentLoc &Arg);

  ExprResult
  BuildExpressionFromDeclTemplateArgument(const TemplateArgument &Arg,
                                          QualType ParamType,
                                          SourceLocation Loc);
  ExprResult
  BuildExpressionFromIntegralTemplateArgument(const TemplateArgument &Arg,
                                              SourceLocation Loc);

  /// \brief Enumeration describing how template parameter lists are compared
  /// for equality.
  enum TemplateParameterListEqualKind {
    /// \brief We are matching the template parameter lists of two templates
    /// that might be redeclarations.
    ///
    /// \code
    /// template<typename T> struct X;
    /// template<typename T> struct X;
    /// \endcode
    TPL_TemplateMatch,

    /// \brief We are matching the template parameter lists of two template
    /// template parameters as part of matching the template parameter lists
    /// of two templates that might be redeclarations.
    ///
    /// \code
    /// template<template<int I> class TT> struct X;
    /// template<template<int Value> class Other> struct X;
    /// \endcode
    TPL_TemplateTemplateParmMatch,

    /// \brief We are matching the template parameter lists of a template
    /// template argument against the template parameter lists of a template
    /// template parameter.
    ///
    /// \code
    /// template<template<int Value> class Metafun> struct X;
    /// template<int Value> struct integer_c;
    /// X<integer_c> xic;
    /// \endcode
    TPL_TemplateTemplateArgumentMatch
  };

  bool TemplateParameterListsAreEqual(TemplateParameterList *New,
                                      TemplateParameterList *Old,
                                      bool Complain,
                                      TemplateParameterListEqualKind Kind,
                                      SourceLocation TemplateArgLoc
                                        = SourceLocation());

  bool CheckTemplateDeclScope(Scope *S, TemplateParameterList *TemplateParams);

  /// \brief Called when the parser has parsed a C++ typename
  /// specifier, e.g., "typename T::type".
  ///
  /// \param S The scope in which this typename type occurs.
  /// \param TypenameLoc the location of the 'typename' keyword
  /// \param SS the nested-name-specifier following the typename (e.g., 'T::').
  /// \param II the identifier we're retrieving (e.g., 'type' in the example).
  /// \param IdLoc the location of the identifier.
  TypeResult
  ActOnTypenameType(Scope *S, SourceLocation TypenameLoc, 
                    const CXXScopeSpec &SS, const IdentifierInfo &II, 
                    SourceLocation IdLoc);

  /// \brief Called when the parser has parsed a C++ typename
  /// specifier that ends in a template-id, e.g.,
  /// "typename MetaFun::template apply<T1, T2>".
  ///
  /// \param S The scope in which this typename type occurs.
  /// \param TypenameLoc the location of the 'typename' keyword
  /// \param SS the nested-name-specifier following the typename (e.g., 'T::').
  /// \param TemplateLoc the location of the 'template' keyword, if any.
  /// \param TemplateName The template name.
  /// \param TemplateNameLoc The location of the template name.
  /// \param LAngleLoc The location of the opening angle bracket  ('<').
  /// \param TemplateArgs The template arguments.
  /// \param RAngleLoc The location of the closing angle bracket  ('>').
  TypeResult
  ActOnTypenameType(Scope *S, SourceLocation TypenameLoc, 
                    const CXXScopeSpec &SS, 
                    SourceLocation TemplateLoc, 
                    TemplateTy Template,
                    SourceLocation TemplateNameLoc,
                    SourceLocation LAngleLoc,
                    ASTTemplateArgsPtr TemplateArgs,
                    SourceLocation RAngleLoc);

  QualType CheckTypenameType(ElaboratedTypeKeyword Keyword,
                             SourceLocation KeywordLoc,
                             NestedNameSpecifierLoc QualifierLoc,
                             const IdentifierInfo &II,
                             SourceLocation IILoc);

  TypeSourceInfo *RebuildTypeInCurrentInstantiation(TypeSourceInfo *T,
                                                    SourceLocation Loc,
                                                    DeclarationName Name);
  bool RebuildNestedNameSpecifierInCurrentInstantiation(CXXScopeSpec &SS);

  ExprResult RebuildExprInCurrentInstantiation(Expr *E);

  std::string
  getTemplateArgumentBindingsText(const TemplateParameterList *Params,
                                  const TemplateArgumentList &Args);

  std::string
  getTemplateArgumentBindingsText(const TemplateParameterList *Params,
                                  const TemplateArgument *Args,
                                  unsigned NumArgs);

  //===--------------------------------------------------------------------===//
  // C++ Variadic Templates (C++0x [temp.variadic])
  //===--------------------------------------------------------------------===//

  /// \brief The context in which an unexpanded parameter pack is
  /// being diagnosed.
  ///
  /// Note that the values of this enumeration line up with the first
  /// argument to the \c err_unexpanded_parameter_pack diagnostic.
  enum UnexpandedParameterPackContext {
    /// \brief An arbitrary expression.
    UPPC_Expression = 0,

    /// \brief The base type of a class type.
    UPPC_BaseType,

    /// \brief The type of an arbitrary declaration.
    UPPC_DeclarationType,

    /// \brief The type of a data member.
    UPPC_DataMemberType,

    /// \brief The size of a bit-field.
    UPPC_BitFieldWidth,

    /// \brief The expression in a static assertion.
    UPPC_StaticAssertExpression,

    /// \brief The fixed underlying type of an enumeration.
    UPPC_FixedUnderlyingType,

    /// \brief The enumerator value.
    UPPC_EnumeratorValue,

    /// \brief A using declaration.
    UPPC_UsingDeclaration,

    /// \brief A friend declaration.
    UPPC_FriendDeclaration,

    /// \brief A declaration qualifier.
    UPPC_DeclarationQualifier,

    /// \brief An initializer.
    UPPC_Initializer,
    
    /// \brief A default argument.
    UPPC_DefaultArgument,
    
    /// \brief The type of a non-type template parameter.
    UPPC_NonTypeTemplateParameterType,

    /// \brief The type of an exception.
    UPPC_ExceptionType,
    
    /// \brief Partial specialization.
    UPPC_PartialSpecialization
  };

  /// \brief If the given type contains an unexpanded parameter pack,
  /// diagnose the error.
  ///
  /// \param Loc The source location where a diagnostc should be emitted.
  ///
  /// \param T The type that is being checked for unexpanded parameter
  /// packs.
  ///
  /// \returns true if an error ocurred, false otherwise.
  bool DiagnoseUnexpandedParameterPack(SourceLocation Loc, TypeSourceInfo *T,
                                       UnexpandedParameterPackContext UPPC);

  /// \brief If the given expression contains an unexpanded parameter
  /// pack, diagnose the error.
  ///
  /// \param E The expression that is being checked for unexpanded
  /// parameter packs.
  ///
  /// \returns true if an error ocurred, false otherwise.
  bool DiagnoseUnexpandedParameterPack(Expr *E,
                       UnexpandedParameterPackContext UPPC = UPPC_Expression);

  /// \brief If the given nested-name-specifier contains an unexpanded
  /// parameter pack, diagnose the error.
  ///
  /// \param SS The nested-name-specifier that is being checked for
  /// unexpanded parameter packs.
  ///
  /// \returns true if an error ocurred, false otherwise.
  bool DiagnoseUnexpandedParameterPack(const CXXScopeSpec &SS,
                                       UnexpandedParameterPackContext UPPC);

  /// \brief If the given name contains an unexpanded parameter pack,
  /// diagnose the error.
  ///
  /// \param NameInfo The name (with source location information) that
  /// is being checked for unexpanded parameter packs.
  ///
  /// \returns true if an error ocurred, false otherwise.
  bool DiagnoseUnexpandedParameterPack(const DeclarationNameInfo &NameInfo,
                                       UnexpandedParameterPackContext UPPC);

  /// \brief If the given template name contains an unexpanded parameter pack,
  /// diagnose the error.
  ///
  /// \param Loc The location of the template name.
  ///
  /// \param Template The template name that is being checked for unexpanded 
  /// parameter packs.
  ///
  /// \returns true if an error ocurred, false otherwise.
  bool DiagnoseUnexpandedParameterPack(SourceLocation Loc,
                                       TemplateName Template,
                                       UnexpandedParameterPackContext UPPC);

  /// \brief If the given template argument contains an unexpanded parameter 
  /// pack, diagnose the error.
  ///
  /// \param Arg The template argument that is being checked for unexpanded 
  /// parameter packs.
  ///
  /// \returns true if an error ocurred, false otherwise.
  bool DiagnoseUnexpandedParameterPack(TemplateArgumentLoc Arg,
                                       UnexpandedParameterPackContext UPPC);
  
  /// \brief Collect the set of unexpanded parameter packs within the given
  /// template argument.  
  ///
  /// \param Arg The template argument that will be traversed to find
  /// unexpanded parameter packs.
  void collectUnexpandedParameterPacks(TemplateArgument Arg,
                   llvm::SmallVectorImpl<UnexpandedParameterPack> &Unexpanded);

  /// \brief Collect the set of unexpanded parameter packs within the given
  /// template argument.  
  ///
  /// \param Arg The template argument that will be traversed to find
  /// unexpanded parameter packs.
  void collectUnexpandedParameterPacks(TemplateArgumentLoc Arg,
                    llvm::SmallVectorImpl<UnexpandedParameterPack> &Unexpanded);

  /// \brief Collect the set of unexpanded parameter packs within the given
  /// type.  
  ///
  /// \param T The type that will be traversed to find
  /// unexpanded parameter packs.
  void collectUnexpandedParameterPacks(QualType T,
                   llvm::SmallVectorImpl<UnexpandedParameterPack> &Unexpanded);

  /// \brief Collect the set of unexpanded parameter packs within the given
  /// type.  
  ///
  /// \param TL The type that will be traversed to find
  /// unexpanded parameter packs.
  void collectUnexpandedParameterPacks(TypeLoc TL,
                   llvm::SmallVectorImpl<UnexpandedParameterPack> &Unexpanded);

  /// \brief Invoked when parsing a template argument followed by an
  /// ellipsis, which creates a pack expansion.
  ///
  /// \param Arg The template argument preceding the ellipsis, which
  /// may already be invalid.
  ///
  /// \param EllipsisLoc The location of the ellipsis.
  ParsedTemplateArgument ActOnPackExpansion(const ParsedTemplateArgument &Arg,
                                            SourceLocation EllipsisLoc);

  /// \brief Invoked when parsing a type followed by an ellipsis, which
  /// creates a pack expansion.
  ///
  /// \param Type The type preceding the ellipsis, which will become
  /// the pattern of the pack expansion.
  ///
  /// \param EllipsisLoc The location of the ellipsis.
  TypeResult ActOnPackExpansion(ParsedType Type, SourceLocation EllipsisLoc);

  /// \brief Construct a pack expansion type from the pattern of the pack
  /// expansion.
  TypeSourceInfo *CheckPackExpansion(TypeSourceInfo *Pattern,
                                     SourceLocation EllipsisLoc,
                                     llvm::Optional<unsigned> NumExpansions);

  /// \brief Construct a pack expansion type from the pattern of the pack
  /// expansion.
  QualType CheckPackExpansion(QualType Pattern,
                              SourceRange PatternRange,
                              SourceLocation EllipsisLoc,
                              llvm::Optional<unsigned> NumExpansions);

  /// \brief Invoked when parsing an expression followed by an ellipsis, which
  /// creates a pack expansion.
  ///
  /// \param Pattern The expression preceding the ellipsis, which will become
  /// the pattern of the pack expansion.
  ///
  /// \param EllipsisLoc The location of the ellipsis.
  ExprResult ActOnPackExpansion(Expr *Pattern, SourceLocation EllipsisLoc);

  /// \brief Invoked when parsing an expression followed by an ellipsis, which
  /// creates a pack expansion.
  ///
  /// \param Pattern The expression preceding the ellipsis, which will become
  /// the pattern of the pack expansion.
  ///
  /// \param EllipsisLoc The location of the ellipsis.
  ExprResult CheckPackExpansion(Expr *Pattern, SourceLocation EllipsisLoc,
                                llvm::Optional<unsigned> NumExpansions);

  /// \brief Determine whether we could expand a pack expansion with the
  /// given set of parameter packs into separate arguments by repeatedly
  /// transforming the pattern.
  ///
  /// \param EllipsisLoc The location of the ellipsis that identifies the
  /// pack expansion.
  ///
  /// \param PatternRange The source range that covers the entire pattern of
  /// the pack expansion.
  ///
  /// \param Unexpanded The set of unexpanded parameter packs within the 
  /// pattern.
  ///
  /// \param NumUnexpanded The number of unexpanded parameter packs in
  /// \p Unexpanded.
  ///
  /// \param ShouldExpand Will be set to \c true if the transformer should
  /// expand the corresponding pack expansions into separate arguments. When
  /// set, \c NumExpansions must also be set.
  ///
  /// \param RetainExpansion Whether the caller should add an unexpanded
  /// pack expansion after all of the expanded arguments. This is used
  /// when extending explicitly-specified template argument packs per
  /// C++0x [temp.arg.explicit]p9.
  ///
  /// \param NumExpansions The number of separate arguments that will be in
  /// the expanded form of the corresponding pack expansion. This is both an
  /// input and an output parameter, which can be set by the caller if the
  /// number of expansions is known a priori (e.g., due to a prior substitution)
  /// and will be set by the callee when the number of expansions is known.
  /// The callee must set this value when \c ShouldExpand is \c true; it may
  /// set this value in other cases.
  ///
  /// \returns true if an error occurred (e.g., because the parameter packs 
  /// are to be instantiated with arguments of different lengths), false 
  /// otherwise. If false, \c ShouldExpand (and possibly \c NumExpansions) 
  /// must be set.
  bool CheckParameterPacksForExpansion(SourceLocation EllipsisLoc,
                                       SourceRange PatternRange,
                                     const UnexpandedParameterPack *Unexpanded,
                                       unsigned NumUnexpanded,
                             const MultiLevelTemplateArgumentList &TemplateArgs,
                                       bool &ShouldExpand,
                                       bool &RetainExpansion,
                                       llvm::Optional<unsigned> &NumExpansions);

  /// \brief Determine the number of arguments in the given pack expansion
  /// type.
  ///
  /// This routine already assumes that the pack expansion type can be
  /// expanded and that the number of arguments in the expansion is
  /// consistent across all of the unexpanded parameter packs in its pattern.
  unsigned getNumArgumentsInExpansion(QualType T, 
                            const MultiLevelTemplateArgumentList &TemplateArgs);
  
  /// \brief Determine whether the given declarator contains any unexpanded
  /// parameter packs.
  ///
  /// This routine is used by the parser to disambiguate function declarators
  /// with an ellipsis prior to the ')', e.g.,
  ///
  /// \code
  ///   void f(T...);
  /// \endcode
  ///
  /// To determine whether we have an (unnamed) function parameter pack or
  /// a variadic function.
  ///
  /// \returns true if the declarator contains any unexpanded parameter packs,
  /// false otherwise.
  bool containsUnexpandedParameterPacks(Declarator &D);
  
  //===--------------------------------------------------------------------===//
  // C++ Template Argument Deduction (C++ [temp.deduct])
  //===--------------------------------------------------------------------===//
  
  /// \brief Describes the result of template argument deduction.
  ///
  /// The TemplateDeductionResult enumeration describes the result of
  /// template argument deduction, as returned from
  /// DeduceTemplateArguments(). The separate TemplateDeductionInfo
  /// structure provides additional information about the results of
  /// template argument deduction, e.g., the deduced template argument
  /// list (if successful) or the specific template parameters or
  /// deduced arguments that were involved in the failure.
  enum TemplateDeductionResult {
    /// \brief Template argument deduction was successful.
    TDK_Success = 0,
    /// \brief Template argument deduction exceeded the maximum template
    /// instantiation depth (which has already been diagnosed).
    TDK_InstantiationDepth,
    /// \brief Template argument deduction did not deduce a value
    /// for every template parameter.
    TDK_Incomplete,
    /// \brief Template argument deduction produced inconsistent
    /// deduced values for the given template parameter.
    TDK_Inconsistent,
    /// \brief Template argument deduction failed due to inconsistent
    /// cv-qualifiers on a template parameter type that would
    /// otherwise be deduced, e.g., we tried to deduce T in "const T"
    /// but were given a non-const "X".
    TDK_Underqualified,
    /// \brief Substitution of the deduced template argument values
    /// resulted in an error.
    TDK_SubstitutionFailure,
    /// \brief Substitution of the deduced template argument values
    /// into a non-deduced context produced a type or value that
    /// produces a type that does not match the original template
    /// arguments provided.
    TDK_NonDeducedMismatch,
    /// \brief When performing template argument deduction for a function
    /// template, there were too many call arguments.
    TDK_TooManyArguments,
    /// \brief When performing template argument deduction for a function
    /// template, there were too few call arguments.
    TDK_TooFewArguments,
    /// \brief The explicitly-specified template arguments were not valid
    /// template arguments for the given template.
    TDK_InvalidExplicitArguments,
    /// \brief The arguments included an overloaded function name that could
    /// not be resolved to a suitable function.
    TDK_FailedOverloadResolution
  };

  TemplateDeductionResult
  DeduceTemplateArguments(ClassTemplatePartialSpecializationDecl *Partial,
                          const TemplateArgumentList &TemplateArgs,
                          sema::TemplateDeductionInfo &Info);

  TemplateDeductionResult
  SubstituteExplicitTemplateArguments(FunctionTemplateDecl *FunctionTemplate,
                              TemplateArgumentListInfo &ExplicitTemplateArgs,
                      llvm::SmallVectorImpl<DeducedTemplateArgument> &Deduced,
                                 llvm::SmallVectorImpl<QualType> &ParamTypes,
                                      QualType *FunctionType,
                                      sema::TemplateDeductionInfo &Info);

  TemplateDeductionResult
  FinishTemplateArgumentDeduction(FunctionTemplateDecl *FunctionTemplate,
                      llvm::SmallVectorImpl<DeducedTemplateArgument> &Deduced,
                                  unsigned NumExplicitlySpecified,
                                  FunctionDecl *&Specialization,
                                  sema::TemplateDeductionInfo &Info);

  TemplateDeductionResult
  DeduceTemplateArguments(FunctionTemplateDecl *FunctionTemplate,
                          TemplateArgumentListInfo *ExplicitTemplateArgs,
                          Expr **Args, unsigned NumArgs,
                          FunctionDecl *&Specialization,
                          sema::TemplateDeductionInfo &Info);

  TemplateDeductionResult
  DeduceTemplateArguments(FunctionTemplateDecl *FunctionTemplate,
                          TemplateArgumentListInfo *ExplicitTemplateArgs,
                          QualType ArgFunctionType,
                          FunctionDecl *&Specialization,
                          sema::TemplateDeductionInfo &Info);

  TemplateDeductionResult
  DeduceTemplateArguments(FunctionTemplateDecl *FunctionTemplate,
                          QualType ToType,
                          CXXConversionDecl *&Specialization,
                          sema::TemplateDeductionInfo &Info);

  TemplateDeductionResult
  DeduceTemplateArguments(FunctionTemplateDecl *FunctionTemplate,
                          TemplateArgumentListInfo *ExplicitTemplateArgs,
                          FunctionDecl *&Specialization,
                          sema::TemplateDeductionInfo &Info);

  bool DeduceAutoType(TypeSourceInfo *AutoType, Expr *Initializer,
                      TypeSourceInfo *&Result);

  FunctionTemplateDecl *getMoreSpecializedTemplate(FunctionTemplateDecl *FT1,
                                                   FunctionTemplateDecl *FT2,
                                                   SourceLocation Loc,
                                           TemplatePartialOrderingContext TPOC,
                                                   unsigned NumCallArguments);
  UnresolvedSetIterator getMostSpecialized(UnresolvedSetIterator SBegin,
                                           UnresolvedSetIterator SEnd,
                                           TemplatePartialOrderingContext TPOC,
                                           unsigned NumCallArguments,
                                           SourceLocation Loc,
                                           const PartialDiagnostic &NoneDiag,
                                           const PartialDiagnostic &AmbigDiag,
                                        const PartialDiagnostic &CandidateDiag,
                                        bool Complain = true);

  ClassTemplatePartialSpecializationDecl *
  getMoreSpecializedPartialSpecialization(
                                  ClassTemplatePartialSpecializationDecl *PS1,
                                  ClassTemplatePartialSpecializationDecl *PS2,
                                  SourceLocation Loc);

  void MarkUsedTemplateParameters(const TemplateArgumentList &TemplateArgs,
                                  bool OnlyDeduced,
                                  unsigned Depth,
                                  llvm::SmallVectorImpl<bool> &Used);
  void MarkDeducedTemplateParameters(FunctionTemplateDecl *FunctionTemplate,
                                     llvm::SmallVectorImpl<bool> &Deduced);

  //===--------------------------------------------------------------------===//
  // C++ Template Instantiation
  //

  MultiLevelTemplateArgumentList getTemplateInstantiationArgs(NamedDecl *D,
                                     const TemplateArgumentList *Innermost = 0,
                                                bool RelativeToPrimary = false,
                                               const FunctionDecl *Pattern = 0);

  /// \brief A template instantiation that is currently in progress.
  struct ActiveTemplateInstantiation {
    /// \brief The kind of template instantiation we are performing
    enum InstantiationKind {
      /// We are instantiating a template declaration. The entity is
      /// the declaration we're instantiating (e.g., a CXXRecordDecl).
      TemplateInstantiation,

      /// We are instantiating a default argument for a template
      /// parameter. The Entity is the template, and
      /// TemplateArgs/NumTemplateArguments provides the template
      /// arguments as specified.
      /// FIXME: Use a TemplateArgumentList
      DefaultTemplateArgumentInstantiation,

      /// We are instantiating a default argument for a function.
      /// The Entity is the ParmVarDecl, and TemplateArgs/NumTemplateArgs
      /// provides the template arguments as specified.
      DefaultFunctionArgumentInstantiation,

      /// We are substituting explicit template arguments provided for
      /// a function template. The entity is a FunctionTemplateDecl.
      ExplicitTemplateArgumentSubstitution,

      /// We are substituting template argument determined as part of
      /// template argument deduction for either a class template
      /// partial specialization or a function template. The
      /// Entity is either a ClassTemplatePartialSpecializationDecl or
      /// a FunctionTemplateDecl.
      DeducedTemplateArgumentSubstitution,

      /// We are substituting prior template arguments into a new
      /// template parameter. The template parameter itself is either a
      /// NonTypeTemplateParmDecl or a TemplateTemplateParmDecl.
      PriorTemplateArgumentSubstitution,

      /// We are checking the validity of a default template argument that
      /// has been used when naming a template-id.
      DefaultTemplateArgumentChecking
    } Kind;

    /// \brief The point of instantiation within the source code.
    SourceLocation PointOfInstantiation;

    /// \brief The template (or partial specialization) in which we are 
    /// performing the instantiation, for substitutions of prior template 
    /// arguments.
    NamedDecl *Template;

    /// \brief The entity that is being instantiated.
    uintptr_t Entity;

    /// \brief The list of template arguments we are substituting, if they
    /// are not part of the entity.
    const TemplateArgument *TemplateArgs;

    /// \brief The number of template arguments in TemplateArgs.
    unsigned NumTemplateArgs;

    /// \brief The template deduction info object associated with the 
    /// substitution or checking of explicit or deduced template arguments.
    sema::TemplateDeductionInfo *DeductionInfo;

    /// \brief The source range that covers the construct that cause
    /// the instantiation, e.g., the template-id that causes a class
    /// template instantiation.
    SourceRange InstantiationRange;

    ActiveTemplateInstantiation()
      : Kind(TemplateInstantiation), Template(0), Entity(0), TemplateArgs(0),
        NumTemplateArgs(0), DeductionInfo(0) {}

    /// \brief Determines whether this template is an actual instantiation
    /// that should be counted toward the maximum instantiation depth.
    bool isInstantiationRecord() const;

    friend bool operator==(const ActiveTemplateInstantiation &X,
                           const ActiveTemplateInstantiation &Y) {
      if (X.Kind != Y.Kind)
        return false;

      if (X.Entity != Y.Entity)
        return false;

      switch (X.Kind) {
      case TemplateInstantiation:
        return true;

      case PriorTemplateArgumentSubstitution:
      case DefaultTemplateArgumentChecking:
        if (X.Template != Y.Template)
          return false;

        // Fall through

      case DefaultTemplateArgumentInstantiation:
      case ExplicitTemplateArgumentSubstitution:
      case DeducedTemplateArgumentSubstitution:
      case DefaultFunctionArgumentInstantiation:
        return X.TemplateArgs == Y.TemplateArgs;

      }

      return true;
    }

    friend bool operator!=(const ActiveTemplateInstantiation &X,
                           const ActiveTemplateInstantiation &Y) {
      return !(X == Y);
    }
  };

  /// \brief List of active template instantiations.
  ///
  /// This vector is treated as a stack. As one template instantiation
  /// requires another template instantiation, additional
  /// instantiations are pushed onto the stack up to a
  /// user-configurable limit LangOptions::InstantiationDepth.
  llvm::SmallVector<ActiveTemplateInstantiation, 16>
    ActiveTemplateInstantiations;

  /// \brief Whether we are in a SFINAE context that is not associated with
  /// template instantiation.
  ///
  /// This is used when setting up a SFINAE trap (\c see SFINAETrap) outside
  /// of a template instantiation or template argument deduction.
  bool InNonInstantiationSFINAEContext;
  
  /// \brief The number of ActiveTemplateInstantiation entries in
  /// \c ActiveTemplateInstantiations that are not actual instantiations and,
  /// therefore, should not be counted as part of the instantiation depth.
  unsigned NonInstantiationEntries;

  /// \brief The last template from which a template instantiation
  /// error or warning was produced.
  ///
  /// This value is used to suppress printing of redundant template
  /// instantiation backtraces when there are multiple errors in the
  /// same instantiation. FIXME: Does this belong in Sema? It's tough
  /// to implement it anywhere else.
  ActiveTemplateInstantiation LastTemplateInstantiationErrorContext;

  /// \brief The current index into pack expansion arguments that will be
  /// used for substitution of parameter packs.
  ///
  /// The pack expansion index will be -1 to indicate that parameter packs 
  /// should be instantiated as themselves. Otherwise, the index specifies
  /// which argument within the parameter pack will be used for substitution.
  int ArgumentPackSubstitutionIndex;

  /// \brief RAII object used to change the argument pack substitution index
  /// within a \c Sema object.
  ///
  /// See \c ArgumentPackSubstitutionIndex for more information.
  class ArgumentPackSubstitutionIndexRAII {
    Sema &Self;
    int OldSubstitutionIndex;
    
  public:
    ArgumentPackSubstitutionIndexRAII(Sema &Self, int NewSubstitutionIndex)
      : Self(Self), OldSubstitutionIndex(Self.ArgumentPackSubstitutionIndex) {
      Self.ArgumentPackSubstitutionIndex = NewSubstitutionIndex;
    }
    
    ~ArgumentPackSubstitutionIndexRAII() {
      Self.ArgumentPackSubstitutionIndex = OldSubstitutionIndex;
    }
  };
  
  friend class ArgumentPackSubstitutionRAII;
  
  /// \brief The stack of calls expression undergoing template instantiation.
  ///
  /// The top of this stack is used by a fixit instantiating unresolved
  /// function calls to fix the AST to match the textual change it prints.
  llvm::SmallVector<CallExpr *, 8> CallsUndergoingInstantiation;
  
  /// \brief For each declaration that involved template argument deduction, the
  /// set of diagnostics that were suppressed during that template argument
  /// deduction.
  ///
  /// FIXME: Serialize this structure to the AST file.
  llvm::DenseMap<Decl *, llvm::SmallVector<PartialDiagnosticAt, 1> >
    SuppressedDiagnostics;
  
  /// \brief A stack object to be created when performing template
  /// instantiation.
  ///
  /// Construction of an object of type \c InstantiatingTemplate
  /// pushes the current instantiation onto the stack of active
  /// instantiations. If the size of this stack exceeds the maximum
  /// number of recursive template instantiations, construction
  /// produces an error and evaluates true.
  ///
  /// Destruction of this object will pop the named instantiation off
  /// the stack.
  struct InstantiatingTemplate {
    /// \brief Note that we are instantiating a class template,
    /// function template, or a member thereof.
    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          Decl *Entity,
                          SourceRange InstantiationRange = SourceRange());

    /// \brief Note that we are instantiating a default argument in a
    /// template-id.
    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          TemplateDecl *Template,
                          const TemplateArgument *TemplateArgs,
                          unsigned NumTemplateArgs,
                          SourceRange InstantiationRange = SourceRange());

    /// \brief Note that we are instantiating a default argument in a
    /// template-id.
    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          FunctionTemplateDecl *FunctionTemplate,
                          const TemplateArgument *TemplateArgs,
                          unsigned NumTemplateArgs,
                          ActiveTemplateInstantiation::InstantiationKind Kind,
                          sema::TemplateDeductionInfo &DeductionInfo,
                          SourceRange InstantiationRange = SourceRange());

    /// \brief Note that we are instantiating as part of template
    /// argument deduction for a class template partial
    /// specialization.
    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          ClassTemplatePartialSpecializationDecl *PartialSpec,
                          const TemplateArgument *TemplateArgs,
                          unsigned NumTemplateArgs,
                          sema::TemplateDeductionInfo &DeductionInfo,
                          SourceRange InstantiationRange = SourceRange());

    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          ParmVarDecl *Param,
                          const TemplateArgument *TemplateArgs,
                          unsigned NumTemplateArgs,
                          SourceRange InstantiationRange = SourceRange());

    /// \brief Note that we are substituting prior template arguments into a
    /// non-type or template template parameter.
    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          NamedDecl *Template,
                          NonTypeTemplateParmDecl *Param,
                          const TemplateArgument *TemplateArgs,
                          unsigned NumTemplateArgs,
                          SourceRange InstantiationRange);

    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          NamedDecl *Template,
                          TemplateTemplateParmDecl *Param,
                          const TemplateArgument *TemplateArgs,
                          unsigned NumTemplateArgs,
                          SourceRange InstantiationRange);

    /// \brief Note that we are checking the default template argument
    /// against the template parameter for a given template-id.
    InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                          TemplateDecl *Template,
                          NamedDecl *Param,
                          const TemplateArgument *TemplateArgs,
                          unsigned NumTemplateArgs,
                          SourceRange InstantiationRange);


    /// \brief Note that we have finished instantiating this template.
    void Clear();

    ~InstantiatingTemplate() { Clear(); }

    /// \brief Determines whether we have exceeded the maximum
    /// recursive template instantiations.
    operator bool() const { return Invalid; }

  private:
    Sema &SemaRef;
    bool Invalid;
    bool SavedInNonInstantiationSFINAEContext;
    bool CheckInstantiationDepth(SourceLocation PointOfInstantiation,
                                 SourceRange InstantiationRange);

    InstantiatingTemplate(const InstantiatingTemplate&); // not implemented

    InstantiatingTemplate&
    operator=(const InstantiatingTemplate&); // not implemented
  };

  void PrintInstantiationStack();

  /// \brief Determines whether we are currently in a context where
  /// template argument substitution failures are not considered
  /// errors.
  ///
  /// \returns An empty \c llvm::Optional if we're not in a SFINAE context.
  /// Otherwise, contains a pointer that, if non-NULL, contains the nearest 
  /// template-deduction context object, which can be used to capture 
  /// diagnostics that will be suppressed. 
  llvm::Optional<sema::TemplateDeductionInfo *> isSFINAEContext() const;

  /// \brief RAII class used to determine whether SFINAE has
  /// trapped any errors that occur during template argument
  /// deduction.`
  class SFINAETrap {
    Sema &SemaRef;
    unsigned PrevSFINAEErrors;
    bool PrevInNonInstantiationSFINAEContext;
    bool PrevAccessCheckingSFINAE;
    
  public:
    explicit SFINAETrap(Sema &SemaRef, bool AccessCheckingSFINAE = false)
      : SemaRef(SemaRef), PrevSFINAEErrors(SemaRef.NumSFINAEErrors),
        PrevInNonInstantiationSFINAEContext(
                                      SemaRef.InNonInstantiationSFINAEContext),
        PrevAccessCheckingSFINAE(SemaRef.AccessCheckingSFINAE)
    { 
      if (!SemaRef.isSFINAEContext())
        SemaRef.InNonInstantiationSFINAEContext = true;
      SemaRef.AccessCheckingSFINAE = AccessCheckingSFINAE;
    }

    ~SFINAETrap() { 
      SemaRef.NumSFINAEErrors = PrevSFINAEErrors; 
      SemaRef.InNonInstantiationSFINAEContext 
        = PrevInNonInstantiationSFINAEContext;
      SemaRef.AccessCheckingSFINAE = PrevAccessCheckingSFINAE;
    }

    /// \brief Determine whether any SFINAE errors have been trapped.
    bool hasErrorOccurred() const {
      return SemaRef.NumSFINAEErrors > PrevSFINAEErrors;
    }
  };

  /// \brief The current instantiation scope used to store local
  /// variables.
  LocalInstantiationScope *CurrentInstantiationScope;

  /// \brief The number of typos corrected by CorrectTypo.
  unsigned TyposCorrected;

  typedef llvm::DenseMap<IdentifierInfo *, std::pair<llvm::StringRef, bool> >
    UnqualifiedTyposCorrectedMap;
  
  /// \brief A cache containing the results of typo correction for unqualified
  /// name lookup.
  ///
  /// The string is the string that we corrected to (which may be empty, if
  /// there was no correction), while the boolean will be true when the
  /// string represents a keyword.
  UnqualifiedTyposCorrectedMap UnqualifiedTyposCorrected;
  
  /// \brief Worker object for performing CFG-based warnings.
  sema::AnalysisBasedWarnings AnalysisWarnings;

  /// \brief An entity for which implicit template instantiation is required.
  ///
  /// The source location associated with the declaration is the first place in
  /// the source code where the declaration was "used". It is not necessarily
  /// the point of instantiation (which will be either before or after the
  /// namespace-scope declaration that triggered this implicit instantiation),
  /// However, it is the location that diagnostics should generally refer to,
  /// because users will need to know what code triggered the instantiation.
  typedef std::pair<ValueDecl *, SourceLocation> PendingImplicitInstantiation;

  /// \brief The queue of implicit template instantiations that are required
  /// but have not yet been performed.
  std::deque<PendingImplicitInstantiation> PendingInstantiations;

  /// \brief The queue of implicit template instantiations that are required
  /// and must be performed within the current local scope.
  ///
  /// This queue is only used for member functions of local classes in
  /// templates, which must be instantiated in the same scope as their
  /// enclosing function, so that they can reference function-local
  /// types, static variables, enumerators, etc.
  std::deque<PendingImplicitInstantiation> PendingLocalImplicitInstantiations;

  void PerformPendingInstantiations(bool LocalOnly = false);

  TypeSourceInfo *SubstType(TypeSourceInfo *T,
                            const MultiLevelTemplateArgumentList &TemplateArgs,
                            SourceLocation Loc, DeclarationName Entity);

  QualType SubstType(QualType T,
                     const MultiLevelTemplateArgumentList &TemplateArgs,
                     SourceLocation Loc, DeclarationName Entity);

  TypeSourceInfo *SubstType(TypeLoc TL,
                            const MultiLevelTemplateArgumentList &TemplateArgs,
                            SourceLocation Loc, DeclarationName Entity);

  TypeSourceInfo *SubstFunctionDeclType(TypeSourceInfo *T,
                            const MultiLevelTemplateArgumentList &TemplateArgs,
                                        SourceLocation Loc,
                                        DeclarationName Entity);
  ParmVarDecl *SubstParmVarDecl(ParmVarDecl *D,
                            const MultiLevelTemplateArgumentList &TemplateArgs,
                                llvm::Optional<unsigned> NumExpansions);
  bool SubstParmTypes(SourceLocation Loc, 
                      ParmVarDecl **Params, unsigned NumParams,
                      const MultiLevelTemplateArgumentList &TemplateArgs,
                      llvm::SmallVectorImpl<QualType> &ParamTypes,
                      llvm::SmallVectorImpl<ParmVarDecl *> *OutParams = 0);
  ExprResult SubstExpr(Expr *E,
                       const MultiLevelTemplateArgumentList &TemplateArgs);
  
  /// \brief Substitute the given template arguments into a list of
  /// expressions, expanding pack expansions if required.
  ///
  /// \param Exprs The list of expressions to substitute into.
  ///
  /// \param NumExprs The number of expressions in \p Exprs.
  ///
  /// \param IsCall Whether this is some form of call, in which case
  /// default arguments will be dropped.
  ///
  /// \param TemplateArgs The set of template arguments to substitute.
  ///
  /// \param Outputs Will receive all of the substituted arguments.
  ///
  /// \returns true if an error occurred, false otherwise.
  bool SubstExprs(Expr **Exprs, unsigned NumExprs, bool IsCall,
                  const MultiLevelTemplateArgumentList &TemplateArgs,
                  llvm::SmallVectorImpl<Expr *> &Outputs);

  StmtResult SubstStmt(Stmt *S,
                       const MultiLevelTemplateArgumentList &TemplateArgs);

  Decl *SubstDecl(Decl *D, DeclContext *Owner,
                  const MultiLevelTemplateArgumentList &TemplateArgs);

  bool
  SubstBaseSpecifiers(CXXRecordDecl *Instantiation,
                      CXXRecordDecl *Pattern,
                      const MultiLevelTemplateArgumentList &TemplateArgs);

  bool
  InstantiateClass(SourceLocation PointOfInstantiation,
                   CXXRecordDecl *Instantiation, CXXRecordDecl *Pattern,
                   const MultiLevelTemplateArgumentList &TemplateArgs,
                   TemplateSpecializationKind TSK,
                   bool Complain = true);

  void InstantiateAttrs(const MultiLevelTemplateArgumentList &TemplateArgs,
                        Decl *Pattern, Decl *Inst);

  bool
  InstantiateClassTemplateSpecialization(SourceLocation PointOfInstantiation,
                           ClassTemplateSpecializationDecl *ClassTemplateSpec,
                           TemplateSpecializationKind TSK,
                           bool Complain = true);

  void InstantiateClassMembers(SourceLocation PointOfInstantiation,
                               CXXRecordDecl *Instantiation,
                            const MultiLevelTemplateArgumentList &TemplateArgs,
                               TemplateSpecializationKind TSK);

  void InstantiateClassTemplateSpecializationMembers(
                                          SourceLocation PointOfInstantiation,
                           ClassTemplateSpecializationDecl *ClassTemplateSpec,
                                                TemplateSpecializationKind TSK);

  NestedNameSpecifierLoc
  SubstNestedNameSpecifierLoc(NestedNameSpecifierLoc NNS,
                           const MultiLevelTemplateArgumentList &TemplateArgs);

  DeclarationNameInfo
  SubstDeclarationNameInfo(const DeclarationNameInfo &NameInfo,
                           const MultiLevelTemplateArgumentList &TemplateArgs);
  TemplateName
  SubstTemplateName(NestedNameSpecifierLoc QualifierLoc, TemplateName Name, 
                    SourceLocation Loc,
                    const MultiLevelTemplateArgumentList &TemplateArgs);
  bool Subst(const TemplateArgumentLoc *Args, unsigned NumArgs,
             TemplateArgumentListInfo &Result,
             const MultiLevelTemplateArgumentList &TemplateArgs);

  void InstantiateFunctionDefinition(SourceLocation PointOfInstantiation,
                                     FunctionDecl *Function,
                                     bool Recursive = false,
                                     bool DefinitionRequired = false);
  void InstantiateStaticDataMemberDefinition(
                                     SourceLocation PointOfInstantiation,
                                     VarDecl *Var,
                                     bool Recursive = false,
                                     bool DefinitionRequired = false);

  void InstantiateMemInitializers(CXXConstructorDecl *New,
                                  const CXXConstructorDecl *Tmpl,
                            const MultiLevelTemplateArgumentList &TemplateArgs);

  NamedDecl *FindInstantiatedDecl(SourceLocation Loc, NamedDecl *D,
                          const MultiLevelTemplateArgumentList &TemplateArgs);
  DeclContext *FindInstantiatedContext(SourceLocation Loc, DeclContext *DC,
                          const MultiLevelTemplateArgumentList &TemplateArgs);

  // Objective-C declarations.
  Decl *ActOnStartClassInterface(SourceLocation AtInterfaceLoc,
                                 IdentifierInfo *ClassName,
                                 SourceLocation ClassLoc,
                                 IdentifierInfo *SuperName,
                                 SourceLocation SuperLoc,
                                 Decl * const *ProtoRefs,
                                 unsigned NumProtoRefs,
                                 const SourceLocation *ProtoLocs,
                                 SourceLocation EndProtoLoc,
                                 AttributeList *AttrList);

  Decl *ActOnCompatiblityAlias(
                    SourceLocation AtCompatibilityAliasLoc,
                    IdentifierInfo *AliasName,  SourceLocation AliasLocation,
                    IdentifierInfo *ClassName, SourceLocation ClassLocation);

  void CheckForwardProtocolDeclarationForCircularDependency(
    IdentifierInfo *PName,
    SourceLocation &PLoc, SourceLocation PrevLoc,
    const ObjCList<ObjCProtocolDecl> &PList);

  Decl *ActOnStartProtocolInterface(
                    SourceLocation AtProtoInterfaceLoc,
                    IdentifierInfo *ProtocolName, SourceLocation ProtocolLoc,
                    Decl * const *ProtoRefNames, unsigned NumProtoRefs,
                    const SourceLocation *ProtoLocs,
                    SourceLocation EndProtoLoc,
                    AttributeList *AttrList);

  Decl *ActOnStartCategoryInterface(SourceLocation AtInterfaceLoc,
                                    IdentifierInfo *ClassName,
                                    SourceLocation ClassLoc,
                                    IdentifierInfo *CategoryName,
                                    SourceLocation CategoryLoc,
                                    Decl * const *ProtoRefs,
                                    unsigned NumProtoRefs,
                                    const SourceLocation *ProtoLocs,
                                    SourceLocation EndProtoLoc);

  Decl *ActOnStartClassImplementation(
                    SourceLocation AtClassImplLoc,
                    IdentifierInfo *ClassName, SourceLocation ClassLoc,
                    IdentifierInfo *SuperClassname,
                    SourceLocation SuperClassLoc);

  Decl *ActOnStartCategoryImplementation(SourceLocation AtCatImplLoc,
                                         IdentifierInfo *ClassName,
                                         SourceLocation ClassLoc,
                                         IdentifierInfo *CatName,
                                         SourceLocation CatLoc);

  Decl *ActOnForwardClassDeclaration(SourceLocation Loc,
                                     IdentifierInfo **IdentList,
                                     SourceLocation *IdentLocs,
                                     unsigned NumElts);

  Decl *ActOnForwardProtocolDeclaration(SourceLocation AtProtoclLoc,
                                        const IdentifierLocPair *IdentList,
                                        unsigned NumElts,
                                        AttributeList *attrList);

  void FindProtocolDeclaration(bool WarnOnDeclarations,
                               const IdentifierLocPair *ProtocolId,
                               unsigned NumProtocols,
                               llvm::SmallVectorImpl<Decl *> &Protocols);

  /// Ensure attributes are consistent with type.
  /// \param [in, out] Attributes The attributes to check; they will
  /// be modified to be consistent with \arg PropertyTy.
  void CheckObjCPropertyAttributes(Decl *PropertyPtrTy,
                                   SourceLocation Loc,
                                   unsigned &Attributes);

  /// Process the specified property declaration and create decls for the
  /// setters and getters as needed.
  /// \param property The property declaration being processed
  /// \param DC The semantic container for the property
  /// \param redeclaredProperty Declaration for property if redeclared 
  ///        in class extension.
  /// \param lexicalDC Container for redeclaredProperty.
  void ProcessPropertyDecl(ObjCPropertyDecl *property,
                           ObjCContainerDecl *DC,
                           ObjCPropertyDecl *redeclaredProperty = 0,
                           ObjCContainerDecl *lexicalDC = 0);

  void DiagnosePropertyMismatch(ObjCPropertyDecl *Property,
                                ObjCPropertyDecl *SuperProperty,
                                const IdentifierInfo *Name);
  void ComparePropertiesInBaseAndSuper(ObjCInterfaceDecl *IDecl);

  void CompareMethodParamsInBaseAndSuper(Decl *IDecl,
                                         ObjCMethodDecl *MethodDecl,
                                         bool IsInstance);

  void CompareProperties(Decl *CDecl, Decl *MergeProtocols);

  void DiagnoseClassExtensionDupMethods(ObjCCategoryDecl *CAT,
                                        ObjCInterfaceDecl *ID);

  void MatchOneProtocolPropertiesInClass(Decl *CDecl,
                                         ObjCProtocolDecl *PDecl);

  void ActOnAtEnd(Scope *S, SourceRange AtEnd, Decl *classDecl,
                  Decl **allMethods = 0, unsigned allNum = 0,
                  Decl **allProperties = 0, unsigned pNum = 0,
                  DeclGroupPtrTy *allTUVars = 0, unsigned tuvNum = 0);

  Decl *ActOnProperty(Scope *S, SourceLocation AtLoc,
                      FieldDeclarator &FD, ObjCDeclSpec &ODS,
                      Selector GetterSel, Selector SetterSel,
                      Decl *ClassCategory,
                      bool *OverridingProperty,
                      tok::ObjCKeywordKind MethodImplKind,
                      DeclContext *lexicalDC = 0);

  Decl *ActOnPropertyImplDecl(Scope *S,
                              SourceLocation AtLoc,
                              SourceLocation PropertyLoc,
                              bool ImplKind,Decl *ClassImplDecl,
                              IdentifierInfo *PropertyId,
                              IdentifierInfo *PropertyIvar,
                              SourceLocation PropertyIvarLoc);

  struct ObjCArgInfo {
    IdentifierInfo *Name;
    SourceLocation NameLoc;
    // The Type is null if no type was specified, and the DeclSpec is invalid
    // in this case.
    ParsedType Type;
    ObjCDeclSpec DeclSpec;

    /// ArgAttrs - Attribute list for this argument.
    AttributeList *ArgAttrs;
  };

  Decl *ActOnMethodDeclaration(
    Scope *S,
    SourceLocation BeginLoc, // location of the + or -.
    SourceLocation EndLoc,   // location of the ; or {.
    tok::TokenKind MethodType,
    Decl *ClassDecl, ObjCDeclSpec &ReturnQT, ParsedType ReturnType,
    Selector Sel,
    // optional arguments. The number of types/arguments is obtained
    // from the Sel.getNumArgs().
    ObjCArgInfo *ArgInfo,
    DeclaratorChunk::ParamInfo *CParamInfo, unsigned CNumArgs, // c-style args
    AttributeList *AttrList, tok::ObjCKeywordKind MethodImplKind,
    bool isVariadic = false);

  // Helper method for ActOnClassMethod/ActOnInstanceMethod.
  // Will search "local" class/category implementations for a method decl.
  // Will also search in class's root looking for instance method.
  // Returns 0 if no method is found.
  ObjCMethodDecl *LookupPrivateClassMethod(Selector Sel,
                                           ObjCInterfaceDecl *CDecl);
  ObjCMethodDecl *LookupPrivateInstanceMethod(Selector Sel,
                                              ObjCInterfaceDecl *ClassDecl);

  ExprResult
  HandleExprPropertyRefExpr(const ObjCObjectPointerType *OPT,
                            Expr *BaseExpr,
                            DeclarationName MemberName,
                            SourceLocation MemberLoc,
                            SourceLocation SuperLoc, QualType SuperType,
                            bool Super);

  ExprResult
  ActOnClassPropertyRefExpr(IdentifierInfo &receiverName,
                            IdentifierInfo &propertyName,
                            SourceLocation receiverNameLoc,
                            SourceLocation propertyNameLoc);

  ObjCMethodDecl *tryCaptureObjCSelf();

  /// \brief Describes the kind of message expression indicated by a message
  /// send that starts with an identifier.
  enum ObjCMessageKind {
    /// \brief The message is sent to 'super'.
    ObjCSuperMessage,
    /// \brief The message is an instance message.
    ObjCInstanceMessage,
    /// \brief The message is a class message, and the identifier is a type
    /// name.
    ObjCClassMessage
  };
  
  ObjCMessageKind getObjCMessageKind(Scope *S,
                                     IdentifierInfo *Name,
                                     SourceLocation NameLoc,
                                     bool IsSuper,
                                     bool HasTrailingDot,
                                     ParsedType &ReceiverType);

  ExprResult ActOnSuperMessage(Scope *S, SourceLocation SuperLoc,
                               Selector Sel,
                               SourceLocation LBracLoc,
                               SourceLocation SelectorLoc,
                               SourceLocation RBracLoc,
                               MultiExprArg Args);

  ExprResult BuildClassMessage(TypeSourceInfo *ReceiverTypeInfo,
                               QualType ReceiverType,
                               SourceLocation SuperLoc,
                               Selector Sel,
                               ObjCMethodDecl *Method,
                               SourceLocation LBracLoc,
                               SourceLocation SelectorLoc,
                               SourceLocation RBracLoc,
                               MultiExprArg Args);

  ExprResult ActOnClassMessage(Scope *S,
                               ParsedType Receiver,
                               Selector Sel,
                               SourceLocation LBracLoc,
                               SourceLocation SelectorLoc,
                               SourceLocation RBracLoc,
                               MultiExprArg Args);

  ExprResult BuildInstanceMessage(Expr *Receiver,
                                  QualType ReceiverType,
                                  SourceLocation SuperLoc,
                                  Selector Sel,
                                  ObjCMethodDecl *Method,
                                  SourceLocation LBracLoc,
                                  SourceLocation SelectorLoc,
                                  SourceLocation RBracLoc,
                                  MultiExprArg Args);

  ExprResult ActOnInstanceMessage(Scope *S,
                                  Expr *Receiver,
                                  Selector Sel,
                                  SourceLocation LBracLoc,
                                  SourceLocation SelectorLoc,
                                  SourceLocation RBracLoc,
                                  MultiExprArg Args);


  enum PragmaOptionsAlignKind {
    POAK_Native,  // #pragma options align=native
    POAK_Natural, // #pragma options align=natural
    POAK_Packed,  // #pragma options align=packed
    POAK_Power,   // #pragma options align=power
    POAK_Mac68k,  // #pragma options align=mac68k
    POAK_Reset    // #pragma options align=reset
  };

  /// ActOnPragmaOptionsAlign - Called on well formed #pragma options align.
  void ActOnPragmaOptionsAlign(PragmaOptionsAlignKind Kind,
                               SourceLocation PragmaLoc,
                               SourceLocation KindLoc);

  enum PragmaPackKind {
    PPK_Default, // #pragma pack([n])
    PPK_Show,    // #pragma pack(show), only supported by MSVC.
    PPK_Push,    // #pragma pack(push, [identifier], [n])
    PPK_Pop      // #pragma pack(pop, [identifier], [n])
  };

  /// ActOnPragmaPack - Called on well formed #pragma pack(...).
  void ActOnPragmaPack(PragmaPackKind Kind,
                       IdentifierInfo *Name,
                       Expr *Alignment,
                       SourceLocation PragmaLoc,
                       SourceLocation LParenLoc,
                       SourceLocation RParenLoc);

  /// ActOnPragmaUnused - Called on well-formed '#pragma unused'.
  void ActOnPragmaUnused(const Token &Identifier,
                         Scope *curScope,
                         SourceLocation PragmaLoc);

  /// ActOnPragmaVisibility - Called on well formed #pragma GCC visibility... .
  void ActOnPragmaVisibility(bool IsPush, const IdentifierInfo* VisType,
                             SourceLocation PragmaLoc);

  NamedDecl *DeclClonePragmaWeak(NamedDecl *ND, IdentifierInfo *II);
  void DeclApplyPragmaWeak(Scope *S, NamedDecl *ND, WeakInfo &W);

  /// ActOnPragmaWeakID - Called on well formed #pragma weak ident.
  void ActOnPragmaWeakID(IdentifierInfo* WeakName,
                         SourceLocation PragmaLoc,
                         SourceLocation WeakNameLoc);

  /// ActOnPragmaWeakAlias - Called on well formed #pragma weak ident = ident.
  void ActOnPragmaWeakAlias(IdentifierInfo* WeakName,
                            IdentifierInfo* AliasName,
                            SourceLocation PragmaLoc,
                            SourceLocation WeakNameLoc,
                            SourceLocation AliasNameLoc);

  /// ActOnPragmaFPContract - Called on well formed
  /// #pragma {STDC,OPENCL} FP_CONTRACT
  void ActOnPragmaFPContract(tok::OnOffSwitch OOS);

  /// AddAlignmentAttributesForRecord - Adds any needed alignment attributes to
  /// a the record decl, to handle '#pragma pack' and '#pragma options align'.
  void AddAlignmentAttributesForRecord(RecordDecl *RD);

  /// FreePackedContext - Deallocate and null out PackContext.
  void FreePackedContext();

  /// PushNamespaceVisibilityAttr - Note that we've entered a
  /// namespace with a visibility attribute.
  void PushNamespaceVisibilityAttr(const VisibilityAttr *Attr);

  /// AddPushedVisibilityAttribute - If '#pragma GCC visibility' was used,
  /// add an appropriate visibility attribute.
  void AddPushedVisibilityAttribute(Decl *RD);

  /// PopPragmaVisibility - Pop the top element of the visibility stack; used
  /// for '#pragma GCC visibility' and visibility attributes on namespaces.
  void PopPragmaVisibility();

  /// FreeVisContext - Deallocate and null out VisContext.
  void FreeVisContext();

  /// AddAlignedAttr - Adds an aligned attribute to a particular declaration.
  void AddAlignedAttr(SourceLocation AttrLoc, Decl *D, Expr *E);
  void AddAlignedAttr(SourceLocation AttrLoc, Decl *D, TypeSourceInfo *T);

  /// CastCategory - Get the correct forwarded implicit cast result category
  /// from the inner expression.
  ExprValueKind CastCategory(Expr *E);

  /// ImpCastExprToType - If Expr is not of type 'Type', insert an implicit
  /// cast.  If there is already an implicit cast, merge into the existing one.
  /// If isLvalue, the result of the cast is an lvalue.
  void ImpCastExprToType(Expr *&Expr, QualType Type, CastKind CK,
                         ExprValueKind VK = VK_RValue,
                         const CXXCastPath *BasePath = 0);

  /// IgnoredValueConversions - Given that an expression's result is
  /// syntactically ignored, perform any conversions that are
  /// required.
  void IgnoredValueConversions(Expr *&expr);

  // UsualUnaryConversions - promotes integers (C99 6.3.1.1p2) and converts
  // functions and arrays to their respective pointers (C99 6.3.2.1).
  Expr *UsualUnaryConversions(Expr *&expr);

  // DefaultFunctionArrayConversion - converts functions and arrays
  // to their respective pointers (C99 6.3.2.1).
  void DefaultFunctionArrayConversion(Expr *&expr);

  // DefaultFunctionArrayLvalueConversion - converts functions and
  // arrays to their respective pointers and performs the
  // lvalue-to-rvalue conversion.
  void DefaultFunctionArrayLvalueConversion(Expr *&expr);

  // DefaultLvalueConversion - performs lvalue-to-rvalue conversion on
  // the operand.  This is DefaultFunctionArrayLvalueConversion,
  // except that it assumes the operand isn't of function or array
  // type.
  void DefaultLvalueConversion(Expr *&expr);

  // DefaultArgumentPromotion (C99 6.5.2.2p6). Used for function calls that
  // do not have a prototype. Integer promotions are performed on each
  // argument, and arguments that have type float are promoted to double.
  void DefaultArgumentPromotion(Expr *&Expr);

  // Used for emitting the right warning by DefaultVariadicArgumentPromotion
  enum VariadicCallType {
    VariadicFunction,
    VariadicBlock,
    VariadicMethod,
    VariadicConstructor,
    VariadicDoesNotApply
  };

  /// GatherArgumentsForCall - Collector argument expressions for various
  /// form of call prototypes.
  bool GatherArgumentsForCall(SourceLocation CallLoc,
                              FunctionDecl *FDecl,
                              const FunctionProtoType *Proto,
                              unsigned FirstProtoArg,
                              Expr **Args, unsigned NumArgs,
                              llvm::SmallVector<Expr *, 8> &AllArgs,
                              VariadicCallType CallType = VariadicDoesNotApply);

  // DefaultVariadicArgumentPromotion - Like DefaultArgumentPromotion, but
  // will warn if the resulting type is not a POD type.
  bool DefaultVariadicArgumentPromotion(Expr *&Expr, VariadicCallType CT,
                                        FunctionDecl *FDecl);

  // UsualArithmeticConversions - performs the UsualUnaryConversions on it's
  // operands and then handles various conversions that are common to binary
  // operators (C99 6.3.1.8). If both operands aren't arithmetic, this
  // routine returns the first non-arithmetic type found. The client is
  // responsible for emitting appropriate error diagnostics.
  QualType UsualArithmeticConversions(Expr *&lExpr, Expr *&rExpr,
                                      bool isCompAssign = false);

  /// AssignConvertType - All of the 'assignment' semantic checks return this
  /// enum to indicate whether the assignment was allowed.  These checks are
  /// done for simple assignments, as well as initialization, return from
  /// function, argument passing, etc.  The query is phrased in terms of a
  /// source and destination type.
  enum AssignConvertType {
    /// Compatible - the types are compatible according to the standard.
    Compatible,

    /// PointerToInt - The assignment converts a pointer to an int, which we
    /// accept as an extension.
    PointerToInt,

    /// IntToPointer - The assignment converts an int to a pointer, which we
    /// accept as an extension.
    IntToPointer,

    /// FunctionVoidPointer - The assignment is between a function pointer and
    /// void*, which the standard doesn't allow, but we accept as an extension.
    FunctionVoidPointer,

    /// IncompatiblePointer - The assignment is between two pointers types that
    /// are not compatible, but we accept them as an extension.
    IncompatiblePointer,

    /// IncompatiblePointer - The assignment is between two pointers types which
    /// point to integers which have a different sign, but are otherwise identical.
    /// This is a subset of the above, but broken out because it's by far the most
    /// common case of incompatible pointers.
    IncompatiblePointerSign,

    /// CompatiblePointerDiscardsQualifiers - The assignment discards
    /// c/v/r qualifiers, which we accept as an extension.
    CompatiblePointerDiscardsQualifiers,

    /// IncompatiblePointerDiscardsQualifiers - The assignment
    /// discards qualifiers that we don't permit to be discarded,
    /// like address spaces.
    IncompatiblePointerDiscardsQualifiers,

    /// IncompatibleNestedPointerQualifiers - The assignment is between two
    /// nested pointer types, and the qualifiers other than the first two
    /// levels differ e.g. char ** -> const char **, but we accept them as an
    /// extension.
    IncompatibleNestedPointerQualifiers,

    /// IncompatibleVectors - The assignment is between two vector types that
    /// have the same size, which we accept as an extension.
    IncompatibleVectors,

    /// IntToBlockPointer - The assignment converts an int to a block
    /// pointer. We disallow this.
    IntToBlockPointer,

    /// IncompatibleBlockPointer - The assignment is between two block
    /// pointers types that are not compatible.
    IncompatibleBlockPointer,

    /// IncompatibleObjCQualifiedId - The assignment is between a qualified
    /// id type and something else (that is incompatible with it). For example,
    /// "id <XXX>" = "Foo *", where "Foo *" doesn't implement the XXX protocol.
    IncompatibleObjCQualifiedId,

    /// Incompatible - We reject this conversion outright, it is invalid to
    /// represent it in the AST.
    Incompatible
  };

  /// DiagnoseAssignmentResult - Emit a diagnostic, if required, for the
  /// assignment conversion type specified by ConvTy.  This returns true if the
  /// conversion was invalid or false if the conversion was accepted.
  bool DiagnoseAssignmentResult(AssignConvertType ConvTy,
                                SourceLocation Loc,
                                QualType DstType, QualType SrcType,
                                Expr *SrcExpr, AssignmentAction Action,
                                bool *Complained = 0);

  /// CheckAssignmentConstraints - Perform type checking for assignment,
  /// argument passing, variable initialization, and function return values.
  /// C99 6.5.16.
  AssignConvertType CheckAssignmentConstraints(SourceLocation Loc,
                                               QualType lhs, QualType rhs);

  /// Check assignment constraints and prepare for a conversion of the
  /// RHS to the LHS type.
  AssignConvertType CheckAssignmentConstraints(QualType lhs, Expr *&rhs,
                                               CastKind &Kind);

  // CheckSingleAssignmentConstraints - Currently used by
  // CheckAssignmentOperands, and ActOnReturnStmt. Prior to type checking,
  // this routine performs the default function/array converions.
  AssignConvertType CheckSingleAssignmentConstraints(QualType lhs,
                                                     Expr *&rExpr);

  // \brief If the lhs type is a transparent union, check whether we
  // can initialize the transparent union with the given expression.
  AssignConvertType CheckTransparentUnionArgumentConstraints(QualType lhs,
                                                             Expr *&rExpr);

  bool IsStringLiteralToNonConstPointerConversion(Expr *From, QualType ToType);

  bool CheckExceptionSpecCompatibility(Expr *From, QualType ToType);

  bool PerformImplicitConversion(Expr *&From, QualType ToType,
                                 AssignmentAction Action,
                                 bool AllowExplicit = false);
  bool PerformImplicitConversion(Expr *&From, QualType ToType,
                                 AssignmentAction Action,
                                 bool AllowExplicit,
                                 ImplicitConversionSequence& ICS);
  bool PerformImplicitConversion(Expr *&From, QualType ToType,
                                 const ImplicitConversionSequence& ICS,
                                 AssignmentAction Action,
                                 bool CStyle = false);
  bool PerformImplicitConversion(Expr *&From, QualType ToType,
                                 const StandardConversionSequence& SCS,
                                 AssignmentAction Action,
                                 bool CStyle);

  /// the following "Check" methods will return a valid/converted QualType
  /// or a null QualType (indicating an error diagnostic was issued).

  /// type checking binary operators (subroutines of CreateBuiltinBinOp).
  QualType InvalidOperands(SourceLocation l, Expr *&lex, Expr *&rex);
  QualType CheckPointerToMemberOperands( // C++ 5.5
    Expr *&lex, Expr *&rex, ExprValueKind &VK,
    SourceLocation OpLoc, bool isIndirect);
  QualType CheckMultiplyDivideOperands( // C99 6.5.5
    Expr *&lex, Expr *&rex, SourceLocation OpLoc, bool isCompAssign,
                                       bool isDivide);
  QualType CheckRemainderOperands( // C99 6.5.5
    Expr *&lex, Expr *&rex, SourceLocation OpLoc, bool isCompAssign = false);
  QualType CheckAdditionOperands( // C99 6.5.6
    Expr *&lex, Expr *&rex, SourceLocation OpLoc, QualType* CompLHSTy = 0);
  QualType CheckSubtractionOperands( // C99 6.5.6
    Expr *&lex, Expr *&rex, SourceLocation OpLoc, QualType* CompLHSTy = 0);
  QualType CheckShiftOperands( // C99 6.5.7
    Expr *&lex, Expr *&rex, SourceLocation OpLoc, unsigned Opc,
    bool isCompAssign = false);
  QualType CheckCompareOperands( // C99 6.5.8/9
    Expr *&lex, Expr *&rex, SourceLocation OpLoc, unsigned Opc,
                                bool isRelational);
  QualType CheckBitwiseOperands( // C99 6.5.[10...12]
    Expr *&lex, Expr *&rex, SourceLocation OpLoc, bool isCompAssign = false);
  QualType CheckLogicalOperands( // C99 6.5.[13,14]
    Expr *&lex, Expr *&rex, SourceLocation OpLoc, unsigned Opc);
  // CheckAssignmentOperands is used for both simple and compound assignment.
  // For simple assignment, pass both expressions and a null converted type.
  // For compound assignment, pass both expressions and the converted type.
  QualType CheckAssignmentOperands( // C99 6.5.16.[1,2]
    Expr *lex, Expr *&rex, SourceLocation OpLoc, QualType convertedType);
  
  void ConvertPropertyForRValue(Expr *&E);
  void ConvertPropertyForLValue(Expr *&LHS, Expr *&RHS, QualType& LHSTy);
                                   
  QualType CheckConditionalOperands( // C99 6.5.15
    Expr *&cond, Expr *&lhs, Expr *&rhs,
    ExprValueKind &VK, ExprObjectKind &OK, SourceLocation questionLoc);
  QualType CXXCheckConditionalOperands( // C++ 5.16
    Expr *&cond, Expr *&lhs, Expr *&rhs,
    ExprValueKind &VK, ExprObjectKind &OK, SourceLocation questionLoc);
  QualType FindCompositePointerType(SourceLocation Loc, Expr *&E1, Expr *&E2,
                                    bool *NonStandardCompositeType = 0);

  QualType FindCompositeObjCPointerType(Expr *&LHS, Expr *&RHS,
                                        SourceLocation questionLoc);

  bool DiagnoseConditionalForNull(Expr *LHS, Expr *RHS,
                                  SourceLocation QuestionLoc);

  /// type checking for vector binary operators.
  QualType CheckVectorOperands(SourceLocation l, Expr *&lex, Expr *&rex);
  QualType CheckVectorCompareOperands(Expr *&lex, Expr *&rx,
                                      SourceLocation l, bool isRel);

  /// type checking declaration initializers (C99 6.7.8)
  bool CheckInitList(const InitializedEntity &Entity,
                     InitListExpr *&InitList, QualType &DeclType);
  bool CheckForConstantInitializer(Expr *e, QualType t);

  // type checking C++ declaration initializers (C++ [dcl.init]).

  /// ReferenceCompareResult - Expresses the result of comparing two
  /// types (cv1 T1 and cv2 T2) to determine their compatibility for the
  /// purposes of initialization by reference (C++ [dcl.init.ref]p4).
  enum ReferenceCompareResult {
    /// Ref_Incompatible - The two types are incompatible, so direct
    /// reference binding is not possible.
    Ref_Incompatible = 0,
    /// Ref_Related - The two types are reference-related, which means
    /// that their unqualified forms (T1 and T2) are either the same
    /// or T1 is a base class of T2.
    Ref_Related,
    /// Ref_Compatible_With_Added_Qualification - The two types are
    /// reference-compatible with added qualification, meaning that
    /// they are reference-compatible and the qualifiers on T1 (cv1)
    /// are greater than the qualifiers on T2 (cv2).
    Ref_Compatible_With_Added_Qualification,
    /// Ref_Compatible - The two types are reference-compatible and
    /// have equivalent qualifiers (cv1 == cv2).
    Ref_Compatible
  };

  ReferenceCompareResult CompareReferenceRelationship(SourceLocation Loc,
                                                      QualType T1, QualType T2,
                                                      bool &DerivedToBase,
                                                      bool &ObjCConversion);

  /// CheckCastTypes - Check type constraints for casting between types under
  /// C semantics, or forward to CXXCheckCStyleCast in C++.
  bool CheckCastTypes(SourceRange TyRange, QualType CastTy, Expr *&CastExpr,
                      CastKind &Kind, ExprValueKind &VK, CXXCastPath &BasePath,
                      bool FunctionalStyle = false);

  // CheckVectorCast - check type constraints for vectors.
  // Since vectors are an extension, there are no C standard reference for this.
  // We allow casting between vectors and integer datatypes of the same size.
  // returns true if the cast is invalid
  bool CheckVectorCast(SourceRange R, QualType VectorTy, QualType Ty,
                       CastKind &Kind);

  // CheckExtVectorCast - check type constraints for extended vectors.
  // Since vectors are an extension, there are no C standard reference for this.
  // We allow casting between vectors and integer datatypes of the same size,
  // or vectors and the element type of that vector.
  // returns true if the cast is invalid
  bool CheckExtVectorCast(SourceRange R, QualType VectorTy, Expr *&CastExpr,
                          CastKind &Kind);

  /// CXXCheckCStyleCast - Check constraints of a C-style or function-style
  /// cast under C++ semantics.
  bool CXXCheckCStyleCast(SourceRange R, QualType CastTy, ExprValueKind &VK,
                          Expr *&CastExpr, CastKind &Kind,
                          CXXCastPath &BasePath, bool FunctionalStyle);

  /// CheckMessageArgumentTypes - Check types in an Obj-C message send.
  /// \param Method - May be null.
  /// \param [out] ReturnType - The return type of the send.
  /// \return true iff there were any incompatible types.
  bool CheckMessageArgumentTypes(Expr **Args, unsigned NumArgs, Selector Sel,
                                 ObjCMethodDecl *Method, bool isClassMessage,
                                 SourceLocation lbrac, SourceLocation rbrac,
                                 QualType &ReturnType, ExprValueKind &VK);

  /// CheckBooleanCondition - Diagnose problems involving the use of
  /// the given expression as a boolean condition (e.g. in an if
  /// statement).  Also performs the standard function and array
  /// decays, possibly changing the input variable.
  ///
  /// \param Loc - A location associated with the condition, e.g. the
  /// 'if' keyword.
  /// \return true iff there were any errors
  bool CheckBooleanCondition(Expr *&CondExpr, SourceLocation Loc);

  ExprResult ActOnBooleanCondition(Scope *S, SourceLocation Loc,
                                           Expr *SubExpr);
  
  /// DiagnoseAssignmentAsCondition - Given that an expression is
  /// being used as a boolean condition, warn if it's an assignment.
  void DiagnoseAssignmentAsCondition(Expr *E);

  /// \brief Redundant parentheses over an equality comparison can indicate
  /// that the user intended an assignment used as condition.
  void DiagnoseEqualityWithExtraParens(ParenExpr *parenE);

  /// CheckCXXBooleanCondition - Returns true if conversion to bool is invalid.
  bool CheckCXXBooleanCondition(Expr *&CondExpr);

  /// ConvertIntegerToTypeWarnOnOverflow - Convert the specified APInt to have
  /// the specified width and sign.  If an overflow occurs, detect it and emit
  /// the specified diagnostic.
  void ConvertIntegerToTypeWarnOnOverflow(llvm::APSInt &OldVal,
                                          unsigned NewWidth, bool NewSign,
                                          SourceLocation Loc, unsigned DiagID);

  /// Checks that the Objective-C declaration is declared in the global scope.
  /// Emits an error and marks the declaration as invalid if it's not declared
  /// in the global scope.
  bool CheckObjCDeclScope(Decl *D);

  /// VerifyIntegerConstantExpression - verifies that an expression is an ICE,
  /// and reports the appropriate diagnostics. Returns false on success.
  /// Can optionally return the value of the expression.
  bool VerifyIntegerConstantExpression(const Expr *E, llvm::APSInt *Result = 0);

  /// VerifyBitField - verifies that a bit field expression is an ICE and has
  /// the correct width, and that the field type is valid.
  /// Returns false on success.
  /// Can optionally return whether the bit-field is of width 0
  bool VerifyBitField(SourceLocation FieldLoc, IdentifierInfo *FieldName,
                      QualType FieldTy, const Expr *BitWidth,
                      bool *ZeroWidth = 0);

  /// \name Code completion
  //@{
  /// \brief Describes the context in which code completion occurs.
  enum ParserCompletionContext {
    /// \brief Code completion occurs at top-level or namespace context.
    PCC_Namespace,
    /// \brief Code completion occurs within a class, struct, or union.
    PCC_Class,
    /// \brief Code completion occurs within an Objective-C interface, protocol,
    /// or category.
    PCC_ObjCInterface,
    /// \brief Code completion occurs within an Objective-C implementation or
    /// category implementation
    PCC_ObjCImplementation,
    /// \brief Code completion occurs within the list of instance variables
    /// in an Objective-C interface, protocol, category, or implementation.
    PCC_ObjCInstanceVariableList,
    /// \brief Code completion occurs following one or more template
    /// headers.
    PCC_Template,
    /// \brief Code completion occurs following one or more template
    /// headers within a class.
    PCC_MemberTemplate,
    /// \brief Code completion occurs within an expression.
    PCC_Expression,
    /// \brief Code completion occurs within a statement, which may
    /// also be an expression or a declaration.
    PCC_Statement,
    /// \brief Code completion occurs at the beginning of the
    /// initialization statement (or expression) in a for loop.
    PCC_ForInit,
    /// \brief Code completion occurs within the condition of an if,
    /// while, switch, or for statement.
    PCC_Condition,
    /// \brief Code completion occurs within the body of a function on a 
    /// recovery path, where we do not have a specific handle on our position
    /// in the grammar.
    PCC_RecoveryInFunction,
    /// \brief Code completion occurs where only a type is permitted.
    PCC_Type,
    /// \brief Code completion occurs in a parenthesized expression, which
    /// might also be a type cast.
    PCC_ParenthesizedExpression,
    /// \brief Code completion occurs within a sequence of declaration 
    /// specifiers within a function, method, or block.
    PCC_LocalDeclarationSpecifiers
  };

  void CodeCompleteOrdinaryName(Scope *S,
                                ParserCompletionContext CompletionContext);
  void CodeCompleteDeclSpec(Scope *S, DeclSpec &DS,
                            bool AllowNonIdentifiers,
                            bool AllowNestedNameSpecifiers);
  
  struct CodeCompleteExpressionData;
  void CodeCompleteExpression(Scope *S, 
                              const CodeCompleteExpressionData &Data);
  void CodeCompleteMemberReferenceExpr(Scope *S, Expr *Base,
                                       SourceLocation OpLoc,
                                       bool IsArrow);
  void CodeCompletePostfixExpression(Scope *S, ExprResult LHS);
  void CodeCompleteTag(Scope *S, unsigned TagSpec);
  void CodeCompleteTypeQualifiers(DeclSpec &DS);
  void CodeCompleteCase(Scope *S);
  void CodeCompleteCall(Scope *S, Expr *Fn, Expr **Args, unsigned NumArgs);
  void CodeCompleteInitializer(Scope *S, Decl *D);
  void CodeCompleteReturn(Scope *S);
  void CodeCompleteAssignmentRHS(Scope *S, Expr *LHS);
  
  void CodeCompleteQualifiedId(Scope *S, CXXScopeSpec &SS,
                               bool EnteringContext);
  void CodeCompleteUsing(Scope *S);
  void CodeCompleteUsingDirective(Scope *S);
  void CodeCompleteNamespaceDecl(Scope *S);
  void CodeCompleteNamespaceAliasDecl(Scope *S);
  void CodeCompleteOperatorName(Scope *S);
  void CodeCompleteConstructorInitializer(Decl *Constructor,
                                          CXXCtorInitializer** Initializers,
                                          unsigned NumInitializers);
  
  void CodeCompleteObjCAtDirective(Scope *S, Decl *ObjCImpDecl,
                                   bool InInterface);
  void CodeCompleteObjCAtVisibility(Scope *S);
  void CodeCompleteObjCAtStatement(Scope *S);
  void CodeCompleteObjCAtExpression(Scope *S);
  void CodeCompleteObjCPropertyFlags(Scope *S, ObjCDeclSpec &ODS);
  void CodeCompleteObjCPropertyGetter(Scope *S, Decl *ClassDecl);
  void CodeCompleteObjCPropertySetter(Scope *S, Decl *ClassDecl);
  void CodeCompleteObjCPassingType(Scope *S, ObjCDeclSpec &DS, 
                                   bool IsParameter);
  void CodeCompleteObjCMessageReceiver(Scope *S);
  void CodeCompleteObjCSuperMessage(Scope *S, SourceLocation SuperLoc,
                                    IdentifierInfo **SelIdents,
                                    unsigned NumSelIdents,
                                    bool AtArgumentExpression);
  void CodeCompleteObjCClassMessage(Scope *S, ParsedType Receiver,
                                    IdentifierInfo **SelIdents,
                                    unsigned NumSelIdents,
                                    bool AtArgumentExpression,
                                    bool IsSuper = false);
  void CodeCompleteObjCInstanceMessage(Scope *S, ExprTy *Receiver,
                                       IdentifierInfo **SelIdents,
                                       unsigned NumSelIdents,
                                       bool AtArgumentExpression,
                                       ObjCInterfaceDecl *Super = 0);
  void CodeCompleteObjCForCollection(Scope *S, 
                                     DeclGroupPtrTy IterationVar);
  void CodeCompleteObjCSelector(Scope *S,
                                IdentifierInfo **SelIdents,
                                unsigned NumSelIdents);
  void CodeCompleteObjCProtocolReferences(IdentifierLocPair *Protocols,
                                          unsigned NumProtocols);
  void CodeCompleteObjCProtocolDecl(Scope *S);
  void CodeCompleteObjCInterfaceDecl(Scope *S);
  void CodeCompleteObjCSuperclass(Scope *S,
                                  IdentifierInfo *ClassName,
                                  SourceLocation ClassNameLoc);
  void CodeCompleteObjCImplementationDecl(Scope *S);
  void CodeCompleteObjCInterfaceCategory(Scope *S,
                                         IdentifierInfo *ClassName,
                                         SourceLocation ClassNameLoc);
  void CodeCompleteObjCImplementationCategory(Scope *S,
                                              IdentifierInfo *ClassName,
                                              SourceLocation ClassNameLoc);
  void CodeCompleteObjCPropertyDefinition(Scope *S, Decl *ObjCImpDecl);
  void CodeCompleteObjCPropertySynthesizeIvar(Scope *S,
                                              IdentifierInfo *PropertyName,
                                              Decl *ObjCImpDecl);
  void CodeCompleteObjCMethodDecl(Scope *S,
                                  bool IsInstanceMethod,
                                  ParsedType ReturnType,
                                  Decl *IDecl);
  void CodeCompleteObjCMethodDeclSelector(Scope *S, 
                                          bool IsInstanceMethod,
                                          bool AtParameterName,
                                          ParsedType ReturnType,
                                          IdentifierInfo **SelIdents,
                                          unsigned NumSelIdents);
  void CodeCompletePreprocessorDirective(bool InConditional);
  void CodeCompleteInPreprocessorConditionalExclusion(Scope *S);
  void CodeCompletePreprocessorMacroName(bool IsDefinition);
  void CodeCompletePreprocessorExpression();
  void CodeCompletePreprocessorMacroArgument(Scope *S,
                                             IdentifierInfo *Macro,
                                             MacroInfo *MacroInfo,
                                             unsigned Argument);
  void CodeCompleteNaturalLanguage();
  void GatherGlobalCodeCompletions(CodeCompletionAllocator &Allocator,
                  llvm::SmallVectorImpl<CodeCompletionResult> &Results);
  //@}

  void PrintStats() const {}

  //===--------------------------------------------------------------------===//
  // Extra semantic analysis beyond the C type system

public:
  SourceLocation getLocationOfStringLiteralByte(const StringLiteral *SL,
                                                unsigned ByteNo) const;

private:  
  void CheckArrayAccess(const Expr *E);
  bool CheckFunctionCall(FunctionDecl *FDecl, CallExpr *TheCall);
  bool CheckBlockCall(NamedDecl *NDecl, CallExpr *TheCall);

  bool CheckablePrintfAttr(const FormatAttr *Format, CallExpr *TheCall);
  bool CheckObjCString(Expr *Arg);

  ExprResult CheckBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall);
  bool CheckARMBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall);

  bool SemaBuiltinVAStart(CallExpr *TheCall);
  bool SemaBuiltinUnorderedCompare(CallExpr *TheCall);
  bool SemaBuiltinFPClassification(CallExpr *TheCall, unsigned NumArgs);

public:
  // Used by C++ template instantiation.
  ExprResult SemaBuiltinShuffleVector(CallExpr *TheCall);

private:
  bool SemaBuiltinPrefetch(CallExpr *TheCall);
  bool SemaBuiltinObjectSize(CallExpr *TheCall);
  bool SemaBuiltinLongjmp(CallExpr *TheCall);
  ExprResult SemaBuiltinAtomicOverloaded(ExprResult TheCallResult);
  bool SemaBuiltinConstantArg(CallExpr *TheCall, int ArgNum,
                              llvm::APSInt &Result);

  bool SemaCheckStringLiteral(const Expr *E, const CallExpr *TheCall,
                              bool HasVAListArg, unsigned format_idx,
                              unsigned firstDataArg, bool isPrintf);

  void CheckFormatString(const StringLiteral *FExpr, const Expr *OrigFormatExpr,
                         const CallExpr *TheCall, bool HasVAListArg,
                         unsigned format_idx, unsigned firstDataArg,
                         bool isPrintf);

  void CheckNonNullArguments(const NonNullAttr *NonNull,
                             const CallExpr *TheCall);

  void CheckPrintfScanfArguments(const CallExpr *TheCall, bool HasVAListArg,
                                 unsigned format_idx, unsigned firstDataArg,
                                 bool isPrintf);

  void CheckReturnStackAddr(Expr *RetValExp, QualType lhsType,
                            SourceLocation ReturnLoc);
  void CheckFloatComparison(SourceLocation loc, Expr* lex, Expr* rex);
  void CheckImplicitConversions(Expr *E, SourceLocation CC = SourceLocation());

  void CheckBitFieldInitialization(SourceLocation InitLoc, FieldDecl *Field,
                                   Expr *Init);

  /// \brief The parser's current scope.
  ///
  /// The parser maintains this state here.
  Scope *CurScope;
  
protected:
  friend class Parser;
  friend class InitializationSequence;
  
  /// \brief Retrieve the parser's current scope.
  Scope *getCurScope() const { return CurScope; }  
};

/// \brief RAII object that enters a new expression evaluation context.
class EnterExpressionEvaluationContext {
  Sema &Actions;

public:
  EnterExpressionEvaluationContext(Sema &Actions,
                                   Sema::ExpressionEvaluationContext NewContext)
    : Actions(Actions) {
    Actions.PushExpressionEvaluationContext(NewContext);
  }

  ~EnterExpressionEvaluationContext() {
    Actions.PopExpressionEvaluationContext();
  }
};

}  // end namespace clang

#endif
