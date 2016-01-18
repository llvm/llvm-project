//===--- ItaniumMangle.cpp - Itanium C++ Name Mangling ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements C++ name mangling according to the Itanium C++ ABI,
// which is used in GCC 3.2 and newer (and many compilers that are
// ABI-compatible with GCC):
//
//   http://mentorembedded.github.io/cxx-abi/abi.html#mangling
//
//===----------------------------------------------------------------------===//
#include "clang/AST/Mangle.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/ABI.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define MANGLE_CHECKER 0

#if MANGLE_CHECKER
#include <cxxabi.h>
#endif

using namespace clang;

namespace {

/// Retrieve the declaration context that should be used when mangling the given
/// declaration.
static const DeclContext *getEffectiveDeclContext(const Decl *D) {
  // The ABI assumes that lambda closure types that occur within 
  // default arguments live in the context of the function. However, due to
  // the way in which Clang parses and creates function declarations, this is
  // not the case: the lambda closure type ends up living in the context 
  // where the function itself resides, because the function declaration itself
  // had not yet been created. Fix the context here.
  if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(D)) {
    if (RD->isLambda())
      if (ParmVarDecl *ContextParam
            = dyn_cast_or_null<ParmVarDecl>(RD->getLambdaContextDecl()))
        return ContextParam->getDeclContext();
  }

  // Perform the same check for block literals.
  if (const BlockDecl *BD = dyn_cast<BlockDecl>(D)) {
    if (ParmVarDecl *ContextParam
          = dyn_cast_or_null<ParmVarDecl>(BD->getBlockManglingContextDecl()))
      return ContextParam->getDeclContext();
  }
  
  const DeclContext *DC = D->getDeclContext();
  if (const CapturedDecl *CD = dyn_cast<CapturedDecl>(DC))
    return getEffectiveDeclContext(CD);

  if (const auto *VD = dyn_cast<VarDecl>(D))
    if (VD->isExternC())
      return VD->getASTContext().getTranslationUnitDecl();

  if (const auto *FD = dyn_cast<FunctionDecl>(D))
    if (FD->isExternC())
      return FD->getASTContext().getTranslationUnitDecl();

  return DC;
}

static const DeclContext *getEffectiveParentContext(const DeclContext *DC) {
  return getEffectiveDeclContext(cast<Decl>(DC));
}

static bool isLocalContainerContext(const DeclContext *DC) {
  return isa<FunctionDecl>(DC) || isa<ObjCMethodDecl>(DC) || isa<BlockDecl>(DC);
}

static const RecordDecl *GetLocalClassDecl(const Decl *D) {
  const DeclContext *DC = getEffectiveDeclContext(D);
  while (!DC->isNamespace() && !DC->isTranslationUnit()) {
    if (isLocalContainerContext(DC))
      return dyn_cast<RecordDecl>(D);
    D = cast<Decl>(DC);
    DC = getEffectiveDeclContext(D);
  }
  return nullptr;
}

static const FunctionDecl *getStructor(const FunctionDecl *fn) {
  if (const FunctionTemplateDecl *ftd = fn->getPrimaryTemplate())
    return ftd->getTemplatedDecl();

  return fn;
}

static const NamedDecl *getStructor(const NamedDecl *decl) {
  const FunctionDecl *fn = dyn_cast_or_null<FunctionDecl>(decl);
  return (fn ? getStructor(fn) : decl);
}

static bool isLambda(const NamedDecl *ND) {
  const CXXRecordDecl *Record = dyn_cast<CXXRecordDecl>(ND);
  if (!Record)
    return false;

  return Record->isLambda();
}

static const unsigned UnknownArity = ~0U;

class ItaniumMangleContextImpl : public ItaniumMangleContext {
  typedef std::pair<const DeclContext*, IdentifierInfo*> DiscriminatorKeyTy;
  llvm::DenseMap<DiscriminatorKeyTy, unsigned> Discriminator;
  llvm::DenseMap<const NamedDecl*, unsigned> Uniquifier;

public:
  explicit ItaniumMangleContextImpl(ASTContext &Context,
                                    DiagnosticsEngine &Diags)
      : ItaniumMangleContext(Context, Diags) {}

  /// @name Mangler Entry Points
  /// @{

  bool shouldMangleCXXName(const NamedDecl *D) override;
  bool shouldMangleStringLiteral(const StringLiteral *) override {
    return false;
  }
  void mangleCXXName(const NamedDecl *D, raw_ostream &) override;
  void mangleThunk(const CXXMethodDecl *MD, const ThunkInfo &Thunk,
                   raw_ostream &) override;
  void mangleCXXDtorThunk(const CXXDestructorDecl *DD, CXXDtorType Type,
                          const ThisAdjustment &ThisAdjustment,
                          raw_ostream &) override;
  void mangleReferenceTemporary(const VarDecl *D, unsigned ManglingNumber,
                                raw_ostream &) override;
  void mangleCXXVTable(const CXXRecordDecl *RD, raw_ostream &) override;
  void mangleCXXVTT(const CXXRecordDecl *RD, raw_ostream &) override;
  void mangleCXXCtorVTable(const CXXRecordDecl *RD, int64_t Offset,
                           const CXXRecordDecl *Type, raw_ostream &) override;
  void mangleCXXRTTI(QualType T, raw_ostream &) override;
  void mangleCXXRTTIName(QualType T, raw_ostream &) override;
  void mangleTypeName(QualType T, raw_ostream &) override;
  void mangleCXXCtor(const CXXConstructorDecl *D, CXXCtorType Type,
                     raw_ostream &) override;
  void mangleCXXDtor(const CXXDestructorDecl *D, CXXDtorType Type,
                     raw_ostream &) override;

  void mangleCXXCtorComdat(const CXXConstructorDecl *D, raw_ostream &) override;
  void mangleCXXDtorComdat(const CXXDestructorDecl *D, raw_ostream &) override;
  void mangleStaticGuardVariable(const VarDecl *D, raw_ostream &) override;
  void mangleDynamicInitializer(const VarDecl *D, raw_ostream &Out) override;
  void mangleDynamicAtExitDestructor(const VarDecl *D,
                                     raw_ostream &Out) override;
  void mangleSEHFilterExpression(const NamedDecl *EnclosingDecl,
                                 raw_ostream &Out) override;
  void mangleSEHFinallyBlock(const NamedDecl *EnclosingDecl,
                             raw_ostream &Out) override;
  void mangleItaniumThreadLocalInit(const VarDecl *D, raw_ostream &) override;
  void mangleItaniumThreadLocalWrapper(const VarDecl *D,
                                       raw_ostream &) override;

  void mangleStringLiteral(const StringLiteral *, raw_ostream &) override;

  bool getNextDiscriminator(const NamedDecl *ND, unsigned &disc) {
    // Lambda closure types are already numbered.
    if (isLambda(ND))
      return false;

    // Anonymous tags are already numbered.
    if (const TagDecl *Tag = dyn_cast<TagDecl>(ND)) {
      if (Tag->getName().empty() && !Tag->getTypedefNameForAnonDecl())
        return false;
    }

    // Use the canonical number for externally visible decls.
    if (ND->isExternallyVisible()) {
      unsigned discriminator = getASTContext().getManglingNumber(ND);
      if (discriminator == 1)
        return false;
      disc = discriminator - 2;
      return true;
    }

    // Make up a reasonable number for internal decls.
    unsigned &discriminator = Uniquifier[ND];
    if (!discriminator) {
      const DeclContext *DC = getEffectiveDeclContext(ND);
      discriminator = ++Discriminator[std::make_pair(DC, ND->getIdentifier())];
    }
    if (discriminator == 1)
      return false;
    disc = discriminator-2;
    return true;
  }
  /// @}
};

/// Manage the mangling of a single name.
class CXXNameMangler {
  ItaniumMangleContextImpl &Context;
  raw_ostream &Out;

  /// The "structor" is the top-level declaration being mangled, if
  /// that's not a template specialization; otherwise it's the pattern
  /// for that specialization.
  const NamedDecl *Structor;
  unsigned StructorType;

  /// The next substitution sequence number.
  unsigned SeqID;

  class FunctionTypeDepthState {
    unsigned Bits;

    enum { InResultTypeMask = 1 };

  public:
    FunctionTypeDepthState() : Bits(0) {}

    /// The number of function types we're inside.
    unsigned getDepth() const {
      return Bits >> 1;
    }

    /// True if we're in the return type of the innermost function type.
    bool isInResultType() const {
      return Bits & InResultTypeMask;
    }

    FunctionTypeDepthState push() {
      FunctionTypeDepthState tmp = *this;
      Bits = (Bits & ~InResultTypeMask) + 2;
      return tmp;
    }

    void enterResultType() {
      Bits |= InResultTypeMask;
    }

    void leaveResultType() {
      Bits &= ~InResultTypeMask;
    }

    void pop(FunctionTypeDepthState saved) {
      assert(getDepth() == saved.getDepth() + 1);
      Bits = saved.Bits;
    }

  } FunctionTypeDepth;

  llvm::DenseMap<uintptr_t, unsigned> Substitutions;

  ASTContext &getASTContext() const { return Context.getASTContext(); }

public:
  CXXNameMangler(ItaniumMangleContextImpl &C, raw_ostream &Out_,
                 const NamedDecl *D = nullptr)
    : Context(C), Out(Out_), Structor(getStructor(D)), StructorType(0),
      SeqID(0) {
    // These can't be mangled without a ctor type or dtor type.
    assert(!D || (!isa<CXXDestructorDecl>(D) &&
                  !isa<CXXConstructorDecl>(D)));
  }
  CXXNameMangler(ItaniumMangleContextImpl &C, raw_ostream &Out_,
                 const CXXConstructorDecl *D, CXXCtorType Type)
    : Context(C), Out(Out_), Structor(getStructor(D)), StructorType(Type),
      SeqID(0) { }
  CXXNameMangler(ItaniumMangleContextImpl &C, raw_ostream &Out_,
                 const CXXDestructorDecl *D, CXXDtorType Type)
    : Context(C), Out(Out_), Structor(getStructor(D)), StructorType(Type),
      SeqID(0) { }

#if MANGLE_CHECKER
  ~CXXNameMangler() {
    if (Out.str()[0] == '\01')
      return;

    int status = 0;
    char *result = abi::__cxa_demangle(Out.str().str().c_str(), 0, 0, &status);
    assert(status == 0 && "Could not demangle mangled name!");
    free(result);
  }
#endif
  raw_ostream &getStream() { return Out; }

  void mangle(const NamedDecl *D);
  void mangleCallOffset(int64_t NonVirtual, int64_t Virtual);
  void mangleNumber(const llvm::APSInt &I);
  void mangleNumber(int64_t Number);
  void mangleFloat(const llvm::APFloat &F);
  void mangleFunctionEncoding(const FunctionDecl *FD);
  void mangleSeqID(unsigned SeqID);
  void mangleName(const NamedDecl *ND);
  void mangleType(QualType T);
  void mangleNameOrStandardSubstitution(const NamedDecl *ND);
  
private:

  bool mangleSubstitution(const NamedDecl *ND);
  bool mangleSubstitution(QualType T);
  bool mangleSubstitution(TemplateName Template);
  bool mangleSubstitution(uintptr_t Ptr);

  void mangleExistingSubstitution(QualType type);
  void mangleExistingSubstitution(TemplateName name);

  bool mangleStandardSubstitution(const NamedDecl *ND);

  void addSubstitution(const NamedDecl *ND) {
    ND = cast<NamedDecl>(ND->getCanonicalDecl());

    addSubstitution(reinterpret_cast<uintptr_t>(ND));
  }
  void addSubstitution(QualType T);
  void addSubstitution(TemplateName Template);
  void addSubstitution(uintptr_t Ptr);

  void mangleUnresolvedPrefix(NestedNameSpecifier *qualifier,
                              bool recursive = false);
  void mangleUnresolvedName(NestedNameSpecifier *qualifier,
                            DeclarationName name,
                            unsigned KnownArity = UnknownArity);

  void mangleName(const TemplateDecl *TD,
                  const TemplateArgument *TemplateArgs,
                  unsigned NumTemplateArgs);
  void mangleUnqualifiedName(const NamedDecl *ND) {
    mangleUnqualifiedName(ND, ND->getDeclName(), UnknownArity);
  }
  void mangleUnqualifiedName(const NamedDecl *ND, DeclarationName Name,
                             unsigned KnownArity);
  void mangleUnscopedName(const NamedDecl *ND);
  void mangleUnscopedTemplateName(const TemplateDecl *ND);
  void mangleUnscopedTemplateName(TemplateName);
  void mangleSourceName(const IdentifierInfo *II);
  void mangleLocalName(const Decl *D);
  void mangleBlockForPrefix(const BlockDecl *Block);
  void mangleUnqualifiedBlock(const BlockDecl *Block);
  void mangleLambda(const CXXRecordDecl *Lambda);
  void mangleNestedName(const NamedDecl *ND, const DeclContext *DC,
                        bool NoFunction=false);
  void mangleNestedName(const TemplateDecl *TD,
                        const TemplateArgument *TemplateArgs,
                        unsigned NumTemplateArgs);
  void manglePrefix(NestedNameSpecifier *qualifier);
  void manglePrefix(const DeclContext *DC, bool NoFunction=false);
  void manglePrefix(QualType type);
  void mangleTemplatePrefix(const TemplateDecl *ND, bool NoFunction=false);
  void mangleTemplatePrefix(TemplateName Template);
  bool mangleUnresolvedTypeOrSimpleId(QualType DestroyedType,
                                      StringRef Prefix = "");
  void mangleOperatorName(DeclarationName Name, unsigned Arity);
  void mangleOperatorName(OverloadedOperatorKind OO, unsigned Arity);
  void mangleQualifiers(Qualifiers Quals);
  void mangleRefQualifier(RefQualifierKind RefQualifier);

  void mangleObjCMethodName(const ObjCMethodDecl *MD);

  // Declare manglers for every type class.
#define ABSTRACT_TYPE(CLASS, PARENT)
#define NON_CANONICAL_TYPE(CLASS, PARENT)
#define TYPE(CLASS, PARENT) void mangleType(const CLASS##Type *T);
#include "clang/AST/TypeNodes.def"

  void mangleType(const TagType*);
  void mangleType(TemplateName);
  void mangleBareFunctionType(const FunctionType *T,
                              bool MangleReturnType);
  void mangleNeonVectorType(const VectorType *T);
  void mangleAArch64NeonVectorType(const VectorType *T);

  void mangleIntegerLiteral(QualType T, const llvm::APSInt &Value);
  void mangleMemberExprBase(const Expr *base, bool isArrow);
  void mangleMemberExpr(const Expr *base, bool isArrow,
                        NestedNameSpecifier *qualifier,
                        NamedDecl *firstQualifierLookup,
                        DeclarationName name,
                        unsigned knownArity);
  void mangleCastExpression(const Expr *E, StringRef CastEncoding);
  void mangleInitListElements(const InitListExpr *InitList);
  void mangleExpression(const Expr *E, unsigned Arity = UnknownArity);
  void mangleCXXCtorType(CXXCtorType T);
  void mangleCXXDtorType(CXXDtorType T);

  void mangleTemplateArgs(const ASTTemplateArgumentListInfo &TemplateArgs);
  void mangleTemplateArgs(const TemplateArgument *TemplateArgs,
                          unsigned NumTemplateArgs);
  void mangleTemplateArgs(const TemplateArgumentList &AL);
  void mangleTemplateArg(TemplateArgument A);

  void mangleTemplateParameter(unsigned Index);

  void mangleFunctionParam(const ParmVarDecl *parm);
};

}

bool ItaniumMangleContextImpl::shouldMangleCXXName(const NamedDecl *D) {
  const FunctionDecl *FD = dyn_cast<FunctionDecl>(D);
  if (FD) {
    LanguageLinkage L = FD->getLanguageLinkage();
    // Overloadable functions need mangling.
    if (FD->hasAttr<OverloadableAttr>())
      return true;

    // "main" is not mangled.
    if (FD->isMain())
      return false;

    // C++ functions and those whose names are not a simple identifier need
    // mangling.
    if (!FD->getDeclName().isIdentifier() || L == CXXLanguageLinkage)
      return true;

    // C functions are not mangled.
    if (L == CLanguageLinkage)
      return false;
  }

  // Otherwise, no mangling is done outside C++ mode.
  if (!getASTContext().getLangOpts().CPlusPlus)
    return false;

  const VarDecl *VD = dyn_cast<VarDecl>(D);
  if (VD) {
    // C variables are not mangled.
    if (VD->isExternC())
      return false;

    // Variables at global scope with non-internal linkage are not mangled
    const DeclContext *DC = getEffectiveDeclContext(D);
    // Check for extern variable declared locally.
    if (DC->isFunctionOrMethod() && D->hasLinkage())
      while (!DC->isNamespace() && !DC->isTranslationUnit())
        DC = getEffectiveParentContext(DC);
    if (DC->isTranslationUnit() && D->getFormalLinkage() != InternalLinkage &&
        !isa<VarTemplateSpecializationDecl>(D))
      return false;
  }

  return true;
}

void CXXNameMangler::mangle(const NamedDecl *D) {
  // <mangled-name> ::= _Z <encoding>
  //            ::= <data name>
  //            ::= <special-name>
  Out << "_Z";
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
    mangleFunctionEncoding(FD);
  else if (const VarDecl *VD = dyn_cast<VarDecl>(D))
    mangleName(VD);
  else if (const IndirectFieldDecl *IFD = dyn_cast<IndirectFieldDecl>(D))
    mangleName(IFD->getAnonField());
  else
    mangleName(cast<FieldDecl>(D));
}

void CXXNameMangler::mangleFunctionEncoding(const FunctionDecl *FD) {
  // <encoding> ::= <function name> <bare-function-type>
  mangleName(FD);

  // Don't mangle in the type if this isn't a decl we should typically mangle.
  if (!Context.shouldMangleDeclName(FD))
    return;

  if (FD->hasAttr<EnableIfAttr>()) {
    FunctionTypeDepthState Saved = FunctionTypeDepth.push();
    Out << "Ua9enable_ifI";
    // FIXME: specific_attr_iterator iterates in reverse order. Fix that and use
    // it here.
    for (AttrVec::const_reverse_iterator I = FD->getAttrs().rbegin(),
                                         E = FD->getAttrs().rend();
         I != E; ++I) {
      EnableIfAttr *EIA = dyn_cast<EnableIfAttr>(*I);
      if (!EIA)
        continue;
      Out << 'X';
      mangleExpression(EIA->getCond());
      Out << 'E';
    }
    Out << 'E';
    FunctionTypeDepth.pop(Saved);
  }

  // Whether the mangling of a function type includes the return type depends on
  // the context and the nature of the function. The rules for deciding whether
  // the return type is included are:
  //
  //   1. Template functions (names or types) have return types encoded, with
  //   the exceptions listed below.
  //   2. Function types not appearing as part of a function name mangling,
  //   e.g. parameters, pointer types, etc., have return type encoded, with the
  //   exceptions listed below.
  //   3. Non-template function names do not have return types encoded.
  //
  // The exceptions mentioned in (1) and (2) above, for which the return type is
  // never included, are
  //   1. Constructors.
  //   2. Destructors.
  //   3. Conversion operator functions, e.g. operator int.
  bool MangleReturnType = false;
  if (FunctionTemplateDecl *PrimaryTemplate = FD->getPrimaryTemplate()) {
    if (!(isa<CXXConstructorDecl>(FD) || isa<CXXDestructorDecl>(FD) ||
          isa<CXXConversionDecl>(FD)))
      MangleReturnType = true;

    // Mangle the type of the primary template.
    FD = PrimaryTemplate->getTemplatedDecl();
  }

  mangleBareFunctionType(FD->getType()->getAs<FunctionType>(), 
                         MangleReturnType);
}

static const DeclContext *IgnoreLinkageSpecDecls(const DeclContext *DC) {
  while (isa<LinkageSpecDecl>(DC)) {
    DC = getEffectiveParentContext(DC);
  }

  return DC;
}

/// Return whether a given namespace is the 'std' namespace.
static bool isStd(const NamespaceDecl *NS) {
  if (!IgnoreLinkageSpecDecls(getEffectiveParentContext(NS))
                                ->isTranslationUnit())
    return false;
  
  const IdentifierInfo *II = NS->getOriginalNamespace()->getIdentifier();
  return II && II->isStr("std");
}

// isStdNamespace - Return whether a given decl context is a toplevel 'std'
// namespace.
static bool isStdNamespace(const DeclContext *DC) {
  if (!DC->isNamespace())
    return false;

  return isStd(cast<NamespaceDecl>(DC));
}

static const TemplateDecl *
isTemplate(const NamedDecl *ND, const TemplateArgumentList *&TemplateArgs) {
  // Check if we have a function template.
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(ND)){
    if (const TemplateDecl *TD = FD->getPrimaryTemplate()) {
      TemplateArgs = FD->getTemplateSpecializationArgs();
      return TD;
    }
  }

  // Check if we have a class template.
  if (const ClassTemplateSpecializationDecl *Spec =
        dyn_cast<ClassTemplateSpecializationDecl>(ND)) {
    TemplateArgs = &Spec->getTemplateArgs();
    return Spec->getSpecializedTemplate();
  }

  // Check if we have a variable template.
  if (const VarTemplateSpecializationDecl *Spec =
          dyn_cast<VarTemplateSpecializationDecl>(ND)) {
    TemplateArgs = &Spec->getTemplateArgs();
    return Spec->getSpecializedTemplate();
  }

  return nullptr;
}

void CXXNameMangler::mangleName(const NamedDecl *ND) {
  //  <name> ::= <nested-name>
  //         ::= <unscoped-name>
  //         ::= <unscoped-template-name> <template-args>
  //         ::= <local-name>
  //
  const DeclContext *DC = getEffectiveDeclContext(ND);

  // If this is an extern variable declared locally, the relevant DeclContext
  // is that of the containing namespace, or the translation unit.
  // FIXME: This is a hack; extern variables declared locally should have
  // a proper semantic declaration context!
  if (isLocalContainerContext(DC) && ND->hasLinkage() && !isLambda(ND))
    while (!DC->isNamespace() && !DC->isTranslationUnit())
      DC = getEffectiveParentContext(DC);
  else if (GetLocalClassDecl(ND)) {
    mangleLocalName(ND);
    return;
  }

  DC = IgnoreLinkageSpecDecls(DC);

  if (DC->isTranslationUnit() || isStdNamespace(DC)) {
    // Check if we have a template.
    const TemplateArgumentList *TemplateArgs = nullptr;
    if (const TemplateDecl *TD = isTemplate(ND, TemplateArgs)) {
      mangleUnscopedTemplateName(TD);
      mangleTemplateArgs(*TemplateArgs);
      return;
    }

    mangleUnscopedName(ND);
    return;
  }

  if (isLocalContainerContext(DC)) {
    mangleLocalName(ND);
    return;
  }

  mangleNestedName(ND, DC);
}
void CXXNameMangler::mangleName(const TemplateDecl *TD,
                                const TemplateArgument *TemplateArgs,
                                unsigned NumTemplateArgs) {
  const DeclContext *DC = IgnoreLinkageSpecDecls(getEffectiveDeclContext(TD));

  if (DC->isTranslationUnit() || isStdNamespace(DC)) {
    mangleUnscopedTemplateName(TD);
    mangleTemplateArgs(TemplateArgs, NumTemplateArgs);
  } else {
    mangleNestedName(TD, TemplateArgs, NumTemplateArgs);
  }
}

void CXXNameMangler::mangleUnscopedName(const NamedDecl *ND) {
  //  <unscoped-name> ::= <unqualified-name>
  //                  ::= St <unqualified-name>   # ::std::

  if (isStdNamespace(IgnoreLinkageSpecDecls(getEffectiveDeclContext(ND))))
    Out << "St";

  mangleUnqualifiedName(ND);
}

void CXXNameMangler::mangleUnscopedTemplateName(const TemplateDecl *ND) {
  //     <unscoped-template-name> ::= <unscoped-name>
  //                              ::= <substitution>
  if (mangleSubstitution(ND))
    return;

  // <template-template-param> ::= <template-param>
  if (const auto *TTP = dyn_cast<TemplateTemplateParmDecl>(ND))
    mangleTemplateParameter(TTP->getIndex());
  else
    mangleUnscopedName(ND->getTemplatedDecl());

  addSubstitution(ND);
}

void CXXNameMangler::mangleUnscopedTemplateName(TemplateName Template) {
  //     <unscoped-template-name> ::= <unscoped-name>
  //                              ::= <substitution>
  if (TemplateDecl *TD = Template.getAsTemplateDecl())
    return mangleUnscopedTemplateName(TD);
  
  if (mangleSubstitution(Template))
    return;

  DependentTemplateName *Dependent = Template.getAsDependentTemplateName();
  assert(Dependent && "Not a dependent template name?");
  if (const IdentifierInfo *Id = Dependent->getIdentifier())
    mangleSourceName(Id);
  else
    mangleOperatorName(Dependent->getOperator(), UnknownArity);
  
  addSubstitution(Template);
}

void CXXNameMangler::mangleFloat(const llvm::APFloat &f) {
  // ABI:
  //   Floating-point literals are encoded using a fixed-length
  //   lowercase hexadecimal string corresponding to the internal
  //   representation (IEEE on Itanium), high-order bytes first,
  //   without leading zeroes. For example: "Lf bf800000 E" is -1.0f
  //   on Itanium.
  // The 'without leading zeroes' thing seems to be an editorial
  // mistake; see the discussion on cxx-abi-dev beginning on
  // 2012-01-16.

  // Our requirements here are just barely weird enough to justify
  // using a custom algorithm instead of post-processing APInt::toString().

  llvm::APInt valueBits = f.bitcastToAPInt();
  unsigned numCharacters = (valueBits.getBitWidth() + 3) / 4;
  assert(numCharacters != 0);

  // Allocate a buffer of the right number of characters.
  SmallVector<char, 20> buffer(numCharacters);

  // Fill the buffer left-to-right.
  for (unsigned stringIndex = 0; stringIndex != numCharacters; ++stringIndex) {
    // The bit-index of the next hex digit.
    unsigned digitBitIndex = 4 * (numCharacters - stringIndex - 1);

    // Project out 4 bits starting at 'digitIndex'.
    llvm::integerPart hexDigit
      = valueBits.getRawData()[digitBitIndex / llvm::integerPartWidth];
    hexDigit >>= (digitBitIndex % llvm::integerPartWidth);
    hexDigit &= 0xF;

    // Map that over to a lowercase hex digit.
    static const char charForHex[16] = {
      '0', '1', '2', '3', '4', '5', '6', '7',
      '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'
    };
    buffer[stringIndex] = charForHex[hexDigit];
  }

  Out.write(buffer.data(), numCharacters);
}

void CXXNameMangler::mangleNumber(const llvm::APSInt &Value) {
  if (Value.isSigned() && Value.isNegative()) {
    Out << 'n';
    Value.abs().print(Out, /*signed*/ false);
  } else {
    Value.print(Out, /*signed*/ false);
  }
}

void CXXNameMangler::mangleNumber(int64_t Number) {
  //  <number> ::= [n] <non-negative decimal integer>
  if (Number < 0) {
    Out << 'n';
    Number = -Number;
  }

  Out << Number;
}

void CXXNameMangler::mangleCallOffset(int64_t NonVirtual, int64_t Virtual) {
  //  <call-offset>  ::= h <nv-offset> _
  //                 ::= v <v-offset> _
  //  <nv-offset>    ::= <offset number>        # non-virtual base override
  //  <v-offset>     ::= <offset number> _ <virtual offset number>
  //                      # virtual base override, with vcall offset
  if (!Virtual) {
    Out << 'h';
    mangleNumber(NonVirtual);
    Out << '_';
    return;
  }

  Out << 'v';
  mangleNumber(NonVirtual);
  Out << '_';
  mangleNumber(Virtual);
  Out << '_';
}

void CXXNameMangler::manglePrefix(QualType type) {
  if (const auto *TST = type->getAs<TemplateSpecializationType>()) {
    if (!mangleSubstitution(QualType(TST, 0))) {
      mangleTemplatePrefix(TST->getTemplateName());
        
      // FIXME: GCC does not appear to mangle the template arguments when
      // the template in question is a dependent template name. Should we
      // emulate that badness?
      mangleTemplateArgs(TST->getArgs(), TST->getNumArgs());
      addSubstitution(QualType(TST, 0));
    }
  } else if (const auto *DTST =
                 type->getAs<DependentTemplateSpecializationType>()) {
    if (!mangleSubstitution(QualType(DTST, 0))) {
      TemplateName Template = getASTContext().getDependentTemplateName(
          DTST->getQualifier(), DTST->getIdentifier());
      mangleTemplatePrefix(Template);

      // FIXME: GCC does not appear to mangle the template arguments when
      // the template in question is a dependent template name. Should we
      // emulate that badness?
      mangleTemplateArgs(DTST->getArgs(), DTST->getNumArgs());
      addSubstitution(QualType(DTST, 0));
    }
  } else {
    // We use the QualType mangle type variant here because it handles
    // substitutions.
    mangleType(type);
  }
}

/// Mangle everything prior to the base-unresolved-name in an unresolved-name.
///
/// \param recursive - true if this is being called recursively,
///   i.e. if there is more prefix "to the right".
void CXXNameMangler::mangleUnresolvedPrefix(NestedNameSpecifier *qualifier,
                                            bool recursive) {

  // x, ::x
  // <unresolved-name> ::= [gs] <base-unresolved-name>

  // T::x / decltype(p)::x
  // <unresolved-name> ::= sr <unresolved-type> <base-unresolved-name>

  // T::N::x /decltype(p)::N::x
  // <unresolved-name> ::= srN <unresolved-type> <unresolved-qualifier-level>+ E
  //                       <base-unresolved-name>

  // A::x, N::y, A<T>::z; "gs" means leading "::"
  // <unresolved-name> ::= [gs] sr <unresolved-qualifier-level>+ E
  //                       <base-unresolved-name>

  switch (qualifier->getKind()) {
  case NestedNameSpecifier::Global:
    Out << "gs";

    // We want an 'sr' unless this is the entire NNS.
    if (recursive)
      Out << "sr";

    // We never want an 'E' here.
    return;

  case NestedNameSpecifier::Super:
    llvm_unreachable("Can't mangle __super specifier");

  case NestedNameSpecifier::Namespace:
    if (qualifier->getPrefix())
      mangleUnresolvedPrefix(qualifier->getPrefix(),
                             /*recursive*/ true);
    else
      Out << "sr";
    mangleSourceName(qualifier->getAsNamespace()->getIdentifier());
    break;
  case NestedNameSpecifier::NamespaceAlias:
    if (qualifier->getPrefix())
      mangleUnresolvedPrefix(qualifier->getPrefix(),
                             /*recursive*/ true);
    else
      Out << "sr";
    mangleSourceName(qualifier->getAsNamespaceAlias()->getIdentifier());
    break;

  case NestedNameSpecifier::TypeSpec:
  case NestedNameSpecifier::TypeSpecWithTemplate: {
    const Type *type = qualifier->getAsType();

    // We only want to use an unresolved-type encoding if this is one of:
    //   - a decltype
    //   - a template type parameter
    //   - a template template parameter with arguments
    // In all of these cases, we should have no prefix.
    if (qualifier->getPrefix()) {
      mangleUnresolvedPrefix(qualifier->getPrefix(),
                             /*recursive*/ true);
    } else {
      // Otherwise, all the cases want this.
      Out << "sr";
    }

    if (mangleUnresolvedTypeOrSimpleId(QualType(type, 0), recursive ? "N" : ""))
      return;

    break;
  }

  case NestedNameSpecifier::Identifier:
    // Member expressions can have these without prefixes.
    if (qualifier->getPrefix())
      mangleUnresolvedPrefix(qualifier->getPrefix(),
                             /*recursive*/ true);
    else
      Out << "sr";

    mangleSourceName(qualifier->getAsIdentifier());
    break;
  }

  // If this was the innermost part of the NNS, and we fell out to
  // here, append an 'E'.
  if (!recursive)
    Out << 'E';
}

/// Mangle an unresolved-name, which is generally used for names which
/// weren't resolved to specific entities.
void CXXNameMangler::mangleUnresolvedName(NestedNameSpecifier *qualifier,
                                          DeclarationName name,
                                          unsigned knownArity) {
  if (qualifier) mangleUnresolvedPrefix(qualifier);
  switch (name.getNameKind()) {
    // <base-unresolved-name> ::= <simple-id>
    case DeclarationName::Identifier:
      mangleSourceName(name.getAsIdentifierInfo());
      break;
    // <base-unresolved-name> ::= dn <destructor-name>
    case DeclarationName::CXXDestructorName:
      Out << "dn";
      mangleUnresolvedTypeOrSimpleId(name.getCXXNameType());
      break;
    // <base-unresolved-name> ::= on <operator-name>
    case DeclarationName::CXXConversionFunctionName:
    case DeclarationName::CXXLiteralOperatorName:
    case DeclarationName::CXXOperatorName:
      Out << "on";
      mangleOperatorName(name, knownArity);
      break;
    case DeclarationName::CXXConstructorName:
      llvm_unreachable("Can't mangle a constructor name!");
    case DeclarationName::CXXUsingDirective:
      llvm_unreachable("Can't mangle a using directive name!");
    case DeclarationName::ObjCMultiArgSelector:
    case DeclarationName::ObjCOneArgSelector:
    case DeclarationName::ObjCZeroArgSelector:
      llvm_unreachable("Can't mangle Objective-C selector names here!");
  }
}

void CXXNameMangler::mangleUnqualifiedName(const NamedDecl *ND,
                                           DeclarationName Name,
                                           unsigned KnownArity) {
  unsigned Arity = KnownArity;
  //  <unqualified-name> ::= <operator-name>
  //                     ::= <ctor-dtor-name>
  //                     ::= <source-name>
  switch (Name.getNameKind()) {
  case DeclarationName::Identifier: {
    if (const IdentifierInfo *II = Name.getAsIdentifierInfo()) {
      // We must avoid conflicts between internally- and externally-
      // linked variable and function declaration names in the same TU:
      //   void test() { extern void foo(); }
      //   static void foo();
      // This naming convention is the same as that followed by GCC,
      // though it shouldn't actually matter.
      if (ND && ND->getFormalLinkage() == InternalLinkage &&
          getEffectiveDeclContext(ND)->isFileContext())
        Out << 'L';

      mangleSourceName(II);
      break;
    }

    // Otherwise, an anonymous entity.  We must have a declaration.
    assert(ND && "mangling empty name without declaration");

    if (const NamespaceDecl *NS = dyn_cast<NamespaceDecl>(ND)) {
      if (NS->isAnonymousNamespace()) {
        // This is how gcc mangles these names.
        Out << "12_GLOBAL__N_1";
        break;
      }
    }

    if (const VarDecl *VD = dyn_cast<VarDecl>(ND)) {
      // We must have an anonymous union or struct declaration.
      const RecordDecl *RD =
        cast<RecordDecl>(VD->getType()->getAs<RecordType>()->getDecl());

      // Itanium C++ ABI 5.1.2:
      //
      //   For the purposes of mangling, the name of an anonymous union is
      //   considered to be the name of the first named data member found by a
      //   pre-order, depth-first, declaration-order walk of the data members of
      //   the anonymous union. If there is no such data member (i.e., if all of
      //   the data members in the union are unnamed), then there is no way for
      //   a program to refer to the anonymous union, and there is therefore no
      //   need to mangle its name.
      assert(RD->isAnonymousStructOrUnion()
             && "Expected anonymous struct or union!");
      const FieldDecl *FD = RD->findFirstNamedDataMember();

      // It's actually possible for various reasons for us to get here
      // with an empty anonymous struct / union.  Fortunately, it
      // doesn't really matter what name we generate.
      if (!FD) break;
      assert(FD->getIdentifier() && "Data member name isn't an identifier!");

      mangleSourceName(FD->getIdentifier());
      break;
    }

    // Class extensions have no name as a category, and it's possible
    // for them to be the semantic parent of certain declarations
    // (primarily, tag decls defined within declarations).  Such
    // declarations will always have internal linkage, so the name
    // doesn't really matter, but we shouldn't crash on them.  For
    // safety, just handle all ObjC containers here.
    if (isa<ObjCContainerDecl>(ND))
      break;
    
    // We must have an anonymous struct.
    const TagDecl *TD = cast<TagDecl>(ND);
    if (const TypedefNameDecl *D = TD->getTypedefNameForAnonDecl()) {
      assert(TD->getDeclContext() == D->getDeclContext() &&
             "Typedef should not be in another decl context!");
      assert(D->getDeclName().getAsIdentifierInfo() &&
             "Typedef was not named!");
      mangleSourceName(D->getDeclName().getAsIdentifierInfo());
      break;
    }

    // <unnamed-type-name> ::= <closure-type-name>
    // 
    // <closure-type-name> ::= Ul <lambda-sig> E [ <nonnegative number> ] _
    // <lambda-sig> ::= <parameter-type>+   # Parameter types or 'v' for 'void'.
    if (const CXXRecordDecl *Record = dyn_cast<CXXRecordDecl>(TD)) {
      if (Record->isLambda() && Record->getLambdaManglingNumber()) {
        mangleLambda(Record);
        break;
      }
    }

    if (TD->isExternallyVisible()) {
      unsigned UnnamedMangle = getASTContext().getManglingNumber(TD);
      Out << "Ut";
      if (UnnamedMangle > 1)
        Out << llvm::utostr(UnnamedMangle - 2);
      Out << '_';
      break;
    }

    // Get a unique id for the anonymous struct.
    unsigned AnonStructId = Context.getAnonymousStructId(TD);

    // Mangle it as a source name in the form
    // [n] $_<id>
    // where n is the length of the string.
    SmallString<8> Str;
    Str += "$_";
    Str += llvm::utostr(AnonStructId);

    Out << Str.size();
    Out << Str;
    break;
  }

  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
    llvm_unreachable("Can't mangle Objective-C selector names here!");

  case DeclarationName::CXXConstructorName:
    if (ND == Structor)
      // If the named decl is the C++ constructor we're mangling, use the type
      // we were given.
      mangleCXXCtorType(static_cast<CXXCtorType>(StructorType));
    else
      // Otherwise, use the complete constructor name. This is relevant if a
      // class with a constructor is declared within a constructor.
      mangleCXXCtorType(Ctor_Complete);
    break;

  case DeclarationName::CXXDestructorName:
    if (ND == Structor)
      // If the named decl is the C++ destructor we're mangling, use the type we
      // were given.
      mangleCXXDtorType(static_cast<CXXDtorType>(StructorType));
    else
      // Otherwise, use the complete destructor name. This is relevant if a
      // class with a destructor is declared within a destructor.
      mangleCXXDtorType(Dtor_Complete);
    break;

  case DeclarationName::CXXOperatorName:
    if (ND && Arity == UnknownArity) {
      Arity = cast<FunctionDecl>(ND)->getNumParams();

      // If we have a member function, we need to include the 'this' pointer.
      if (const auto *MD = dyn_cast<CXXMethodDecl>(ND))
        if (!MD->isStatic())
          Arity++;
    }
  // FALLTHROUGH
  case DeclarationName::CXXConversionFunctionName:
  case DeclarationName::CXXLiteralOperatorName:
    mangleOperatorName(Name, Arity);
    break;

  case DeclarationName::CXXUsingDirective:
    llvm_unreachable("Can't mangle a using directive name!");
  }
}

void CXXNameMangler::mangleSourceName(const IdentifierInfo *II) {
  // <source-name> ::= <positive length number> <identifier>
  // <number> ::= [n] <non-negative decimal integer>
  // <identifier> ::= <unqualified source code identifier>
  Out << II->getLength() << II->getName();
}

void CXXNameMangler::mangleNestedName(const NamedDecl *ND,
                                      const DeclContext *DC,
                                      bool NoFunction) {
  // <nested-name> 
  //   ::= N [<CV-qualifiers>] [<ref-qualifier>] <prefix> <unqualified-name> E
  //   ::= N [<CV-qualifiers>] [<ref-qualifier>] <template-prefix> 
  //       <template-args> E

  Out << 'N';
  if (const CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(ND)) {
    Qualifiers MethodQuals =
        Qualifiers::fromCVRMask(Method->getTypeQualifiers());
    // We do not consider restrict a distinguishing attribute for overloading
    // purposes so we must not mangle it.
    MethodQuals.removeRestrict();
    mangleQualifiers(MethodQuals);
    mangleRefQualifier(Method->getRefQualifier());
  }
  
  // Check if we have a template.
  const TemplateArgumentList *TemplateArgs = nullptr;
  if (const TemplateDecl *TD = isTemplate(ND, TemplateArgs)) {
    mangleTemplatePrefix(TD, NoFunction);
    mangleTemplateArgs(*TemplateArgs);
  }
  else {
    manglePrefix(DC, NoFunction);
    mangleUnqualifiedName(ND);
  }

  Out << 'E';
}
void CXXNameMangler::mangleNestedName(const TemplateDecl *TD,
                                      const TemplateArgument *TemplateArgs,
                                      unsigned NumTemplateArgs) {
  // <nested-name> ::= N [<CV-qualifiers>] <template-prefix> <template-args> E

  Out << 'N';

  mangleTemplatePrefix(TD);
  mangleTemplateArgs(TemplateArgs, NumTemplateArgs);

  Out << 'E';
}

void CXXNameMangler::mangleLocalName(const Decl *D) {
  // <local-name> := Z <function encoding> E <entity name> [<discriminator>]
  //              := Z <function encoding> E s [<discriminator>]
  // <local-name> := Z <function encoding> E d [ <parameter number> ] 
  //                 _ <entity name>
  // <discriminator> := _ <non-negative number>
  assert(isa<NamedDecl>(D) || isa<BlockDecl>(D));
  const RecordDecl *RD = GetLocalClassDecl(D);
  const DeclContext *DC = getEffectiveDeclContext(RD ? RD : D);

  Out << 'Z';

  if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(DC))
    mangleObjCMethodName(MD);
  else if (const BlockDecl *BD = dyn_cast<BlockDecl>(DC))
    mangleBlockForPrefix(BD);
  else
    mangleFunctionEncoding(cast<FunctionDecl>(DC));

  Out << 'E';

  if (RD) {
    // The parameter number is omitted for the last parameter, 0 for the 
    // second-to-last parameter, 1 for the third-to-last parameter, etc. The 
    // <entity name> will of course contain a <closure-type-name>: Its 
    // numbering will be local to the particular argument in which it appears
    // -- other default arguments do not affect its encoding.
    const CXXRecordDecl *CXXRD = dyn_cast<CXXRecordDecl>(RD);
    if (CXXRD->isLambda()) {
      if (const ParmVarDecl *Parm
              = dyn_cast_or_null<ParmVarDecl>(CXXRD->getLambdaContextDecl())) {
        if (const FunctionDecl *Func
              = dyn_cast<FunctionDecl>(Parm->getDeclContext())) {
          Out << 'd';
          unsigned Num = Func->getNumParams() - Parm->getFunctionScopeIndex();
          if (Num > 1)
            mangleNumber(Num - 2);
          Out << '_';
        }
      }
    }
    
    // Mangle the name relative to the closest enclosing function.
    // equality ok because RD derived from ND above
    if (D == RD)  {
      mangleUnqualifiedName(RD);
    } else if (const BlockDecl *BD = dyn_cast<BlockDecl>(D)) {
      manglePrefix(getEffectiveDeclContext(BD), true /*NoFunction*/);
      mangleUnqualifiedBlock(BD);
    } else {
      const NamedDecl *ND = cast<NamedDecl>(D);
      mangleNestedName(ND, getEffectiveDeclContext(ND), true /*NoFunction*/);
    }
  } else if (const BlockDecl *BD = dyn_cast<BlockDecl>(D)) {
    // Mangle a block in a default parameter; see above explanation for
    // lambdas.
    if (const ParmVarDecl *Parm
            = dyn_cast_or_null<ParmVarDecl>(BD->getBlockManglingContextDecl())) {
      if (const FunctionDecl *Func
            = dyn_cast<FunctionDecl>(Parm->getDeclContext())) {
        Out << 'd';
        unsigned Num = Func->getNumParams() - Parm->getFunctionScopeIndex();
        if (Num > 1)
          mangleNumber(Num - 2);
        Out << '_';
      }
    }

    mangleUnqualifiedBlock(BD);
  } else {
    mangleUnqualifiedName(cast<NamedDecl>(D));
  }

  if (const NamedDecl *ND = dyn_cast<NamedDecl>(RD ? RD : D)) {
    unsigned disc;
    if (Context.getNextDiscriminator(ND, disc)) {
      if (disc < 10)
        Out << '_' << disc;
      else
        Out << "__" << disc << '_';
    }
  }
}

void CXXNameMangler::mangleBlockForPrefix(const BlockDecl *Block) {
  if (GetLocalClassDecl(Block)) {
    mangleLocalName(Block);
    return;
  }
  const DeclContext *DC = getEffectiveDeclContext(Block);
  if (isLocalContainerContext(DC)) {
    mangleLocalName(Block);
    return;
  }
  manglePrefix(getEffectiveDeclContext(Block));
  mangleUnqualifiedBlock(Block);
}

void CXXNameMangler::mangleUnqualifiedBlock(const BlockDecl *Block) {
  if (Decl *Context = Block->getBlockManglingContextDecl()) {
    if ((isa<VarDecl>(Context) || isa<FieldDecl>(Context)) &&
        Context->getDeclContext()->isRecord()) {
      if (const IdentifierInfo *Name
            = cast<NamedDecl>(Context)->getIdentifier()) {
        mangleSourceName(Name);
        Out << 'M';            
      }
    }
  }

  // If we have a block mangling number, use it.
  unsigned Number = Block->getBlockManglingNumber();
  // Otherwise, just make up a number. It doesn't matter what it is because
  // the symbol in question isn't externally visible.
  if (!Number)
    Number = Context.getBlockId(Block, false);
  Out << "Ub";
  if (Number > 0)
    Out << Number - 1;
  Out << '_';
}

void CXXNameMangler::mangleLambda(const CXXRecordDecl *Lambda) {
  // If the context of a closure type is an initializer for a class member 
  // (static or nonstatic), it is encoded in a qualified name with a final 
  // <prefix> of the form:
  //
  //   <data-member-prefix> := <member source-name> M
  //
  // Technically, the data-member-prefix is part of the <prefix>. However,
  // since a closure type will always be mangled with a prefix, it's easier
  // to emit that last part of the prefix here.
  if (Decl *Context = Lambda->getLambdaContextDecl()) {
    if ((isa<VarDecl>(Context) || isa<FieldDecl>(Context)) &&
        Context->getDeclContext()->isRecord()) {
      if (const IdentifierInfo *Name
            = cast<NamedDecl>(Context)->getIdentifier()) {
        mangleSourceName(Name);
        Out << 'M';            
      }
    }
  }

  Out << "Ul";
  const FunctionProtoType *Proto = Lambda->getLambdaTypeInfo()->getType()->
                                   getAs<FunctionProtoType>();
  mangleBareFunctionType(Proto, /*MangleReturnType=*/false);        
  Out << "E";
  
  // The number is omitted for the first closure type with a given 
  // <lambda-sig> in a given context; it is n-2 for the nth closure type 
  // (in lexical order) with that same <lambda-sig> and context.
  //
  // The AST keeps track of the number for us.
  unsigned Number = Lambda->getLambdaManglingNumber();
  assert(Number > 0 && "Lambda should be mangled as an unnamed class");
  if (Number > 1)
    mangleNumber(Number - 2);
  Out << '_';  
}

void CXXNameMangler::manglePrefix(NestedNameSpecifier *qualifier) {
  switch (qualifier->getKind()) {
  case NestedNameSpecifier::Global:
    // nothing
    return;

  case NestedNameSpecifier::Super:
    llvm_unreachable("Can't mangle __super specifier");

  case NestedNameSpecifier::Namespace:
    mangleName(qualifier->getAsNamespace());
    return;

  case NestedNameSpecifier::NamespaceAlias:
    mangleName(qualifier->getAsNamespaceAlias()->getNamespace());
    return;

  case NestedNameSpecifier::TypeSpec:
  case NestedNameSpecifier::TypeSpecWithTemplate:
    manglePrefix(QualType(qualifier->getAsType(), 0));
    return;

  case NestedNameSpecifier::Identifier:
    // Member expressions can have these without prefixes, but that
    // should end up in mangleUnresolvedPrefix instead.
    assert(qualifier->getPrefix());
    manglePrefix(qualifier->getPrefix());

    mangleSourceName(qualifier->getAsIdentifier());
    return;
  }

  llvm_unreachable("unexpected nested name specifier");
}

void CXXNameMangler::manglePrefix(const DeclContext *DC, bool NoFunction) {
  //  <prefix> ::= <prefix> <unqualified-name>
  //           ::= <template-prefix> <template-args>
  //           ::= <template-param>
  //           ::= # empty
  //           ::= <substitution>

  DC = IgnoreLinkageSpecDecls(DC);

  if (DC->isTranslationUnit())
    return;

  if (NoFunction && isLocalContainerContext(DC))
    return;

  assert(!isLocalContainerContext(DC));

  const NamedDecl *ND = cast<NamedDecl>(DC);  
  if (mangleSubstitution(ND))
    return;
  
  // Check if we have a template.
  const TemplateArgumentList *TemplateArgs = nullptr;
  if (const TemplateDecl *TD = isTemplate(ND, TemplateArgs)) {
    mangleTemplatePrefix(TD);
    mangleTemplateArgs(*TemplateArgs);
  } else {
    manglePrefix(getEffectiveDeclContext(ND), NoFunction);
    mangleUnqualifiedName(ND);
  }

  addSubstitution(ND);
}

void CXXNameMangler::mangleTemplatePrefix(TemplateName Template) {
  // <template-prefix> ::= <prefix> <template unqualified-name>
  //                   ::= <template-param>
  //                   ::= <substitution>
  if (TemplateDecl *TD = Template.getAsTemplateDecl())
    return mangleTemplatePrefix(TD);

  if (QualifiedTemplateName *Qualified = Template.getAsQualifiedTemplateName())
    manglePrefix(Qualified->getQualifier());
  
  if (OverloadedTemplateStorage *Overloaded
                                      = Template.getAsOverloadedTemplate()) {
    mangleUnqualifiedName(nullptr, (*Overloaded->begin())->getDeclName(),
                          UnknownArity);
    return;
  }
   
  DependentTemplateName *Dependent = Template.getAsDependentTemplateName();
  assert(Dependent && "Unknown template name kind?");
  if (NestedNameSpecifier *Qualifier = Dependent->getQualifier())
    manglePrefix(Qualifier);
  mangleUnscopedTemplateName(Template);
}

void CXXNameMangler::mangleTemplatePrefix(const TemplateDecl *ND,
                                          bool NoFunction) {
  // <template-prefix> ::= <prefix> <template unqualified-name>
  //                   ::= <template-param>
  //                   ::= <substitution>
  // <template-template-param> ::= <template-param>
  //                               <substitution>

  if (mangleSubstitution(ND))
    return;

  // <template-template-param> ::= <template-param>
  if (const auto *TTP = dyn_cast<TemplateTemplateParmDecl>(ND)) {
    mangleTemplateParameter(TTP->getIndex());
  } else {
    manglePrefix(getEffectiveDeclContext(ND), NoFunction);
    mangleUnqualifiedName(ND->getTemplatedDecl());
  }

  addSubstitution(ND);
}

/// Mangles a template name under the production <type>.  Required for
/// template template arguments.
///   <type> ::= <class-enum-type>
///          ::= <template-param>
///          ::= <substitution>
void CXXNameMangler::mangleType(TemplateName TN) {
  if (mangleSubstitution(TN))
    return;

  TemplateDecl *TD = nullptr;

  switch (TN.getKind()) {
  case TemplateName::QualifiedTemplate:
    TD = TN.getAsQualifiedTemplateName()->getTemplateDecl();
    goto HaveDecl;

  case TemplateName::Template:
    TD = TN.getAsTemplateDecl();
    goto HaveDecl;

  HaveDecl:
    if (isa<TemplateTemplateParmDecl>(TD))
      mangleTemplateParameter(cast<TemplateTemplateParmDecl>(TD)->getIndex());
    else
      mangleName(TD);
    break;

  case TemplateName::OverloadedTemplate:
    llvm_unreachable("can't mangle an overloaded template name as a <type>");

  case TemplateName::DependentTemplate: {
    const DependentTemplateName *Dependent = TN.getAsDependentTemplateName();
    assert(Dependent->isIdentifier());

    // <class-enum-type> ::= <name>
    // <name> ::= <nested-name>
    mangleUnresolvedPrefix(Dependent->getQualifier());
    mangleSourceName(Dependent->getIdentifier());
    break;
  }

  case TemplateName::SubstTemplateTemplateParm: {
    // Substituted template parameters are mangled as the substituted
    // template.  This will check for the substitution twice, which is
    // fine, but we have to return early so that we don't try to *add*
    // the substitution twice.
    SubstTemplateTemplateParmStorage *subst
      = TN.getAsSubstTemplateTemplateParm();
    mangleType(subst->getReplacement());
    return;
  }

  case TemplateName::SubstTemplateTemplateParmPack: {
    // FIXME: not clear how to mangle this!
    // template <template <class> class T...> class A {
    //   template <template <class> class U...> void foo(B<T,U> x...);
    // };
    Out << "_SUBSTPACK_";
    break;
  }
  }

  addSubstitution(TN);
}

bool CXXNameMangler::mangleUnresolvedTypeOrSimpleId(QualType Ty,
                                                    StringRef Prefix) {
  // Only certain other types are valid as prefixes;  enumerate them.
  switch (Ty->getTypeClass()) {
  case Type::Builtin:
  case Type::Complex:
  case Type::Adjusted:
  case Type::Decayed:
  case Type::Pointer:
  case Type::BlockPointer:
  case Type::LValueReference:
  case Type::RValueReference:
  case Type::MemberPointer:
  case Type::ConstantArray:
  case Type::IncompleteArray:
  case Type::VariableArray:
  case Type::DependentSizedArray:
  case Type::DependentSizedExtVector:
  case Type::Vector:
  case Type::ExtVector:
  case Type::FunctionProto:
  case Type::FunctionNoProto:
  case Type::Paren:
  case Type::Attributed:
  case Type::Auto:
  case Type::PackExpansion:
  case Type::ObjCObject:
  case Type::ObjCInterface:
  case Type::ObjCObjectPointer:
  case Type::Atomic:
    llvm_unreachable("type is illegal as a nested name specifier");

  case Type::SubstTemplateTypeParmPack:
    // FIXME: not clear how to mangle this!
    // template <class T...> class A {
    //   template <class U...> void foo(decltype(T::foo(U())) x...);
    // };
    Out << "_SUBSTPACK_";
    break;

  // <unresolved-type> ::= <template-param>
  //                   ::= <decltype>
  //                   ::= <template-template-param> <template-args>
  // (this last is not official yet)
  case Type::TypeOfExpr:
  case Type::TypeOf:
  case Type::Decltype:
  case Type::TemplateTypeParm:
  case Type::UnaryTransform:
  case Type::SubstTemplateTypeParm:
  unresolvedType:
    // Some callers want a prefix before the mangled type.
    Out << Prefix;

    // This seems to do everything we want.  It's not really
    // sanctioned for a substituted template parameter, though.
    mangleType(Ty);

    // We never want to print 'E' directly after an unresolved-type,
    // so we return directly.
    return true;

  case Type::Typedef:
    mangleSourceName(cast<TypedefType>(Ty)->getDecl()->getIdentifier());
    break;

  case Type::UnresolvedUsing:
    mangleSourceName(
        cast<UnresolvedUsingType>(Ty)->getDecl()->getIdentifier());
    break;

  case Type::Enum:
  case Type::Record:
    mangleSourceName(cast<TagType>(Ty)->getDecl()->getIdentifier());
    break;

  case Type::TemplateSpecialization: {
    const TemplateSpecializationType *TST =
        cast<TemplateSpecializationType>(Ty);
    TemplateName TN = TST->getTemplateName();
    switch (TN.getKind()) {
    case TemplateName::Template:
    case TemplateName::QualifiedTemplate: {
      TemplateDecl *TD = TN.getAsTemplateDecl();

      // If the base is a template template parameter, this is an
      // unresolved type.
      assert(TD && "no template for template specialization type");
      if (isa<TemplateTemplateParmDecl>(TD))
        goto unresolvedType;

      mangleSourceName(TD->getIdentifier());
      break;
    }

    case TemplateName::OverloadedTemplate:
    case TemplateName::DependentTemplate:
      llvm_unreachable("invalid base for a template specialization type");

    case TemplateName::SubstTemplateTemplateParm: {
      SubstTemplateTemplateParmStorage *subst =
          TN.getAsSubstTemplateTemplateParm();
      mangleExistingSubstitution(subst->getReplacement());
      break;
    }

    case TemplateName::SubstTemplateTemplateParmPack: {
      // FIXME: not clear how to mangle this!
      // template <template <class U> class T...> class A {
      //   template <class U...> void foo(decltype(T<U>::foo) x...);
      // };
      Out << "_SUBSTPACK_";
      break;
    }
    }

    mangleTemplateArgs(TST->getArgs(), TST->getNumArgs());
    break;
  }

  case Type::InjectedClassName:
    mangleSourceName(
        cast<InjectedClassNameType>(Ty)->getDecl()->getIdentifier());
    break;

  case Type::DependentName:
    mangleSourceName(cast<DependentNameType>(Ty)->getIdentifier());
    break;

  case Type::DependentTemplateSpecialization: {
    const DependentTemplateSpecializationType *DTST =
        cast<DependentTemplateSpecializationType>(Ty);
    mangleSourceName(DTST->getIdentifier());
    mangleTemplateArgs(DTST->getArgs(), DTST->getNumArgs());
    break;
  }

  case Type::Elaborated:
    return mangleUnresolvedTypeOrSimpleId(
        cast<ElaboratedType>(Ty)->getNamedType(), Prefix);
  }

  return false;
}

void CXXNameMangler::mangleOperatorName(DeclarationName Name, unsigned Arity) {
  switch (Name.getNameKind()) {
  case DeclarationName::CXXConstructorName:
  case DeclarationName::CXXDestructorName:
  case DeclarationName::CXXUsingDirective:
  case DeclarationName::Identifier:
  case DeclarationName::ObjCMultiArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCZeroArgSelector:
    llvm_unreachable("Not an operator name");

  case DeclarationName::CXXConversionFunctionName:
    // <operator-name> ::= cv <type>    # (cast)
    Out << "cv";
    mangleType(Name.getCXXNameType());
    break;

  case DeclarationName::CXXLiteralOperatorName:
    Out << "li";
    mangleSourceName(Name.getCXXLiteralIdentifier());
    return;

  case DeclarationName::CXXOperatorName:
    mangleOperatorName(Name.getCXXOverloadedOperator(), Arity);
    break;
  }
}



void
CXXNameMangler::mangleOperatorName(OverloadedOperatorKind OO, unsigned Arity) {
  switch (OO) {
  // <operator-name> ::= nw     # new
  case OO_New: Out << "nw"; break;
  //              ::= na        # new[]
  case OO_Array_New: Out << "na"; break;
  //              ::= dl        # delete
  case OO_Delete: Out << "dl"; break;
  //              ::= da        # delete[]
  case OO_Array_Delete: Out << "da"; break;
  //              ::= ps        # + (unary)
  //              ::= pl        # + (binary or unknown)
  case OO_Plus:
    Out << (Arity == 1? "ps" : "pl"); break;
  //              ::= ng        # - (unary)
  //              ::= mi        # - (binary or unknown)
  case OO_Minus:
    Out << (Arity == 1? "ng" : "mi"); break;
  //              ::= ad        # & (unary)
  //              ::= an        # & (binary or unknown)
  case OO_Amp:
    Out << (Arity == 1? "ad" : "an"); break;
  //              ::= de        # * (unary)
  //              ::= ml        # * (binary or unknown)
  case OO_Star:
    // Use binary when unknown.
    Out << (Arity == 1? "de" : "ml"); break;
  //              ::= co        # ~
  case OO_Tilde: Out << "co"; break;
  //              ::= dv        # /
  case OO_Slash: Out << "dv"; break;
  //              ::= rm        # %
  case OO_Percent: Out << "rm"; break;
  //              ::= or        # |
  case OO_Pipe: Out << "or"; break;
  //              ::= eo        # ^
  case OO_Caret: Out << "eo"; break;
  //              ::= aS        # =
  case OO_Equal: Out << "aS"; break;
  //              ::= pL        # +=
  case OO_PlusEqual: Out << "pL"; break;
  //              ::= mI        # -=
  case OO_MinusEqual: Out << "mI"; break;
  //              ::= mL        # *=
  case OO_StarEqual: Out << "mL"; break;
  //              ::= dV        # /=
  case OO_SlashEqual: Out << "dV"; break;
  //              ::= rM        # %=
  case OO_PercentEqual: Out << "rM"; break;
  //              ::= aN        # &=
  case OO_AmpEqual: Out << "aN"; break;
  //              ::= oR        # |=
  case OO_PipeEqual: Out << "oR"; break;
  //              ::= eO        # ^=
  case OO_CaretEqual: Out << "eO"; break;
  //              ::= ls        # <<
  case OO_LessLess: Out << "ls"; break;
  //              ::= rs        # >>
  case OO_GreaterGreater: Out << "rs"; break;
  //              ::= lS        # <<=
  case OO_LessLessEqual: Out << "lS"; break;
  //              ::= rS        # >>=
  case OO_GreaterGreaterEqual: Out << "rS"; break;
  //              ::= eq        # ==
  case OO_EqualEqual: Out << "eq"; break;
  //              ::= ne        # !=
  case OO_ExclaimEqual: Out << "ne"; break;
  //              ::= lt        # <
  case OO_Less: Out << "lt"; break;
  //              ::= gt        # >
  case OO_Greater: Out << "gt"; break;
  //              ::= le        # <=
  case OO_LessEqual: Out << "le"; break;
  //              ::= ge        # >=
  case OO_GreaterEqual: Out << "ge"; break;
  //              ::= nt        # !
  case OO_Exclaim: Out << "nt"; break;
  //              ::= aa        # &&
  case OO_AmpAmp: Out << "aa"; break;
  //              ::= oo        # ||
  case OO_PipePipe: Out << "oo"; break;
  //              ::= pp        # ++
  case OO_PlusPlus: Out << "pp"; break;
  //              ::= mm        # --
  case OO_MinusMinus: Out << "mm"; break;
  //              ::= cm        # ,
  case OO_Comma: Out << "cm"; break;
  //              ::= pm        # ->*
  case OO_ArrowStar: Out << "pm"; break;
  //              ::= pt        # ->
  case OO_Arrow: Out << "pt"; break;
  //              ::= cl        # ()
  case OO_Call: Out << "cl"; break;
  //              ::= ix        # []
  case OO_Subscript: Out << "ix"; break;

  //              ::= qu        # ?
  // The conditional operator can't be overloaded, but we still handle it when
  // mangling expressions.
  case OO_Conditional: Out << "qu"; break;
  // Proposal on cxx-abi-dev, 2015-10-21.
  //              ::= aw        # co_await
  case OO_Coawait: Out << "aw"; break;

  case OO_None:
  case NUM_OVERLOADED_OPERATORS:
    llvm_unreachable("Not an overloaded operator");
  }
}

void CXXNameMangler::mangleQualifiers(Qualifiers Quals) {
  // <CV-qualifiers> ::= [r] [V] [K]    # restrict (C99), volatile, const
  if (Quals.hasRestrict())
    Out << 'r';
  if (Quals.hasVolatile())
    Out << 'V';
  if (Quals.hasConst())
    Out << 'K';

  if (Quals.hasAddressSpace()) {
    // Address space extension:
    //
    //   <type> ::= U <target-addrspace>
    //   <type> ::= U <OpenCL-addrspace>
    //   <type> ::= U <CUDA-addrspace>

    SmallString<64> ASString;
    unsigned AS = Quals.getAddressSpace();

    if (Context.getASTContext().addressSpaceMapManglingFor(AS)) {
      //  <target-addrspace> ::= "AS" <address-space-number>
      unsigned TargetAS = Context.getASTContext().getTargetAddressSpace(AS);
      ASString = "AS" + llvm::utostr_32(TargetAS);
    } else {
      switch (AS) {
      default: llvm_unreachable("Not a language specific address space");
      //  <OpenCL-addrspace> ::= "CL" [ "global" | "local" | "constant" ]
      case LangAS::opencl_global:   ASString = "CLglobal";   break;
      case LangAS::opencl_local:    ASString = "CLlocal";    break;
      case LangAS::opencl_constant: ASString = "CLconstant"; break;
      //  <CUDA-addrspace> ::= "CU" [ "device" | "constant" | "shared" ]
      case LangAS::cuda_device:     ASString = "CUdevice";   break;
      case LangAS::cuda_constant:   ASString = "CUconstant"; break;
      case LangAS::cuda_shared:     ASString = "CUshared";   break;
      }
    }
    Out << 'U' << ASString.size() << ASString;
  }
  
  StringRef LifetimeName;
  switch (Quals.getObjCLifetime()) {
  // Objective-C ARC Extension:
  //
  //   <type> ::= U "__strong"
  //   <type> ::= U "__weak"
  //   <type> ::= U "__autoreleasing"
  case Qualifiers::OCL_None:
    break;
    
  case Qualifiers::OCL_Weak:
    LifetimeName = "__weak";
    break;
    
  case Qualifiers::OCL_Strong:
    LifetimeName = "__strong";
    break;
    
  case Qualifiers::OCL_Autoreleasing:
    LifetimeName = "__autoreleasing";
    break;
    
  case Qualifiers::OCL_ExplicitNone:
    // The __unsafe_unretained qualifier is *not* mangled, so that
    // __unsafe_unretained types in ARC produce the same manglings as the
    // equivalent (but, naturally, unqualified) types in non-ARC, providing
    // better ABI compatibility.
    //
    // It's safe to do this because unqualified 'id' won't show up
    // in any type signatures that need to be mangled.
    break;
  }
  if (!LifetimeName.empty())
    Out << 'U' << LifetimeName.size() << LifetimeName;
}

void CXXNameMangler::mangleRefQualifier(RefQualifierKind RefQualifier) {
  // <ref-qualifier> ::= R                # lvalue reference
  //                 ::= O                # rvalue-reference
  switch (RefQualifier) {
  case RQ_None:
    break;
      
  case RQ_LValue:
    Out << 'R';
    break;
      
  case RQ_RValue:
    Out << 'O';
    break;
  }
}

void CXXNameMangler::mangleObjCMethodName(const ObjCMethodDecl *MD) {
  Context.mangleObjCMethodName(MD, Out);
}

static bool isTypeSubstitutable(Qualifiers Quals, const Type *Ty) {
  if (Quals)
    return true;
  if (Ty->isSpecificBuiltinType(BuiltinType::ObjCSel))
    return true;
  if (Ty->isOpenCLSpecificType())
    return true;
  if (Ty->isBuiltinType())
    return false;

  return true;
}

void CXXNameMangler::mangleType(QualType T) {
  // If our type is instantiation-dependent but not dependent, we mangle
  // it as it was written in the source, removing any top-level sugar. 
  // Otherwise, use the canonical type.
  //
  // FIXME: This is an approximation of the instantiation-dependent name 
  // mangling rules, since we should really be using the type as written and
  // augmented via semantic analysis (i.e., with implicit conversions and
  // default template arguments) for any instantiation-dependent type. 
  // Unfortunately, that requires several changes to our AST:
  //   - Instantiation-dependent TemplateSpecializationTypes will need to be 
  //     uniqued, so that we can handle substitutions properly
  //   - Default template arguments will need to be represented in the
  //     TemplateSpecializationType, since they need to be mangled even though
  //     they aren't written.
  //   - Conversions on non-type template arguments need to be expressed, since
  //     they can affect the mangling of sizeof/alignof.
  if (!T->isInstantiationDependentType() || T->isDependentType())
    T = T.getCanonicalType();
  else {
    // Desugar any types that are purely sugar.
    do {
      // Don't desugar through template specialization types that aren't
      // type aliases. We need to mangle the template arguments as written.
      if (const TemplateSpecializationType *TST 
                                      = dyn_cast<TemplateSpecializationType>(T))
        if (!TST->isTypeAlias())
          break;

      QualType Desugared 
        = T.getSingleStepDesugaredType(Context.getASTContext());
      if (Desugared == T)
        break;
      
      T = Desugared;
    } while (true);
  }
  SplitQualType split = T.split();
  Qualifiers quals = split.Quals;
  const Type *ty = split.Ty;

  bool isSubstitutable = isTypeSubstitutable(quals, ty);
  if (isSubstitutable && mangleSubstitution(T))
    return;

  // If we're mangling a qualified array type, push the qualifiers to
  // the element type.
  if (quals && isa<ArrayType>(T)) {
    ty = Context.getASTContext().getAsArrayType(T);
    quals = Qualifiers();

    // Note that we don't update T: we want to add the
    // substitution at the original type.
  }

  if (quals) {
    mangleQualifiers(quals);
    // Recurse:  even if the qualified type isn't yet substitutable,
    // the unqualified type might be.
    mangleType(QualType(ty, 0));
  } else {
    switch (ty->getTypeClass()) {
#define ABSTRACT_TYPE(CLASS, PARENT)
#define NON_CANONICAL_TYPE(CLASS, PARENT) \
    case Type::CLASS: \
      llvm_unreachable("can't mangle non-canonical type " #CLASS "Type"); \
      return;
#define TYPE(CLASS, PARENT) \
    case Type::CLASS: \
      mangleType(static_cast<const CLASS##Type*>(ty)); \
      break;
#include "clang/AST/TypeNodes.def"
    }
  }

  // Add the substitution.
  if (isSubstitutable)
    addSubstitution(T);
}

void CXXNameMangler::mangleNameOrStandardSubstitution(const NamedDecl *ND) {
  if (!mangleStandardSubstitution(ND))
    mangleName(ND);
}

void CXXNameMangler::mangleType(const BuiltinType *T) {
  //  <type>         ::= <builtin-type>
  //  <builtin-type> ::= v  # void
  //                 ::= w  # wchar_t
  //                 ::= b  # bool
  //                 ::= c  # char
  //                 ::= a  # signed char
  //                 ::= h  # unsigned char
  //                 ::= s  # short
  //                 ::= t  # unsigned short
  //                 ::= i  # int
  //                 ::= j  # unsigned int
  //                 ::= l  # long
  //                 ::= m  # unsigned long
  //                 ::= x  # long long, __int64
  //                 ::= y  # unsigned long long, __int64
  //                 ::= n  # __int128
  //                 ::= o  # unsigned __int128
  //                 ::= f  # float
  //                 ::= d  # double
  //                 ::= e  # long double, __float80
  // UNSUPPORTED:    ::= g  # __float128
  // UNSUPPORTED:    ::= Dd # IEEE 754r decimal floating point (64 bits)
  // UNSUPPORTED:    ::= De # IEEE 754r decimal floating point (128 bits)
  // UNSUPPORTED:    ::= Df # IEEE 754r decimal floating point (32 bits)
  //                 ::= Dh # IEEE 754r half-precision floating point (16 bits)
  //                 ::= Di # char32_t
  //                 ::= Ds # char16_t
  //                 ::= Dn # std::nullptr_t (i.e., decltype(nullptr))
  //                 ::= u <source-name>    # vendor extended type
  switch (T->getKind()) {
  case BuiltinType::Void:
    Out << 'v';
    break;
  case BuiltinType::Bool:
    Out << 'b';
    break;
  case BuiltinType::Char_U:
  case BuiltinType::Char_S:
    Out << 'c';
    break;
  case BuiltinType::UChar:
    Out << 'h';
    break;
  case BuiltinType::UShort:
    Out << 't';
    break;
  case BuiltinType::UInt:
    Out << 'j';
    break;
  case BuiltinType::ULong:
    Out << 'm';
    break;
  case BuiltinType::ULongLong:
    Out << 'y';
    break;
  case BuiltinType::UInt128:
    Out << 'o';
    break;
  case BuiltinType::SChar:
    Out << 'a';
    break;
  case BuiltinType::WChar_S:
  case BuiltinType::WChar_U:
    Out << 'w';
    break;
  case BuiltinType::Char16:
    Out << "Ds";
    break;
  case BuiltinType::Char32:
    Out << "Di";
    break;
  case BuiltinType::Short:
    Out << 's';
    break;
  case BuiltinType::Int:
    Out << 'i';
    break;
  case BuiltinType::Long:
    Out << 'l';
    break;
  case BuiltinType::LongLong:
    Out << 'x';
    break;
  case BuiltinType::Int128:
    Out << 'n';
    break;
  case BuiltinType::Half:
    Out << "Dh";
    break;
  case BuiltinType::Float:
    Out << 'f';
    break;
  case BuiltinType::Double:
    Out << 'd';
    break;
  case BuiltinType::LongDouble:
    Out << (getASTContext().getTargetInfo().useFloat128ManglingForLongDouble()
                ? 'g'
                : 'e');
    break;
  case BuiltinType::NullPtr:
    Out << "Dn";
    break;

#define BUILTIN_TYPE(Id, SingletonId)
#define PLACEHOLDER_TYPE(Id, SingletonId) \
  case BuiltinType::Id:
#include "clang/AST/BuiltinTypes.def"
  case BuiltinType::Dependent:
    llvm_unreachable("mangling a placeholder type");
  case BuiltinType::ObjCId:
    Out << "11objc_object";
    break;
  case BuiltinType::ObjCClass:
    Out << "10objc_class";
    break;
  case BuiltinType::ObjCSel:
    Out << "13objc_selector";
    break;
  case BuiltinType::OCLImage1d:
    Out << "11ocl_image1d";
    break;
  case BuiltinType::OCLImage1dArray:
    Out << "16ocl_image1darray";
    break;
  case BuiltinType::OCLImage1dBuffer:
    Out << "17ocl_image1dbuffer";
    break;
  case BuiltinType::OCLImage2d:
    Out << "11ocl_image2d";
    break;
  case BuiltinType::OCLImage2dArray:
    Out << "16ocl_image2darray";
    break;
  case BuiltinType::OCLImage2dDepth:
    Out << "16ocl_image2ddepth";
    break;
  case BuiltinType::OCLImage2dArrayDepth:
    Out << "21ocl_image2darraydepth";
    break;
  case BuiltinType::OCLImage2dMSAA:
    Out << "15ocl_image2dmsaa";
    break;
  case BuiltinType::OCLImage2dArrayMSAA:
    Out << "20ocl_image2darraymsaa";
    break;
  case BuiltinType::OCLImage2dMSAADepth:
    Out << "20ocl_image2dmsaadepth";
    break;
  case BuiltinType::OCLImage2dArrayMSAADepth:
    Out << "35ocl_image2darraymsaadepth";
    break;
  case BuiltinType::OCLImage3d:
    Out << "11ocl_image3d";
    break;
  case BuiltinType::OCLSampler:
    Out << "11ocl_sampler";
    break;
  case BuiltinType::OCLEvent:
    Out << "9ocl_event";
    break;
  case BuiltinType::OCLClkEvent:
    Out << "12ocl_clkevent";
    break;
  case BuiltinType::OCLQueue:
    Out << "9ocl_queue";
    break;
  case BuiltinType::OCLNDRange:
    Out << "11ocl_ndrange";
    break;
  case BuiltinType::OCLReserveID:
    Out << "13ocl_reserveid";
    break;
  }
}

// <type>          ::= <function-type>
// <function-type> ::= [<CV-qualifiers>] F [Y]
//                      <bare-function-type> [<ref-qualifier>] E
void CXXNameMangler::mangleType(const FunctionProtoType *T) {
  // Mangle CV-qualifiers, if present.  These are 'this' qualifiers,
  // e.g. "const" in "int (A::*)() const".
  mangleQualifiers(Qualifiers::fromCVRMask(T->getTypeQuals()));

  Out << 'F';

  // FIXME: We don't have enough information in the AST to produce the 'Y'
  // encoding for extern "C" function types.
  mangleBareFunctionType(T, /*MangleReturnType=*/true);

  // Mangle the ref-qualifier, if present.
  mangleRefQualifier(T->getRefQualifier());

  Out << 'E';
}

void CXXNameMangler::mangleType(const FunctionNoProtoType *T) {
  // Function types without prototypes can arise when mangling a function type
  // within an overloadable function in C. We mangle these as the absence of any
  // parameter types (not even an empty parameter list).
  Out << 'F';

  FunctionTypeDepthState saved = FunctionTypeDepth.push();

  FunctionTypeDepth.enterResultType();
  mangleType(T->getReturnType());
  FunctionTypeDepth.leaveResultType();

  FunctionTypeDepth.pop(saved);
  Out << 'E';
}

void CXXNameMangler::mangleBareFunctionType(const FunctionType *T,
                                            bool MangleReturnType) {
  // We should never be mangling something without a prototype.
  const FunctionProtoType *Proto = cast<FunctionProtoType>(T);

  // Record that we're in a function type.  See mangleFunctionParam
  // for details on what we're trying to achieve here.
  FunctionTypeDepthState saved = FunctionTypeDepth.push();

  // <bare-function-type> ::= <signature type>+
  if (MangleReturnType) {
    FunctionTypeDepth.enterResultType();
    mangleType(Proto->getReturnType());
    FunctionTypeDepth.leaveResultType();
  }

  if (Proto->getNumParams() == 0 && !Proto->isVariadic()) {
    //   <builtin-type> ::= v   # void
    Out << 'v';

    FunctionTypeDepth.pop(saved);
    return;
  }

  for (const auto &Arg : Proto->param_types())
    mangleType(Context.getASTContext().getSignatureParameterType(Arg));

  FunctionTypeDepth.pop(saved);

  // <builtin-type>      ::= z  # ellipsis
  if (Proto->isVariadic())
    Out << 'z';
}

// <type>            ::= <class-enum-type>
// <class-enum-type> ::= <name>
void CXXNameMangler::mangleType(const UnresolvedUsingType *T) {
  mangleName(T->getDecl());
}

// <type>            ::= <class-enum-type>
// <class-enum-type> ::= <name>
void CXXNameMangler::mangleType(const EnumType *T) {
  mangleType(static_cast<const TagType*>(T));
}
void CXXNameMangler::mangleType(const RecordType *T) {
  mangleType(static_cast<const TagType*>(T));
}
void CXXNameMangler::mangleType(const TagType *T) {
  mangleName(T->getDecl());
}

// <type>       ::= <array-type>
// <array-type> ::= A <positive dimension number> _ <element type>
//              ::= A [<dimension expression>] _ <element type>
void CXXNameMangler::mangleType(const ConstantArrayType *T) {
  Out << 'A' << T->getSize() << '_';
  mangleType(T->getElementType());
}
void CXXNameMangler::mangleType(const VariableArrayType *T) {
  Out << 'A';
  // decayed vla types (size 0) will just be skipped.
  if (T->getSizeExpr())
    mangleExpression(T->getSizeExpr());
  Out << '_';
  mangleType(T->getElementType());
}
void CXXNameMangler::mangleType(const DependentSizedArrayType *T) {
  Out << 'A';
  mangleExpression(T->getSizeExpr());
  Out << '_';
  mangleType(T->getElementType());
}
void CXXNameMangler::mangleType(const IncompleteArrayType *T) {
  Out << "A_";
  mangleType(T->getElementType());
}

// <type>                   ::= <pointer-to-member-type>
// <pointer-to-member-type> ::= M <class type> <member type>
void CXXNameMangler::mangleType(const MemberPointerType *T) {
  Out << 'M';
  mangleType(QualType(T->getClass(), 0));
  QualType PointeeType = T->getPointeeType();
  if (const FunctionProtoType *FPT = dyn_cast<FunctionProtoType>(PointeeType)) {
    mangleType(FPT);
    
    // Itanium C++ ABI 5.1.8:
    //
    //   The type of a non-static member function is considered to be different,
    //   for the purposes of substitution, from the type of a namespace-scope or
    //   static member function whose type appears similar. The types of two
    //   non-static member functions are considered to be different, for the
    //   purposes of substitution, if the functions are members of different
    //   classes. In other words, for the purposes of substitution, the class of 
    //   which the function is a member is considered part of the type of 
    //   function.

    // Given that we already substitute member function pointers as a
    // whole, the net effect of this rule is just to unconditionally
    // suppress substitution on the function type in a member pointer.
    // We increment the SeqID here to emulate adding an entry to the
    // substitution table.
    ++SeqID;
  } else
    mangleType(PointeeType);
}

// <type>           ::= <template-param>
void CXXNameMangler::mangleType(const TemplateTypeParmType *T) {
  mangleTemplateParameter(T->getIndex());
}

// <type>           ::= <template-param>
void CXXNameMangler::mangleType(const SubstTemplateTypeParmPackType *T) {
  // FIXME: not clear how to mangle this!
  // template <class T...> class A {
  //   template <class U...> void foo(T(*)(U) x...);
  // };
  Out << "_SUBSTPACK_";
}

// <type> ::= P <type>   # pointer-to
void CXXNameMangler::mangleType(const PointerType *T) {
  Out << 'P';
  mangleType(T->getPointeeType());
}
void CXXNameMangler::mangleType(const ObjCObjectPointerType *T) {
  Out << 'P';
  mangleType(T->getPointeeType());
}

// <type> ::= R <type>   # reference-to
void CXXNameMangler::mangleType(const LValueReferenceType *T) {
  Out << 'R';
  mangleType(T->getPointeeType());
}

// <type> ::= O <type>   # rvalue reference-to (C++0x)
void CXXNameMangler::mangleType(const RValueReferenceType *T) {
  Out << 'O';
  mangleType(T->getPointeeType());
}

// <type> ::= C <type>   # complex pair (C 2000)
void CXXNameMangler::mangleType(const ComplexType *T) {
  Out << 'C';
  mangleType(T->getElementType());
}

// ARM's ABI for Neon vector types specifies that they should be mangled as
// if they are structs (to match ARM's initial implementation).  The
// vector type must be one of the special types predefined by ARM.
void CXXNameMangler::mangleNeonVectorType(const VectorType *T) {
  QualType EltType = T->getElementType();
  assert(EltType->isBuiltinType() && "Neon vector element not a BuiltinType");
  const char *EltName = nullptr;
  if (T->getVectorKind() == VectorType::NeonPolyVector) {
    switch (cast<BuiltinType>(EltType)->getKind()) {
    case BuiltinType::SChar:
    case BuiltinType::UChar:
      EltName = "poly8_t";
      break;
    case BuiltinType::Short:
    case BuiltinType::UShort:
      EltName = "poly16_t";
      break;
    case BuiltinType::ULongLong:
      EltName = "poly64_t";
      break;
    default: llvm_unreachable("unexpected Neon polynomial vector element type");
    }
  } else {
    switch (cast<BuiltinType>(EltType)->getKind()) {
    case BuiltinType::SChar:     EltName = "int8_t"; break;
    case BuiltinType::UChar:     EltName = "uint8_t"; break;
    case BuiltinType::Short:     EltName = "int16_t"; break;
    case BuiltinType::UShort:    EltName = "uint16_t"; break;
    case BuiltinType::Int:       EltName = "int32_t"; break;
    case BuiltinType::UInt:      EltName = "uint32_t"; break;
    case BuiltinType::LongLong:  EltName = "int64_t"; break;
    case BuiltinType::ULongLong: EltName = "uint64_t"; break;
    case BuiltinType::Double:    EltName = "float64_t"; break;
    case BuiltinType::Float:     EltName = "float32_t"; break;
    case BuiltinType::Half:      EltName = "float16_t";break;
    default:
      llvm_unreachable("unexpected Neon vector element type");
    }
  }
  const char *BaseName = nullptr;
  unsigned BitSize = (T->getNumElements() *
                      getASTContext().getTypeSize(EltType));
  if (BitSize == 64)
    BaseName = "__simd64_";
  else {
    assert(BitSize == 128 && "Neon vector type not 64 or 128 bits");
    BaseName = "__simd128_";
  }
  Out << strlen(BaseName) + strlen(EltName);
  Out << BaseName << EltName;
}

static StringRef mangleAArch64VectorBase(const BuiltinType *EltType) {
  switch (EltType->getKind()) {
  case BuiltinType::SChar:
    return "Int8";
  case BuiltinType::Short:
    return "Int16";
  case BuiltinType::Int:
    return "Int32";
  case BuiltinType::Long:
  case BuiltinType::LongLong:
    return "Int64";
  case BuiltinType::UChar:
    return "Uint8";
  case BuiltinType::UShort:
    return "Uint16";
  case BuiltinType::UInt:
    return "Uint32";
  case BuiltinType::ULong:
  case BuiltinType::ULongLong:
    return "Uint64";
  case BuiltinType::Half:
    return "Float16";
  case BuiltinType::Float:
    return "Float32";
  case BuiltinType::Double:
    return "Float64";
  default:
    llvm_unreachable("Unexpected vector element base type");
  }
}

// AArch64's ABI for Neon vector types specifies that they should be mangled as
// the equivalent internal name. The vector type must be one of the special
// types predefined by ARM.
void CXXNameMangler::mangleAArch64NeonVectorType(const VectorType *T) {
  QualType EltType = T->getElementType();
  assert(EltType->isBuiltinType() && "Neon vector element not a BuiltinType");
  unsigned BitSize =
      (T->getNumElements() * getASTContext().getTypeSize(EltType));
  (void)BitSize; // Silence warning.

  assert((BitSize == 64 || BitSize == 128) &&
         "Neon vector type not 64 or 128 bits");

  StringRef EltName;
  if (T->getVectorKind() == VectorType::NeonPolyVector) {
    switch (cast<BuiltinType>(EltType)->getKind()) {
    case BuiltinType::UChar:
      EltName = "Poly8";
      break;
    case BuiltinType::UShort:
      EltName = "Poly16";
      break;
    case BuiltinType::ULong:
    case BuiltinType::ULongLong:
      EltName = "Poly64";
      break;
    default:
      llvm_unreachable("unexpected Neon polynomial vector element type");
    }
  } else
    EltName = mangleAArch64VectorBase(cast<BuiltinType>(EltType));

  std::string TypeName =
      ("__" + EltName + "x" + llvm::utostr(T->getNumElements()) + "_t").str();
  Out << TypeName.length() << TypeName;
}

// GNU extension: vector types
// <type>                  ::= <vector-type>
// <vector-type>           ::= Dv <positive dimension number> _
//                                    <extended element type>
//                         ::= Dv [<dimension expression>] _ <element type>
// <extended element type> ::= <element type>
//                         ::= p # AltiVec vector pixel
//                         ::= b # Altivec vector bool
void CXXNameMangler::mangleType(const VectorType *T) {
  if ((T->getVectorKind() == VectorType::NeonVector ||
       T->getVectorKind() == VectorType::NeonPolyVector)) {
    llvm::Triple Target = getASTContext().getTargetInfo().getTriple();
    llvm::Triple::ArchType Arch =
        getASTContext().getTargetInfo().getTriple().getArch();
    if ((Arch == llvm::Triple::aarch64 ||
         Arch == llvm::Triple::aarch64_be) && !Target.isOSDarwin())
      mangleAArch64NeonVectorType(T);
    else
      mangleNeonVectorType(T);
    return;
  }
  Out << "Dv" << T->getNumElements() << '_';
  if (T->getVectorKind() == VectorType::AltiVecPixel)
    Out << 'p';
  else if (T->getVectorKind() == VectorType::AltiVecBool)
    Out << 'b';
  else
    mangleType(T->getElementType());
}
void CXXNameMangler::mangleType(const ExtVectorType *T) {
  mangleType(static_cast<const VectorType*>(T));
}
void CXXNameMangler::mangleType(const DependentSizedExtVectorType *T) {
  Out << "Dv";
  mangleExpression(T->getSizeExpr());
  Out << '_';
  mangleType(T->getElementType());
}

void CXXNameMangler::mangleType(const PackExpansionType *T) {
  // <type>  ::= Dp <type>          # pack expansion (C++0x)
  Out << "Dp";
  mangleType(T->getPattern());
}

void CXXNameMangler::mangleType(const ObjCInterfaceType *T) {
  mangleSourceName(T->getDecl()->getIdentifier());
}

void CXXNameMangler::mangleType(const ObjCObjectType *T) {
  // Treat __kindof as a vendor extended type qualifier.
  if (T->isKindOfType())
    Out << "U8__kindof";

  if (!T->qual_empty()) {
    // Mangle protocol qualifiers.
    SmallString<64> QualStr;
    llvm::raw_svector_ostream QualOS(QualStr);
    QualOS << "objcproto";
    for (const auto *I : T->quals()) {
      StringRef name = I->getName();
      QualOS << name.size() << name;
    }
    Out << 'U' << QualStr.size() << QualStr;
  }

  mangleType(T->getBaseType());

  if (T->isSpecialized()) {
    // Mangle type arguments as I <type>+ E
    Out << 'I';
    for (auto typeArg : T->getTypeArgs())
      mangleType(typeArg);
    Out << 'E';
  }
}

void CXXNameMangler::mangleType(const BlockPointerType *T) {
  Out << "U13block_pointer";
  mangleType(T->getPointeeType());
}

void CXXNameMangler::mangleType(const InjectedClassNameType *T) {
  // Mangle injected class name types as if the user had written the
  // specialization out fully.  It may not actually be possible to see
  // this mangling, though.
  mangleType(T->getInjectedSpecializationType());
}

void CXXNameMangler::mangleType(const TemplateSpecializationType *T) {
  if (TemplateDecl *TD = T->getTemplateName().getAsTemplateDecl()) {
    mangleName(TD, T->getArgs(), T->getNumArgs());
  } else {
    if (mangleSubstitution(QualType(T, 0)))
      return;
    
    mangleTemplatePrefix(T->getTemplateName());
    
    // FIXME: GCC does not appear to mangle the template arguments when
    // the template in question is a dependent template name. Should we
    // emulate that badness?
    mangleTemplateArgs(T->getArgs(), T->getNumArgs());
    addSubstitution(QualType(T, 0));
  }
}

void CXXNameMangler::mangleType(const DependentNameType *T) {
  // Proposal by cxx-abi-dev, 2014-03-26
  // <class-enum-type> ::= <name>    # non-dependent or dependent type name or
  //                                 # dependent elaborated type specifier using
  //                                 # 'typename'
  //                   ::= Ts <name> # dependent elaborated type specifier using
  //                                 # 'struct' or 'class'
  //                   ::= Tu <name> # dependent elaborated type specifier using
  //                                 # 'union'
  //                   ::= Te <name> # dependent elaborated type specifier using
  //                                 # 'enum'
  switch (T->getKeyword()) {
    case ETK_Typename:
      break;
    case ETK_Struct:
    case ETK_Class:
    case ETK_Interface:
      Out << "Ts";
      break;
    case ETK_Union:
      Out << "Tu";
      break;
    case ETK_Enum:
      Out << "Te";
      break;
    default:
      llvm_unreachable("unexpected keyword for dependent type name");
  }
  // Typename types are always nested
  Out << 'N';
  manglePrefix(T->getQualifier());
  mangleSourceName(T->getIdentifier());
  Out << 'E';
}

void CXXNameMangler::mangleType(const DependentTemplateSpecializationType *T) {
  // Dependently-scoped template types are nested if they have a prefix.
  Out << 'N';

  // TODO: avoid making this TemplateName.
  TemplateName Prefix =
    getASTContext().getDependentTemplateName(T->getQualifier(),
                                             T->getIdentifier());
  mangleTemplatePrefix(Prefix);

  // FIXME: GCC does not appear to mangle the template arguments when
  // the template in question is a dependent template name. Should we
  // emulate that badness?
  mangleTemplateArgs(T->getArgs(), T->getNumArgs());    
  Out << 'E';
}

void CXXNameMangler::mangleType(const TypeOfType *T) {
  // FIXME: this is pretty unsatisfactory, but there isn't an obvious
  // "extension with parameters" mangling.
  Out << "u6typeof";
}

void CXXNameMangler::mangleType(const TypeOfExprType *T) {
  // FIXME: this is pretty unsatisfactory, but there isn't an obvious
  // "extension with parameters" mangling.
  Out << "u6typeof";
}

void CXXNameMangler::mangleType(const DecltypeType *T) {
  Expr *E = T->getUnderlyingExpr();

  // type ::= Dt <expression> E  # decltype of an id-expression
  //                             #   or class member access
  //      ::= DT <expression> E  # decltype of an expression

  // This purports to be an exhaustive list of id-expressions and
  // class member accesses.  Note that we do not ignore parentheses;
  // parentheses change the semantics of decltype for these
  // expressions (and cause the mangler to use the other form).
  if (isa<DeclRefExpr>(E) ||
      isa<MemberExpr>(E) ||
      isa<UnresolvedLookupExpr>(E) ||
      isa<DependentScopeDeclRefExpr>(E) ||
      isa<CXXDependentScopeMemberExpr>(E) ||
      isa<UnresolvedMemberExpr>(E))
    Out << "Dt";
  else
    Out << "DT";
  mangleExpression(E);
  Out << 'E';
}

void CXXNameMangler::mangleType(const UnaryTransformType *T) {
  // If this is dependent, we need to record that. If not, we simply
  // mangle it as the underlying type since they are equivalent.
  if (T->isDependentType()) {
    Out << 'U';
    
    switch (T->getUTTKind()) {
      case UnaryTransformType::EnumUnderlyingType:
        Out << "3eut";
        break;
    }
  }

  mangleType(T->getUnderlyingType());
}

void CXXNameMangler::mangleType(const AutoType *T) {
  QualType D = T->getDeducedType();
  // <builtin-type> ::= Da  # dependent auto
  if (D.isNull())
    Out << (T->isDecltypeAuto() ? "Dc" : "Da");
  else
    mangleType(D);
}

void CXXNameMangler::mangleType(const AtomicType *T) {
  // <type> ::= U <source-name> <type>  # vendor extended type qualifier
  // (Until there's a standardized mangling...)
  Out << "U7_Atomic";
  mangleType(T->getValueType());
}

void CXXNameMangler::mangleIntegerLiteral(QualType T,
                                          const llvm::APSInt &Value) {
  //  <expr-primary> ::= L <type> <value number> E # integer literal
  Out << 'L';

  mangleType(T);
  if (T->isBooleanType()) {
    // Boolean values are encoded as 0/1.
    Out << (Value.getBoolValue() ? '1' : '0');
  } else {
    mangleNumber(Value);
  }
  Out << 'E';

}

void CXXNameMangler::mangleMemberExprBase(const Expr *Base, bool IsArrow) {
  // Ignore member expressions involving anonymous unions.
  while (const auto *RT = Base->getType()->getAs<RecordType>()) {
    if (!RT->getDecl()->isAnonymousStructOrUnion())
      break;
    const auto *ME = dyn_cast<MemberExpr>(Base);
    if (!ME)
      break;
    Base = ME->getBase();
    IsArrow = ME->isArrow();
  }

  if (Base->isImplicitCXXThis()) {
    // Note: GCC mangles member expressions to the implicit 'this' as
    // *this., whereas we represent them as this->. The Itanium C++ ABI
    // does not specify anything here, so we follow GCC.
    Out << "dtdefpT";
  } else {
    Out << (IsArrow ? "pt" : "dt");
    mangleExpression(Base);
  }
}

/// Mangles a member expression.
void CXXNameMangler::mangleMemberExpr(const Expr *base,
                                      bool isArrow,
                                      NestedNameSpecifier *qualifier,
                                      NamedDecl *firstQualifierLookup,
                                      DeclarationName member,
                                      unsigned arity) {
  // <expression> ::= dt <expression> <unresolved-name>
  //              ::= pt <expression> <unresolved-name>
  if (base)
    mangleMemberExprBase(base, isArrow);
  mangleUnresolvedName(qualifier, member, arity);
}

/// Look at the callee of the given call expression and determine if
/// it's a parenthesized id-expression which would have triggered ADL
/// otherwise.
static bool isParenthesizedADLCallee(const CallExpr *call) {
  const Expr *callee = call->getCallee();
  const Expr *fn = callee->IgnoreParens();

  // Must be parenthesized.  IgnoreParens() skips __extension__ nodes,
  // too, but for those to appear in the callee, it would have to be
  // parenthesized.
  if (callee == fn) return false;

  // Must be an unresolved lookup.
  const UnresolvedLookupExpr *lookup = dyn_cast<UnresolvedLookupExpr>(fn);
  if (!lookup) return false;

  assert(!lookup->requiresADL());

  // Must be an unqualified lookup.
  if (lookup->getQualifier()) return false;

  // Must not have found a class member.  Note that if one is a class
  // member, they're all class members.
  if (lookup->getNumDecls() > 0 &&
      (*lookup->decls_begin())->isCXXClassMember())
    return false;

  // Otherwise, ADL would have been triggered.
  return true;
}

void CXXNameMangler::mangleCastExpression(const Expr *E, StringRef CastEncoding) {
  const ExplicitCastExpr *ECE = cast<ExplicitCastExpr>(E);
  Out << CastEncoding;
  mangleType(ECE->getType());
  mangleExpression(ECE->getSubExpr());
}

void CXXNameMangler::mangleInitListElements(const InitListExpr *InitList) {
  if (auto *Syntactic = InitList->getSyntacticForm())
    InitList = Syntactic;
  for (unsigned i = 0, e = InitList->getNumInits(); i != e; ++i)
    mangleExpression(InitList->getInit(i));
}

void CXXNameMangler::mangleExpression(const Expr *E, unsigned Arity) {
  // <expression> ::= <unary operator-name> <expression>
  //              ::= <binary operator-name> <expression> <expression>
  //              ::= <trinary operator-name> <expression> <expression> <expression>
  //              ::= cv <type> expression           # conversion with one argument
  //              ::= cv <type> _ <expression>* E # conversion with a different number of arguments
  //              ::= dc <type> <expression>         # dynamic_cast<type> (expression)
  //              ::= sc <type> <expression>         # static_cast<type> (expression)
  //              ::= cc <type> <expression>         # const_cast<type> (expression)
  //              ::= rc <type> <expression>         # reinterpret_cast<type> (expression)
  //              ::= st <type>                      # sizeof (a type)
  //              ::= at <type>                      # alignof (a type)
  //              ::= <template-param>
  //              ::= <function-param>
  //              ::= sr <type> <unqualified-name>                   # dependent name
  //              ::= sr <type> <unqualified-name> <template-args>   # dependent template-id
  //              ::= ds <expression> <expression>                   # expr.*expr
  //              ::= sZ <template-param>                            # size of a parameter pack
  //              ::= sZ <function-param>    # size of a function parameter pack
  //              ::= <expr-primary>
  // <expr-primary> ::= L <type> <value number> E    # integer literal
  //                ::= L <type <value float> E      # floating literal
  //                ::= L <mangled-name> E           # external name
  //                ::= fpT                          # 'this' expression
  QualType ImplicitlyConvertedToType;
  
recurse:
  switch (E->getStmtClass()) {
  case Expr::NoStmtClass:
#define ABSTRACT_STMT(Type)
#define EXPR(Type, Base)
#define STMT(Type, Base) \
  case Expr::Type##Class:
#include "clang/AST/StmtNodes.inc"
    // fallthrough

  // These all can only appear in local or variable-initialization
  // contexts and so should never appear in a mangling.
  case Expr::AddrLabelExprClass:
  case Expr::DesignatedInitUpdateExprClass:
  case Expr::ImplicitValueInitExprClass:
  case Expr::NoInitExprClass:
  case Expr::ParenListExprClass:
  case Expr::LambdaExprClass:
  case Expr::MSPropertyRefExprClass:
  case Expr::TypoExprClass:  // This should no longer exist in the AST by now.
  case Expr::OMPArraySectionExprClass:
    llvm_unreachable("unexpected statement kind");

  // FIXME: invent manglings for all these.
  case Expr::BlockExprClass:
  case Expr::ChooseExprClass:
  case Expr::CompoundLiteralExprClass:
  case Expr::DesignatedInitExprClass:
  case Expr::ExtVectorElementExprClass:
  case Expr::GenericSelectionExprClass:
  case Expr::ObjCEncodeExprClass:
  case Expr::ObjCIsaExprClass:
  case Expr::ObjCIvarRefExprClass:
  case Expr::ObjCMessageExprClass:
  case Expr::ObjCPropertyRefExprClass:
  case Expr::ObjCProtocolExprClass:
  case Expr::ObjCSelectorExprClass:
  case Expr::ObjCStringLiteralClass:
  case Expr::ObjCBoxedExprClass:
  case Expr::ObjCArrayLiteralClass:
  case Expr::ObjCDictionaryLiteralClass:
  case Expr::ObjCSubscriptRefExprClass:
  case Expr::ObjCIndirectCopyRestoreExprClass:
  case Expr::OffsetOfExprClass:
  case Expr::PredefinedExprClass:
  case Expr::ShuffleVectorExprClass:
  case Expr::ConvertVectorExprClass:
  case Expr::StmtExprClass:
  case Expr::TypeTraitExprClass:
  case Expr::ArrayTypeTraitExprClass:
  case Expr::ExpressionTraitExprClass:
  case Expr::VAArgExprClass:
  case Expr::CUDAKernelCallExprClass:
  case Expr::AsTypeExprClass:
  case Expr::PseudoObjectExprClass:
  case Expr::AtomicExprClass:
  {
    // As bad as this diagnostic is, it's better than crashing.
    DiagnosticsEngine &Diags = Context.getDiags();
    unsigned DiagID = Diags.getCustomDiagID(DiagnosticsEngine::Error,
                                     "cannot yet mangle expression type %0");
    Diags.Report(E->getExprLoc(), DiagID)
      << E->getStmtClassName() << E->getSourceRange();
    break;
  }

  case Expr::CXXUuidofExprClass: {
    const CXXUuidofExpr *UE = cast<CXXUuidofExpr>(E);
    if (UE->isTypeOperand()) {
      QualType UuidT = UE->getTypeOperand(Context.getASTContext());
      Out << "u8__uuidoft";
      mangleType(UuidT);
    } else {
      Expr *UuidExp = UE->getExprOperand();
      Out << "u8__uuidofz";
      mangleExpression(UuidExp, Arity);
    }
    break;
  }

  // Even gcc-4.5 doesn't mangle this.
  case Expr::BinaryConditionalOperatorClass: {
    DiagnosticsEngine &Diags = Context.getDiags();
    unsigned DiagID =
      Diags.getCustomDiagID(DiagnosticsEngine::Error,
                "?: operator with omitted middle operand cannot be mangled");
    Diags.Report(E->getExprLoc(), DiagID)
      << E->getStmtClassName() << E->getSourceRange();
    break;
  }

  // These are used for internal purposes and cannot be meaningfully mangled.
  case Expr::OpaqueValueExprClass:
    llvm_unreachable("cannot mangle opaque value; mangling wrong thing?");

  case Expr::InitListExprClass: {
    Out << "il";
    mangleInitListElements(cast<InitListExpr>(E));
    Out << "E";
    break;
  }

  case Expr::CXXDefaultArgExprClass:
    mangleExpression(cast<CXXDefaultArgExpr>(E)->getExpr(), Arity);
    break;

  case Expr::CXXDefaultInitExprClass:
    mangleExpression(cast<CXXDefaultInitExpr>(E)->getExpr(), Arity);
    break;

  case Expr::CXXStdInitializerListExprClass:
    mangleExpression(cast<CXXStdInitializerListExpr>(E)->getSubExpr(), Arity);
    break;

  case Expr::SubstNonTypeTemplateParmExprClass:
    mangleExpression(cast<SubstNonTypeTemplateParmExpr>(E)->getReplacement(),
                     Arity);
    break;

  case Expr::UserDefinedLiteralClass:
    // We follow g++'s approach of mangling a UDL as a call to the literal
    // operator.
  case Expr::CXXMemberCallExprClass: // fallthrough
  case Expr::CallExprClass: {
    const CallExpr *CE = cast<CallExpr>(E);

    // <expression> ::= cp <simple-id> <expression>* E
    // We use this mangling only when the call would use ADL except
    // for being parenthesized.  Per discussion with David
    // Vandervoorde, 2011.04.25.
    if (isParenthesizedADLCallee(CE)) {
      Out << "cp";
      // The callee here is a parenthesized UnresolvedLookupExpr with
      // no qualifier and should always get mangled as a <simple-id>
      // anyway.

    // <expression> ::= cl <expression>* E
    } else {
      Out << "cl";
    }

    unsigned CallArity = CE->getNumArgs();
    for (const Expr *Arg : CE->arguments())
      if (isa<PackExpansionExpr>(Arg))
        CallArity = UnknownArity;

    mangleExpression(CE->getCallee(), CallArity);
    for (const Expr *Arg : CE->arguments())
      mangleExpression(Arg);
    Out << 'E';
    break;
  }

  case Expr::CXXNewExprClass: {
    const CXXNewExpr *New = cast<CXXNewExpr>(E);
    if (New->isGlobalNew()) Out << "gs";
    Out << (New->isArray() ? "na" : "nw");
    for (CXXNewExpr::const_arg_iterator I = New->placement_arg_begin(),
           E = New->placement_arg_end(); I != E; ++I)
      mangleExpression(*I);
    Out << '_';
    mangleType(New->getAllocatedType());
    if (New->hasInitializer()) {
      if (New->getInitializationStyle() == CXXNewExpr::ListInit)
        Out << "il";
      else
        Out << "pi";
      const Expr *Init = New->getInitializer();
      if (const CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(Init)) {
        // Directly inline the initializers.
        for (CXXConstructExpr::const_arg_iterator I = CCE->arg_begin(),
                                                  E = CCE->arg_end();
             I != E; ++I)
          mangleExpression(*I);
      } else if (const ParenListExpr *PLE = dyn_cast<ParenListExpr>(Init)) {
        for (unsigned i = 0, e = PLE->getNumExprs(); i != e; ++i)
          mangleExpression(PLE->getExpr(i));
      } else if (New->getInitializationStyle() == CXXNewExpr::ListInit &&
                 isa<InitListExpr>(Init)) {
        // Only take InitListExprs apart for list-initialization.
        mangleInitListElements(cast<InitListExpr>(Init));
      } else
        mangleExpression(Init);
    }
    Out << 'E';
    break;
  }

  case Expr::CXXPseudoDestructorExprClass: {
    const auto *PDE = cast<CXXPseudoDestructorExpr>(E);
    if (const Expr *Base = PDE->getBase())
      mangleMemberExprBase(Base, PDE->isArrow());
    NestedNameSpecifier *Qualifier = PDE->getQualifier();
    QualType ScopeType;
    if (TypeSourceInfo *ScopeInfo = PDE->getScopeTypeInfo()) {
      if (Qualifier) {
        mangleUnresolvedPrefix(Qualifier,
                               /*Recursive=*/true);
        mangleUnresolvedTypeOrSimpleId(ScopeInfo->getType());
        Out << 'E';
      } else {
        Out << "sr";
        if (!mangleUnresolvedTypeOrSimpleId(ScopeInfo->getType()))
          Out << 'E';
      }
    } else if (Qualifier) {
      mangleUnresolvedPrefix(Qualifier);
    }
    // <base-unresolved-name> ::= dn <destructor-name>
    Out << "dn";
    QualType DestroyedType = PDE->getDestroyedType();
    mangleUnresolvedTypeOrSimpleId(DestroyedType);
    break;
  }

  case Expr::MemberExprClass: {
    const MemberExpr *ME = cast<MemberExpr>(E);
    mangleMemberExpr(ME->getBase(), ME->isArrow(),
                     ME->getQualifier(), nullptr,
                     ME->getMemberDecl()->getDeclName(), Arity);
    break;
  }

  case Expr::UnresolvedMemberExprClass: {
    const UnresolvedMemberExpr *ME = cast<UnresolvedMemberExpr>(E);
    mangleMemberExpr(ME->isImplicitAccess() ? nullptr : ME->getBase(),
                     ME->isArrow(), ME->getQualifier(), nullptr,
                     ME->getMemberName(), Arity);
    if (ME->hasExplicitTemplateArgs())
      mangleTemplateArgs(ME->getExplicitTemplateArgs());
    break;
  }

  case Expr::CXXDependentScopeMemberExprClass: {
    const CXXDependentScopeMemberExpr *ME
      = cast<CXXDependentScopeMemberExpr>(E);
    mangleMemberExpr(ME->isImplicitAccess() ? nullptr : ME->getBase(),
                     ME->isArrow(), ME->getQualifier(),
                     ME->getFirstQualifierFoundInScope(),
                     ME->getMember(), Arity);
    if (ME->hasExplicitTemplateArgs())
      mangleTemplateArgs(ME->getExplicitTemplateArgs());
    break;
  }

  case Expr::UnresolvedLookupExprClass: {
    const UnresolvedLookupExpr *ULE = cast<UnresolvedLookupExpr>(E);
    mangleUnresolvedName(ULE->getQualifier(), ULE->getName(), Arity);

    // All the <unresolved-name> productions end in a
    // base-unresolved-name, where <template-args> are just tacked
    // onto the end.
    if (ULE->hasExplicitTemplateArgs())
      mangleTemplateArgs(ULE->getExplicitTemplateArgs());
    break;
  }

  case Expr::CXXUnresolvedConstructExprClass: {
    const CXXUnresolvedConstructExpr *CE = cast<CXXUnresolvedConstructExpr>(E);
    unsigned N = CE->arg_size();

    Out << "cv";
    mangleType(CE->getType());
    if (N != 1) Out << '_';
    for (unsigned I = 0; I != N; ++I) mangleExpression(CE->getArg(I));
    if (N != 1) Out << 'E';
    break;
  }

  case Expr::CXXConstructExprClass: {
    const auto *CE = cast<CXXConstructExpr>(E);
    if (!CE->isListInitialization() || CE->isStdInitListInitialization()) {
      assert(
          CE->getNumArgs() >= 1 &&
          (CE->getNumArgs() == 1 || isa<CXXDefaultArgExpr>(CE->getArg(1))) &&
          "implicit CXXConstructExpr must have one argument");
      return mangleExpression(cast<CXXConstructExpr>(E)->getArg(0));
    }
    Out << "il";
    for (auto *E : CE->arguments())
      mangleExpression(E);
    Out << "E";
    break;
  }

  case Expr::CXXTemporaryObjectExprClass: {
    const auto *CE = cast<CXXTemporaryObjectExpr>(E);
    unsigned N = CE->getNumArgs();
    bool List = CE->isListInitialization();

    if (List)
      Out << "tl";
    else
      Out << "cv";
    mangleType(CE->getType());
    if (!List && N != 1)
      Out << '_';
    if (CE->isStdInitListInitialization()) {
      // We implicitly created a std::initializer_list<T> for the first argument
      // of a constructor of type U in an expression of the form U{a, b, c}.
      // Strip all the semantic gunk off the initializer list.
      auto *SILE =
          cast<CXXStdInitializerListExpr>(CE->getArg(0)->IgnoreImplicit());
      auto *ILE = cast<InitListExpr>(SILE->getSubExpr()->IgnoreImplicit());
      mangleInitListElements(ILE);
    } else {
      for (auto *E : CE->arguments())
        mangleExpression(E);
    }
    if (List || N != 1)
      Out << 'E';
    break;
  }

  case Expr::CXXScalarValueInitExprClass:
    Out << "cv";
    mangleType(E->getType());
    Out << "_E";
    break;

  case Expr::CXXNoexceptExprClass:
    Out << "nx";
    mangleExpression(cast<CXXNoexceptExpr>(E)->getOperand());
    break;

  case Expr::UnaryExprOrTypeTraitExprClass: {
    const UnaryExprOrTypeTraitExpr *SAE = cast<UnaryExprOrTypeTraitExpr>(E);
    
    if (!SAE->isInstantiationDependent()) {
      // Itanium C++ ABI:
      //   If the operand of a sizeof or alignof operator is not 
      //   instantiation-dependent it is encoded as an integer literal 
      //   reflecting the result of the operator.
      //
      //   If the result of the operator is implicitly converted to a known 
      //   integer type, that type is used for the literal; otherwise, the type 
      //   of std::size_t or std::ptrdiff_t is used.
      QualType T = (ImplicitlyConvertedToType.isNull() || 
                    !ImplicitlyConvertedToType->isIntegerType())? SAE->getType()
                                                    : ImplicitlyConvertedToType;
      llvm::APSInt V = SAE->EvaluateKnownConstInt(Context.getASTContext());
      mangleIntegerLiteral(T, V);
      break;
    }
    
    switch(SAE->getKind()) {
    case UETT_SizeOf:
      Out << 's';
      break;
    case UETT_AlignOf:
      Out << 'a';
      break;
    case UETT_VecStep: {
      DiagnosticsEngine &Diags = Context.getDiags();
      unsigned DiagID = Diags.getCustomDiagID(DiagnosticsEngine::Error,
                                     "cannot yet mangle vec_step expression");
      Diags.Report(DiagID);
      return;
    }
    case UETT_OpenMPRequiredSimdAlign:
      DiagnosticsEngine &Diags = Context.getDiags();
      unsigned DiagID = Diags.getCustomDiagID(
          DiagnosticsEngine::Error,
          "cannot yet mangle __builtin_omp_required_simd_align expression");
      Diags.Report(DiagID);
      return;
    }
    if (SAE->isArgumentType()) {
      Out << 't';
      mangleType(SAE->getArgumentType());
    } else {
      Out << 'z';
      mangleExpression(SAE->getArgumentExpr());
    }
    break;
  }

  case Expr::CXXThrowExprClass: {
    const CXXThrowExpr *TE = cast<CXXThrowExpr>(E);
    //  <expression> ::= tw <expression>  # throw expression
    //               ::= tr               # rethrow
    if (TE->getSubExpr()) {
      Out << "tw";
      mangleExpression(TE->getSubExpr());
    } else {
      Out << "tr";
    }
    break;
  }

  case Expr::CXXTypeidExprClass: {
    const CXXTypeidExpr *TIE = cast<CXXTypeidExpr>(E);
    //  <expression> ::= ti <type>        # typeid (type)
    //               ::= te <expression>  # typeid (expression)
    if (TIE->isTypeOperand()) {
      Out << "ti";
      mangleType(TIE->getTypeOperand(Context.getASTContext()));
    } else {
      Out << "te";
      mangleExpression(TIE->getExprOperand());
    }
    break;
  }

  case Expr::CXXDeleteExprClass: {
    const CXXDeleteExpr *DE = cast<CXXDeleteExpr>(E);
    //  <expression> ::= [gs] dl <expression>  # [::] delete expr
    //               ::= [gs] da <expression>  # [::] delete [] expr
    if (DE->isGlobalDelete()) Out << "gs";
    Out << (DE->isArrayForm() ? "da" : "dl");
    mangleExpression(DE->getArgument());
    break;
  }

  case Expr::UnaryOperatorClass: {
    const UnaryOperator *UO = cast<UnaryOperator>(E);
    mangleOperatorName(UnaryOperator::getOverloadedOperator(UO->getOpcode()),
                       /*Arity=*/1);
    mangleExpression(UO->getSubExpr());
    break;
  }

  case Expr::ArraySubscriptExprClass: {
    const ArraySubscriptExpr *AE = cast<ArraySubscriptExpr>(E);

    // Array subscript is treated as a syntactically weird form of
    // binary operator.
    Out << "ix";
    mangleExpression(AE->getLHS());
    mangleExpression(AE->getRHS());
    break;
  }

  case Expr::CompoundAssignOperatorClass: // fallthrough
  case Expr::BinaryOperatorClass: {
    const BinaryOperator *BO = cast<BinaryOperator>(E);
    if (BO->getOpcode() == BO_PtrMemD)
      Out << "ds";
    else
      mangleOperatorName(BinaryOperator::getOverloadedOperator(BO->getOpcode()),
                         /*Arity=*/2);
    mangleExpression(BO->getLHS());
    mangleExpression(BO->getRHS());
    break;
  }

  case Expr::ConditionalOperatorClass: {
    const ConditionalOperator *CO = cast<ConditionalOperator>(E);
    mangleOperatorName(OO_Conditional, /*Arity=*/3);
    mangleExpression(CO->getCond());
    mangleExpression(CO->getLHS(), Arity);
    mangleExpression(CO->getRHS(), Arity);
    break;
  }

  case Expr::ImplicitCastExprClass: {
    ImplicitlyConvertedToType = E->getType();
    E = cast<ImplicitCastExpr>(E)->getSubExpr();
    goto recurse;
  }
      
  case Expr::ObjCBridgedCastExprClass: {
    // Mangle ownership casts as a vendor extended operator __bridge, 
    // __bridge_transfer, or __bridge_retain.
    StringRef Kind = cast<ObjCBridgedCastExpr>(E)->getBridgeKindName();
    Out << "v1U" << Kind.size() << Kind;
  }
  // Fall through to mangle the cast itself.
      
  case Expr::CStyleCastExprClass:
    mangleCastExpression(E, "cv");
    break;

  case Expr::CXXFunctionalCastExprClass: {
    auto *Sub = cast<ExplicitCastExpr>(E)->getSubExpr()->IgnoreImplicit();
    // FIXME: Add isImplicit to CXXConstructExpr.
    if (auto *CCE = dyn_cast<CXXConstructExpr>(Sub))
      if (CCE->getParenOrBraceRange().isInvalid())
        Sub = CCE->getArg(0)->IgnoreImplicit();
    if (auto *StdInitList = dyn_cast<CXXStdInitializerListExpr>(Sub))
      Sub = StdInitList->getSubExpr()->IgnoreImplicit();
    if (auto *IL = dyn_cast<InitListExpr>(Sub)) {
      Out << "tl";
      mangleType(E->getType());
      mangleInitListElements(IL);
      Out << "E";
    } else {
      mangleCastExpression(E, "cv");
    }
    break;
  }

  case Expr::CXXStaticCastExprClass:
    mangleCastExpression(E, "sc");
    break;
  case Expr::CXXDynamicCastExprClass:
    mangleCastExpression(E, "dc");
    break;
  case Expr::CXXReinterpretCastExprClass:
    mangleCastExpression(E, "rc");
    break;
  case Expr::CXXConstCastExprClass:
    mangleCastExpression(E, "cc");
    break;

  case Expr::CXXOperatorCallExprClass: {
    const CXXOperatorCallExpr *CE = cast<CXXOperatorCallExpr>(E);
    unsigned NumArgs = CE->getNumArgs();
    mangleOperatorName(CE->getOperator(), /*Arity=*/NumArgs);
    // Mangle the arguments.
    for (unsigned i = 0; i != NumArgs; ++i)
      mangleExpression(CE->getArg(i));
    break;
  }

  case Expr::ParenExprClass:
    mangleExpression(cast<ParenExpr>(E)->getSubExpr(), Arity);
    break;

  case Expr::DeclRefExprClass: {
    const NamedDecl *D = cast<DeclRefExpr>(E)->getDecl();

    switch (D->getKind()) {
    default:
      //  <expr-primary> ::= L <mangled-name> E # external name
      Out << 'L';
      mangle(D);
      Out << 'E';
      break;

    case Decl::ParmVar:
      mangleFunctionParam(cast<ParmVarDecl>(D));
      break;

    case Decl::EnumConstant: {
      const EnumConstantDecl *ED = cast<EnumConstantDecl>(D);
      mangleIntegerLiteral(ED->getType(), ED->getInitVal());
      break;
    }

    case Decl::NonTypeTemplateParm: {
      const NonTypeTemplateParmDecl *PD = cast<NonTypeTemplateParmDecl>(D);
      mangleTemplateParameter(PD->getIndex());
      break;
    }

    }

    break;
  }

  case Expr::SubstNonTypeTemplateParmPackExprClass:
    // FIXME: not clear how to mangle this!
    // template <unsigned N...> class A {
    //   template <class U...> void foo(U (&x)[N]...);
    // };
    Out << "_SUBSTPACK_";
    break;

  case Expr::FunctionParmPackExprClass: {
    // FIXME: not clear how to mangle this!
    const FunctionParmPackExpr *FPPE = cast<FunctionParmPackExpr>(E);
    Out << "v110_SUBSTPACK";
    mangleFunctionParam(FPPE->getParameterPack());
    break;
  }

  case Expr::DependentScopeDeclRefExprClass: {
    const DependentScopeDeclRefExpr *DRE = cast<DependentScopeDeclRefExpr>(E);
    mangleUnresolvedName(DRE->getQualifier(), DRE->getDeclName(), Arity);

    // All the <unresolved-name> productions end in a
    // base-unresolved-name, where <template-args> are just tacked
    // onto the end.
    if (DRE->hasExplicitTemplateArgs())
      mangleTemplateArgs(DRE->getExplicitTemplateArgs());
    break;
  }

  case Expr::CXXBindTemporaryExprClass:
    mangleExpression(cast<CXXBindTemporaryExpr>(E)->getSubExpr());
    break;

  case Expr::ExprWithCleanupsClass:
    mangleExpression(cast<ExprWithCleanups>(E)->getSubExpr(), Arity);
    break;

  case Expr::FloatingLiteralClass: {
    const FloatingLiteral *FL = cast<FloatingLiteral>(E);
    Out << 'L';
    mangleType(FL->getType());
    mangleFloat(FL->getValue());
    Out << 'E';
    break;
  }

  case Expr::CharacterLiteralClass:
    Out << 'L';
    mangleType(E->getType());
    Out << cast<CharacterLiteral>(E)->getValue();
    Out << 'E';
    break;

  // FIXME. __objc_yes/__objc_no are mangled same as true/false
  case Expr::ObjCBoolLiteralExprClass:
    Out << "Lb";
    Out << (cast<ObjCBoolLiteralExpr>(E)->getValue() ? '1' : '0');
    Out << 'E';
    break;
  
  case Expr::CXXBoolLiteralExprClass:
    Out << "Lb";
    Out << (cast<CXXBoolLiteralExpr>(E)->getValue() ? '1' : '0');
    Out << 'E';
    break;

  case Expr::IntegerLiteralClass: {
    llvm::APSInt Value(cast<IntegerLiteral>(E)->getValue());
    if (E->getType()->isSignedIntegerType())
      Value.setIsSigned(true);
    mangleIntegerLiteral(E->getType(), Value);
    break;
  }

  case Expr::ImaginaryLiteralClass: {
    const ImaginaryLiteral *IE = cast<ImaginaryLiteral>(E);
    // Mangle as if a complex literal.
    // Proposal from David Vandevoorde, 2010.06.30.
    Out << 'L';
    mangleType(E->getType());
    if (const FloatingLiteral *Imag =
          dyn_cast<FloatingLiteral>(IE->getSubExpr())) {
      // Mangle a floating-point zero of the appropriate type.
      mangleFloat(llvm::APFloat(Imag->getValue().getSemantics()));
      Out << '_';
      mangleFloat(Imag->getValue());
    } else {
      Out << "0_";
      llvm::APSInt Value(cast<IntegerLiteral>(IE->getSubExpr())->getValue());
      if (IE->getSubExpr()->getType()->isSignedIntegerType())
        Value.setIsSigned(true);
      mangleNumber(Value);
    }
    Out << 'E';
    break;
  }

  case Expr::StringLiteralClass: {
    // Revised proposal from David Vandervoorde, 2010.07.15.
    Out << 'L';
    assert(isa<ConstantArrayType>(E->getType()));
    mangleType(E->getType());
    Out << 'E';
    break;
  }

  case Expr::GNUNullExprClass:
    // FIXME: should this really be mangled the same as nullptr?
    // fallthrough

  case Expr::CXXNullPtrLiteralExprClass: {
    Out << "LDnE";
    break;
  }
      
  case Expr::PackExpansionExprClass:
    Out << "sp";
    mangleExpression(cast<PackExpansionExpr>(E)->getPattern());
    break;
      
  case Expr::SizeOfPackExprClass: {
    auto *SPE = cast<SizeOfPackExpr>(E);
    if (SPE->isPartiallySubstituted()) {
      Out << "sP";
      for (const auto &A : SPE->getPartialArguments())
        mangleTemplateArg(A);
      Out << "E";
      break;
    }

    Out << "sZ";
    const NamedDecl *Pack = SPE->getPack();
    if (const TemplateTypeParmDecl *TTP = dyn_cast<TemplateTypeParmDecl>(Pack))
      mangleTemplateParameter(TTP->getIndex());
    else if (const NonTypeTemplateParmDecl *NTTP
                = dyn_cast<NonTypeTemplateParmDecl>(Pack))
      mangleTemplateParameter(NTTP->getIndex());
    else if (const TemplateTemplateParmDecl *TempTP
                                    = dyn_cast<TemplateTemplateParmDecl>(Pack))
      mangleTemplateParameter(TempTP->getIndex());
    else
      mangleFunctionParam(cast<ParmVarDecl>(Pack));
    break;
  }

  case Expr::MaterializeTemporaryExprClass: {
    mangleExpression(cast<MaterializeTemporaryExpr>(E)->GetTemporaryExpr());
    break;
  }

  case Expr::CXXFoldExprClass: {
    auto *FE = cast<CXXFoldExpr>(E);
    if (FE->isLeftFold())
      Out << (FE->getInit() ? "fL" : "fl");
    else
      Out << (FE->getInit() ? "fR" : "fr");

    if (FE->getOperator() == BO_PtrMemD)
      Out << "ds";
    else
      mangleOperatorName(
          BinaryOperator::getOverloadedOperator(FE->getOperator()),
          /*Arity=*/2);

    if (FE->getLHS())
      mangleExpression(FE->getLHS());
    if (FE->getRHS())
      mangleExpression(FE->getRHS());
    break;
  }

  case Expr::CXXThisExprClass:
    Out << "fpT";
    break;

  case Expr::CoawaitExprClass:
    // FIXME: Propose a non-vendor mangling.
    Out << "v18co_await";
    mangleExpression(cast<CoawaitExpr>(E)->getOperand());
    break;

  case Expr::CoyieldExprClass:
    // FIXME: Propose a non-vendor mangling.
    Out << "v18co_yield";
    mangleExpression(cast<CoawaitExpr>(E)->getOperand());
    break;
  }
}

/// Mangle an expression which refers to a parameter variable.
///
/// <expression>     ::= <function-param>
/// <function-param> ::= fp <top-level CV-qualifiers> _      # L == 0, I == 0
/// <function-param> ::= fp <top-level CV-qualifiers>
///                      <parameter-2 non-negative number> _ # L == 0, I > 0
/// <function-param> ::= fL <L-1 non-negative number>
///                      p <top-level CV-qualifiers> _       # L > 0, I == 0
/// <function-param> ::= fL <L-1 non-negative number>
///                      p <top-level CV-qualifiers>
///                      <I-1 non-negative number> _         # L > 0, I > 0
///
/// L is the nesting depth of the parameter, defined as 1 if the
/// parameter comes from the innermost function prototype scope
/// enclosing the current context, 2 if from the next enclosing
/// function prototype scope, and so on, with one special case: if
/// we've processed the full parameter clause for the innermost
/// function type, then L is one less.  This definition conveniently
/// makes it irrelevant whether a function's result type was written
/// trailing or leading, but is otherwise overly complicated; the
/// numbering was first designed without considering references to
/// parameter in locations other than return types, and then the
/// mangling had to be generalized without changing the existing
/// manglings.
///
/// I is the zero-based index of the parameter within its parameter
/// declaration clause.  Note that the original ABI document describes
/// this using 1-based ordinals.
void CXXNameMangler::mangleFunctionParam(const ParmVarDecl *parm) {
  unsigned parmDepth = parm->getFunctionScopeDepth();
  unsigned parmIndex = parm->getFunctionScopeIndex();

  // Compute 'L'.
  // parmDepth does not include the declaring function prototype.
  // FunctionTypeDepth does account for that.
  assert(parmDepth < FunctionTypeDepth.getDepth());
  unsigned nestingDepth = FunctionTypeDepth.getDepth() - parmDepth;
  if (FunctionTypeDepth.isInResultType())
    nestingDepth--;

  if (nestingDepth == 0) {
    Out << "fp";
  } else {
    Out << "fL" << (nestingDepth - 1) << 'p';
  }

  // Top-level qualifiers.  We don't have to worry about arrays here,
  // because parameters declared as arrays should already have been
  // transformed to have pointer type. FIXME: apparently these don't
  // get mangled if used as an rvalue of a known non-class type?
  assert(!parm->getType()->isArrayType()
         && "parameter's type is still an array type?");
  mangleQualifiers(parm->getType().getQualifiers());

  // Parameter index.
  if (parmIndex != 0) {
    Out << (parmIndex - 1);
  }
  Out << '_';
}

void CXXNameMangler::mangleCXXCtorType(CXXCtorType T) {
  // <ctor-dtor-name> ::= C1  # complete object constructor
  //                  ::= C2  # base object constructor
  //
  // In addition, C5 is a comdat name with C1 and C2 in it.
  switch (T) {
  case Ctor_Complete:
    Out << "C1";
    break;
  case Ctor_Base:
    Out << "C2";
    break;
  case Ctor_Comdat:
    Out << "C5";
    break;
  case Ctor_DefaultClosure:
  case Ctor_CopyingClosure:
    llvm_unreachable("closure constructors don't exist for the Itanium ABI!");
  }
}

void CXXNameMangler::mangleCXXDtorType(CXXDtorType T) {
  // <ctor-dtor-name> ::= D0  # deleting destructor
  //                  ::= D1  # complete object destructor
  //                  ::= D2  # base object destructor
  //
  // In addition, D5 is a comdat name with D1, D2 and, if virtual, D0 in it.
  switch (T) {
  case Dtor_Deleting:
    Out << "D0";
    break;
  case Dtor_Complete:
    Out << "D1";
    break;
  case Dtor_Base:
    Out << "D2";
    break;
  case Dtor_Comdat:
    Out << "D5";
    break;
  }
}

void CXXNameMangler::mangleTemplateArgs(
                          const ASTTemplateArgumentListInfo &TemplateArgs) {
  // <template-args> ::= I <template-arg>+ E
  Out << 'I';
  for (unsigned i = 0, e = TemplateArgs.NumTemplateArgs; i != e; ++i)
    mangleTemplateArg(TemplateArgs.getTemplateArgs()[i].getArgument());
  Out << 'E';
}

void CXXNameMangler::mangleTemplateArgs(const TemplateArgumentList &AL) {
  // <template-args> ::= I <template-arg>+ E
  Out << 'I';
  for (unsigned i = 0, e = AL.size(); i != e; ++i)
    mangleTemplateArg(AL[i]);
  Out << 'E';
}

void CXXNameMangler::mangleTemplateArgs(const TemplateArgument *TemplateArgs,
                                        unsigned NumTemplateArgs) {
  // <template-args> ::= I <template-arg>+ E
  Out << 'I';
  for (unsigned i = 0; i != NumTemplateArgs; ++i)
    mangleTemplateArg(TemplateArgs[i]);
  Out << 'E';
}

void CXXNameMangler::mangleTemplateArg(TemplateArgument A) {
  // <template-arg> ::= <type>              # type or template
  //                ::= X <expression> E    # expression
  //                ::= <expr-primary>      # simple expressions
  //                ::= J <template-arg>* E # argument pack
  if (!A.isInstantiationDependent() || A.isDependent())
    A = Context.getASTContext().getCanonicalTemplateArgument(A);
  
  switch (A.getKind()) {
  case TemplateArgument::Null:
    llvm_unreachable("Cannot mangle NULL template argument");
      
  case TemplateArgument::Type:
    mangleType(A.getAsType());
    break;
  case TemplateArgument::Template:
    // This is mangled as <type>.
    mangleType(A.getAsTemplate());
    break;
  case TemplateArgument::TemplateExpansion:
    // <type>  ::= Dp <type>          # pack expansion (C++0x)
    Out << "Dp";
    mangleType(A.getAsTemplateOrTemplatePattern());
    break;
  case TemplateArgument::Expression: {
    // It's possible to end up with a DeclRefExpr here in certain
    // dependent cases, in which case we should mangle as a
    // declaration.
    const Expr *E = A.getAsExpr()->IgnoreParens();
    if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
      const ValueDecl *D = DRE->getDecl();
      if (isa<VarDecl>(D) || isa<FunctionDecl>(D)) {
        Out << 'L';
        mangle(D);
        Out << 'E';
        break;
      }
    }
    
    Out << 'X';
    mangleExpression(E);
    Out << 'E';
    break;
  }
  case TemplateArgument::Integral:
    mangleIntegerLiteral(A.getIntegralType(), A.getAsIntegral());
    break;
  case TemplateArgument::Declaration: {
    //  <expr-primary> ::= L <mangled-name> E # external name
    // Clang produces AST's where pointer-to-member-function expressions
    // and pointer-to-function expressions are represented as a declaration not
    // an expression. We compensate for it here to produce the correct mangling.
    ValueDecl *D = A.getAsDecl();
    bool compensateMangling = !A.getParamTypeForDecl()->isReferenceType();
    if (compensateMangling) {
      Out << 'X';
      mangleOperatorName(OO_Amp, 1);
    }

    Out << 'L';
    // References to external entities use the mangled name; if the name would
    // not normally be manged then mangle it as unqualified.
    mangle(D);
    Out << 'E';

    if (compensateMangling)
      Out << 'E';

    break;
  }
  case TemplateArgument::NullPtr: {
    //  <expr-primary> ::= L <type> 0 E
    Out << 'L';
    mangleType(A.getNullPtrType());
    Out << "0E";
    break;
  }
  case TemplateArgument::Pack: {
    //  <template-arg> ::= J <template-arg>* E
    Out << 'J';
    for (const auto &P : A.pack_elements())
      mangleTemplateArg(P);
    Out << 'E';
  }
  }
}

void CXXNameMangler::mangleTemplateParameter(unsigned Index) {
  // <template-param> ::= T_    # first template parameter
  //                  ::= T <parameter-2 non-negative number> _
  if (Index == 0)
    Out << "T_";
  else
    Out << 'T' << (Index - 1) << '_';
}

void CXXNameMangler::mangleSeqID(unsigned SeqID) {
  if (SeqID == 1)
    Out << '0';
  else if (SeqID > 1) {
    SeqID--;

    // <seq-id> is encoded in base-36, using digits and upper case letters.
    char Buffer[7]; // log(2**32) / log(36) ~= 7
    MutableArrayRef<char> BufferRef(Buffer);
    MutableArrayRef<char>::reverse_iterator I = BufferRef.rbegin();

    for (; SeqID != 0; SeqID /= 36) {
      unsigned C = SeqID % 36;
      *I++ = (C < 10 ? '0' + C : 'A' + C - 10);
    }

    Out.write(I.base(), I - BufferRef.rbegin());
  }
  Out << '_';
}

void CXXNameMangler::mangleExistingSubstitution(QualType type) {
  bool result = mangleSubstitution(type);
  assert(result && "no existing substitution for type");
  (void) result;
}

void CXXNameMangler::mangleExistingSubstitution(TemplateName tname) {
  bool result = mangleSubstitution(tname);
  assert(result && "no existing substitution for template name");
  (void) result;
}

// <substitution> ::= S <seq-id> _
//                ::= S_
bool CXXNameMangler::mangleSubstitution(const NamedDecl *ND) {
  // Try one of the standard substitutions first.
  if (mangleStandardSubstitution(ND))
    return true;

  ND = cast<NamedDecl>(ND->getCanonicalDecl());
  return mangleSubstitution(reinterpret_cast<uintptr_t>(ND));
}

/// Determine whether the given type has any qualifiers that are relevant for
/// substitutions.
static bool hasMangledSubstitutionQualifiers(QualType T) {
  Qualifiers Qs = T.getQualifiers();
  return Qs.getCVRQualifiers() || Qs.hasAddressSpace();
}

bool CXXNameMangler::mangleSubstitution(QualType T) {
  if (!hasMangledSubstitutionQualifiers(T)) {
    if (const RecordType *RT = T->getAs<RecordType>())
      return mangleSubstitution(RT->getDecl());
  }

  uintptr_t TypePtr = reinterpret_cast<uintptr_t>(T.getAsOpaquePtr());

  return mangleSubstitution(TypePtr);
}

bool CXXNameMangler::mangleSubstitution(TemplateName Template) {
  if (TemplateDecl *TD = Template.getAsTemplateDecl())
    return mangleSubstitution(TD);
  
  Template = Context.getASTContext().getCanonicalTemplateName(Template);
  return mangleSubstitution(
                      reinterpret_cast<uintptr_t>(Template.getAsVoidPointer()));
}

bool CXXNameMangler::mangleSubstitution(uintptr_t Ptr) {
  llvm::DenseMap<uintptr_t, unsigned>::iterator I = Substitutions.find(Ptr);
  if (I == Substitutions.end())
    return false;

  unsigned SeqID = I->second;
  Out << 'S';
  mangleSeqID(SeqID);

  return true;
}

static bool isCharType(QualType T) {
  if (T.isNull())
    return false;

  return T->isSpecificBuiltinType(BuiltinType::Char_S) ||
    T->isSpecificBuiltinType(BuiltinType::Char_U);
}

/// Returns whether a given type is a template specialization of a given name
/// with a single argument of type char.
static bool isCharSpecialization(QualType T, const char *Name) {
  if (T.isNull())
    return false;

  const RecordType *RT = T->getAs<RecordType>();
  if (!RT)
    return false;

  const ClassTemplateSpecializationDecl *SD =
    dyn_cast<ClassTemplateSpecializationDecl>(RT->getDecl());
  if (!SD)
    return false;

  if (!isStdNamespace(getEffectiveDeclContext(SD)))
    return false;

  const TemplateArgumentList &TemplateArgs = SD->getTemplateArgs();
  if (TemplateArgs.size() != 1)
    return false;

  if (!isCharType(TemplateArgs[0].getAsType()))
    return false;

  return SD->getIdentifier()->getName() == Name;
}

template <std::size_t StrLen>
static bool isStreamCharSpecialization(const ClassTemplateSpecializationDecl*SD,
                                       const char (&Str)[StrLen]) {
  if (!SD->getIdentifier()->isStr(Str))
    return false;

  const TemplateArgumentList &TemplateArgs = SD->getTemplateArgs();
  if (TemplateArgs.size() != 2)
    return false;

  if (!isCharType(TemplateArgs[0].getAsType()))
    return false;

  if (!isCharSpecialization(TemplateArgs[1].getAsType(), "char_traits"))
    return false;

  return true;
}

bool CXXNameMangler::mangleStandardSubstitution(const NamedDecl *ND) {
  // <substitution> ::= St # ::std::
  if (const NamespaceDecl *NS = dyn_cast<NamespaceDecl>(ND)) {
    if (isStd(NS)) {
      Out << "St";
      return true;
    }
  }

  if (const ClassTemplateDecl *TD = dyn_cast<ClassTemplateDecl>(ND)) {
    if (!isStdNamespace(getEffectiveDeclContext(TD)))
      return false;

    // <substitution> ::= Sa # ::std::allocator
    if (TD->getIdentifier()->isStr("allocator")) {
      Out << "Sa";
      return true;
    }

    // <<substitution> ::= Sb # ::std::basic_string
    if (TD->getIdentifier()->isStr("basic_string")) {
      Out << "Sb";
      return true;
    }
  }

  if (const ClassTemplateSpecializationDecl *SD =
        dyn_cast<ClassTemplateSpecializationDecl>(ND)) {
    if (!isStdNamespace(getEffectiveDeclContext(SD)))
      return false;

    //    <substitution> ::= Ss # ::std::basic_string<char,
    //                            ::std::char_traits<char>,
    //                            ::std::allocator<char> >
    if (SD->getIdentifier()->isStr("basic_string")) {
      const TemplateArgumentList &TemplateArgs = SD->getTemplateArgs();

      if (TemplateArgs.size() != 3)
        return false;

      if (!isCharType(TemplateArgs[0].getAsType()))
        return false;

      if (!isCharSpecialization(TemplateArgs[1].getAsType(), "char_traits"))
        return false;

      if (!isCharSpecialization(TemplateArgs[2].getAsType(), "allocator"))
        return false;

      Out << "Ss";
      return true;
    }

    //    <substitution> ::= Si # ::std::basic_istream<char,
    //                            ::std::char_traits<char> >
    if (isStreamCharSpecialization(SD, "basic_istream")) {
      Out << "Si";
      return true;
    }

    //    <substitution> ::= So # ::std::basic_ostream<char,
    //                            ::std::char_traits<char> >
    if (isStreamCharSpecialization(SD, "basic_ostream")) {
      Out << "So";
      return true;
    }

    //    <substitution> ::= Sd # ::std::basic_iostream<char,
    //                            ::std::char_traits<char> >
    if (isStreamCharSpecialization(SD, "basic_iostream")) {
      Out << "Sd";
      return true;
    }
  }
  return false;
}

void CXXNameMangler::addSubstitution(QualType T) {
  if (!hasMangledSubstitutionQualifiers(T)) {
    if (const RecordType *RT = T->getAs<RecordType>()) {
      addSubstitution(RT->getDecl());
      return;
    }
  }

  uintptr_t TypePtr = reinterpret_cast<uintptr_t>(T.getAsOpaquePtr());
  addSubstitution(TypePtr);
}

void CXXNameMangler::addSubstitution(TemplateName Template) {
  if (TemplateDecl *TD = Template.getAsTemplateDecl())
    return addSubstitution(TD);
  
  Template = Context.getASTContext().getCanonicalTemplateName(Template);
  addSubstitution(reinterpret_cast<uintptr_t>(Template.getAsVoidPointer()));
}

void CXXNameMangler::addSubstitution(uintptr_t Ptr) {
  assert(!Substitutions.count(Ptr) && "Substitution already exists!");
  Substitutions[Ptr] = SeqID++;
}

//

/// Mangles the name of the declaration D and emits that name to the given
/// output stream.
///
/// If the declaration D requires a mangled name, this routine will emit that
/// mangled name to \p os and return true. Otherwise, \p os will be unchanged
/// and this routine will return false. In this case, the caller should just
/// emit the identifier of the declaration (\c D->getIdentifier()) as its
/// name.
void ItaniumMangleContextImpl::mangleCXXName(const NamedDecl *D,
                                             raw_ostream &Out) {
  assert((isa<FunctionDecl>(D) || isa<VarDecl>(D)) &&
          "Invalid mangleName() call, argument is not a variable or function!");
  assert(!isa<CXXConstructorDecl>(D) && !isa<CXXDestructorDecl>(D) &&
         "Invalid mangleName() call on 'structor decl!");

  PrettyStackTraceDecl CrashInfo(D, SourceLocation(),
                                 getASTContext().getSourceManager(),
                                 "Mangling declaration");

  CXXNameMangler Mangler(*this, Out, D);
  Mangler.mangle(D);
}

void ItaniumMangleContextImpl::mangleCXXCtor(const CXXConstructorDecl *D,
                                             CXXCtorType Type,
                                             raw_ostream &Out) {
  CXXNameMangler Mangler(*this, Out, D, Type);
  Mangler.mangle(D);
}

void ItaniumMangleContextImpl::mangleCXXDtor(const CXXDestructorDecl *D,
                                             CXXDtorType Type,
                                             raw_ostream &Out) {
  CXXNameMangler Mangler(*this, Out, D, Type);
  Mangler.mangle(D);
}

void ItaniumMangleContextImpl::mangleCXXCtorComdat(const CXXConstructorDecl *D,
                                                   raw_ostream &Out) {
  CXXNameMangler Mangler(*this, Out, D, Ctor_Comdat);
  Mangler.mangle(D);
}

void ItaniumMangleContextImpl::mangleCXXDtorComdat(const CXXDestructorDecl *D,
                                                   raw_ostream &Out) {
  CXXNameMangler Mangler(*this, Out, D, Dtor_Comdat);
  Mangler.mangle(D);
}

void ItaniumMangleContextImpl::mangleThunk(const CXXMethodDecl *MD,
                                           const ThunkInfo &Thunk,
                                           raw_ostream &Out) {
  //  <special-name> ::= T <call-offset> <base encoding>
  //                      # base is the nominal target function of thunk
  //  <special-name> ::= Tc <call-offset> <call-offset> <base encoding>
  //                      # base is the nominal target function of thunk
  //                      # first call-offset is 'this' adjustment
  //                      # second call-offset is result adjustment
  
  assert(!isa<CXXDestructorDecl>(MD) &&
         "Use mangleCXXDtor for destructor decls!");
  CXXNameMangler Mangler(*this, Out);
  Mangler.getStream() << "_ZT";
  if (!Thunk.Return.isEmpty())
    Mangler.getStream() << 'c';
  
  // Mangle the 'this' pointer adjustment.
  Mangler.mangleCallOffset(Thunk.This.NonVirtual,
                           Thunk.This.Virtual.Itanium.VCallOffsetOffset);

  // Mangle the return pointer adjustment if there is one.
  if (!Thunk.Return.isEmpty())
    Mangler.mangleCallOffset(Thunk.Return.NonVirtual,
                             Thunk.Return.Virtual.Itanium.VBaseOffsetOffset);

  Mangler.mangleFunctionEncoding(MD);
}

void ItaniumMangleContextImpl::mangleCXXDtorThunk(
    const CXXDestructorDecl *DD, CXXDtorType Type,
    const ThisAdjustment &ThisAdjustment, raw_ostream &Out) {
  //  <special-name> ::= T <call-offset> <base encoding>
  //                      # base is the nominal target function of thunk
  CXXNameMangler Mangler(*this, Out, DD, Type);
  Mangler.getStream() << "_ZT";

  // Mangle the 'this' pointer adjustment.
  Mangler.mangleCallOffset(ThisAdjustment.NonVirtual, 
                           ThisAdjustment.Virtual.Itanium.VCallOffsetOffset);

  Mangler.mangleFunctionEncoding(DD);
}

/// Returns the mangled name for a guard variable for the passed in VarDecl.
void ItaniumMangleContextImpl::mangleStaticGuardVariable(const VarDecl *D,
                                                         raw_ostream &Out) {
  //  <special-name> ::= GV <object name>       # Guard variable for one-time
  //                                            # initialization
  CXXNameMangler Mangler(*this, Out);
  Mangler.getStream() << "_ZGV";
  Mangler.mangleName(D);
}

void ItaniumMangleContextImpl::mangleDynamicInitializer(const VarDecl *MD,
                                                        raw_ostream &Out) {
  // These symbols are internal in the Itanium ABI, so the names don't matter.
  // Clang has traditionally used this symbol and allowed LLVM to adjust it to
  // avoid duplicate symbols.
  Out << "__cxx_global_var_init";
}

void ItaniumMangleContextImpl::mangleDynamicAtExitDestructor(const VarDecl *D,
                                                             raw_ostream &Out) {
  // Prefix the mangling of D with __dtor_.
  CXXNameMangler Mangler(*this, Out);
  Mangler.getStream() << "__dtor_";
  if (shouldMangleDeclName(D))
    Mangler.mangle(D);
  else
    Mangler.getStream() << D->getName();
}

void ItaniumMangleContextImpl::mangleSEHFilterExpression(
    const NamedDecl *EnclosingDecl, raw_ostream &Out) {
  CXXNameMangler Mangler(*this, Out);
  Mangler.getStream() << "__filt_";
  if (shouldMangleDeclName(EnclosingDecl))
    Mangler.mangle(EnclosingDecl);
  else
    Mangler.getStream() << EnclosingDecl->getName();
}

void ItaniumMangleContextImpl::mangleSEHFinallyBlock(
    const NamedDecl *EnclosingDecl, raw_ostream &Out) {
  CXXNameMangler Mangler(*this, Out);
  Mangler.getStream() << "__fin_";
  if (shouldMangleDeclName(EnclosingDecl))
    Mangler.mangle(EnclosingDecl);
  else
    Mangler.getStream() << EnclosingDecl->getName();
}

void ItaniumMangleContextImpl::mangleItaniumThreadLocalInit(const VarDecl *D,
                                                            raw_ostream &Out) {
  //  <special-name> ::= TH <object name>
  CXXNameMangler Mangler(*this, Out);
  Mangler.getStream() << "_ZTH";
  Mangler.mangleName(D);
}

void
ItaniumMangleContextImpl::mangleItaniumThreadLocalWrapper(const VarDecl *D,
                                                          raw_ostream &Out) {
  //  <special-name> ::= TW <object name>
  CXXNameMangler Mangler(*this, Out);
  Mangler.getStream() << "_ZTW";
  Mangler.mangleName(D);
}

void ItaniumMangleContextImpl::mangleReferenceTemporary(const VarDecl *D,
                                                        unsigned ManglingNumber,
                                                        raw_ostream &Out) {
  // We match the GCC mangling here.
  //  <special-name> ::= GR <object name>
  CXXNameMangler Mangler(*this, Out);
  Mangler.getStream() << "_ZGR";
  Mangler.mangleName(D);
  assert(ManglingNumber > 0 && "Reference temporary mangling number is zero!");
  Mangler.mangleSeqID(ManglingNumber - 1);
}

void ItaniumMangleContextImpl::mangleCXXVTable(const CXXRecordDecl *RD,
                                               raw_ostream &Out) {
  // <special-name> ::= TV <type>  # virtual table
  CXXNameMangler Mangler(*this, Out);
  Mangler.getStream() << "_ZTV";
  Mangler.mangleNameOrStandardSubstitution(RD);
}

void ItaniumMangleContextImpl::mangleCXXVTT(const CXXRecordDecl *RD,
                                            raw_ostream &Out) {
  // <special-name> ::= TT <type>  # VTT structure
  CXXNameMangler Mangler(*this, Out);
  Mangler.getStream() << "_ZTT";
  Mangler.mangleNameOrStandardSubstitution(RD);
}

void ItaniumMangleContextImpl::mangleCXXCtorVTable(const CXXRecordDecl *RD,
                                                   int64_t Offset,
                                                   const CXXRecordDecl *Type,
                                                   raw_ostream &Out) {
  // <special-name> ::= TC <type> <offset number> _ <base type>
  CXXNameMangler Mangler(*this, Out);
  Mangler.getStream() << "_ZTC";
  Mangler.mangleNameOrStandardSubstitution(RD);
  Mangler.getStream() << Offset;
  Mangler.getStream() << '_';
  Mangler.mangleNameOrStandardSubstitution(Type);
}

void ItaniumMangleContextImpl::mangleCXXRTTI(QualType Ty, raw_ostream &Out) {
  // <special-name> ::= TI <type>  # typeinfo structure
  assert(!Ty.hasQualifiers() && "RTTI info cannot have top-level qualifiers");
  CXXNameMangler Mangler(*this, Out);
  Mangler.getStream() << "_ZTI";
  Mangler.mangleType(Ty);
}

void ItaniumMangleContextImpl::mangleCXXRTTIName(QualType Ty,
                                                 raw_ostream &Out) {
  // <special-name> ::= TS <type>  # typeinfo name (null terminated byte string)
  CXXNameMangler Mangler(*this, Out);
  Mangler.getStream() << "_ZTS";
  Mangler.mangleType(Ty);
}

void ItaniumMangleContextImpl::mangleTypeName(QualType Ty, raw_ostream &Out) {
  mangleCXXRTTIName(Ty, Out);
}

void ItaniumMangleContextImpl::mangleStringLiteral(const StringLiteral *, raw_ostream &) {
  llvm_unreachable("Can't mangle string literals");
}

ItaniumMangleContext *
ItaniumMangleContext::create(ASTContext &Context, DiagnosticsEngine &Diags) {
  return new ItaniumMangleContextImpl(Context, Diags);
}

