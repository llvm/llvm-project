//===--- HLSLBuiltinTypeDeclBuilder.cpp - HLSL Builtin Type Decl Builder --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper classes for creating HLSL builtin class types. Used by external HLSL
// sema source.
//
//===----------------------------------------------------------------------===//

#include "HLSLBuiltinTypeDeclBuilder.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/TypeBase.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaHLSL.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm::hlsl;

namespace clang {

namespace hlsl {

namespace {

static FunctionDecl *lookupBuiltinFunction(Sema &S, StringRef Name) {
  IdentifierInfo &II =
      S.getASTContext().Idents.get(Name, tok::TokenKind::identifier);
  DeclarationNameInfo NameInfo =
      DeclarationNameInfo(DeclarationName(&II), SourceLocation());
  LookupResult R(S, NameInfo, Sema::LookupOrdinaryName);
  // AllowBuiltinCreation is false but LookupDirect will create
  // the builtin when searching the global scope anyways...
  S.LookupName(R, S.getCurScope());
  // FIXME: If the builtin function was user-declared in global scope,
  // this assert *will* fail. Should this call LookupBuiltin instead?
  assert(R.isSingleResult() &&
         "Since this is a builtin it should always resolve!");
  return cast<FunctionDecl>(R.getFoundDecl());
}
} // namespace

// Builder for template arguments of builtin types. Used internally
// by BuiltinTypeDeclBuilder.
struct TemplateParameterListBuilder {
  BuiltinTypeDeclBuilder &Builder;
  llvm::SmallVector<NamedDecl *> Params;

  TemplateParameterListBuilder(BuiltinTypeDeclBuilder &RB) : Builder(RB) {}
  ~TemplateParameterListBuilder();

  TemplateParameterListBuilder &
  addTypeParameter(StringRef Name, QualType DefaultValue = QualType());

  ConceptSpecializationExpr *
  constructConceptSpecializationExpr(Sema &S, ConceptDecl *CD);

  BuiltinTypeDeclBuilder &finalizeTemplateArgs(ConceptDecl *CD = nullptr);
};

// Builder for methods or constructors of builtin types. Allows creating methods
// or constructors of builtin types using the builder pattern like this:
//
//   BuiltinTypeMethodBuilder(RecordBuilder, "MethodName", ReturnType)
//       .addParam("param_name", Type, InOutModifier)
//       .callBuiltin("builtin_name", BuiltinParams...)
//       .finalize();
//
// The builder needs to have all of the parameters before it can create
// a CXXMethodDecl or CXXConstructorDecl. It collects them in addParam calls and
// when a first method that builds the body is called or when access to 'this`
// is needed it creates the CXXMethodDecl/CXXConstructorDecl and ParmVarDecls
// instances. These can then be referenced from the body building methods.
// Destructor or an explicit call to finalize() will complete the method
// definition.
//
// The callBuiltin helper method accepts constants via `Expr *` or placeholder
// value arguments to indicate which function arguments to forward to the
// builtin.
//
// If the method that is being built has a non-void return type the
// finalize() will create a return statement with the value of the last
// statement (unless the last statement is already a ReturnStmt or the return
// value is void).
struct BuiltinTypeMethodBuilder {
private:
  struct Param {
    const IdentifierInfo &NameII;
    QualType Ty;
    HLSLParamModifierAttr::Spelling Modifier;
    Param(const IdentifierInfo &NameII, QualType Ty,
          HLSLParamModifierAttr::Spelling Modifier)
        : NameII(NameII), Ty(Ty), Modifier(Modifier) {}
  };

  BuiltinTypeDeclBuilder &DeclBuilder;
  DeclarationName Name;
  QualType ReturnTy;
  // method or constructor declaration
  // (CXXConstructorDecl derives from CXXMethodDecl)
  CXXMethodDecl *Method;
  bool IsConst;
  bool IsCtor;
  llvm::SmallVector<Param> Params;
  llvm::SmallVector<Stmt *> StmtsList;

  // Argument placeholders, inspired by std::placeholder. These are the indices
  // of arguments to forward to `callBuiltin` and other method builder methods.
  // Additional special values are:
  //   Handle   - refers to the resource handle.
  //   LastStmt - refers to the last statement in the method body; referencing
  //              LastStmt will remove the statement from the method body since
  //              it will be linked from the new expression being constructed.
  enum class PlaceHolder { _0, _1, _2, _3, _4, Handle = 128, LastStmt };

  Expr *convertPlaceholder(PlaceHolder PH);
  Expr *convertPlaceholder(Expr *E) { return E; }

public:
  friend BuiltinTypeDeclBuilder;

  BuiltinTypeMethodBuilder(BuiltinTypeDeclBuilder &DB, DeclarationName &Name,
                           QualType ReturnTy, bool IsConst = false,
                           bool IsCtor = false)
      : DeclBuilder(DB), Name(Name), ReturnTy(ReturnTy), Method(nullptr),
        IsConst(IsConst), IsCtor(IsCtor) {}

  BuiltinTypeMethodBuilder(BuiltinTypeDeclBuilder &DB, StringRef NameStr,
                           QualType ReturnTy, bool IsConst = false,
                           bool IsCtor = false);
  BuiltinTypeMethodBuilder(const BuiltinTypeMethodBuilder &Other) = delete;

  ~BuiltinTypeMethodBuilder() { finalize(); }

  BuiltinTypeMethodBuilder &
  operator=(const BuiltinTypeMethodBuilder &Other) = delete;

  BuiltinTypeMethodBuilder &addParam(StringRef Name, QualType Ty,
                                     HLSLParamModifierAttr::Spelling Modifier =
                                         HLSLParamModifierAttr::Keyword_in);
  template <typename... Ts>
  BuiltinTypeMethodBuilder &callBuiltin(StringRef BuiltinName,
                                        QualType ReturnType, Ts... ArgSpecs);
  template <typename TLHS, typename TRHS>
  BuiltinTypeMethodBuilder &assign(TLHS LHS, TRHS RHS);
  template <typename T> BuiltinTypeMethodBuilder &dereference(T Ptr);
  BuiltinTypeDeclBuilder &finalize();
  Expr *getResourceHandleExpr();

private:
  void createDecl();

  // Makes sure the declaration is created; should be called before any
  // statement added to the body or when access to 'this' is needed.
  void ensureCompleteDecl() {
    if (!Method)
      createDecl();
  }
};

TemplateParameterListBuilder::~TemplateParameterListBuilder() {
  finalizeTemplateArgs();
}

TemplateParameterListBuilder &
TemplateParameterListBuilder::addTypeParameter(StringRef Name,
                                               QualType DefaultValue) {
  assert(!Builder.Record->isCompleteDefinition() &&
         "record is already complete");
  ASTContext &AST = Builder.SemaRef.getASTContext();
  unsigned Position = static_cast<unsigned>(Params.size());
  auto *Decl = TemplateTypeParmDecl::Create(
      AST, Builder.Record->getDeclContext(), SourceLocation(), SourceLocation(),
      /* TemplateDepth */ 0, Position,
      &AST.Idents.get(Name, tok::TokenKind::identifier),
      /* Typename */ true,
      /* ParameterPack */ false,
      /* HasTypeConstraint*/ false);
  if (!DefaultValue.isNull())
    Decl->setDefaultArgument(AST,
                             Builder.SemaRef.getTrivialTemplateArgumentLoc(
                                 DefaultValue, QualType(), SourceLocation()));

  Params.emplace_back(Decl);
  return *this;
}

// The concept specialization expression (CSE) constructed in
// constructConceptSpecializationExpr is constructed so that it
// matches the CSE that is constructed when parsing the below C++ code:
//
// template<typename T>
// concept is_typed_resource_element_compatible =
// __builtin_hlsl_typed_resource_element_compatible<T>
//
// template<typename element_type> requires
// is_typed_resource_element_compatible<element_type>
// struct RWBuffer {
//     element_type Val;
// };
//
// int fn() {
//     RWBuffer<int> Buf;
// }
//
// When dumping the AST and filtering for "RWBuffer", the resulting AST
// structure is what we're trying to construct below, specifically the
// CSE portion.
ConceptSpecializationExpr *
TemplateParameterListBuilder::constructConceptSpecializationExpr(
    Sema &S, ConceptDecl *CD) {
  ASTContext &Context = S.getASTContext();
  SourceLocation Loc = Builder.Record->getBeginLoc();
  DeclarationNameInfo DNI(CD->getDeclName(), Loc);
  NestedNameSpecifierLoc NNSLoc;
  DeclContext *DC = Builder.Record->getDeclContext();
  TemplateArgumentListInfo TALI(Loc, Loc);

  // Assume that the concept decl has just one template parameter
  // This parameter should have been added when CD was constructed
  // in getTypedBufferConceptDecl
  assert(CD->getTemplateParameters()->size() == 1 &&
         "unexpected concept decl parameter count");
  TemplateTypeParmDecl *ConceptTTPD =
      dyn_cast<TemplateTypeParmDecl>(CD->getTemplateParameters()->getParam(0));

  // this TemplateTypeParmDecl is the template for the resource, and is
  // used to construct a template argumentthat will be used
  // to construct the ImplicitConceptSpecializationDecl
  TemplateTypeParmDecl *T = TemplateTypeParmDecl::Create(
      Context,                          // AST context
      Builder.Record->getDeclContext(), // DeclContext
      SourceLocation(), SourceLocation(),
      /*D=*/0,                    // Depth in the template parameter list
      /*P=*/0,                    // Position in the template parameter list
      /*Id=*/nullptr,             // Identifier for 'T'
      /*Typename=*/true,          // Indicates this is a 'typename' or 'class'
      /*ParameterPack=*/false,    // Not a parameter pack
      /*HasTypeConstraint=*/false // Has no type constraint
  );

  T->setDeclContext(DC);

  QualType ConceptTType = Context.getTypeDeclType(ConceptTTPD);

  // this is the 2nd template argument node, on which
  // the concept constraint is actually being applied: 'element_type'
  TemplateArgument ConceptTA = TemplateArgument(ConceptTType);

  QualType CSETType = Context.getTypeDeclType(T);

  // this is the 1st template argument node, which represents
  // the abstract type that a concept would refer to: 'T'
  TemplateArgument CSETA = TemplateArgument(CSETType);

  ImplicitConceptSpecializationDecl *ImplicitCSEDecl =
      ImplicitConceptSpecializationDecl::Create(
          Context, Builder.Record->getDeclContext(), Loc, {CSETA});

  // Constraint satisfaction is used to construct the
  // ConceptSpecailizationExpr, and represents the 2nd Template Argument,
  // located at the bottom of the sample AST above.
  const ConstraintSatisfaction CS(CD, {ConceptTA});
  TemplateArgumentLoc TAL =
      S.getTrivialTemplateArgumentLoc(ConceptTA, QualType(), SourceLocation());

  TALI.addArgument(TAL);
  const ASTTemplateArgumentListInfo *ATALI =
      ASTTemplateArgumentListInfo::Create(Context, TALI);

  // In the concept reference, ATALI is what adds the extra
  // TemplateArgument node underneath CSE
  ConceptReference *CR =
      ConceptReference::Create(Context, NNSLoc, Loc, DNI, CD, CD, ATALI);

  ConceptSpecializationExpr *CSE =
      ConceptSpecializationExpr::Create(Context, CR, ImplicitCSEDecl, &CS);

  return CSE;
}

BuiltinTypeDeclBuilder &
TemplateParameterListBuilder::finalizeTemplateArgs(ConceptDecl *CD) {
  if (Params.empty())
    return Builder;

  ASTContext &AST = Builder.SemaRef.Context;
  ConceptSpecializationExpr *CSE =
      CD ? constructConceptSpecializationExpr(Builder.SemaRef, CD) : nullptr;
  auto *ParamList = TemplateParameterList::Create(
      AST, SourceLocation(), SourceLocation(), Params, SourceLocation(), CSE);
  Builder.Template = ClassTemplateDecl::Create(
      AST, Builder.Record->getDeclContext(), SourceLocation(),
      DeclarationName(Builder.Record->getIdentifier()), ParamList,
      Builder.Record);

  Builder.Record->setDescribedClassTemplate(Builder.Template);
  Builder.Template->setImplicit(true);
  Builder.Template->setLexicalDeclContext(Builder.Record->getDeclContext());

  // NOTE: setPreviousDecl before addDecl so new decl replace old decl when
  // make visible.
  Builder.Template->setPreviousDecl(Builder.PrevTemplate);
  Builder.Record->getDeclContext()->addDecl(Builder.Template);
  Params.clear();

  return Builder;
}

Expr *BuiltinTypeMethodBuilder::convertPlaceholder(PlaceHolder PH) {
  if (PH == PlaceHolder::Handle)
    return getResourceHandleExpr();

  if (PH == PlaceHolder::LastStmt) {
    assert(!StmtsList.empty() && "no statements in the list");
    Stmt *LastStmt = StmtsList.pop_back_val();
    assert(isa<ValueStmt>(LastStmt) && "last statement does not have a value");
    return cast<ValueStmt>(LastStmt)->getExprStmt();
  }

  ASTContext &AST = DeclBuilder.SemaRef.getASTContext();
  ParmVarDecl *ParamDecl = Method->getParamDecl(static_cast<unsigned>(PH));
  return DeclRefExpr::Create(
      AST, NestedNameSpecifierLoc(), SourceLocation(), ParamDecl, false,
      DeclarationNameInfo(ParamDecl->getDeclName(), SourceLocation()),
      ParamDecl->getType(), VK_PRValue);
}

BuiltinTypeMethodBuilder::BuiltinTypeMethodBuilder(BuiltinTypeDeclBuilder &DB,
                                                   StringRef NameStr,
                                                   QualType ReturnTy,
                                                   bool IsConst, bool IsCtor)
    : DeclBuilder(DB), ReturnTy(ReturnTy), Method(nullptr), IsConst(IsConst),
      IsCtor(IsCtor) {

  assert((!NameStr.empty() || IsCtor) && "method needs a name");
  assert(((IsCtor && !IsConst) || !IsCtor) && "constructor cannot be const");

  ASTContext &AST = DB.SemaRef.getASTContext();
  if (IsCtor) {
    Name = AST.DeclarationNames.getCXXConstructorName(
        AST.getCanonicalTagType(DB.Record));
  } else {
    const IdentifierInfo &II =
        AST.Idents.get(NameStr, tok::TokenKind::identifier);
    Name = DeclarationName(&II);
  }
}

BuiltinTypeMethodBuilder &
BuiltinTypeMethodBuilder::addParam(StringRef Name, QualType Ty,
                                   HLSLParamModifierAttr::Spelling Modifier) {
  assert(Method == nullptr && "Cannot add param, method already created");
  const IdentifierInfo &II = DeclBuilder.SemaRef.getASTContext().Idents.get(
      Name, tok::TokenKind::identifier);
  Params.emplace_back(II, Ty, Modifier);
  return *this;
}

void BuiltinTypeMethodBuilder::createDecl() {
  assert(Method == nullptr && "Method or constructor is already created");

  // create method or constructor type
  ASTContext &AST = DeclBuilder.SemaRef.getASTContext();
  SmallVector<QualType> ParamTypes;
  for (Param &MP : Params)
    ParamTypes.emplace_back(MP.Ty);

  FunctionProtoType::ExtProtoInfo ExtInfo;
  if (IsConst)
    ExtInfo.TypeQuals.addConst();

  QualType FuncTy = AST.getFunctionType(ReturnTy, ParamTypes, ExtInfo);

  // create method or constructor decl
  auto *TSInfo = AST.getTrivialTypeSourceInfo(FuncTy, SourceLocation());
  DeclarationNameInfo NameInfo = DeclarationNameInfo(Name, SourceLocation());
  if (IsCtor)
    Method = CXXConstructorDecl::Create(
        AST, DeclBuilder.Record, SourceLocation(), NameInfo, FuncTy, TSInfo,
        ExplicitSpecifier(), false, true, false,
        ConstexprSpecKind::Unspecified);
  else
    Method =
        CXXMethodDecl::Create(AST, DeclBuilder.Record, SourceLocation(),
                              NameInfo, FuncTy, TSInfo, SC_None, false, false,
                              ConstexprSpecKind::Unspecified, SourceLocation());

  // create params & set them to the function prototype
  SmallVector<ParmVarDecl *> ParmDecls;
  unsigned CurScopeDepth = DeclBuilder.SemaRef.getCurScope()->getDepth();
  auto FnProtoLoc =
      Method->getTypeSourceInfo()->getTypeLoc().getAs<FunctionProtoTypeLoc>();
  for (int I = 0, E = Params.size(); I != E; I++) {
    Param &MP = Params[I];
    ParmVarDecl *Parm = ParmVarDecl::Create(
        AST, Method->getDeclContext(), SourceLocation(), SourceLocation(),
        &MP.NameII, MP.Ty,
        AST.getTrivialTypeSourceInfo(MP.Ty, SourceLocation()), SC_None,
        nullptr);
    if (MP.Modifier != HLSLParamModifierAttr::Keyword_in) {
      auto *Mod =
          HLSLParamModifierAttr::Create(AST, SourceRange(), MP.Modifier);
      Parm->addAttr(Mod);
    }
    Parm->setScopeInfo(CurScopeDepth, I);
    ParmDecls.push_back(Parm);
    FnProtoLoc.setParam(I, Parm);
  }
  Method->setParams({ParmDecls});
}

Expr *BuiltinTypeMethodBuilder::getResourceHandleExpr() {
  ensureCompleteDecl();

  ASTContext &AST = DeclBuilder.SemaRef.getASTContext();
  CXXThisExpr *This = CXXThisExpr::Create(
      AST, SourceLocation(), Method->getFunctionObjectParameterType(), true);
  FieldDecl *HandleField = DeclBuilder.getResourceHandleField();
  return MemberExpr::CreateImplicit(AST, This, false, HandleField,
                                    HandleField->getType(), VK_LValue,
                                    OK_Ordinary);
}

template <typename... Ts>
BuiltinTypeMethodBuilder &
BuiltinTypeMethodBuilder::callBuiltin(StringRef BuiltinName,
                                      QualType ReturnType, Ts... ArgSpecs) {
  std::array<Expr *, sizeof...(ArgSpecs)> Args{
      convertPlaceholder(std::forward<Ts>(ArgSpecs))...};

  ensureCompleteDecl();

  ASTContext &AST = DeclBuilder.SemaRef.getASTContext();
  FunctionDecl *FD = lookupBuiltinFunction(DeclBuilder.SemaRef, BuiltinName);
  DeclRefExpr *DRE = DeclRefExpr::Create(
      AST, NestedNameSpecifierLoc(), SourceLocation(), FD, false,
      FD->getNameInfo(), AST.BuiltinFnTy, VK_PRValue);

  auto *ImpCast = ImplicitCastExpr::Create(
      AST, AST.getPointerType(FD->getType()), CK_BuiltinFnToFnPtr, DRE, nullptr,
      VK_PRValue, FPOptionsOverride());

  if (ReturnType.isNull())
    ReturnType = FD->getReturnType();

  Expr *Call = CallExpr::Create(AST, ImpCast, Args, ReturnType, VK_PRValue,
                                SourceLocation(), FPOptionsOverride());
  StmtsList.push_back(Call);
  return *this;
}

template <typename TLHS, typename TRHS>
BuiltinTypeMethodBuilder &BuiltinTypeMethodBuilder::assign(TLHS LHS, TRHS RHS) {
  Expr *LHSExpr = convertPlaceholder(LHS);
  Expr *RHSExpr = convertPlaceholder(RHS);
  Stmt *AssignStmt = BinaryOperator::Create(
      DeclBuilder.SemaRef.getASTContext(), LHSExpr, RHSExpr, BO_Assign,
      LHSExpr->getType(), ExprValueKind::VK_PRValue,
      ExprObjectKind::OK_Ordinary, SourceLocation(), FPOptionsOverride());
  StmtsList.push_back(AssignStmt);
  return *this;
}

template <typename T>
BuiltinTypeMethodBuilder &BuiltinTypeMethodBuilder::dereference(T Ptr) {
  Expr *PtrExpr = convertPlaceholder(Ptr);
  Expr *Deref =
      UnaryOperator::Create(DeclBuilder.SemaRef.getASTContext(), PtrExpr,
                            UO_Deref, PtrExpr->getType()->getPointeeType(),
                            VK_PRValue, OK_Ordinary, SourceLocation(),
                            /*CanOverflow=*/false, FPOptionsOverride());
  StmtsList.push_back(Deref);
  return *this;
}

BuiltinTypeDeclBuilder &BuiltinTypeMethodBuilder::finalize() {
  assert(!DeclBuilder.Record->isCompleteDefinition() &&
         "record is already complete");

  ensureCompleteDecl();

  if (!Method->hasBody()) {
    ASTContext &AST = DeclBuilder.SemaRef.getASTContext();
    assert((ReturnTy == AST.VoidTy || !StmtsList.empty()) &&
           "nothing to return from non-void method");
    if (ReturnTy != AST.VoidTy) {
      if (Expr *LastExpr = dyn_cast<Expr>(StmtsList.back())) {
        assert(AST.hasSameUnqualifiedType(LastExpr->getType(),
                                          ReturnTy.getNonReferenceType()) &&
               "Return type of the last statement must match the return type "
               "of the method");
        if (!isa<ReturnStmt>(LastExpr)) {
          StmtsList.pop_back();
          StmtsList.push_back(
              ReturnStmt::Create(AST, SourceLocation(), LastExpr, nullptr));
        }
      }
    }

    Method->setBody(CompoundStmt::Create(AST, StmtsList, FPOptionsOverride(),
                                         SourceLocation(), SourceLocation()));
    Method->setLexicalDeclContext(DeclBuilder.Record);
    Method->setAccess(AccessSpecifier::AS_public);
    Method->addAttr(AlwaysInlineAttr::CreateImplicit(
        AST, SourceRange(), AlwaysInlineAttr::CXX11_clang_always_inline));
    DeclBuilder.Record->addDecl(Method);
  }
  return DeclBuilder;
}

BuiltinTypeDeclBuilder::BuiltinTypeDeclBuilder(Sema &SemaRef, CXXRecordDecl *R)
    : SemaRef(SemaRef), Record(R) {
  Record->startDefinition();
  Template = Record->getDescribedClassTemplate();
}

BuiltinTypeDeclBuilder::BuiltinTypeDeclBuilder(Sema &SemaRef,
                                               NamespaceDecl *Namespace,
                                               StringRef Name)
    : SemaRef(SemaRef), HLSLNamespace(Namespace) {
  ASTContext &AST = SemaRef.getASTContext();
  IdentifierInfo &II = AST.Idents.get(Name, tok::TokenKind::identifier);

  LookupResult Result(SemaRef, &II, SourceLocation(), Sema::LookupTagName);
  CXXRecordDecl *PrevDecl = nullptr;
  if (SemaRef.LookupQualifiedName(Result, HLSLNamespace)) {
    // Declaration already exists (from precompiled headers)
    NamedDecl *Found = Result.getFoundDecl();
    if (auto *TD = dyn_cast<ClassTemplateDecl>(Found)) {
      PrevDecl = TD->getTemplatedDecl();
      PrevTemplate = TD;
    } else
      PrevDecl = dyn_cast<CXXRecordDecl>(Found);
    assert(PrevDecl && "Unexpected lookup result type.");
  }

  if (PrevDecl && PrevDecl->isCompleteDefinition()) {
    Record = PrevDecl;
    Template = PrevTemplate;
    return;
  }

  Record =
      CXXRecordDecl::Create(AST, TagDecl::TagKind::Class, HLSLNamespace,
                            SourceLocation(), SourceLocation(), &II, PrevDecl);
  Record->setImplicit(true);
  Record->setLexicalDeclContext(HLSLNamespace);
  Record->setHasExternalLexicalStorage();

  // Don't let anyone derive from built-in types.
  Record->addAttr(
      FinalAttr::CreateImplicit(AST, SourceRange(), FinalAttr::Keyword_final));
}

BuiltinTypeDeclBuilder::~BuiltinTypeDeclBuilder() {
  if (HLSLNamespace && !Template && Record->getDeclContext() == HLSLNamespace)
    HLSLNamespace->addDecl(Record);
}

BuiltinTypeDeclBuilder &
BuiltinTypeDeclBuilder::addMemberVariable(StringRef Name, QualType Type,
                                          llvm::ArrayRef<Attr *> Attrs,
                                          AccessSpecifier Access) {
  assert(!Record->isCompleteDefinition() && "record is already complete");
  assert(Record->isBeingDefined() &&
         "Definition must be started before adding members!");
  ASTContext &AST = Record->getASTContext();

  IdentifierInfo &II = AST.Idents.get(Name, tok::TokenKind::identifier);
  TypeSourceInfo *MemTySource =
      AST.getTrivialTypeSourceInfo(Type, SourceLocation());
  auto *Field = FieldDecl::Create(
      AST, Record, SourceLocation(), SourceLocation(), &II, Type, MemTySource,
      nullptr, false, InClassInitStyle::ICIS_NoInit);
  Field->setAccess(Access);
  Field->setImplicit(true);
  for (Attr *A : Attrs) {
    if (A)
      Field->addAttr(A);
  }

  Record->addDecl(Field);
  Fields[Name] = Field;
  return *this;
}

BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addHandleMember(
    ResourceClass RC, bool IsROV, bool RawBuffer, AccessSpecifier Access) {
  assert(!Record->isCompleteDefinition() && "record is already complete");

  ASTContext &Ctx = SemaRef.getASTContext();
  TypeSourceInfo *ElementTypeInfo =
      Ctx.getTrivialTypeSourceInfo(getHandleElementType(), SourceLocation());

  // add handle member with resource type attributes
  QualType AttributedResTy = QualType();
  SmallVector<const Attr *> Attrs = {
      HLSLResourceClassAttr::CreateImplicit(Ctx, RC),
      IsROV ? HLSLROVAttr::CreateImplicit(Ctx) : nullptr,
      RawBuffer ? HLSLRawBufferAttr::CreateImplicit(Ctx) : nullptr,
      ElementTypeInfo
          ? HLSLContainedTypeAttr::CreateImplicit(Ctx, ElementTypeInfo)
          : nullptr};
  if (CreateHLSLAttributedResourceType(SemaRef, Ctx.HLSLResourceTy, Attrs,
                                       AttributedResTy))
    addMemberVariable("__handle", AttributedResTy, {}, Access);
  return *this;
}

// Adds default constructor to the resource class:
// Resource::Resource()
BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addDefaultHandleConstructor() {
  if (Record->isCompleteDefinition())
    return *this;

  using PH = BuiltinTypeMethodBuilder::PlaceHolder;
  QualType HandleType = getResourceHandleField()->getType();
  return BuiltinTypeMethodBuilder(*this, "", SemaRef.getASTContext().VoidTy,
                                  false, true)
      .callBuiltin("__builtin_hlsl_resource_uninitializedhandle", HandleType,
                   PH::Handle)
      .assign(PH::Handle, PH::LastStmt)
      .finalize();
}

BuiltinTypeDeclBuilder &
BuiltinTypeDeclBuilder::addHandleConstructorFromBinding() {
  if (Record->isCompleteDefinition())
    return *this;

  using PH = BuiltinTypeMethodBuilder::PlaceHolder;
  ASTContext &AST = SemaRef.getASTContext();
  QualType HandleType = getResourceHandleField()->getType();

  return BuiltinTypeMethodBuilder(*this, "", AST.VoidTy, false, true)
      .addParam("registerNo", AST.UnsignedIntTy)
      .addParam("spaceNo", AST.UnsignedIntTy)
      .addParam("range", AST.IntTy)
      .addParam("index", AST.UnsignedIntTy)
      .addParam("name", AST.getPointerType(AST.CharTy.withConst()))
      .callBuiltin("__builtin_hlsl_resource_handlefrombinding", HandleType,
                   PH::Handle, PH::_0, PH::_1, PH::_2, PH::_3, PH::_4)
      .assign(PH::Handle, PH::LastStmt)
      .finalize();
}

BuiltinTypeDeclBuilder &
BuiltinTypeDeclBuilder::addHandleConstructorFromImplicitBinding() {
  if (Record->isCompleteDefinition())
    return *this;

  using PH = BuiltinTypeMethodBuilder::PlaceHolder;
  ASTContext &AST = SemaRef.getASTContext();
  QualType HandleType = getResourceHandleField()->getType();

  return BuiltinTypeMethodBuilder(*this, "", AST.VoidTy, false, true)
      .addParam("spaceNo", AST.UnsignedIntTy)
      .addParam("range", AST.IntTy)
      .addParam("index", AST.UnsignedIntTy)
      .addParam("orderId", AST.UnsignedIntTy)
      .addParam("name", AST.getPointerType(AST.CharTy.withConst()))
      .callBuiltin("__builtin_hlsl_resource_handlefromimplicitbinding",
                   HandleType, PH::Handle, PH::_0, PH::_1, PH::_2, PH::_3,
                   PH::_4)
      .assign(PH::Handle, PH::LastStmt)
      .finalize();
}

BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addArraySubscriptOperators() {
  ASTContext &AST = Record->getASTContext();
  DeclarationName Subscript =
      AST.DeclarationNames.getCXXOperatorName(OO_Subscript);

  addHandleAccessFunction(Subscript, /*IsConst=*/true, /*IsRef=*/true);
  if (getResourceAttrs().ResourceClass == llvm::dxil::ResourceClass::UAV)
    addHandleAccessFunction(Subscript, /*IsConst=*/false, /*IsRef=*/true);

  return *this;
}

BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addLoadMethods() {
  if (Record->isCompleteDefinition())
    return *this;

  ASTContext &AST = Record->getASTContext();
  IdentifierInfo &II = AST.Idents.get("Load", tok::TokenKind::identifier);
  DeclarationName Load(&II);
  // TODO: We also need versions with status for CheckAccessFullyMapped.
  addHandleAccessFunction(Load, /*IsConst=*/false, /*IsRef=*/false);

  return *this;
}

FieldDecl *BuiltinTypeDeclBuilder::getResourceHandleField() const {
  auto I = Fields.find("__handle");
  assert(I != Fields.end() &&
         I->second->getType()->isHLSLAttributedResourceType() &&
         "record does not have resource handle field");
  return I->second;
}

QualType BuiltinTypeDeclBuilder::getFirstTemplateTypeParam() {
  assert(Template && "record it not a template");
  if (const auto *TTD = dyn_cast<TemplateTypeParmDecl>(
          Template->getTemplateParameters()->getParam(0))) {
    return QualType(TTD->getTypeForDecl(), 0);
  }
  return QualType();
}

QualType BuiltinTypeDeclBuilder::getHandleElementType() {
  if (Template)
    return getFirstTemplateTypeParam();
  // TODO: Should we default to VoidTy? Using `i8` is arguably ambiguous.
  return SemaRef.getASTContext().Char8Ty;
}

HLSLAttributedResourceType::Attributes
BuiltinTypeDeclBuilder::getResourceAttrs() const {
  QualType HandleType = getResourceHandleField()->getType();
  return cast<HLSLAttributedResourceType>(HandleType)->getAttrs();
}

// BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::startDefinition() {
//   assert(!Record->isCompleteDefinition() && "record is already complete");
//   Record->startDefinition();
//   return *this;
// }

BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::completeDefinition() {
  assert(!Record->isCompleteDefinition() && "record is already complete");
  assert(Record->isBeingDefined() &&
         "Definition must be started before completing it.");

  Record->completeDefinition();
  return *this;
}

Expr *BuiltinTypeDeclBuilder::getConstantIntExpr(int value) {
  ASTContext &AST = SemaRef.getASTContext();
  return IntegerLiteral::Create(
      AST, llvm::APInt(AST.getTypeSize(AST.IntTy), value, true), AST.IntTy,
      SourceLocation());
}

BuiltinTypeDeclBuilder &
BuiltinTypeDeclBuilder::addSimpleTemplateParams(ArrayRef<StringRef> Names,
                                                ConceptDecl *CD = nullptr) {
  if (Record->isCompleteDefinition()) {
    assert(Template && "existing record it not a template");
    assert(Template->getTemplateParameters()->size() == Names.size() &&
           "template param count mismatch");
    return *this;
  }

  TemplateParameterListBuilder Builder = TemplateParameterListBuilder(*this);
  for (StringRef Name : Names)
    Builder.addTypeParameter(Name);
  return Builder.finalizeTemplateArgs(CD);
}

BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addIncrementCounterMethod() {
  using PH = BuiltinTypeMethodBuilder::PlaceHolder;
  return BuiltinTypeMethodBuilder(*this, "IncrementCounter",
                                  SemaRef.getASTContext().UnsignedIntTy)
      .callBuiltin("__builtin_hlsl_buffer_update_counter", QualType(),
                   PH::Handle, getConstantIntExpr(1))
      .finalize();
}

BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addDecrementCounterMethod() {
  using PH = BuiltinTypeMethodBuilder::PlaceHolder;
  return BuiltinTypeMethodBuilder(*this, "DecrementCounter",
                                  SemaRef.getASTContext().UnsignedIntTy)
      .callBuiltin("__builtin_hlsl_buffer_update_counter", QualType(),
                   PH::Handle, getConstantIntExpr(-1))
      .finalize();
}

BuiltinTypeDeclBuilder &
BuiltinTypeDeclBuilder::addHandleAccessFunction(DeclarationName &Name,
                                                bool IsConst, bool IsRef) {
  assert(!Record->isCompleteDefinition() && "record is already complete");
  ASTContext &AST = SemaRef.getASTContext();
  using PH = BuiltinTypeMethodBuilder::PlaceHolder;

  QualType ElemTy = getHandleElementType();
  QualType AddrSpaceElemTy =
      AST.getAddrSpaceQualType(ElemTy, LangAS::hlsl_device);
  QualType ElemPtrTy = AST.getPointerType(AddrSpaceElemTy);
  QualType ReturnTy;

  if (IsRef) {
    ReturnTy = AddrSpaceElemTy;
    if (IsConst)
      ReturnTy.addConst();
    ReturnTy = AST.getLValueReferenceType(ReturnTy);
  } else {
    ReturnTy = ElemTy;
    if (IsConst)
      ReturnTy.addConst();
  }

  return BuiltinTypeMethodBuilder(*this, Name, ReturnTy, IsConst)
      .addParam("Index", AST.UnsignedIntTy)
      .callBuiltin("__builtin_hlsl_resource_getpointer", ElemPtrTy, PH::Handle,
                   PH::_0)
      .dereference(PH::LastStmt)
      .finalize();
}

BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addAppendMethod() {
  using PH = BuiltinTypeMethodBuilder::PlaceHolder;
  ASTContext &AST = SemaRef.getASTContext();
  QualType ElemTy = getHandleElementType();
  QualType AddrSpaceElemTy =
      AST.getAddrSpaceQualType(ElemTy, LangAS::hlsl_device);
  return BuiltinTypeMethodBuilder(*this, "Append", AST.VoidTy)
      .addParam("value", ElemTy)
      .callBuiltin("__builtin_hlsl_buffer_update_counter", AST.UnsignedIntTy,
                   PH::Handle, getConstantIntExpr(1))
      .callBuiltin("__builtin_hlsl_resource_getpointer",
                   AST.getPointerType(AddrSpaceElemTy), PH::Handle,
                   PH::LastStmt)
      .dereference(PH::LastStmt)
      .assign(PH::LastStmt, PH::_0)
      .finalize();
}

BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addConsumeMethod() {
  using PH = BuiltinTypeMethodBuilder::PlaceHolder;
  ASTContext &AST = SemaRef.getASTContext();
  QualType ElemTy = getHandleElementType();
  QualType AddrSpaceElemTy =
      AST.getAddrSpaceQualType(ElemTy, LangAS::hlsl_device);
  return BuiltinTypeMethodBuilder(*this, "Consume", ElemTy)
      .callBuiltin("__builtin_hlsl_buffer_update_counter", AST.UnsignedIntTy,
                   PH::Handle, getConstantIntExpr(-1))
      .callBuiltin("__builtin_hlsl_resource_getpointer",
                   AST.getPointerType(AddrSpaceElemTy), PH::Handle,
                   PH::LastStmt)
      .dereference(PH::LastStmt)
      .finalize();
}

} // namespace hlsl
} // namespace clang
