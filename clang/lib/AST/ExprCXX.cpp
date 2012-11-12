//===--- ExprCXX.cpp - (C++) Expression AST Node Implementation -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the subclesses of Expr class declared in ExprCXX.h
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/IdentifierTable.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/TypeLoc.h"
using namespace clang;


//===----------------------------------------------------------------------===//
//  Child Iterators for iterating over subexpressions/substatements
//===----------------------------------------------------------------------===//

bool CXXTypeidExpr::isPotentiallyEvaluated() const {
  if (isTypeOperand())
    return false;

  // C++11 [expr.typeid]p3:
  //   When typeid is applied to an expression other than a glvalue of
  //   polymorphic class type, [...] the expression is an unevaluated operand.
  const Expr *E = getExprOperand();
  if (const CXXRecordDecl *RD = E->getType()->getAsCXXRecordDecl())
    if (RD->isPolymorphic() && E->isGLValue())
      return true;

  return false;
}

QualType CXXTypeidExpr::getTypeOperand() const {
  assert(isTypeOperand() && "Cannot call getTypeOperand for typeid(expr)");
  return Operand.get<TypeSourceInfo *>()->getType().getNonReferenceType()
                                                        .getUnqualifiedType();
}

QualType CXXUuidofExpr::getTypeOperand() const {
  assert(isTypeOperand() && "Cannot call getTypeOperand for __uuidof(expr)");
  return Operand.get<TypeSourceInfo *>()->getType().getNonReferenceType()
                                                        .getUnqualifiedType();
}

// static
UuidAttr *CXXUuidofExpr::GetUuidAttrOfType(QualType QT) {
  // Optionally remove one level of pointer, reference or array indirection.
  const Type *Ty = QT.getTypePtr();
  if (QT->isPointerType() || QT->isReferenceType())
    Ty = QT->getPointeeType().getTypePtr();
  else if (QT->isArrayType())
    Ty = cast<ArrayType>(QT)->getElementType().getTypePtr();

  // Loop all record redeclaration looking for an uuid attribute.
  CXXRecordDecl *RD = Ty->getAsCXXRecordDecl();
  for (CXXRecordDecl::redecl_iterator I = RD->redecls_begin(),
       E = RD->redecls_end(); I != E; ++I) {
    if (UuidAttr *Uuid = I->getAttr<UuidAttr>())
      return Uuid;
  }

  return 0;
}

// CXXScalarValueInitExpr
SourceRange CXXScalarValueInitExpr::getSourceRange() const {
  SourceLocation Start = RParenLoc;
  if (TypeInfo)
    Start = TypeInfo->getTypeLoc().getBeginLoc();
  return SourceRange(Start, RParenLoc);
}

// CXXNewExpr
CXXNewExpr::CXXNewExpr(ASTContext &C, bool globalNew, FunctionDecl *operatorNew,
                       FunctionDecl *operatorDelete,
                       bool usualArrayDeleteWantsSize,
                       ArrayRef<Expr*> placementArgs,
                       SourceRange typeIdParens, Expr *arraySize,
                       InitializationStyle initializationStyle,
                       Expr *initializer, QualType ty,
                       TypeSourceInfo *allocatedTypeInfo,
                       SourceRange Range, SourceRange directInitRange)
  : Expr(CXXNewExprClass, ty, VK_RValue, OK_Ordinary,
         ty->isDependentType(), ty->isDependentType(),
         ty->isInstantiationDependentType(),
         ty->containsUnexpandedParameterPack()),
    SubExprs(0), OperatorNew(operatorNew), OperatorDelete(operatorDelete),
    AllocatedTypeInfo(allocatedTypeInfo), TypeIdParens(typeIdParens),
    Range(Range), DirectInitRange(directInitRange),
    GlobalNew(globalNew), UsualArrayDeleteWantsSize(usualArrayDeleteWantsSize) {
  assert((initializer != 0 || initializationStyle == NoInit) &&
         "Only NoInit can have no initializer.");
  StoredInitializationStyle = initializer ? initializationStyle + 1 : 0;
  AllocateArgsArray(C, arraySize != 0, placementArgs.size(), initializer != 0);
  unsigned i = 0;
  if (Array) {
    if (arraySize->isInstantiationDependent())
      ExprBits.InstantiationDependent = true;
    
    if (arraySize->containsUnexpandedParameterPack())
      ExprBits.ContainsUnexpandedParameterPack = true;

    SubExprs[i++] = arraySize;
  }

  if (initializer) {
    if (initializer->isInstantiationDependent())
      ExprBits.InstantiationDependent = true;

    if (initializer->containsUnexpandedParameterPack())
      ExprBits.ContainsUnexpandedParameterPack = true;

    SubExprs[i++] = initializer;
  }

  for (unsigned j = 0; j != placementArgs.size(); ++j) {
    if (placementArgs[j]->isInstantiationDependent())
      ExprBits.InstantiationDependent = true;
    if (placementArgs[j]->containsUnexpandedParameterPack())
      ExprBits.ContainsUnexpandedParameterPack = true;

    SubExprs[i++] = placementArgs[j];
  }

  switch (getInitializationStyle()) {
  case CallInit:
    this->Range.setEnd(DirectInitRange.getEnd()); break;
  case ListInit:
    this->Range.setEnd(getInitializer()->getSourceRange().getEnd()); break;
  default: break;
  }
}

void CXXNewExpr::AllocateArgsArray(ASTContext &C, bool isArray,
                                   unsigned numPlaceArgs, bool hasInitializer){
  assert(SubExprs == 0 && "SubExprs already allocated");
  Array = isArray;
  NumPlacementArgs = numPlaceArgs;

  unsigned TotalSize = Array + hasInitializer + NumPlacementArgs;
  SubExprs = new (C) Stmt*[TotalSize];
}

bool CXXNewExpr::shouldNullCheckAllocation(ASTContext &Ctx) const {
  return getOperatorNew()->getType()->
    castAs<FunctionProtoType>()->isNothrow(Ctx);
}

// CXXDeleteExpr
QualType CXXDeleteExpr::getDestroyedType() const {
  const Expr *Arg = getArgument();
  // The type-to-delete may not be a pointer if it's a dependent type.
  const QualType ArgType = Arg->getType();

  if (ArgType->isDependentType() && !ArgType->isPointerType())
    return QualType();

  return ArgType->getAs<PointerType>()->getPointeeType();
}

// CXXPseudoDestructorExpr
PseudoDestructorTypeStorage::PseudoDestructorTypeStorage(TypeSourceInfo *Info)
 : Type(Info) 
{
  Location = Info->getTypeLoc().getLocalSourceRange().getBegin();
}

CXXPseudoDestructorExpr::CXXPseudoDestructorExpr(ASTContext &Context,
                Expr *Base, bool isArrow, SourceLocation OperatorLoc,
                NestedNameSpecifierLoc QualifierLoc, TypeSourceInfo *ScopeType, 
                SourceLocation ColonColonLoc, SourceLocation TildeLoc, 
                PseudoDestructorTypeStorage DestroyedType)
  : Expr(CXXPseudoDestructorExprClass,
         Context.getPointerType(Context.getFunctionType(Context.VoidTy, 0, 0,
                                         FunctionProtoType::ExtProtoInfo())),
         VK_RValue, OK_Ordinary,
         /*isTypeDependent=*/(Base->isTypeDependent() ||
           (DestroyedType.getTypeSourceInfo() &&
            DestroyedType.getTypeSourceInfo()->getType()->isDependentType())),
         /*isValueDependent=*/Base->isValueDependent(),
         (Base->isInstantiationDependent() ||
          (QualifierLoc &&
           QualifierLoc.getNestedNameSpecifier()->isInstantiationDependent()) ||
          (ScopeType &&
           ScopeType->getType()->isInstantiationDependentType()) ||
          (DestroyedType.getTypeSourceInfo() &&
           DestroyedType.getTypeSourceInfo()->getType()
                                             ->isInstantiationDependentType())),
         // ContainsUnexpandedParameterPack
         (Base->containsUnexpandedParameterPack() ||
          (QualifierLoc && 
           QualifierLoc.getNestedNameSpecifier()
                                        ->containsUnexpandedParameterPack()) ||
          (ScopeType && 
           ScopeType->getType()->containsUnexpandedParameterPack()) ||
          (DestroyedType.getTypeSourceInfo() &&
           DestroyedType.getTypeSourceInfo()->getType()
                                   ->containsUnexpandedParameterPack()))),
    Base(static_cast<Stmt *>(Base)), IsArrow(isArrow),
    OperatorLoc(OperatorLoc), QualifierLoc(QualifierLoc),
    ScopeType(ScopeType), ColonColonLoc(ColonColonLoc), TildeLoc(TildeLoc),
    DestroyedType(DestroyedType) { }

QualType CXXPseudoDestructorExpr::getDestroyedType() const {
  if (TypeSourceInfo *TInfo = DestroyedType.getTypeSourceInfo())
    return TInfo->getType();
  
  return QualType();
}

SourceRange CXXPseudoDestructorExpr::getSourceRange() const {
  SourceLocation End = DestroyedType.getLocation();
  if (TypeSourceInfo *TInfo = DestroyedType.getTypeSourceInfo())
    End = TInfo->getTypeLoc().getLocalSourceRange().getEnd();
  return SourceRange(Base->getLocStart(), End);
}

// UnresolvedLookupExpr
UnresolvedLookupExpr *
UnresolvedLookupExpr::Create(ASTContext &C, 
                             CXXRecordDecl *NamingClass,
                             NestedNameSpecifierLoc QualifierLoc,
                             SourceLocation TemplateKWLoc,
                             const DeclarationNameInfo &NameInfo,
                             bool ADL,
                             const TemplateArgumentListInfo *Args,
                             UnresolvedSetIterator Begin,
                             UnresolvedSetIterator End)
{
  assert(Args || TemplateKWLoc.isValid());
  unsigned num_args = Args ? Args->size() : 0;
  void *Mem = C.Allocate(sizeof(UnresolvedLookupExpr) +
                         ASTTemplateKWAndArgsInfo::sizeFor(num_args));
  return new (Mem) UnresolvedLookupExpr(C, NamingClass, QualifierLoc,
                                        TemplateKWLoc, NameInfo,
                                        ADL, /*Overload*/ true, Args,
                                        Begin, End);
}

UnresolvedLookupExpr *
UnresolvedLookupExpr::CreateEmpty(ASTContext &C,
                                  bool HasTemplateKWAndArgsInfo,
                                  unsigned NumTemplateArgs) {
  std::size_t size = sizeof(UnresolvedLookupExpr);
  if (HasTemplateKWAndArgsInfo)
    size += ASTTemplateKWAndArgsInfo::sizeFor(NumTemplateArgs);

  void *Mem = C.Allocate(size, llvm::alignOf<UnresolvedLookupExpr>());
  UnresolvedLookupExpr *E = new (Mem) UnresolvedLookupExpr(EmptyShell());
  E->HasTemplateKWAndArgsInfo = HasTemplateKWAndArgsInfo;
  return E;
}

OverloadExpr::OverloadExpr(StmtClass K, ASTContext &C, 
                           NestedNameSpecifierLoc QualifierLoc,
                           SourceLocation TemplateKWLoc,
                           const DeclarationNameInfo &NameInfo,
                           const TemplateArgumentListInfo *TemplateArgs,
                           UnresolvedSetIterator Begin, 
                           UnresolvedSetIterator End,
                           bool KnownDependent,
                           bool KnownInstantiationDependent,
                           bool KnownContainsUnexpandedParameterPack)
  : Expr(K, C.OverloadTy, VK_LValue, OK_Ordinary, KnownDependent, 
         KnownDependent,
         (KnownInstantiationDependent ||
          NameInfo.isInstantiationDependent() ||
          (QualifierLoc &&
           QualifierLoc.getNestedNameSpecifier()->isInstantiationDependent())),
         (KnownContainsUnexpandedParameterPack ||
          NameInfo.containsUnexpandedParameterPack() ||
          (QualifierLoc && 
           QualifierLoc.getNestedNameSpecifier()
                                      ->containsUnexpandedParameterPack()))),
    NameInfo(NameInfo), QualifierLoc(QualifierLoc),
    Results(0), NumResults(End - Begin),
    HasTemplateKWAndArgsInfo(TemplateArgs != 0 || TemplateKWLoc.isValid())
{
  NumResults = End - Begin;
  if (NumResults) {
    // Determine whether this expression is type-dependent.
    for (UnresolvedSetImpl::const_iterator I = Begin; I != End; ++I) {
      if ((*I)->getDeclContext()->isDependentContext() ||
          isa<UnresolvedUsingValueDecl>(*I)) {
        ExprBits.TypeDependent = true;
        ExprBits.ValueDependent = true;
        ExprBits.InstantiationDependent = true;
      }
    }

    Results = static_cast<DeclAccessPair *>(
                                C.Allocate(sizeof(DeclAccessPair) * NumResults, 
                                           llvm::alignOf<DeclAccessPair>()));
    memcpy(Results, &*Begin.getIterator(), 
           NumResults * sizeof(DeclAccessPair));
  }

  // If we have explicit template arguments, check for dependent
  // template arguments and whether they contain any unexpanded pack
  // expansions.
  if (TemplateArgs) {
    bool Dependent = false;
    bool InstantiationDependent = false;
    bool ContainsUnexpandedParameterPack = false;
    getTemplateKWAndArgsInfo()->initializeFrom(TemplateKWLoc, *TemplateArgs,
                                               Dependent,
                                               InstantiationDependent,
                                               ContainsUnexpandedParameterPack);

    if (Dependent) {
      ExprBits.TypeDependent = true;
      ExprBits.ValueDependent = true;
    }
    if (InstantiationDependent)
      ExprBits.InstantiationDependent = true;
    if (ContainsUnexpandedParameterPack)
      ExprBits.ContainsUnexpandedParameterPack = true;
  } else if (TemplateKWLoc.isValid()) {
    getTemplateKWAndArgsInfo()->initializeFrom(TemplateKWLoc);
  }

  if (isTypeDependent())
    setType(C.DependentTy);
}

void OverloadExpr::initializeResults(ASTContext &C,
                                     UnresolvedSetIterator Begin,
                                     UnresolvedSetIterator End) {
  assert(Results == 0 && "Results already initialized!");
  NumResults = End - Begin;
  if (NumResults) {
     Results = static_cast<DeclAccessPair *>(
                               C.Allocate(sizeof(DeclAccessPair) * NumResults,
 
                                          llvm::alignOf<DeclAccessPair>()));
     memcpy(Results, &*Begin.getIterator(), 
            NumResults * sizeof(DeclAccessPair));
  }
}

CXXRecordDecl *OverloadExpr::getNamingClass() const {
  if (isa<UnresolvedLookupExpr>(this))
    return cast<UnresolvedLookupExpr>(this)->getNamingClass();
  else
    return cast<UnresolvedMemberExpr>(this)->getNamingClass();
}

// DependentScopeDeclRefExpr
DependentScopeDeclRefExpr::DependentScopeDeclRefExpr(QualType T,
                            NestedNameSpecifierLoc QualifierLoc,
                            SourceLocation TemplateKWLoc,
                            const DeclarationNameInfo &NameInfo,
                            const TemplateArgumentListInfo *Args)
  : Expr(DependentScopeDeclRefExprClass, T, VK_LValue, OK_Ordinary,
         true, true,
         (NameInfo.isInstantiationDependent() ||
          (QualifierLoc && 
           QualifierLoc.getNestedNameSpecifier()->isInstantiationDependent())),
         (NameInfo.containsUnexpandedParameterPack() ||
          (QualifierLoc && 
           QualifierLoc.getNestedNameSpecifier()
                            ->containsUnexpandedParameterPack()))),
    QualifierLoc(QualifierLoc), NameInfo(NameInfo), 
    HasTemplateKWAndArgsInfo(Args != 0 || TemplateKWLoc.isValid())
{
  if (Args) {
    bool Dependent = true;
    bool InstantiationDependent = true;
    bool ContainsUnexpandedParameterPack
      = ExprBits.ContainsUnexpandedParameterPack;
    getTemplateKWAndArgsInfo()->initializeFrom(TemplateKWLoc, *Args,
                                               Dependent,
                                               InstantiationDependent,
                                               ContainsUnexpandedParameterPack);
    ExprBits.ContainsUnexpandedParameterPack = ContainsUnexpandedParameterPack;
  } else if (TemplateKWLoc.isValid()) {
    getTemplateKWAndArgsInfo()->initializeFrom(TemplateKWLoc);
  }
}

DependentScopeDeclRefExpr *
DependentScopeDeclRefExpr::Create(ASTContext &C,
                                  NestedNameSpecifierLoc QualifierLoc,
                                  SourceLocation TemplateKWLoc,
                                  const DeclarationNameInfo &NameInfo,
                                  const TemplateArgumentListInfo *Args) {
  std::size_t size = sizeof(DependentScopeDeclRefExpr);
  if (Args)
    size += ASTTemplateKWAndArgsInfo::sizeFor(Args->size());
  else if (TemplateKWLoc.isValid())
    size += ASTTemplateKWAndArgsInfo::sizeFor(0);
  void *Mem = C.Allocate(size);
  return new (Mem) DependentScopeDeclRefExpr(C.DependentTy, QualifierLoc,
                                             TemplateKWLoc, NameInfo, Args);
}

DependentScopeDeclRefExpr *
DependentScopeDeclRefExpr::CreateEmpty(ASTContext &C,
                                       bool HasTemplateKWAndArgsInfo,
                                       unsigned NumTemplateArgs) {
  std::size_t size = sizeof(DependentScopeDeclRefExpr);
  if (HasTemplateKWAndArgsInfo)
    size += ASTTemplateKWAndArgsInfo::sizeFor(NumTemplateArgs);
  void *Mem = C.Allocate(size);
  DependentScopeDeclRefExpr *E
    = new (Mem) DependentScopeDeclRefExpr(QualType(), NestedNameSpecifierLoc(),
                                          SourceLocation(),
                                          DeclarationNameInfo(), 0);
  E->HasTemplateKWAndArgsInfo = HasTemplateKWAndArgsInfo;
  return E;
}

SourceRange CXXConstructExpr::getSourceRange() const {
  if (isa<CXXTemporaryObjectExpr>(this))
    return cast<CXXTemporaryObjectExpr>(this)->getSourceRange();

  if (ParenRange.isValid())
    return SourceRange(Loc, ParenRange.getEnd());

  SourceLocation End = Loc;
  for (unsigned I = getNumArgs(); I > 0; --I) {
    const Expr *Arg = getArg(I-1);
    if (!Arg->isDefaultArgument()) {
      SourceLocation NewEnd = Arg->getLocEnd();
      if (NewEnd.isValid()) {
        End = NewEnd;
        break;
      }
    }
  }

  return SourceRange(Loc, End);
}

SourceRange CXXOperatorCallExpr::getSourceRangeImpl() const {
  OverloadedOperatorKind Kind = getOperator();
  if (Kind == OO_PlusPlus || Kind == OO_MinusMinus) {
    if (getNumArgs() == 1)
      // Prefix operator
      return SourceRange(getOperatorLoc(), getArg(0)->getLocEnd());
    else
      // Postfix operator
      return SourceRange(getArg(0)->getLocStart(), getOperatorLoc());
  } else if (Kind == OO_Arrow) {
    return getArg(0)->getSourceRange();
  } else if (Kind == OO_Call) {
    return SourceRange(getArg(0)->getLocStart(), getRParenLoc());
  } else if (Kind == OO_Subscript) {
    return SourceRange(getArg(0)->getLocStart(), getRParenLoc());
  } else if (getNumArgs() == 1) {
    return SourceRange(getOperatorLoc(), getArg(0)->getLocEnd());
  } else if (getNumArgs() == 2) {
    return SourceRange(getArg(0)->getLocStart(), getArg(1)->getLocEnd());
  } else {
    return getOperatorLoc();
  }
}

Expr *CXXMemberCallExpr::getImplicitObjectArgument() const {
  const Expr *Callee = getCallee()->IgnoreParens();
  if (const MemberExpr *MemExpr = dyn_cast<MemberExpr>(Callee))
    return MemExpr->getBase();
  if (const BinaryOperator *BO = dyn_cast<BinaryOperator>(Callee))
    if (BO->getOpcode() == BO_PtrMemD || BO->getOpcode() == BO_PtrMemI)
      return BO->getLHS();

  // FIXME: Will eventually need to cope with member pointers.
  return 0;
}

CXXMethodDecl *CXXMemberCallExpr::getMethodDecl() const {
  if (const MemberExpr *MemExpr = 
      dyn_cast<MemberExpr>(getCallee()->IgnoreParens()))
    return cast<CXXMethodDecl>(MemExpr->getMemberDecl());

  // FIXME: Will eventually need to cope with member pointers.
  return 0;
}


CXXRecordDecl *CXXMemberCallExpr::getRecordDecl() const {
  Expr* ThisArg = getImplicitObjectArgument();
  if (!ThisArg)
    return 0;

  if (ThisArg->getType()->isAnyPointerType())
    return ThisArg->getType()->getPointeeType()->getAsCXXRecordDecl();

  return ThisArg->getType()->getAsCXXRecordDecl();
}


//===----------------------------------------------------------------------===//
//  Named casts
//===----------------------------------------------------------------------===//

/// getCastName - Get the name of the C++ cast being used, e.g.,
/// "static_cast", "dynamic_cast", "reinterpret_cast", or
/// "const_cast". The returned pointer must not be freed.
const char *CXXNamedCastExpr::getCastName() const {
  switch (getStmtClass()) {
  case CXXStaticCastExprClass:      return "static_cast";
  case CXXDynamicCastExprClass:     return "dynamic_cast";
  case CXXReinterpretCastExprClass: return "reinterpret_cast";
  case CXXConstCastExprClass:       return "const_cast";
  default:                          return "<invalid cast>";
  }
}

CXXStaticCastExpr *CXXStaticCastExpr::Create(ASTContext &C, QualType T,
                                             ExprValueKind VK,
                                             CastKind K, Expr *Op,
                                             const CXXCastPath *BasePath,
                                             TypeSourceInfo *WrittenTy,
                                             SourceLocation L, 
                                             SourceLocation RParenLoc) {
  unsigned PathSize = (BasePath ? BasePath->size() : 0);
  void *Buffer = C.Allocate(sizeof(CXXStaticCastExpr)
                            + PathSize * sizeof(CXXBaseSpecifier*));
  CXXStaticCastExpr *E =
    new (Buffer) CXXStaticCastExpr(T, VK, K, Op, PathSize, WrittenTy, L,
                                   RParenLoc);
  if (PathSize) E->setCastPath(*BasePath);
  return E;
}

CXXStaticCastExpr *CXXStaticCastExpr::CreateEmpty(ASTContext &C,
                                                  unsigned PathSize) {
  void *Buffer =
    C.Allocate(sizeof(CXXStaticCastExpr) + PathSize * sizeof(CXXBaseSpecifier*));
  return new (Buffer) CXXStaticCastExpr(EmptyShell(), PathSize);
}

CXXDynamicCastExpr *CXXDynamicCastExpr::Create(ASTContext &C, QualType T,
                                               ExprValueKind VK,
                                               CastKind K, Expr *Op,
                                               const CXXCastPath *BasePath,
                                               TypeSourceInfo *WrittenTy,
                                               SourceLocation L, 
                                               SourceLocation RParenLoc) {
  unsigned PathSize = (BasePath ? BasePath->size() : 0);
  void *Buffer = C.Allocate(sizeof(CXXDynamicCastExpr)
                            + PathSize * sizeof(CXXBaseSpecifier*));
  CXXDynamicCastExpr *E =
    new (Buffer) CXXDynamicCastExpr(T, VK, K, Op, PathSize, WrittenTy, L,
                                    RParenLoc);
  if (PathSize) E->setCastPath(*BasePath);
  return E;
}

CXXDynamicCastExpr *CXXDynamicCastExpr::CreateEmpty(ASTContext &C,
                                                    unsigned PathSize) {
  void *Buffer =
    C.Allocate(sizeof(CXXDynamicCastExpr) + PathSize * sizeof(CXXBaseSpecifier*));
  return new (Buffer) CXXDynamicCastExpr(EmptyShell(), PathSize);
}

/// isAlwaysNull - Return whether the result of the dynamic_cast is proven
/// to always be null. For example:
///
/// struct A { };
/// struct B final : A { };
/// struct C { };
///
/// C *f(B* b) { return dynamic_cast<C*>(b); }
bool CXXDynamicCastExpr::isAlwaysNull() const
{
  QualType SrcType = getSubExpr()->getType();
  QualType DestType = getType();

  if (const PointerType *SrcPTy = SrcType->getAs<PointerType>()) {
    SrcType = SrcPTy->getPointeeType();
    DestType = DestType->castAs<PointerType>()->getPointeeType();
  }

  if (DestType->isVoidType())
    return false;

  const CXXRecordDecl *SrcRD = 
    cast<CXXRecordDecl>(SrcType->castAs<RecordType>()->getDecl());

  if (!SrcRD->hasAttr<FinalAttr>())
    return false;

  const CXXRecordDecl *DestRD = 
    cast<CXXRecordDecl>(DestType->castAs<RecordType>()->getDecl());

  return !DestRD->isDerivedFrom(SrcRD);
}

CXXReinterpretCastExpr *
CXXReinterpretCastExpr::Create(ASTContext &C, QualType T, ExprValueKind VK,
                               CastKind K, Expr *Op,
                               const CXXCastPath *BasePath,
                               TypeSourceInfo *WrittenTy, SourceLocation L, 
                               SourceLocation RParenLoc) {
  unsigned PathSize = (BasePath ? BasePath->size() : 0);
  void *Buffer =
    C.Allocate(sizeof(CXXReinterpretCastExpr) + PathSize * sizeof(CXXBaseSpecifier*));
  CXXReinterpretCastExpr *E =
    new (Buffer) CXXReinterpretCastExpr(T, VK, K, Op, PathSize, WrittenTy, L,
                                        RParenLoc);
  if (PathSize) E->setCastPath(*BasePath);
  return E;
}

CXXReinterpretCastExpr *
CXXReinterpretCastExpr::CreateEmpty(ASTContext &C, unsigned PathSize) {
  void *Buffer = C.Allocate(sizeof(CXXReinterpretCastExpr)
                            + PathSize * sizeof(CXXBaseSpecifier*));
  return new (Buffer) CXXReinterpretCastExpr(EmptyShell(), PathSize);
}

CXXConstCastExpr *CXXConstCastExpr::Create(ASTContext &C, QualType T,
                                           ExprValueKind VK, Expr *Op,
                                           TypeSourceInfo *WrittenTy,
                                           SourceLocation L, 
                                           SourceLocation RParenLoc) {
  return new (C) CXXConstCastExpr(T, VK, Op, WrittenTy, L, RParenLoc);
}

CXXConstCastExpr *CXXConstCastExpr::CreateEmpty(ASTContext &C) {
  return new (C) CXXConstCastExpr(EmptyShell());
}

CXXFunctionalCastExpr *
CXXFunctionalCastExpr::Create(ASTContext &C, QualType T, ExprValueKind VK,
                              TypeSourceInfo *Written, SourceLocation L,
                              CastKind K, Expr *Op, const CXXCastPath *BasePath,
                               SourceLocation R) {
  unsigned PathSize = (BasePath ? BasePath->size() : 0);
  void *Buffer = C.Allocate(sizeof(CXXFunctionalCastExpr)
                            + PathSize * sizeof(CXXBaseSpecifier*));
  CXXFunctionalCastExpr *E =
    new (Buffer) CXXFunctionalCastExpr(T, VK, Written, L, K, Op, PathSize, R);
  if (PathSize) E->setCastPath(*BasePath);
  return E;
}

CXXFunctionalCastExpr *
CXXFunctionalCastExpr::CreateEmpty(ASTContext &C, unsigned PathSize) {
  void *Buffer = C.Allocate(sizeof(CXXFunctionalCastExpr)
                            + PathSize * sizeof(CXXBaseSpecifier*));
  return new (Buffer) CXXFunctionalCastExpr(EmptyShell(), PathSize);
}

UserDefinedLiteral::LiteralOperatorKind
UserDefinedLiteral::getLiteralOperatorKind() const {
  if (getNumArgs() == 0)
    return LOK_Template;
  if (getNumArgs() == 2)
    return LOK_String;

  assert(getNumArgs() == 1 && "unexpected #args in literal operator call");
  QualType ParamTy =
    cast<FunctionDecl>(getCalleeDecl())->getParamDecl(0)->getType();
  if (ParamTy->isPointerType())
    return LOK_Raw;
  if (ParamTy->isAnyCharacterType())
    return LOK_Character;
  if (ParamTy->isIntegerType())
    return LOK_Integer;
  if (ParamTy->isFloatingType())
    return LOK_Floating;

  llvm_unreachable("unknown kind of literal operator");
}

Expr *UserDefinedLiteral::getCookedLiteral() {
#ifndef NDEBUG
  LiteralOperatorKind LOK = getLiteralOperatorKind();
  assert(LOK != LOK_Template && LOK != LOK_Raw && "not a cooked literal");
#endif
  return getArg(0);
}

const IdentifierInfo *UserDefinedLiteral::getUDSuffix() const {
  return cast<FunctionDecl>(getCalleeDecl())->getLiteralIdentifier();
}

CXXDefaultArgExpr *
CXXDefaultArgExpr::Create(ASTContext &C, SourceLocation Loc, 
                          ParmVarDecl *Param, Expr *SubExpr) {
  void *Mem = C.Allocate(sizeof(CXXDefaultArgExpr) + sizeof(Stmt *));
  return new (Mem) CXXDefaultArgExpr(CXXDefaultArgExprClass, Loc, Param, 
                                     SubExpr);
}

CXXTemporary *CXXTemporary::Create(ASTContext &C,
                                   const CXXDestructorDecl *Destructor) {
  return new (C) CXXTemporary(Destructor);
}

CXXBindTemporaryExpr *CXXBindTemporaryExpr::Create(ASTContext &C,
                                                   CXXTemporary *Temp,
                                                   Expr* SubExpr) {
  assert((SubExpr->getType()->isRecordType() ||
          SubExpr->getType()->isArrayType()) &&
         "Expression bound to a temporary must have record or array type!");

  return new (C) CXXBindTemporaryExpr(Temp, SubExpr);
}

CXXTemporaryObjectExpr::CXXTemporaryObjectExpr(ASTContext &C,
                                               CXXConstructorDecl *Cons,
                                               TypeSourceInfo *Type,
                                               ArrayRef<Expr*> Args,
                                               SourceRange parenRange,
                                               bool HadMultipleCandidates,
                                               bool ZeroInitialization)
  : CXXConstructExpr(C, CXXTemporaryObjectExprClass, 
                     Type->getType().getNonReferenceType(), 
                     Type->getTypeLoc().getBeginLoc(),
                     Cons, false, Args,
                     HadMultipleCandidates, /*FIXME*/false, ZeroInitialization,
                     CXXConstructExpr::CK_Complete, parenRange),
    Type(Type) {
}

SourceRange CXXTemporaryObjectExpr::getSourceRange() const {
  return SourceRange(Type->getTypeLoc().getBeginLoc(),
                     getParenRange().getEnd());
}

CXXConstructExpr *CXXConstructExpr::Create(ASTContext &C, QualType T,
                                           SourceLocation Loc,
                                           CXXConstructorDecl *D, bool Elidable,
                                           ArrayRef<Expr*> Args,
                                           bool HadMultipleCandidates,
                                           bool ListInitialization,
                                           bool ZeroInitialization,
                                           ConstructionKind ConstructKind,
                                           SourceRange ParenRange) {
  return new (C) CXXConstructExpr(C, CXXConstructExprClass, T, Loc, D, 
                                  Elidable, Args,
                                  HadMultipleCandidates, ListInitialization,
                                  ZeroInitialization, ConstructKind,
                                  ParenRange);
}

CXXConstructExpr::CXXConstructExpr(ASTContext &C, StmtClass SC, QualType T,
                                   SourceLocation Loc,
                                   CXXConstructorDecl *D, bool elidable,
                                   ArrayRef<Expr*> args,
                                   bool HadMultipleCandidates,
                                   bool ListInitialization,
                                   bool ZeroInitialization,
                                   ConstructionKind ConstructKind,
                                   SourceRange ParenRange)
  : Expr(SC, T, VK_RValue, OK_Ordinary,
         T->isDependentType(), T->isDependentType(),
         T->isInstantiationDependentType(),
         T->containsUnexpandedParameterPack()),
    Constructor(D), Loc(Loc), ParenRange(ParenRange),  NumArgs(args.size()),
    Elidable(elidable), HadMultipleCandidates(HadMultipleCandidates),
    ListInitialization(ListInitialization),
    ZeroInitialization(ZeroInitialization),
    ConstructKind(ConstructKind), Args(0)
{
  if (NumArgs) {
    Args = new (C) Stmt*[args.size()];
    
    for (unsigned i = 0; i != args.size(); ++i) {
      assert(args[i] && "NULL argument in CXXConstructExpr");

      if (args[i]->isValueDependent())
        ExprBits.ValueDependent = true;
      if (args[i]->isInstantiationDependent())
        ExprBits.InstantiationDependent = true;
      if (args[i]->containsUnexpandedParameterPack())
        ExprBits.ContainsUnexpandedParameterPack = true;
  
      Args[i] = args[i];
    }
  }
}

LambdaExpr::Capture::Capture(SourceLocation Loc, bool Implicit,
                             LambdaCaptureKind Kind, VarDecl *Var,
                             SourceLocation EllipsisLoc)
  : VarAndBits(Var, 0), Loc(Loc), EllipsisLoc(EllipsisLoc)
{
  unsigned Bits = 0;
  if (Implicit)
    Bits |= Capture_Implicit;
  
  switch (Kind) {
  case LCK_This: 
    assert(Var == 0 && "'this' capture cannot have a variable!");
    break;

  case LCK_ByCopy:
    Bits |= Capture_ByCopy;
    // Fall through 
  case LCK_ByRef:
    assert(Var && "capture must have a variable!");
    break;
  }
  VarAndBits.setInt(Bits);
}

LambdaCaptureKind LambdaExpr::Capture::getCaptureKind() const {
  if (capturesThis())
    return LCK_This;

  return (VarAndBits.getInt() & Capture_ByCopy)? LCK_ByCopy : LCK_ByRef;
}

LambdaExpr::LambdaExpr(QualType T, 
                       SourceRange IntroducerRange,
                       LambdaCaptureDefault CaptureDefault,
                       ArrayRef<Capture> Captures, 
                       bool ExplicitParams,
                       bool ExplicitResultType,
                       ArrayRef<Expr *> CaptureInits,
                       ArrayRef<VarDecl *> ArrayIndexVars,
                       ArrayRef<unsigned> ArrayIndexStarts,
                       SourceLocation ClosingBrace,
                       bool ContainsUnexpandedParameterPack)
  : Expr(LambdaExprClass, T, VK_RValue, OK_Ordinary,
         T->isDependentType(), T->isDependentType(), T->isDependentType(),
         ContainsUnexpandedParameterPack),
    IntroducerRange(IntroducerRange),
    NumCaptures(Captures.size()),
    CaptureDefault(CaptureDefault),
    ExplicitParams(ExplicitParams),
    ExplicitResultType(ExplicitResultType),
    ClosingBrace(ClosingBrace)
{
  assert(CaptureInits.size() == Captures.size() && "Wrong number of arguments");
  CXXRecordDecl *Class = getLambdaClass();
  CXXRecordDecl::LambdaDefinitionData &Data = Class->getLambdaData();
  
  // FIXME: Propagate "has unexpanded parameter pack" bit.
  
  // Copy captures.
  ASTContext &Context = Class->getASTContext();
  Data.NumCaptures = NumCaptures;
  Data.NumExplicitCaptures = 0;
  Data.Captures = (Capture *)Context.Allocate(sizeof(Capture) * NumCaptures);
  Capture *ToCapture = Data.Captures;
  for (unsigned I = 0, N = Captures.size(); I != N; ++I) {
    if (Captures[I].isExplicit())
      ++Data.NumExplicitCaptures;
    
    *ToCapture++ = Captures[I];
  }
 
  // Copy initialization expressions for the non-static data members.
  Stmt **Stored = getStoredStmts();
  for (unsigned I = 0, N = CaptureInits.size(); I != N; ++I)
    *Stored++ = CaptureInits[I];
  
  // Copy the body of the lambda.
  *Stored++ = getCallOperator()->getBody();

  // Copy the array index variables, if any.
  HasArrayIndexVars = !ArrayIndexVars.empty();
  if (HasArrayIndexVars) {
    assert(ArrayIndexStarts.size() == NumCaptures);
    memcpy(getArrayIndexVars(), ArrayIndexVars.data(),
           sizeof(VarDecl *) * ArrayIndexVars.size());
    memcpy(getArrayIndexStarts(), ArrayIndexStarts.data(), 
           sizeof(unsigned) * Captures.size());
    getArrayIndexStarts()[Captures.size()] = ArrayIndexVars.size();
  }
}

LambdaExpr *LambdaExpr::Create(ASTContext &Context, 
                               CXXRecordDecl *Class,
                               SourceRange IntroducerRange,
                               LambdaCaptureDefault CaptureDefault,
                               ArrayRef<Capture> Captures, 
                               bool ExplicitParams,
                               bool ExplicitResultType,
                               ArrayRef<Expr *> CaptureInits,
                               ArrayRef<VarDecl *> ArrayIndexVars,
                               ArrayRef<unsigned> ArrayIndexStarts,
                               SourceLocation ClosingBrace,
                               bool ContainsUnexpandedParameterPack) {
  // Determine the type of the expression (i.e., the type of the
  // function object we're creating).
  QualType T = Context.getTypeDeclType(Class);

  unsigned Size = sizeof(LambdaExpr) + sizeof(Stmt *) * (Captures.size() + 1);
  if (!ArrayIndexVars.empty()) {
    Size += sizeof(unsigned) * (Captures.size() + 1);
    // Realign for following VarDecl array.
    Size = llvm::RoundUpToAlignment(Size, llvm::alignOf<VarDecl*>());
    Size += sizeof(VarDecl *) * ArrayIndexVars.size();
  }
  void *Mem = Context.Allocate(Size);
  return new (Mem) LambdaExpr(T, IntroducerRange, CaptureDefault, 
                              Captures, ExplicitParams, ExplicitResultType,
                              CaptureInits, ArrayIndexVars, ArrayIndexStarts,
                              ClosingBrace, ContainsUnexpandedParameterPack);
}

LambdaExpr *LambdaExpr::CreateDeserialized(ASTContext &C, unsigned NumCaptures,
                                           unsigned NumArrayIndexVars) {
  unsigned Size = sizeof(LambdaExpr) + sizeof(Stmt *) * (NumCaptures + 1);
  if (NumArrayIndexVars)
    Size += sizeof(VarDecl) * NumArrayIndexVars
          + sizeof(unsigned) * (NumCaptures + 1);
  void *Mem = C.Allocate(Size);
  return new (Mem) LambdaExpr(EmptyShell(), NumCaptures, NumArrayIndexVars > 0);
}

LambdaExpr::capture_iterator LambdaExpr::capture_begin() const {
  return getLambdaClass()->getLambdaData().Captures;
}

LambdaExpr::capture_iterator LambdaExpr::capture_end() const {
  return capture_begin() + NumCaptures;
}

LambdaExpr::capture_iterator LambdaExpr::explicit_capture_begin() const {
  return capture_begin();
}

LambdaExpr::capture_iterator LambdaExpr::explicit_capture_end() const {
  struct CXXRecordDecl::LambdaDefinitionData &Data
    = getLambdaClass()->getLambdaData();
  return Data.Captures + Data.NumExplicitCaptures;
}

LambdaExpr::capture_iterator LambdaExpr::implicit_capture_begin() const {
  return explicit_capture_end();
}

LambdaExpr::capture_iterator LambdaExpr::implicit_capture_end() const {
  return capture_end();
}

ArrayRef<VarDecl *> 
LambdaExpr::getCaptureInitIndexVars(capture_init_iterator Iter) const {
  assert(HasArrayIndexVars && "No array index-var data?");
  
  unsigned Index = Iter - capture_init_begin();
  assert(Index < getLambdaClass()->getLambdaData().NumCaptures &&
         "Capture index out-of-range");
  VarDecl **IndexVars = getArrayIndexVars();
  unsigned *IndexStarts = getArrayIndexStarts();
  return ArrayRef<VarDecl *>(IndexVars + IndexStarts[Index],
                             IndexVars + IndexStarts[Index + 1]);
}

CXXRecordDecl *LambdaExpr::getLambdaClass() const {
  return getType()->getAsCXXRecordDecl();
}

CXXMethodDecl *LambdaExpr::getCallOperator() const {
  CXXRecordDecl *Record = getLambdaClass();
  DeclarationName Name
    = Record->getASTContext().DeclarationNames.getCXXOperatorName(OO_Call);
  DeclContext::lookup_result Calls = Record->lookup(Name);
  assert(Calls.first != Calls.second && "Missing lambda call operator!");
  CXXMethodDecl *Result = cast<CXXMethodDecl>(*Calls.first++);
  assert(Calls.first == Calls.second && "More than lambda one call operator?");
  return Result;
}

CompoundStmt *LambdaExpr::getBody() const {
  if (!getStoredStmts()[NumCaptures])
    getStoredStmts()[NumCaptures] = getCallOperator()->getBody();
    
  return reinterpret_cast<CompoundStmt *>(getStoredStmts()[NumCaptures]);
}

bool LambdaExpr::isMutable() const {
  return !getCallOperator()->isConst();
}

ExprWithCleanups::ExprWithCleanups(Expr *subexpr,
                                   ArrayRef<CleanupObject> objects)
  : Expr(ExprWithCleanupsClass, subexpr->getType(),
         subexpr->getValueKind(), subexpr->getObjectKind(),
         subexpr->isTypeDependent(), subexpr->isValueDependent(),
         subexpr->isInstantiationDependent(),
         subexpr->containsUnexpandedParameterPack()),
    SubExpr(subexpr) {
  ExprWithCleanupsBits.NumObjects = objects.size();
  for (unsigned i = 0, e = objects.size(); i != e; ++i)
    getObjectsBuffer()[i] = objects[i];
}

ExprWithCleanups *ExprWithCleanups::Create(ASTContext &C, Expr *subexpr,
                                           ArrayRef<CleanupObject> objects) {
  size_t size = sizeof(ExprWithCleanups)
              + objects.size() * sizeof(CleanupObject);
  void *buffer = C.Allocate(size, llvm::alignOf<ExprWithCleanups>());
  return new (buffer) ExprWithCleanups(subexpr, objects);
}

ExprWithCleanups::ExprWithCleanups(EmptyShell empty, unsigned numObjects)
  : Expr(ExprWithCleanupsClass, empty) {
  ExprWithCleanupsBits.NumObjects = numObjects;
}

ExprWithCleanups *ExprWithCleanups::Create(ASTContext &C, EmptyShell empty,
                                           unsigned numObjects) {
  size_t size = sizeof(ExprWithCleanups) + numObjects * sizeof(CleanupObject);
  void *buffer = C.Allocate(size, llvm::alignOf<ExprWithCleanups>());
  return new (buffer) ExprWithCleanups(empty, numObjects);
}

CXXUnresolvedConstructExpr::CXXUnresolvedConstructExpr(TypeSourceInfo *Type,
                                                 SourceLocation LParenLoc,
                                                 ArrayRef<Expr*> Args,
                                                 SourceLocation RParenLoc)
  : Expr(CXXUnresolvedConstructExprClass, 
         Type->getType().getNonReferenceType(),
         (Type->getType()->isLValueReferenceType() ? VK_LValue
          :Type->getType()->isRValueReferenceType()? VK_XValue
          :VK_RValue),
         OK_Ordinary,
         Type->getType()->isDependentType(), true, true,
         Type->getType()->containsUnexpandedParameterPack()),
    Type(Type),
    LParenLoc(LParenLoc),
    RParenLoc(RParenLoc),
    NumArgs(Args.size()) {
  Stmt **StoredArgs = reinterpret_cast<Stmt **>(this + 1);
  for (unsigned I = 0; I != Args.size(); ++I) {
    if (Args[I]->containsUnexpandedParameterPack())
      ExprBits.ContainsUnexpandedParameterPack = true;

    StoredArgs[I] = Args[I];
  }
}

CXXUnresolvedConstructExpr *
CXXUnresolvedConstructExpr::Create(ASTContext &C,
                                   TypeSourceInfo *Type,
                                   SourceLocation LParenLoc,
                                   ArrayRef<Expr*> Args,
                                   SourceLocation RParenLoc) {
  void *Mem = C.Allocate(sizeof(CXXUnresolvedConstructExpr) +
                         sizeof(Expr *) * Args.size());
  return new (Mem) CXXUnresolvedConstructExpr(Type, LParenLoc, Args, RParenLoc);
}

CXXUnresolvedConstructExpr *
CXXUnresolvedConstructExpr::CreateEmpty(ASTContext &C, unsigned NumArgs) {
  Stmt::EmptyShell Empty;
  void *Mem = C.Allocate(sizeof(CXXUnresolvedConstructExpr) +
                         sizeof(Expr *) * NumArgs);
  return new (Mem) CXXUnresolvedConstructExpr(Empty, NumArgs);
}

SourceRange CXXUnresolvedConstructExpr::getSourceRange() const {
  return SourceRange(Type->getTypeLoc().getBeginLoc(), RParenLoc);
}

CXXDependentScopeMemberExpr::CXXDependentScopeMemberExpr(ASTContext &C,
                                                 Expr *Base, QualType BaseType,
                                                 bool IsArrow,
                                                 SourceLocation OperatorLoc,
                                          NestedNameSpecifierLoc QualifierLoc,
                                          SourceLocation TemplateKWLoc,
                                          NamedDecl *FirstQualifierFoundInScope,
                                          DeclarationNameInfo MemberNameInfo,
                                   const TemplateArgumentListInfo *TemplateArgs)
  : Expr(CXXDependentScopeMemberExprClass, C.DependentTy,
         VK_LValue, OK_Ordinary, true, true, true,
         ((Base && Base->containsUnexpandedParameterPack()) ||
          (QualifierLoc && 
           QualifierLoc.getNestedNameSpecifier()
                                       ->containsUnexpandedParameterPack()) ||
          MemberNameInfo.containsUnexpandedParameterPack())),
    Base(Base), BaseType(BaseType), IsArrow(IsArrow),
    HasTemplateKWAndArgsInfo(TemplateArgs != 0 || TemplateKWLoc.isValid()),
    OperatorLoc(OperatorLoc), QualifierLoc(QualifierLoc), 
    FirstQualifierFoundInScope(FirstQualifierFoundInScope),
    MemberNameInfo(MemberNameInfo) {
  if (TemplateArgs) {
    bool Dependent = true;
    bool InstantiationDependent = true;
    bool ContainsUnexpandedParameterPack = false;
    getTemplateKWAndArgsInfo()->initializeFrom(TemplateKWLoc, *TemplateArgs,
                                               Dependent,
                                               InstantiationDependent,
                                               ContainsUnexpandedParameterPack);
    if (ContainsUnexpandedParameterPack)
      ExprBits.ContainsUnexpandedParameterPack = true;
  } else if (TemplateKWLoc.isValid()) {
    getTemplateKWAndArgsInfo()->initializeFrom(TemplateKWLoc);
  }
}

CXXDependentScopeMemberExpr::CXXDependentScopeMemberExpr(ASTContext &C,
                          Expr *Base, QualType BaseType,
                          bool IsArrow,
                          SourceLocation OperatorLoc,
                          NestedNameSpecifierLoc QualifierLoc,
                          NamedDecl *FirstQualifierFoundInScope,
                          DeclarationNameInfo MemberNameInfo)
  : Expr(CXXDependentScopeMemberExprClass, C.DependentTy,
         VK_LValue, OK_Ordinary, true, true, true,
         ((Base && Base->containsUnexpandedParameterPack()) ||
          (QualifierLoc && 
           QualifierLoc.getNestedNameSpecifier()->
                                         containsUnexpandedParameterPack()) ||
          MemberNameInfo.containsUnexpandedParameterPack())),
    Base(Base), BaseType(BaseType), IsArrow(IsArrow),
    HasTemplateKWAndArgsInfo(false),
    OperatorLoc(OperatorLoc), QualifierLoc(QualifierLoc),
    FirstQualifierFoundInScope(FirstQualifierFoundInScope),
    MemberNameInfo(MemberNameInfo) { }

CXXDependentScopeMemberExpr *
CXXDependentScopeMemberExpr::Create(ASTContext &C,
                                Expr *Base, QualType BaseType, bool IsArrow,
                                SourceLocation OperatorLoc,
                                NestedNameSpecifierLoc QualifierLoc,
                                SourceLocation TemplateKWLoc,
                                NamedDecl *FirstQualifierFoundInScope,
                                DeclarationNameInfo MemberNameInfo,
                                const TemplateArgumentListInfo *TemplateArgs) {
  if (!TemplateArgs && !TemplateKWLoc.isValid())
    return new (C) CXXDependentScopeMemberExpr(C, Base, BaseType,
                                               IsArrow, OperatorLoc,
                                               QualifierLoc,
                                               FirstQualifierFoundInScope,
                                               MemberNameInfo);

  unsigned NumTemplateArgs = TemplateArgs ? TemplateArgs->size() : 0;
  std::size_t size = sizeof(CXXDependentScopeMemberExpr)
    + ASTTemplateKWAndArgsInfo::sizeFor(NumTemplateArgs);

  void *Mem = C.Allocate(size, llvm::alignOf<CXXDependentScopeMemberExpr>());
  return new (Mem) CXXDependentScopeMemberExpr(C, Base, BaseType,
                                               IsArrow, OperatorLoc,
                                               QualifierLoc,
                                               TemplateKWLoc,
                                               FirstQualifierFoundInScope,
                                               MemberNameInfo, TemplateArgs);
}

CXXDependentScopeMemberExpr *
CXXDependentScopeMemberExpr::CreateEmpty(ASTContext &C,
                                         bool HasTemplateKWAndArgsInfo,
                                         unsigned NumTemplateArgs) {
  if (!HasTemplateKWAndArgsInfo)
    return new (C) CXXDependentScopeMemberExpr(C, 0, QualType(),
                                               0, SourceLocation(),
                                               NestedNameSpecifierLoc(), 0,
                                               DeclarationNameInfo());

  std::size_t size = sizeof(CXXDependentScopeMemberExpr) +
                     ASTTemplateKWAndArgsInfo::sizeFor(NumTemplateArgs);
  void *Mem = C.Allocate(size, llvm::alignOf<CXXDependentScopeMemberExpr>());
  CXXDependentScopeMemberExpr *E
    =  new (Mem) CXXDependentScopeMemberExpr(C, 0, QualType(),
                                             0, SourceLocation(),
                                             NestedNameSpecifierLoc(),
                                             SourceLocation(), 0,
                                             DeclarationNameInfo(), 0);
  E->HasTemplateKWAndArgsInfo = true;
  return E;
}

bool CXXDependentScopeMemberExpr::isImplicitAccess() const {
  if (Base == 0)
    return true;
  
  return cast<Expr>(Base)->isImplicitCXXThis();
}

static bool hasOnlyNonStaticMemberFunctions(UnresolvedSetIterator begin,
                                            UnresolvedSetIterator end) {
  do {
    NamedDecl *decl = *begin;
    if (isa<UnresolvedUsingValueDecl>(decl))
      return false;
    if (isa<UsingShadowDecl>(decl))
      decl = cast<UsingShadowDecl>(decl)->getUnderlyingDecl();

    // Unresolved member expressions should only contain methods and
    // method templates.
    assert(isa<CXXMethodDecl>(decl) || isa<FunctionTemplateDecl>(decl));

    if (isa<FunctionTemplateDecl>(decl))
      decl = cast<FunctionTemplateDecl>(decl)->getTemplatedDecl();
    if (cast<CXXMethodDecl>(decl)->isStatic())
      return false;
  } while (++begin != end);

  return true;
}

UnresolvedMemberExpr::UnresolvedMemberExpr(ASTContext &C, 
                                           bool HasUnresolvedUsing,
                                           Expr *Base, QualType BaseType,
                                           bool IsArrow,
                                           SourceLocation OperatorLoc,
                                           NestedNameSpecifierLoc QualifierLoc,
                                           SourceLocation TemplateKWLoc,
                                   const DeclarationNameInfo &MemberNameInfo,
                                   const TemplateArgumentListInfo *TemplateArgs,
                                           UnresolvedSetIterator Begin, 
                                           UnresolvedSetIterator End)
  : OverloadExpr(UnresolvedMemberExprClass, C, QualifierLoc, TemplateKWLoc,
                 MemberNameInfo, TemplateArgs, Begin, End,
                 // Dependent
                 ((Base && Base->isTypeDependent()) ||
                  BaseType->isDependentType()),
                 ((Base && Base->isInstantiationDependent()) ||
                   BaseType->isInstantiationDependentType()),
                 // Contains unexpanded parameter pack
                 ((Base && Base->containsUnexpandedParameterPack()) ||
                  BaseType->containsUnexpandedParameterPack())),
    IsArrow(IsArrow), HasUnresolvedUsing(HasUnresolvedUsing),
    Base(Base), BaseType(BaseType), OperatorLoc(OperatorLoc) {

  // Check whether all of the members are non-static member functions,
  // and if so, mark give this bound-member type instead of overload type.
  if (hasOnlyNonStaticMemberFunctions(Begin, End))
    setType(C.BoundMemberTy);
}

bool UnresolvedMemberExpr::isImplicitAccess() const {
  if (Base == 0)
    return true;
  
  return cast<Expr>(Base)->isImplicitCXXThis();
}

UnresolvedMemberExpr *
UnresolvedMemberExpr::Create(ASTContext &C, 
                             bool HasUnresolvedUsing,
                             Expr *Base, QualType BaseType, bool IsArrow,
                             SourceLocation OperatorLoc,
                             NestedNameSpecifierLoc QualifierLoc,
                             SourceLocation TemplateKWLoc,
                             const DeclarationNameInfo &MemberNameInfo,
                             const TemplateArgumentListInfo *TemplateArgs,
                             UnresolvedSetIterator Begin, 
                             UnresolvedSetIterator End) {
  std::size_t size = sizeof(UnresolvedMemberExpr);
  if (TemplateArgs)
    size += ASTTemplateKWAndArgsInfo::sizeFor(TemplateArgs->size());
  else if (TemplateKWLoc.isValid())
    size += ASTTemplateKWAndArgsInfo::sizeFor(0);

  void *Mem = C.Allocate(size, llvm::alignOf<UnresolvedMemberExpr>());
  return new (Mem) UnresolvedMemberExpr(C, 
                             HasUnresolvedUsing, Base, BaseType,
                             IsArrow, OperatorLoc, QualifierLoc, TemplateKWLoc,
                             MemberNameInfo, TemplateArgs, Begin, End);
}

UnresolvedMemberExpr *
UnresolvedMemberExpr::CreateEmpty(ASTContext &C, bool HasTemplateKWAndArgsInfo,
                                  unsigned NumTemplateArgs) {
  std::size_t size = sizeof(UnresolvedMemberExpr);
  if (HasTemplateKWAndArgsInfo)
    size += ASTTemplateKWAndArgsInfo::sizeFor(NumTemplateArgs);

  void *Mem = C.Allocate(size, llvm::alignOf<UnresolvedMemberExpr>());
  UnresolvedMemberExpr *E = new (Mem) UnresolvedMemberExpr(EmptyShell());
  E->HasTemplateKWAndArgsInfo = HasTemplateKWAndArgsInfo;
  return E;
}

CXXRecordDecl *UnresolvedMemberExpr::getNamingClass() const {
  // Unlike for UnresolvedLookupExpr, it is very easy to re-derive this.

  // If there was a nested name specifier, it names the naming class.
  // It can't be dependent: after all, we were actually able to do the
  // lookup.
  CXXRecordDecl *Record = 0;
  if (getQualifier()) {
    const Type *T = getQualifier()->getAsType();
    assert(T && "qualifier in member expression does not name type");
    Record = T->getAsCXXRecordDecl();
    assert(Record && "qualifier in member expression does not name record");
  }
  // Otherwise the naming class must have been the base class.
  else {
    QualType BaseType = getBaseType().getNonReferenceType();
    if (isArrow()) {
      const PointerType *PT = BaseType->getAs<PointerType>();
      assert(PT && "base of arrow member access is not pointer");
      BaseType = PT->getPointeeType();
    }
    
    Record = BaseType->getAsCXXRecordDecl();
    assert(Record && "base of member expression does not name record");
  }
  
  return Record;
}

SubstNonTypeTemplateParmPackExpr::
SubstNonTypeTemplateParmPackExpr(QualType T, 
                                 NonTypeTemplateParmDecl *Param,
                                 SourceLocation NameLoc,
                                 const TemplateArgument &ArgPack)
  : Expr(SubstNonTypeTemplateParmPackExprClass, T, VK_RValue, OK_Ordinary, 
         true, true, true, true),
    Param(Param), Arguments(ArgPack.pack_begin()), 
    NumArguments(ArgPack.pack_size()), NameLoc(NameLoc) { }

TemplateArgument SubstNonTypeTemplateParmPackExpr::getArgumentPack() const {
  return TemplateArgument(Arguments, NumArguments);
}

FunctionParmPackExpr::FunctionParmPackExpr(QualType T, ParmVarDecl *ParamPack,
                                           SourceLocation NameLoc,
                                           unsigned NumParams,
                                           Decl * const *Params)
  : Expr(FunctionParmPackExprClass, T, VK_LValue, OK_Ordinary,
         true, true, true, true),
    ParamPack(ParamPack), NameLoc(NameLoc), NumParameters(NumParams) {
  if (Params)
    std::uninitialized_copy(Params, Params + NumParams,
                            reinterpret_cast<Decl**>(this+1));
}

FunctionParmPackExpr *
FunctionParmPackExpr::Create(ASTContext &Context, QualType T,
                             ParmVarDecl *ParamPack, SourceLocation NameLoc,
                             llvm::ArrayRef<Decl*> Params) {
  return new (Context.Allocate(sizeof(FunctionParmPackExpr) +
                               sizeof(ParmVarDecl*) * Params.size()))
    FunctionParmPackExpr(T, ParamPack, NameLoc, Params.size(), Params.data());
}

FunctionParmPackExpr *
FunctionParmPackExpr::CreateEmpty(ASTContext &Context, unsigned NumParams) {
  return new (Context.Allocate(sizeof(FunctionParmPackExpr) +
                               sizeof(ParmVarDecl*) * NumParams))
    FunctionParmPackExpr(QualType(), 0, SourceLocation(), 0, 0);
}

TypeTraitExpr::TypeTraitExpr(QualType T, SourceLocation Loc, TypeTrait Kind,
                             ArrayRef<TypeSourceInfo *> Args,
                             SourceLocation RParenLoc,
                             bool Value)
  : Expr(TypeTraitExprClass, T, VK_RValue, OK_Ordinary,
         /*TypeDependent=*/false,
         /*ValueDependent=*/false,
         /*InstantiationDependent=*/false,
         /*ContainsUnexpandedParameterPack=*/false),
    Loc(Loc), RParenLoc(RParenLoc)
{
  TypeTraitExprBits.Kind = Kind;
  TypeTraitExprBits.Value = Value;
  TypeTraitExprBits.NumArgs = Args.size();

  TypeSourceInfo **ToArgs = getTypeSourceInfos();
  
  for (unsigned I = 0, N = Args.size(); I != N; ++I) {
    if (Args[I]->getType()->isDependentType())
      setValueDependent(true);
    if (Args[I]->getType()->isInstantiationDependentType())
      setInstantiationDependent(true);
    if (Args[I]->getType()->containsUnexpandedParameterPack())
      setContainsUnexpandedParameterPack(true);
    
    ToArgs[I] = Args[I];
  }
}

TypeTraitExpr *TypeTraitExpr::Create(ASTContext &C, QualType T, 
                                     SourceLocation Loc, 
                                     TypeTrait Kind,
                                     ArrayRef<TypeSourceInfo *> Args,
                                     SourceLocation RParenLoc,
                                     bool Value) {
  unsigned Size = sizeof(TypeTraitExpr) + sizeof(TypeSourceInfo*) * Args.size();
  void *Mem = C.Allocate(Size);
  return new (Mem) TypeTraitExpr(T, Loc, Kind, Args, RParenLoc, Value);
}

TypeTraitExpr *TypeTraitExpr::CreateDeserialized(ASTContext &C,
                                                 unsigned NumArgs) {
  unsigned Size = sizeof(TypeTraitExpr) + sizeof(TypeSourceInfo*) * NumArgs;
  void *Mem = C.Allocate(Size);
  return new (Mem) TypeTraitExpr(EmptyShell());
}

void ArrayTypeTraitExpr::anchor() { }
