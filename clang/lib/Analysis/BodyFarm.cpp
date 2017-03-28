//== BodyFarm.cpp  - Factory for conjuring up fake bodies ----------*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// BodyFarm is a factory for creating faux implementations for functions/methods
// for analysis purposes.
//
//===----------------------------------------------------------------------===//

#include "BodyFarm.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprObjC.h"
#include "clang/Analysis/CodeInjector.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// Helper creation functions for constructing faux ASTs.
//===----------------------------------------------------------------------===//

static bool isDispatchBlock(QualType Ty) {
  // Is it a block pointer?
  const BlockPointerType *BPT = Ty->getAs<BlockPointerType>();
  if (!BPT)
    return false;

  // Check if the block pointer type takes no arguments and
  // returns void.
  const FunctionProtoType *FT =
  BPT->getPointeeType()->getAs<FunctionProtoType>();
  return FT && FT->getReturnType()->isVoidType() && FT->getNumParams() == 0;
}

namespace {
class ASTMaker {
public:
  ASTMaker(ASTContext &C) : C(C) {}
  
  /// Create a new BinaryOperator representing a simple assignment.
  BinaryOperator *makeAssignment(const Expr *LHS, const Expr *RHS, QualType Ty);
  
  /// Create a new BinaryOperator representing a comparison.
  BinaryOperator *makeComparison(const Expr *LHS, const Expr *RHS,
                                 BinaryOperator::Opcode Op);
  
  /// Create a new compound stmt using the provided statements.
  CompoundStmt *makeCompound(ArrayRef<Stmt*>);
  
  /// Create a new DeclRefExpr for the referenced variable.
  DeclRefExpr *makeDeclRefExpr(const VarDecl *D);
  
  /// Create a new UnaryOperator representing a dereference.
  UnaryOperator *makeDereference(const Expr *Arg, QualType Ty);
  
  /// Create an implicit cast for an integer conversion.
  Expr *makeIntegralCast(const Expr *Arg, QualType Ty);
  
  /// Create an implicit cast to a builtin boolean type.
  ImplicitCastExpr *makeIntegralCastToBoolean(const Expr *Arg);
  
  // Create an implicit cast for lvalue-to-rvaluate conversions.
  ImplicitCastExpr *makeLvalueToRvalue(const Expr *Arg, QualType Ty);
  
  /// Create an Objective-C bool literal.
  ObjCBoolLiteralExpr *makeObjCBool(bool Val);

  /// Create an Objective-C ivar reference.
  ObjCIvarRefExpr *makeObjCIvarRef(const Expr *Base, const ObjCIvarDecl *IVar);
  
  /// Create a Return statement.
  ReturnStmt *makeReturn(const Expr *RetVal);
  
private:
  ASTContext &C;
};
}

BinaryOperator *ASTMaker::makeAssignment(const Expr *LHS, const Expr *RHS,
                                         QualType Ty) {
 return new (C) BinaryOperator(const_cast<Expr*>(LHS), const_cast<Expr*>(RHS),
                               BO_Assign, Ty, VK_RValue,
                               OK_Ordinary, SourceLocation(), FPOptions());
}

BinaryOperator *ASTMaker::makeComparison(const Expr *LHS, const Expr *RHS,
                                         BinaryOperator::Opcode Op) {
  assert(BinaryOperator::isLogicalOp(Op) ||
         BinaryOperator::isComparisonOp(Op));
  return new (C) BinaryOperator(const_cast<Expr*>(LHS),
                                const_cast<Expr*>(RHS),
                                Op,
                                C.getLogicalOperationType(),
                                VK_RValue,
                                OK_Ordinary, SourceLocation(), FPOptions());
}

CompoundStmt *ASTMaker::makeCompound(ArrayRef<Stmt *> Stmts) {
  return new (C) CompoundStmt(C, Stmts, SourceLocation(), SourceLocation());
}

DeclRefExpr *ASTMaker::makeDeclRefExpr(const VarDecl *D) {
  DeclRefExpr *DR =
    DeclRefExpr::Create(/* Ctx = */ C,
                        /* QualifierLoc = */ NestedNameSpecifierLoc(),
                        /* TemplateKWLoc = */ SourceLocation(),
                        /* D = */ const_cast<VarDecl*>(D),
                        /* RefersToEnclosingVariableOrCapture = */ false,
                        /* NameLoc = */ SourceLocation(),
                        /* T = */ D->getType(),
                        /* VK = */ VK_LValue);
  return DR;
}

UnaryOperator *ASTMaker::makeDereference(const Expr *Arg, QualType Ty) {
  return new (C) UnaryOperator(const_cast<Expr*>(Arg), UO_Deref, Ty,
                               VK_LValue, OK_Ordinary, SourceLocation());
}

ImplicitCastExpr *ASTMaker::makeLvalueToRvalue(const Expr *Arg, QualType Ty) {
  return ImplicitCastExpr::Create(C, Ty, CK_LValueToRValue,
                                  const_cast<Expr*>(Arg), nullptr, VK_RValue);
}

Expr *ASTMaker::makeIntegralCast(const Expr *Arg, QualType Ty) {
  if (Arg->getType() == Ty)
    return const_cast<Expr*>(Arg);

  return ImplicitCastExpr::Create(C, Ty, CK_IntegralCast,
                                  const_cast<Expr*>(Arg), nullptr, VK_RValue);
}

ImplicitCastExpr *ASTMaker::makeIntegralCastToBoolean(const Expr *Arg) {
  return ImplicitCastExpr::Create(C, C.BoolTy, CK_IntegralToBoolean,
                                  const_cast<Expr*>(Arg), nullptr, VK_RValue);
}

ObjCBoolLiteralExpr *ASTMaker::makeObjCBool(bool Val) {
  QualType Ty = C.getBOOLDecl() ? C.getBOOLType() : C.ObjCBuiltinBoolTy;
  return new (C) ObjCBoolLiteralExpr(Val, Ty, SourceLocation());
}

ObjCIvarRefExpr *ASTMaker::makeObjCIvarRef(const Expr *Base,
                                           const ObjCIvarDecl *IVar) {
  return new (C) ObjCIvarRefExpr(const_cast<ObjCIvarDecl*>(IVar),
                                 IVar->getType(), SourceLocation(),
                                 SourceLocation(), const_cast<Expr*>(Base),
                                 /*arrow=*/true, /*free=*/false);
}


ReturnStmt *ASTMaker::makeReturn(const Expr *RetVal) {
  return new (C) ReturnStmt(SourceLocation(), const_cast<Expr*>(RetVal),
                            nullptr);
}

//===----------------------------------------------------------------------===//
// Creation functions for faux ASTs.
//===----------------------------------------------------------------------===//

typedef Stmt *(*FunctionFarmer)(ASTContext &C, const FunctionDecl *D);

/// Create a fake body for dispatch_once.
static Stmt *create_dispatch_once(ASTContext &C, const FunctionDecl *D) {
  // Check if we have at least two parameters.
  if (D->param_size() != 2)
    return nullptr;

  // Check if the first parameter is a pointer to integer type.
  const ParmVarDecl *Predicate = D->getParamDecl(0);
  QualType PredicateQPtrTy = Predicate->getType();
  const PointerType *PredicatePtrTy = PredicateQPtrTy->getAs<PointerType>();
  if (!PredicatePtrTy)
    return nullptr;
  QualType PredicateTy = PredicatePtrTy->getPointeeType();
  if (!PredicateTy->isIntegerType())
    return nullptr;

  // Check if the second parameter is the proper block type.
  const ParmVarDecl *Block = D->getParamDecl(1);
  QualType Ty = Block->getType();
  if (!isDispatchBlock(Ty))
    return nullptr;

  // Everything checks out.  Create a fakse body that checks the predicate,
  // sets it, and calls the block.  Basically, an AST dump of:
  //
  // void dispatch_once(dispatch_once_t *predicate, dispatch_block_t block) {
  //  if (!*predicate) {
  //    *predicate = 1;
  //    block();
  //  }
  // }
  
  ASTMaker M(C);
  
  // (1) Create the call.
  DeclRefExpr *DR = M.makeDeclRefExpr(Block);
  ImplicitCastExpr *ICE = M.makeLvalueToRvalue(DR, Ty);
  CallExpr *CE = new (C) CallExpr(C, ICE, None, C.VoidTy, VK_RValue,
                                  SourceLocation());

  // (2) Create the assignment to the predicate.
  IntegerLiteral *IL =
    IntegerLiteral::Create(C, llvm::APInt(C.getTypeSize(C.IntTy), (uint64_t) 1),
                           C.IntTy, SourceLocation());
  BinaryOperator *B =
    M.makeAssignment(
       M.makeDereference(
          M.makeLvalueToRvalue(
            M.makeDeclRefExpr(Predicate), PredicateQPtrTy),
            PredicateTy),
       M.makeIntegralCast(IL, PredicateTy),
       PredicateTy);
  
  // (3) Create the compound statement.
  Stmt *Stmts[] = { B, CE };
  CompoundStmt *CS = M.makeCompound(Stmts);
  
  // (4) Create the 'if' condition.
  ImplicitCastExpr *LValToRval =
    M.makeLvalueToRvalue(
      M.makeDereference(
        M.makeLvalueToRvalue(
          M.makeDeclRefExpr(Predicate),
          PredicateQPtrTy),
        PredicateTy),
    PredicateTy);
  
  UnaryOperator *UO = new (C) UnaryOperator(LValToRval, UO_LNot, C.IntTy,
                                           VK_RValue, OK_Ordinary,
                                           SourceLocation());
  
  // (5) Create the 'if' statement.
  IfStmt *If = new (C) IfStmt(C, SourceLocation(), false, nullptr, nullptr,
                              UO, CS);
  return If;
}

/// Create a fake body for dispatch_sync.
static Stmt *create_dispatch_sync(ASTContext &C, const FunctionDecl *D) {
  // Check if we have at least two parameters.
  if (D->param_size() != 2)
    return nullptr;

  // Check if the second parameter is a block.
  const ParmVarDecl *PV = D->getParamDecl(1);
  QualType Ty = PV->getType();
  if (!isDispatchBlock(Ty))
    return nullptr;

  // Everything checks out.  Create a fake body that just calls the block.
  // This is basically just an AST dump of:
  //
  // void dispatch_sync(dispatch_queue_t queue, void (^block)(void)) {
  //   block();
  // }
  //  
  ASTMaker M(C);
  DeclRefExpr *DR = M.makeDeclRefExpr(PV);
  ImplicitCastExpr *ICE = M.makeLvalueToRvalue(DR, Ty);
  CallExpr *CE = new (C) CallExpr(C, ICE, None, C.VoidTy, VK_RValue,
                                  SourceLocation());
  return CE;
}

static Stmt *create_OSAtomicCompareAndSwap(ASTContext &C, const FunctionDecl *D)
{
  // There are exactly 3 arguments.
  if (D->param_size() != 3)
    return nullptr;

  // Signature:
  // _Bool OSAtomicCompareAndSwapPtr(void *__oldValue,
  //                                 void *__newValue,
  //                                 void * volatile *__theValue)
  // Generate body:
  //   if (oldValue == *theValue) {
  //    *theValue = newValue;
  //    return YES;
  //   }
  //   else return NO;

  QualType ResultTy = D->getReturnType();
  bool isBoolean = ResultTy->isBooleanType();
  if (!isBoolean && !ResultTy->isIntegralType(C))
    return nullptr;

  const ParmVarDecl *OldValue = D->getParamDecl(0);
  QualType OldValueTy = OldValue->getType();

  const ParmVarDecl *NewValue = D->getParamDecl(1);
  QualType NewValueTy = NewValue->getType();
  
  assert(OldValueTy == NewValueTy);
  
  const ParmVarDecl *TheValue = D->getParamDecl(2);
  QualType TheValueTy = TheValue->getType();
  const PointerType *PT = TheValueTy->getAs<PointerType>();
  if (!PT)
    return nullptr;
  QualType PointeeTy = PT->getPointeeType();
  
  ASTMaker M(C);
  // Construct the comparison.
  Expr *Comparison =
    M.makeComparison(
      M.makeLvalueToRvalue(M.makeDeclRefExpr(OldValue), OldValueTy),
      M.makeLvalueToRvalue(
        M.makeDereference(
          M.makeLvalueToRvalue(M.makeDeclRefExpr(TheValue), TheValueTy),
          PointeeTy),
        PointeeTy),
      BO_EQ);

  // Construct the body of the IfStmt.
  Stmt *Stmts[2];
  Stmts[0] =
    M.makeAssignment(
      M.makeDereference(
        M.makeLvalueToRvalue(M.makeDeclRefExpr(TheValue), TheValueTy),
        PointeeTy),
      M.makeLvalueToRvalue(M.makeDeclRefExpr(NewValue), NewValueTy),
      NewValueTy);
  
  Expr *BoolVal = M.makeObjCBool(true);
  Expr *RetVal = isBoolean ? M.makeIntegralCastToBoolean(BoolVal)
                           : M.makeIntegralCast(BoolVal, ResultTy);
  Stmts[1] = M.makeReturn(RetVal);
  CompoundStmt *Body = M.makeCompound(Stmts);
  
  // Construct the else clause.
  BoolVal = M.makeObjCBool(false);
  RetVal = isBoolean ? M.makeIntegralCastToBoolean(BoolVal)
                     : M.makeIntegralCast(BoolVal, ResultTy);
  Stmt *Else = M.makeReturn(RetVal);
  
  /// Construct the If.
  Stmt *If = new (C) IfStmt(C, SourceLocation(), false, nullptr, nullptr,
                            Comparison, Body, SourceLocation(), Else);

  return If;  
}

Stmt *BodyFarm::getBody(const FunctionDecl *D) {
  D = D->getCanonicalDecl();
  
  Optional<Stmt *> &Val = Bodies[D];
  if (Val.hasValue())
    return Val.getValue();

  Val = nullptr;

  if (D->getIdentifier() == nullptr)
    return nullptr;

  StringRef Name = D->getName();
  if (Name.empty())
    return nullptr;

  FunctionFarmer FF;

  if (Name.startswith("OSAtomicCompareAndSwap") ||
      Name.startswith("objc_atomicCompareAndSwap")) {
    FF = create_OSAtomicCompareAndSwap;
  }
  else {
    FF = llvm::StringSwitch<FunctionFarmer>(Name)
          .Case("dispatch_sync", create_dispatch_sync)
          .Case("dispatch_once", create_dispatch_once)
          .Default(nullptr);
  }
  
  if (FF) { Val = FF(C, D); }
  else if (Injector) { Val = Injector->getBody(D); }
  return Val.getValue();
}

static const ObjCIvarDecl *findBackingIvar(const ObjCPropertyDecl *Prop) {
  const ObjCIvarDecl *IVar = Prop->getPropertyIvarDecl();

  if (IVar)
    return IVar;

  // When a readonly property is shadowed in a class extensions with a
  // a readwrite property, the instance variable belongs to the shadowing
  // property rather than the shadowed property. If there is no instance
  // variable on a readonly property, check to see whether the property is
  // shadowed and if so try to get the instance variable from shadowing
  // property.
  if (!Prop->isReadOnly())
    return nullptr;

  auto *Container = cast<ObjCContainerDecl>(Prop->getDeclContext());
  const ObjCInterfaceDecl *PrimaryInterface = nullptr;
  if (auto *InterfaceDecl = dyn_cast<ObjCInterfaceDecl>(Container)) {
    PrimaryInterface = InterfaceDecl;
  } else if (auto *CategoryDecl = dyn_cast<ObjCCategoryDecl>(Container)) {
    PrimaryInterface = CategoryDecl->getClassInterface();
  } else if (auto *ImplDecl = dyn_cast<ObjCImplDecl>(Container)) {
    PrimaryInterface = ImplDecl->getClassInterface();
  } else {
    return nullptr;
  }

  // FindPropertyVisibleInPrimaryClass() looks first in class extensions, so it
  // is guaranteed to find the shadowing property, if it exists, rather than
  // the shadowed property.
  auto *ShadowingProp = PrimaryInterface->FindPropertyVisibleInPrimaryClass(
      Prop->getIdentifier(), Prop->getQueryKind());
  if (ShadowingProp && ShadowingProp != Prop) {
    IVar = ShadowingProp->getPropertyIvarDecl();
  }

  return IVar;
}

static Stmt *createObjCPropertyGetter(ASTContext &Ctx,
                                      const ObjCPropertyDecl *Prop) {
  // First, find the backing ivar.
  const ObjCIvarDecl *IVar = findBackingIvar(Prop);
  if (!IVar)
    return nullptr;

  // Ignore weak variables, which have special behavior.
  if (Prop->getPropertyAttributes() & ObjCPropertyDecl::OBJC_PR_weak)
    return nullptr;

  // Look to see if Sema has synthesized a body for us. This happens in
  // Objective-C++ because the return value may be a C++ class type with a
  // non-trivial copy constructor. We can only do this if we can find the
  // @synthesize for this property, though (or if we know it's been auto-
  // synthesized).
  const ObjCImplementationDecl *ImplDecl =
    IVar->getContainingInterface()->getImplementation();
  if (ImplDecl) {
    for (const auto *I : ImplDecl->property_impls()) {
      if (I->getPropertyDecl() != Prop)
        continue;

      if (I->getGetterCXXConstructor()) {
        ASTMaker M(Ctx);
        return M.makeReturn(I->getGetterCXXConstructor());
      }
    }
  }

  // Sanity check that the property is the same type as the ivar, or a
  // reference to it, and that it is either an object pointer or trivially
  // copyable.
  if (!Ctx.hasSameUnqualifiedType(IVar->getType(),
                                  Prop->getType().getNonReferenceType()))
    return nullptr;
  if (!IVar->getType()->isObjCLifetimeType() &&
      !IVar->getType().isTriviallyCopyableType(Ctx))
    return nullptr;

  // Generate our body:
  //   return self->_ivar;
  ASTMaker M(Ctx);

  const VarDecl *selfVar = Prop->getGetterMethodDecl()->getSelfDecl();
  if (!selfVar)
    return nullptr;

  Expr *loadedIVar =
    M.makeObjCIvarRef(
      M.makeLvalueToRvalue(
        M.makeDeclRefExpr(selfVar),
        selfVar->getType()),
      IVar);

  if (!Prop->getType()->isReferenceType())
    loadedIVar = M.makeLvalueToRvalue(loadedIVar, IVar->getType());

  return M.makeReturn(loadedIVar);
}

Stmt *BodyFarm::getBody(const ObjCMethodDecl *D) {
  // We currently only know how to synthesize property accessors.
  if (!D->isPropertyAccessor())
    return nullptr;

  D = D->getCanonicalDecl();

  Optional<Stmt *> &Val = Bodies[D];
  if (Val.hasValue())
    return Val.getValue();
  Val = nullptr;

  const ObjCPropertyDecl *Prop = D->findPropertyDecl();
  if (!Prop)
    return nullptr;

  // For now, we only synthesize getters.
  // Synthesizing setters would cause false negatives in the
  // RetainCountChecker because the method body would bind the parameter
  // to an instance variable, causing it to escape. This would prevent
  // warning in the following common scenario:
  //
  //  id foo = [[NSObject alloc] init];
  //  self.foo = foo; // We should warn that foo leaks here.
  //
  if (D->param_size() != 0)
    return nullptr;

  Val = createObjCPropertyGetter(C, Prop);

  return Val.getValue();
}

