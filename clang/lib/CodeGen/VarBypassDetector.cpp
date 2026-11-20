//===--- VarBypassDetector.cpp - Bypass jumps detector ------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VarBypassDetector.h"

#include "CodeGenModule.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/ScopeExit.h"

using namespace clang;
using namespace CodeGen;

/// True if the body contains a goto, switch, indirect goto, or asm goto. Lets
/// Init() skip scope building for the common jump-free function, this is a
/// minor optimization win.
static bool hasJumpStmts(const Stmt *Body) {
  llvm::SmallVector<const Stmt *, 32> Worklist;
  Worklist.push_back(Body);
  while (!Worklist.empty()) {
    const Stmt *S = Worklist.pop_back_val();
    if (!S)
      continue;
    switch (S->getStmtClass()) {
    case Stmt::GotoStmtClass:
    case Stmt::SwitchStmtClass:
    case Stmt::IndirectGotoStmtClass:
      return true;
    case Stmt::GCCAsmStmtClass:
      if (cast<GCCAsmStmt>(S)->isAsmGoto())
        return true;
      break;
    case Stmt::LambdaExprClass:
      for (const Expr *Init : cast<LambdaExpr>(S)->capture_inits())
        Worklist.push_back(Init);
      continue;
    default:
      break;
    }
    for (const Stmt *Child : S->children())
      Worklist.push_back(Child);
  }
  return false;
}

/// Clear the object and pre-process for the given statement, usually function
/// body statement.
void VarBypassDetector::Init(CodeGenModule &CGM, const Stmt *Body) {
  FromScopes.clear();
  ToScopes.clear();
  Bypasses.clear();
  BackwardBypassedVars.clear();
  BackwardBypassedCLs.clear();
  BypassedVarsAtSource.clear();
  Scopes = {{~0U, 0, nullptr}};
  CurEnclosingBlockScope = 0;
  AlwaysBypassed = false;
  Initialized = true;
  if (!hasJumpStmts(Body))
    return;
  unsigned ParentScope = 0;
  AlwaysBypassed = !BuildScopeInformation(CGM, Body, ParentScope);
  if (!AlwaysBypassed)
    Detect(CGM.getLangOpts().CPlusPlus);
}

bool VarBypassDetector::BuildScopeInformation(CodeGenModule &CGM, QualType Ty,
                                              unsigned &ParentScope) {
  // Mirrors CodeGenFunction::EmitVariablyModifiedType: walk variably modified
  // types to find any evaluated VLA size expressions or typeof(expr) operands.
  while (!Ty.isNull() && Ty->isVariablyModifiedType()) {
    const Type *T = Ty.getTypePtr();
    switch (T->getTypeClass()) {
#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define NON_CANONICAL_TYPE(Class, Base)
#define DEPENDENT_TYPE(Class, Base) case Type::Class:
#define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(Class, Base)
#include "clang/AST/TypeNodes.inc"
      llvm_unreachable("unexpected dependent type!");

    // These types are never variably-modified.
    case Type::Builtin:
    case Type::Complex:
    case Type::Vector:
    case Type::ExtVector:
    case Type::ConstantMatrix:
    case Type::Record:
    case Type::Enum:
    case Type::Using:
    case Type::TemplateSpecialization:
    case Type::ObjCTypeParam:
    case Type::ObjCObject:
    case Type::ObjCInterface:
    case Type::ObjCObjectPointer:
    case Type::BitInt:
    case Type::HLSLInlineSpirv:
    case Type::PredefinedSugar:
      llvm_unreachable("type class is never variably-modified!");

    case Type::Adjusted:
      Ty = cast<AdjustedType>(T)->getAdjustedType();
      break;

    case Type::Decayed:
      Ty = cast<DecayedType>(T)->getPointeeType();
      break;

    case Type::Pointer:
      Ty = cast<PointerType>(T)->getPointeeType();
      break;

    case Type::BlockPointer:
      Ty = cast<BlockPointerType>(T)->getPointeeType();
      break;

    case Type::LValueReference:
    case Type::RValueReference:
      Ty = cast<ReferenceType>(T)->getPointeeType();
      break;

    case Type::MemberPointer:
      Ty = cast<MemberPointerType>(T)->getPointeeType();
      break;

    case Type::ArrayParameter:
    case Type::ConstantArray:
    case Type::IncompleteArray:
      Ty = cast<ArrayType>(T)->getElementType();
      break;

    case Type::VariableArray: {
      const auto *VAT = cast<VariableArrayType>(T);
      if (const Expr *Size = VAT->getSizeExpr())
        if (!BuildScopeInformation(CGM, Size, ParentScope))
          return false;
      Ty = VAT->getElementType();
      break;
    }

    case Type::FunctionProto:
    case Type::FunctionNoProto:
      Ty = cast<FunctionType>(T)->getReturnType();
      break;

    case Type::Paren:
    case Type::TypeOf:
    case Type::UnaryTransform:
    case Type::Attributed:
    case Type::BTFTagAttributed:
    case Type::OverflowBehavior:
    case Type::HLSLAttributedResource:
    case Type::SubstTemplateTypeParm:
    case Type::MacroQualified:
    case Type::CountAttributed:
    case Type::LateParsedAttr:
      Ty = Ty.getSingleStepDesugaredType(CGM.getContext());
      break;

    case Type::Typedef:
    case Type::Decltype:
    case Type::Auto:
    case Type::DeducedTemplateSpecialization:
    case Type::PackIndexing:
      // Stop walking: CodeGenFunction::EmitVariablyModifiedType does not
      // evaluate anything for these (e.g., typedefs are evaluated at their
      // declaration and decltype operands are unevaluated).
      return true;

    case Type::TypeOfExpr:
      return BuildScopeInformation(
          CGM, cast<TypeOfExprType>(T)->getUnderlyingExpr(), ParentScope);

    case Type::Atomic:
      Ty = cast<AtomicType>(T)->getValueType();
      break;

    case Type::Pipe:
      Ty = cast<PipeType>(T)->getElementType();
      break;
    }
  }
  return true;
}

/// Build scope information for a declaration that is part of a DeclStmt.
/// Returns false if we failed to build scope information and can't tell for
/// which vars are being bypassed.
bool VarBypassDetector::BuildScopeInformation(CodeGenModule &CGM, const Decl *D,
                                              unsigned &ParentScope) {
  if (const auto *TD = dyn_cast<TypedefNameDecl>(D))
    return BuildScopeInformation(CGM, TD->getUnderlyingType(), ParentScope);

  const VarDecl *VD = dyn_cast<VarDecl>(D);
  if (!VD)
    return true;

  if (VD->hasLocalStorage()) {
    Scopes.push_back({ParentScope, CurEnclosingBlockScope, VD});
    ParentScope = Scopes.size() - 1;
    if (!BuildScopeInformation(CGM, VD->getType(), ParentScope))
      return false;
  }

  if (VD->hasLocalStorage() || CGM.getLangOpts().CPlusPlus)
    if (const Expr *Init = VD->getInit())
      return BuildScopeInformation(CGM, Init, ParentScope);

  return true;
}

/// Returns true if SubStmt is a non-compound child statement of Parent that
/// introduces its own lexical cleanup scope (e.g. the body or branch of a
/// loop or conditional statement).
static bool needsChildCleanupScope(const Stmt *Parent, const Stmt *SubStmt) {
  if (isa<CompoundStmt>(SubStmt))
    return false;

  switch (Parent->getStmtClass()) {
  case Stmt::IfStmtClass: {
    const auto *IS = cast<IfStmt>(Parent);
    return SubStmt == IS->getThen() || SubStmt == IS->getElse();
  }
  case Stmt::WhileStmtClass:
    return SubStmt == cast<WhileStmt>(Parent)->getBody();
  case Stmt::DoStmtClass:
    return SubStmt == cast<DoStmt>(Parent)->getBody();
  case Stmt::ForStmtClass:
    return SubStmt == cast<ForStmt>(Parent)->getBody();
  default:
    return false;
  }
}

/// Walk through the statements, adding any labels or gotos to
/// LabelAndGotoScopes and recursively walking the AST as needed.
/// Returns false if we failed to build scope information and can't tell for
/// which vars are being bypassed.
bool VarBypassDetector::BuildScopeInformation(CodeGenModule &CGM, const Stmt *S,
                                              unsigned &origParentScope) {
  // If this is a statement, rather than an expression, scopes within it don't
  // propagate out into the enclosing scope. Otherwise we have to worry about
  // block literals, which have the lifetime of their enclosing statement.
  unsigned independentParentScope = origParentScope;
  unsigned &ParentScope = ((isa<Expr, AttributedStmt>(S) && !isa<StmtExpr>(S))
                               ? origParentScope
                               : independentParentScope);

  unsigned PrevBlockScope = CurEnclosingBlockScope;
  llvm::scope_exit RestoreBlock(
      [&] { CurEnclosingBlockScope = PrevBlockScope; });

  unsigned StmtsToSkip = 0u;

  switch (S->getStmtClass()) {
  case Stmt::IndirectGotoStmtClass:
    return false;

  case Stmt::CompoundStmtClass:
    Scopes.push_back({ParentScope, ParentScope, nullptr});
    ParentScope = Scopes.size() - 1;
    CurEnclosingBlockScope = ParentScope;
    break;

  case Stmt::CompoundLiteralExprClass: {
    const auto *CLE = cast<CompoundLiteralExpr>(S);
    if (!BuildScopeInformation(CGM, CLE->getType(), origParentScope))
      return false;
    if (!CGM.getLangOpts().CPlusPlus && !CLE->isFileScope()) {
      Scopes.push_back({origParentScope, CurEnclosingBlockScope, CLE});
      origParentScope = Scopes.size() - 1;
    }
    break;
  }

  case Stmt::UnaryExprOrTypeTraitExprClass: {
    const auto *UETT = cast<UnaryExprOrTypeTraitExpr>(S);
    if (UETT->isArgumentType())
      return BuildScopeInformation(CGM, UETT->getArgumentType(),
                                   origParentScope);
    if (!UETT->getTypeOfArgument()->isVariablyModifiedType())
      return true;
    break;
  }

  case Stmt::GenericSelectionExprClass: {
    const auto *GSE = cast<GenericSelectionExpr>(S);
    if (!GSE->isResultDependent())
      return BuildScopeInformation(CGM, GSE->getResultExpr(), origParentScope);
    return true;
  }

  case Stmt::ChooseExprClass:
    return BuildScopeInformation(CGM, cast<ChooseExpr>(S)->getChosenSubExpr(),
                                 origParentScope);

  case Stmt::LambdaExprClass:
    // Only walk capture initializers; the lambda body is a separate function
    // analyzed by its own VarBypassDetector.
    for (const Expr *Init : cast<LambdaExpr>(S)->capture_inits())
      if (Init && !BuildScopeInformation(CGM, Init, origParentScope))
        return false;
    return true;

  case Stmt::ForStmtClass:
    if (!CGM.getLangOpts().C99 && !CGM.getLangOpts().CPlusPlus) {
      if (const Stmt *Init = cast<ForStmt>(S)->getInit()) {
        if (!BuildScopeInformation(CGM, Init, origParentScope))
          return false;
        ParentScope = origParentScope;
        ++StmtsToSkip;
      }
    }
    [[fallthrough]];

  case Stmt::IfStmtClass:
  case Stmt::WhileStmtClass:
    Scopes.push_back({ParentScope, ParentScope, nullptr});
    ParentScope = Scopes.size() - 1;
    CurEnclosingBlockScope = ParentScope;
    break;

  case Stmt::SwitchStmtClass:
    Scopes.push_back({ParentScope, ParentScope, nullptr});
    ParentScope = Scopes.size() - 1;
    CurEnclosingBlockScope = ParentScope;
    if (const Stmt *Init = cast<SwitchStmt>(S)->getInit()) {
      if (!BuildScopeInformation(CGM, Init, ParentScope))
        return false;
      ++StmtsToSkip;
    }
    if (const VarDecl *Var = cast<SwitchStmt>(S)->getConditionVariable()) {
      if (!BuildScopeInformation(CGM, Var, ParentScope))
        return false;
      ++StmtsToSkip;
    }
    if (const Expr *Cond = cast<SwitchStmt>(S)->getCond()) {
      if (!BuildScopeInformation(CGM, Cond, ParentScope))
        return false;
      ++StmtsToSkip;
    }
    [[fallthrough]];

  case Stmt::GotoStmtClass:
    FromScopes.push_back({S, ParentScope});
    break;

  case Stmt::DeclStmtClass: {
    const DeclStmt *DS = cast<DeclStmt>(S);
    for (auto *I : DS->decls())
      if (!BuildScopeInformation(CGM, I, origParentScope))
        return false;
    return true;
  }

  case Stmt::CaseStmtClass:
  case Stmt::DefaultStmtClass:
  case Stmt::LabelStmtClass:
    llvm_unreachable("the loop below handles labels and cases");
    break;

  default:
    if (const auto *ECE = dyn_cast<ExplicitCastExpr>(S))
      if (!BuildScopeInformation(CGM, ECE->getTypeAsWritten(), origParentScope))
        return false;
    break;
  }

  for (const Stmt *SubStmt : S->children()) {
    if (!SubStmt)
      continue;
    if (StmtsToSkip) {
      --StmtsToSkip;
      continue;
    }

    bool IsDoCond = isa<DoStmt>(S) && SubStmt == cast<DoStmt>(S)->getCond();
    bool NeedsChildCleanupScope = needsChildCleanupScope(S, SubStmt);

    unsigned SubCleanupScope = ParentScope;
    unsigned SavedBlockScope = CurEnclosingBlockScope;
    llvm::scope_exit RestoreChildBlock(
        [&] { CurEnclosingBlockScope = SavedBlockScope; });
    if (NeedsChildCleanupScope) {
      Scopes.push_back({SubCleanupScope, SubCleanupScope, nullptr});
      SubCleanupScope = Scopes.size() - 1;
      CurEnclosingBlockScope = SubCleanupScope;
    }

    unsigned &ChildScope =
        IsDoCond ? origParentScope
                 : (NeedsChildCleanupScope ? SubCleanupScope : ParentScope);

    // Cases, labels, and defaults aren't "scope parents".  It's also
    // important to handle these iteratively instead of recursively in
    // order to avoid blowing out the stack.
    while (true) {
      const Stmt *Next;
      if (const SwitchCase *SC = dyn_cast<SwitchCase>(SubStmt))
        Next = SC->getSubStmt();
      else if (const LabelStmt *LS = dyn_cast<LabelStmt>(SubStmt))
        Next = LS->getSubStmt();
      else
        break;

      ToScopes[SubStmt] = ChildScope;
      SubStmt = Next;
    }

    // Recursively walk the AST.
    bool Result;
    CGM.runWithSufficientStackSpace(S->getEndLoc(), [&] {
      Result = BuildScopeInformation(CGM, SubStmt, ChildScope);
    });
    if (!Result)
      return false;
  }

  if (const auto *GAS = dyn_cast<GCCAsmStmt>(S); GAS && GAS->isAsmGoto())
    FromScopes.push_back({S, ParentScope});

  return true;
}

/// Checks each jump and stores each variable declaration they bypass.
void VarBypassDetector::Detect(bool IsCPlusPlus) {
  for (const auto &S : FromScopes) {
    const Stmt *St = S.first;
    unsigned From = S.second;
    if (const GotoStmt *GS = dyn_cast<GotoStmt>(St)) {
      if (const LabelStmt *LS = GS->getLabel()->getStmt())
        Detect(IsCPlusPlus, From, ToScopes[LS], GS);
    } else if (const SwitchStmt *SS = dyn_cast<SwitchStmt>(St)) {
      // Keyed by the switch, not by each case: the dispatch is the jump
      // source, and that is where the initialization is emitted.
      for (const SwitchCase *SC = SS->getSwitchCaseList(); SC;
           SC = SC->getNextSwitchCase()) {
        Detect(IsCPlusPlus, From, ToScopes[SC], SS);
      }
    } else if (const GCCAsmStmt *GAS = dyn_cast<GCCAsmStmt>(St)) {
      for (const AddrLabelExpr *L : GAS->labels())
        if (const LabelStmt *LS = L->getLabel()->getStmt())
          Detect(IsCPlusPlus, From, ToScopes[LS], GAS);
    } else {
      llvm_unreachable("goto, switch, or asm goto was expected");
    }
  }
}

/// Checks the jump and stores each variable declaration or compound literal it
/// bypasses.
void VarBypassDetector::Detect(bool IsCPlusPlus, unsigned From, unsigned To,
                               const Stmt *Source) {
  while (From != To) {
    if (From < To) {
      assert(Scopes[To].ParentScope < To &&
             "Scope tree must be topologically ordered");
      const ScopeEntry &ScopeTo = Scopes[To];
      To = ScopeTo.ParentScope;
      if (const auto *VD = ScopeTo.Item.dyn_cast<const VarDecl *>()) {
        Bypasses.insert(VD);
        BypassedVarsAtSource[Source].insert(VD);
      }
    } else {
      assert(Scopes[From].ParentScope < From &&
             "Scope tree must be topologically ordered");
      const ScopeEntry &ScopeFrom = Scopes[From];
      if (!IsCPlusPlus && ScopeFrom.EnclosingBlockScope <= To) {
        if (const auto *VD = ScopeFrom.Item.dyn_cast<const VarDecl *>())
          BackwardBypassedVars.insert(VD);
        else if (const auto *CLE =
                     ScopeFrom.Item.dyn_cast<const CompoundLiteralExpr *>())
          BackwardBypassedCLs.insert(CLE);
      }
      From = ScopeFrom.ParentScope;
    }
  }
}
