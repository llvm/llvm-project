//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Decl nodes as CIR code.
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"
#include "CIRGenOpenACCHelpers.h"

#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "clang/AST/DeclOpenACC.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace clang;
using namespace clang::CIRGen;

namespace {
struct OpenACCDeclareCleanup final : EHScopeStack::Cleanup {
  mlir::acc::DeclareEnterOp enterOp;

  OpenACCDeclareCleanup(mlir::acc::DeclareEnterOp enterOp) : enterOp(enterOp) {}

  template <typename OutTy, typename InTy>
  void createOutOp(CIRGenFunction &cgf, InTy inOp) {
    if constexpr (std::is_same_v<OutTy, mlir::acc::DeleteOp>) {
      auto outOp =
          OutTy::create(cgf.getBuilder(), inOp.getLoc(), inOp,
                        inOp.getStructured(), inOp.getImplicit(),
                        llvm::Twine(inOp.getNameAttr()), inOp.getBounds());
      outOp.setDataClause(inOp.getDataClause());
      outOp.setModifiers(inOp.getModifiers());
    } else {
      auto outOp =
          OutTy::create(cgf.getBuilder(), inOp.getLoc(), inOp, inOp.getVarPtr(),
                        inOp.getStructured(), inOp.getImplicit(),
                        llvm::Twine(inOp.getNameAttr()), inOp.getBounds());
      outOp.setDataClause(inOp.getDataClause());
      outOp.setModifiers(inOp.getModifiers());
    }
  }

  void emit(CIRGenFunction &cgf) override {
    auto exitOp = mlir::acc::DeclareExitOp::create(
        cgf.getBuilder(), enterOp.getLoc(), enterOp, {});

    // Some data clauses need to be referenced in 'exit', AND need to have an
    // operation after the exit.  Copy these from the enter operation.
    for (mlir::Value val : enterOp.getDataClauseOperands()) {
      if (auto copyin = val.getDefiningOp<mlir::acc::CopyinOp>()) {
        switch (copyin.getDataClause()) {
        default:
          llvm_unreachable(
              "OpenACC local declare clause copyin unexpected data clause");
          break;
        case mlir::acc::DataClause::acc_copy:
          createOutOp<mlir::acc::CopyoutOp>(cgf, copyin);
          break;
        case mlir::acc::DataClause::acc_copyin:
          createOutOp<mlir::acc::DeleteOp>(cgf, copyin);
          break;
        }
      } else if (auto create = val.getDefiningOp<mlir::acc::CreateOp>()) {
        switch (create.getDataClause()) {
        default:
          llvm_unreachable(
              "OpenACC local declare clause create unexpected data clause");
          break;
        case mlir::acc::DataClause::acc_copyout:
          createOutOp<mlir::acc::CopyoutOp>(cgf, create);
          break;
        case mlir::acc::DataClause::acc_create:
          createOutOp<mlir::acc::DeleteOp>(cgf, create);
          break;
        }
      } else if (auto present = val.getDefiningOp<mlir::acc::PresentOp>()) {
        createOutOp<mlir::acc::DeleteOp>(cgf, present);
      } else if (auto dev_res =
                     val.getDefiningOp<mlir::acc::DeclareDeviceResidentOp>()) {
        createOutOp<mlir::acc::DeleteOp>(cgf, dev_res);
      } else if (val.getDefiningOp<mlir::acc::DeclareLinkOp>()) {
        // Link has no exit clauses, and shouldn't be copied.
        continue;
      } else if (val.getDefiningOp<mlir::acc::DevicePtrOp>()) {
        // DevicePtr has no exit clauses, and shouldn't be copied.
        continue;
      } else {
        llvm_unreachable("OpenACC local declare clause unexpected defining op");
        continue;
      }
      exitOp.getDataClauseOperandsMutable().append(val);
    }
  }
};
} // namespace

void CIRGenModule::emitGlobalOpenACCDecl(const OpenACCConstructDecl *d) {
  if (const auto *rd = dyn_cast<OpenACCRoutineDecl>(d))
    emitGlobalOpenACCRoutineDecl(rd);
  else
    emitGlobalOpenACCDeclareDecl(cast<OpenACCDeclareDecl>(d));
}

void CIRGenFunction::emitOpenACCDeclare(const OpenACCDeclareDecl &d) {
  mlir::Location exprLoc = cgm.getLoc(d.getBeginLoc());
  auto enterOp = mlir::acc::DeclareEnterOp::create(
      builder, exprLoc, mlir::acc::DeclareTokenType::get(&cgm.getMLIRContext()),
      {});

  emitOpenACCClauses(enterOp, OpenACCDirectiveKind::Declare, d.clauses());

  ehStack.pushCleanup<OpenACCDeclareCleanup>(CleanupKind::NormalCleanup,
                                             enterOp);
}

// Helper function that gets the declaration referenced by the declare clause.
// This is a simplified verison of the work that `getOpenACCDataOperandInfo`
// does, as it only has to get forms that 'declare' does.
static const Decl *getDeclareReferencedDecl(const Expr *e) {
  const Expr *curVarExpr = e->IgnoreParenImpCasts();

  // Since we allow array sections, we have to unpack the array sections here.
  // We don't have to worry about other bounds, since only variable or array
  // name (plus array sections as an extension) are permitted.
  while (const auto *ase = dyn_cast<ArraySectionExpr>(curVarExpr))
    curVarExpr = ase->getBase()->IgnoreParenImpCasts();

  if (const auto *dre = dyn_cast<DeclRefExpr>(curVarExpr))
    return dre->getFoundDecl()->getCanonicalDecl();

  // MemberExpr is allowed when it is implicit 'this'.
  return cast<MemberExpr>(curVarExpr)->getMemberDecl()->getCanonicalDecl();
}

template <typename BeforeOpTy, typename DataClauseTy>
void CIRGenModule::emitGlobalOpenACCDeclareDataOperands(
    const Expr *varOperand, DataClauseTy dataClause,
    OpenACCModifierKind modifiers, bool structured, bool implicit,
    bool requiresDtor) {
  // This is a template argument so that we don't have to include all of
  // mlir::acc into CIRGenModule.
  static_assert(std::is_same_v<DataClauseTy, mlir::acc::DataClause>);
  mlir::Location exprLoc = getLoc(varOperand->getBeginLoc());
  const Decl *refedDecl = getDeclareReferencedDecl(varOperand);
  StringRef varName = getMangledName(GlobalDecl{cast<VarDecl>(refedDecl)});

  // We have to emit two separate functions in this case, an acc_ctor and an
  // acc_dtor. These two sections are/should remain reasonably equal, however
  // the order of the clauses/vs-enter&exit in them makes combining these two
  // sections not particularly attractive, so we have a bit of repetition.
  {
    mlir::OpBuilder::InsertionGuard guardCase(builder);
    auto ctorOp = mlir::acc::GlobalConstructorOp::create(
        builder, exprLoc, (varName + "_acc_ctor").str());
    getModule().push_back(ctorOp);
    mlir::Block *block = builder.createBlock(&ctorOp.getRegion(),
                                             ctorOp.getRegion().end(), {}, {});
    builder.setInsertionPointToEnd(block);
    // These things are close enough to a function handling-wise we can just
    // create this here.
    CIRGenFunction cgf{*this, builder, true};
    llvm::SaveAndRestore<CIRGenFunction *> savedCGF(curCGF, &cgf);
    cgf.curFn = ctorOp;
    CIRGenFunction::SourceLocRAIIObject fnLoc{cgf, exprLoc};

    // This gets the information we need, PLUS emits the bounds correctly, so we
    // have to do this in both enter and exit.
    CIRGenFunction::OpenACCDataOperandInfo inf =
        cgf.getOpenACCDataOperandInfo(varOperand);
    auto beforeOp =
        BeforeOpTy::create(builder, exprLoc, inf.varValue, structured, implicit,
                           inf.name, inf.bounds);
    beforeOp.setDataClause(dataClause);
    beforeOp.setModifiers(convertOpenACCModifiers(modifiers));

    mlir::acc::DeclareEnterOp::create(
        builder, exprLoc, mlir::acc::DeclareTokenType::get(&getMLIRContext()),
        beforeOp.getResult());

    mlir::acc::TerminatorOp::create(builder, exprLoc);
  }

  // copyin, create, and device_resident require a destructor, link does not. In
  // the case of the first three, they are all a 'getdeviceptr', followed by the
  // declare_exit, followed by a delete op in the destructor region.
  if (requiresDtor) {
    mlir::OpBuilder::InsertionGuard guardCase(builder);
    auto ctorOp = mlir::acc::GlobalDestructorOp::create(
        builder, exprLoc, (varName + "_acc_dtor").str());
    getModule().push_back(ctorOp);
    mlir::Block *block = builder.createBlock(&ctorOp.getRegion(),
                                             ctorOp.getRegion().end(), {}, {});
    builder.setInsertionPointToEnd(block);

    // These things are close enough to a function handling-wise we can just
    // create this here.
    CIRGenFunction cgf{*this, builder, true};
    llvm::SaveAndRestore<CIRGenFunction *> savedCGF(curCGF, &cgf);
    cgf.curFn = ctorOp;
    CIRGenFunction::SourceLocRAIIObject fnLoc{cgf, exprLoc};

    CIRGenFunction::OpenACCDataOperandInfo inf =
        cgf.getOpenACCDataOperandInfo(varOperand);
    auto getDevPtr = mlir::acc::GetDevicePtrOp::create(
        builder, exprLoc, inf.varValue, structured, implicit, inf.name,
        inf.bounds);
    getDevPtr.setDataClause(dataClause);
    getDevPtr.setModifiers(convertOpenACCModifiers(modifiers));

    mlir::acc::DeclareExitOp::create(builder, exprLoc, /*token=*/mlir::Value{},
                                     getDevPtr.getResult());
    auto deleteOp = mlir::acc::DeleteOp::create(
        builder, exprLoc, getDevPtr, structured, implicit, inf.name, {});
    deleteOp.setDataClause(dataClause);
    deleteOp.setModifiers(convertOpenACCModifiers(modifiers));
    mlir::acc::TerminatorOp::create(builder, exprLoc);
  }
}
namespace {
// This class emits all of the information for a 'declare' at a global/ns/class
// scope. Each clause results in its own acc_ctor and acc_dtor for the variable.
// This class creates those and emits them properly.
// This behavior is unique/special enough from the emission of statement-level
// clauses that it doesn't really make sense to use that clause visitor.
class OpenACCGlobalDeclareClauseEmitter final
    : public OpenACCClauseVisitor<OpenACCGlobalDeclareClauseEmitter> {
  CIRGenModule &cgm;

public:
  OpenACCGlobalDeclareClauseEmitter(CIRGenModule &cgm) : cgm(cgm) {}

  void VisitClause(const OpenACCClause &clause) {
    llvm_unreachable("Invalid OpenACC clause on global Declare");
  }

  void emitClauses(ArrayRef<const OpenACCClause *> clauses) {
    this->VisitClauseList(clauses);
  }

  void VisitCopyInClause(const OpenACCCopyInClause &clause) {
    for (const Expr *var : clause.getVarList())
      cgm.emitGlobalOpenACCDeclareDataOperands<mlir::acc::CopyinOp>(
          var, mlir::acc::DataClause::acc_copyin, clause.getModifierList(),
          /*structured=*/true,
          /*implicit=*/false, /*requiresDtor=*/true);
  }

  void VisitCreateClause(const OpenACCCreateClause &clause) {
    for (const Expr *var : clause.getVarList())
      cgm.emitGlobalOpenACCDeclareDataOperands<mlir::acc::CreateOp>(
          var, mlir::acc::DataClause::acc_create, clause.getModifierList(),
          /*structured=*/true,
          /*implicit=*/false, /*requiresDtor=*/true);
  }

  void VisitDeviceResidentClause(const OpenACCDeviceResidentClause &clause) {
    for (const Expr *var : clause.getVarList())
      cgm.emitGlobalOpenACCDeclareDataOperands<
          mlir::acc::DeclareDeviceResidentOp>(
          var, mlir::acc::DataClause::acc_declare_device_resident, {},
          /*structured=*/true,
          /*implicit=*/false, /*requiresDtor=*/true);
  }

  void VisitLinkClause(const OpenACCLinkClause &clause) {
    for (const Expr *var : clause.getVarList())
      cgm.emitGlobalOpenACCDeclareDataOperands<mlir::acc::DeclareLinkOp>(
          var, mlir::acc::DataClause::acc_declare_link, {},
          /*structured=*/true,
          /*implicit=*/false, /*requiresDtor=*/false);
  }
};
} // namespace

void CIRGenModule::emitGlobalOpenACCDeclareDecl(const OpenACCDeclareDecl *d) {
  // Declare creates 1 'acc_ctor' and 0-1 'acc_dtor' per clause, since it needs
  // a unique one on a per-variable basis. We can just use a clause emitter to
  // do all the work.
  mlir::OpBuilder::InsertionGuard guardCase(builder);
  OpenACCGlobalDeclareClauseEmitter em{*this};
  em.emitClauses(d->clauses());
}

void CIRGenFunction::emitOpenACCRoutine(const OpenACCRoutineDecl &d) {
  getCIRGenModule().errorNYI(d.getSourceRange(), "OpenACC Routine Construct");
}

void CIRGenModule::emitGlobalOpenACCRoutineDecl(const OpenACCRoutineDecl *d) {
  errorNYI(d->getSourceRange(), "OpenACC Global Routine Construct");
}
