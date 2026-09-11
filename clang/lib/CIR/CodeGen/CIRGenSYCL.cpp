//===--------- CIRGenSYCL.cpp - Emit CIR for SYCL kernels -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code required for the generation of SYCL kernel code.
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"
#include "CIRGenModule.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/SYCLKernelInfo.h"
#include "clang/AST/StmtSYCL.h"

#include "llvm/Support/SaveAndRestore.h"

using namespace clang;
using namespace clang::CIRGen;

mlir::LogicalResult
CIRGenFunction::emitSYCLKernelCallStmt(const SYCLKernelCallStmt &s) {
  // SYCLKernelCallStmt nodes are only present in the bodies of functions
  // declared with the sycl_kernel_entry_point attribute. ODR-use of such a
  // function in code emitted during device compilation should be diagnosed.
  // During device compilation, the offload kernel entry point is emitted in
  // place of such a function (see CIRGenModule::emitDeferred), so this
  // function is only reached during host compilation.
  assert(!getLangOpts().SYCLIsDevice &&
         "Attempt to emit a SYCL kernel call statement during device "
         "compilation");

  // During host compilation, the kernel launch statement is emitted in place
  // of the original function body.
  return emitStmt(s.getKernelLaunchStmt(), /*useCurrentScope=*/true);
}

// Emit the body of a SYCL kernel caller offload entry point. Mirrors the tail
// of generateCode, but is driven by an OutlinedFunctionDecl and an explicit
// argument list rather than a FunctionDecl.
void CIRGenFunction::emitSYCLKernelCaller(
    const OutlinedFunctionDecl *outlinedFnDecl, cir::FuncOp funcOp,
    cir::FuncType funcType, FunctionArgList &args) {
  const Stmt *body = outlinedFnDecl->getBody();
  SourceLocation loc = outlinedFnDecl->getLocation();
  SourceRange bodyRange = body->getSourceRange();

  // Synthesized entry point: no FunctionDecl, emitted with an empty GlobalDecl.
  curGD = GlobalDecl();

  SourceLocRAIIObject fnLoc{*this, loc.isValid() ? getLoc(loc)
                                                 : builder.getUnknownLoc()};

  mlir::Location fusedLoc = getLoc(bodyRange);
  mlir::Block *entryBB = funcOp.addEntryBlock();

  SymTableScopeTy varScope(symbolTable);
  {
    LexicalScope lexScope(*this, fusedLoc, entryBB);
    startFunction(GlobalDecl(), getContext().VoidTy, funcOp, funcType, args,
                  loc, bodyRange.getBegin());
    if (mlir::failed(emitFunctionBody(body)))
      return;
    if (mlir::failed(funcOp.verifyBody()))
      return;
    finishFunction(body->getEndLoc());
  }

  eraseEmptyAndUnusedBlocks(funcOp);
}

void CIRGenModule::emitSYCLKernelCaller(const FunctionDecl *kernelEntryPointFn,
                                        ASTContext &ctx) {
  assert(ctx.getLangOpts().SYCLIsDevice &&
         "SYCL kernel caller offload entry point functions can only be emitted"
         " during device compilation");

  const auto *kernelEntryPointAttr =
      kernelEntryPointFn->getAttr<SYCLKernelEntryPointAttr>();
  assert(kernelEntryPointAttr && "Missing sycl_kernel_entry_point attribute");
  assert(!kernelEntryPointAttr->isInvalidAttr() &&
         "sycl_kernel_entry_point attribute is invalid");

  // Find the SYCLKernelCallStmt.
  SYCLKernelCallStmt *kernelCallStmt =
      cast<SYCLKernelCallStmt>(kernelEntryPointFn->getBody());

  // Retrieve the SYCL kernel caller parameters from the OutlinedFunctionDecl.
  FunctionArgList args;
  const OutlinedFunctionDecl *outlinedFnDecl =
      kernelCallStmt->getOutlinedFunctionDecl();
  args.append(outlinedFnDecl->param_begin(), outlinedFnDecl->param_end());

  // Compute the function info and CIR function type.
  const CIRGenFunctionInfo &fnInfo =
      getTypes().arrangeDeviceKernelCallerDeclaration(ctx.VoidTy, args);
  cir::FuncType funcType = getTypes().getFunctionType(fnInfo);

  // Retrieve the generated name for the SYCL kernel caller function.
  CanQualType kernelNameType =
      ctx.getCanonicalType(kernelEntryPointAttr->getKernelName());
  const SYCLKernelInfo &kernelInfo = ctx.getSYCLKernelInfo(kernelNameType);

  // Synthesized from the OutlinedFunctionDecl, not a FunctionDecl, so create
  // the function directly with a null FunctionDecl (mirrors classic CodeGen's
  // llvm::Function::Create).
  cir::FuncOp funcOp = createCIRFunction(
      getLoc(kernelEntryPointFn->getSourceRange()), kernelInfo.GetKernelName(),
      funcType, /*funcDecl=*/nullptr);
  funcOp.setLinkage(cir::GlobalLinkageKind::ExternalLinkage);

  // Emit as a device kernel (e.g. spir_kernel). Classic CodeGen derives this
  // from CC_DeviceKernel via SetLLVMFunctionAttributes; CIR does not yet route
  // opFuncCallingConv onto the FuncOp, so set it from the target hook.
  funcOp.setCallingConv(getTargetCIRGenInfo().getDeviceKernelCallingConv());

  // Route through the shared attribute path so generic function attributes
  // (e.g. convergent) are applied, matching classic CodeGen's
  // SetLLVMFunctionAttributes. There is no FunctionDecl, so pass an empty
  // GlobalDecl.
  setCIRFunctionAttributes(GlobalDecl(), fnInfo, funcOp, /*isThunk=*/false);

  // TODO: attributes applied by classic CodeGen not yet handled in CIR:
  // SetSYCLKernelAttributes (norecurse, mustprogress), addSYCLModuleIdAttr.
  assert(!cir::MissingFeatures::setLLVMFunctionFEnvAttributes());

  // Emit the SYCL kernel caller function.
  CIRGenFunction cgf(*this, builder);
  llvm::SaveAndRestore<CIRGenFunction *> savedCGF(curCGF, &cgf);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    cgf.emitSYCLKernelCaller(outlinedFnDecl, funcOp, funcType, args);
  }

  setDSOLocal(static_cast<mlir::Operation *>(funcOp));

  setNonAliasAttributes(GlobalDecl(), funcOp);
  // CIR's setter takes a FunctionDecl; nullptr skips OutlinedFunctionDecl-
  // derived attributes (e.g. inline hints), not yet handled.
  assert(!cir::MissingFeatures::opFuncExtraAttrs());
  setCIRFunctionAttributesForDefinition(/*fd=*/nullptr, funcOp);
}
