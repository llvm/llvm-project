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
#include "clang/AST/StmtSYCL.h"
#include "clang/AST/SYCLKernelInfo.h"

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

// Emit the body of a SYCL kernel caller offload entry point function. The body
// is the transformed body held by the OutlinedFunctionDecl associated with the
// sycl_kernel_entry_point attributed function. This mirrors the tail of
// CIRGenFunction::generateCode, but is driven by an OutlinedFunctionDecl and an
// explicit argument list rather than a FunctionDecl.
void CIRGenFunction::emitSYCLKernelCaller(
    const OutlinedFunctionDecl *outlinedFnDecl, cir::FuncOp funcOp,
    cir::FuncType funcType, FunctionArgList &args) {
  const Stmt *body = outlinedFnDecl->getBody();
  SourceLocation loc = outlinedFnDecl->getLocation();
  SourceRange bodyRange = body->getSourceRange();

  // The offload entry point is synthesized and has no FunctionDecl of its own.
  // As in classic CodeGen's EmitSYCLKernelCaller, it is emitted with an empty
  // GlobalDecl: it is a free function (never an implicit-object member) and
  // must not run a C++ instance-function prologue. LexicalScope's implicit
  // return handles a null curGD (the entry point returns void).
  curGD = GlobalDecl();

  // Establish a source location for the function so that the prologue can
  // inherit one (see CIRGenFunction::getLoc / currSrcLoc).
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

  // Mirror the tail of generateCode: drop leftover empty/unreachable blocks the
  // lexical-scope machinery may have created.
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
      getTypes().arrangeSYCLKernelCallerDeclaration(ctx.VoidTy, args);
  cir::FuncType funcType = getTypes().getFunctionType(fnInfo);

  // Retrieve the generated name for the SYCL kernel caller function.
  CanQualType kernelNameType =
      ctx.getCanonicalType(kernelEntryPointAttr->getKernelName());
  const SYCLKernelInfo &kernelInfo = ctx.getSYCLKernelInfo(kernelNameType);

  // Create the SYCL kernel caller function. Unlike an ordinary function, this
  // offload entry point is synthesized from the OutlinedFunctionDecl held by
  // the SYCLKernelCallStmt rather than emitted from a FunctionDecl, so it is
  // constructed with an empty GlobalDecl.
  cir::FuncOp funcOp = getOrCreateCIRFunction(
      kernelInfo.GetKernelName(), funcType, GlobalDecl(), /*forVTable=*/false,
      /*dontDefer=*/true, /*isThunk=*/false, ForDefinition);

  // The kernel caller offload entry point has external linkage. Classic
  // CodeGen creates it with ExternalLinkage explicitly (EmitSYCLKernelCaller);
  // createCIRFunction already applies ExternalLinkage by default, so set it
  // explicitly here to make the contract clear rather than rely on the default.
  funcOp.setLinkage(cir::GlobalLinkageKind::ExternalLinkage);

  // Set the device kernel calling convention so the entry point is emitted as
  // a kernel (e.g. spir_kernel) rather than an ordinary device function.
  // Classic CodeGen derives this from the CC_DeviceKernel function info via
  // SetLLVMFunctionAttributes; CIR does not yet route the function-info calling
  // convention onto the FuncOp (opFuncCallingConv), so set it directly from the
  // target hook, matching how CIRGen sets kernel calling conventions elsewhere.
  funcOp.setCallingConv(getTargetCIRGenInfo().getDeviceKernelCallingConv());

  // TODO: The following attributes applied by classic CodeGen's
  // EmitSYCLKernelCaller are not yet applied in CIR:
  //  - SetSYCLKernelAttributes: norecurse and mustprogress.
  //  - addSYCLModuleIdAttr: the "sycl-module-id" attribute.
  //  - setDSOLocal.
  assert(!cir::MissingFeatures::setLLVMFunctionFEnvAttributes());
  assert(!cir::MissingFeatures::setDSOLocal());

  // Emit the SYCL kernel caller function.
  CIRGenFunction cgf(*this, builder);
  curCGF = &cgf;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    cgf.emitSYCLKernelCaller(outlinedFnDecl, funcOp, funcType, args);
  }
  curCGF = nullptr;

  setNonAliasAttributes(GlobalDecl(), funcOp);
  // The SYCL kernel caller is synthesized from an OutlinedFunctionDecl rather
  // than a FunctionDecl. Classic CodeGen passes the OutlinedFunctionDecl to
  // SetLLVMFunctionAttributesForDefinition, but CIR's setter takes a
  // FunctionDecl; passing nullptr here skips OutlinedFunctionDecl-derived
  // attributes (e.g. inline hints), which are not yet handled.
  assert(!cir::MissingFeatures::opFuncExtraAttrs());
  setCIRFunctionAttributesForDefinition(/*fd=*/nullptr, funcOp);
}
