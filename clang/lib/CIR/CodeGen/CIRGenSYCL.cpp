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

#include "clang/AST/StmtSYCL.h"

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
