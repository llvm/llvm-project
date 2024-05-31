//===----- SemaOpenCL.h --- Semantic Analysis for OpenCL constructs -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares semantic analysis routines for OpenCL.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMAOPENCL_H
#define LLVM_CLANG_SEMA_SEMAOPENCL_H

#include "clang/Sema/SemaBase.h"

namespace clang {
class Decl;
class ParsedAttr;

class SemaOpenCL : public SemaBase {
public:
  SemaOpenCL(Sema &S);

  void handleNoSVMAttr(Decl *D, const ParsedAttr &AL);
  void handleAccessAttr(Decl *D, const ParsedAttr &AL);

  // Handles intel_reqd_sub_group_size.
  void handleSubGroupSize(Decl *D, const ParsedAttr &AL);
};

} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMAOPENCL_H
