//===---- UnimplementedFeatureGuarding.h - Checks against NYI ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file introduces some helper classes to guard against features that
// CodeGen supports that we do not have and also do not have great ways to
// assert against.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_UFG
#define LLVM_CLANG_LIB_CIR_UFG

namespace cir {
struct UnimplementedFeature {
  // TODO(CIR): Implement the CIRGenFunction::buildTypeCheck method that handles
  // sanitizer related type check features
  static bool buildTypeCheck() { return false; }
  static bool tbaa() { return false; }
  static bool cleanups() { return false; }
  // This is for whether or not we've implemented a cir::VectorType
  // corresponding to `llvm::VectorType`
  static bool cirVectorType() { return false; }

  // CIR still unware of address space
  static bool addressSpaceInGlobalVar() { return false; }

  // Unhandled global/linkage information.
  static bool unnamedAddr() { return false; }
  static bool setComdat() { return false; }
  static bool setDSOLocal() { return false; }
  static bool threadLocal() { return false; }
  static bool setDLLStorageClass() { return false; }

  // Sanitizers
  static bool reportGlobalToASan() { return false; }
  static bool emitCheckedInBoundsGEP() { return false; }

  // ObjC
  static bool setObjCGCLValueClass() { return false; }

  // Debug info
  static bool generateDebugInfo() { return false; }
};
} // namespace cir

#endif
