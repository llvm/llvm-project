//===--- CGRecordLayout.h - LLVM Record Layout Information ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/CodeGen/CGRecordLayout.h. The queries
// are adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRRECORDLAYOUT_H
#define LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRRECORDLAYOUT_H

namespace mlir {
namespace cir {

class CIRLowerContext;

/// This class contains layout information for one RecordDecl, which is a
/// struct/union/class.  The decl represented must be a definition, not a
/// forward declaration. This class is also used to contain layout information
/// for one ObjCInterfaceDecl.
class CIRRecordLayout {
  CIRRecordLayout();
};

} // namespace cir
} // namespace mlir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRRECORDLAYOUT_H
