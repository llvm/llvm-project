//===- HLSLBufferLayoutBuilder.h ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DerivedTypes.h"

namespace clang {
class RecordType;
class FieldDecl;

namespace CodeGen {
class CodeGenModule;

//===----------------------------------------------------------------------===//
// Implementation of constant buffer layout common between DirectX and
// SPIR/SPIR-V.
//===----------------------------------------------------------------------===//

class HLSLBufferLayoutBuilder {
private:
  CodeGenModule &CGM;

public:
  HLSLBufferLayoutBuilder(CodeGenModule &CGM) : CGM(CGM) {}

  // Returns an explicitly padded type for the given structure type and layout
  // data.
  llvm::StructType *
  createLayoutType(const RecordType *StructType,
                   const llvm::SmallVector<int32_t> *Packoffsets = nullptr);

private:
  bool layoutField(const clang::FieldDecl *FD, unsigned &EndOffset,
                   unsigned &FieldOffset, llvm::Type *&FieldType,
                   int Packoffset = -1);
};

} // namespace CodeGen
} // namespace clang
