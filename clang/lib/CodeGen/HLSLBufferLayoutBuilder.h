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
  llvm::StringRef LayoutTypeName;

public:
  HLSLBufferLayoutBuilder(CodeGenModule &CGM, llvm::StringRef LayoutTypeName)
      : CGM(CGM), LayoutTypeName(LayoutTypeName) {}

  // Returns LLVM target extension type with the name LayoutTypeName
  // for given structure type and layout data. The first number in
  // the Layout is the size followed by offsets for each struct element.
  llvm::TargetExtType *
  createLayoutType(const RecordType *StructType,
                   const llvm::SmallVector<unsigned> *Packoffsets = nullptr);

private:
  bool layoutField(const clang::FieldDecl *FD, unsigned &EndOffset,
                   llvm::SmallVector<unsigned> &Layout,
                   llvm::SmallVector<llvm::Type *> &LayoutElements,
                   int Packoffset);
};

} // namespace CodeGen
} // namespace clang
