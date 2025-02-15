//===- HLSLTargetInfo.h ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABIInfoImpl.h"
#include "TargetInfo.h"

using namespace clang;
using namespace clang::CodeGen;

//===----------------------------------------------------------------------===//
// Target codegen info implementation common between DirectX and SPIR/SPIR-V.
//===----------------------------------------------------------------------===//

class CommonHLSLTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  CommonHLSLTargetCodeGenInfo(std::unique_ptr<ABIInfo> Info)
      : TargetCodeGenInfo(std::move(Info)) {}

  // Returns LLVM target extension type "dx.Layout" or "spv.Layout"
  // for given structure type and layout data. The first number in
  // the Layout is the size followed by offsets for each struct element.
  virtual llvm::Type *getHLSLLayoutType(CodeGenModule &CGM,
                                        llvm::StructType *LayoutStructTy,
                                        SmallVector<unsigned> Layout) const {
    return nullptr;
  };

protected:
  // Creates a layout type for given struct with HLSL constant buffer layout
  // taking into account Packoffsets, if provided.
  virtual llvm::Type *createHLSLBufferLayoutType(
      CodeGenModule &CGM, const clang::RecordType *StructType,
      const SmallVector<unsigned> *Packoffsets = nullptr) const;
};
