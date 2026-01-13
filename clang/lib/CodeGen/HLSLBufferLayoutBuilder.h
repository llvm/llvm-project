//===- HLSLBufferLayoutBuilder.h ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/TypeBase.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DerivedTypes.h"

namespace clang {
namespace CodeGen {
class CGHLSLOffsetInfo;
class CodeGenModule;
class CGHLSLOffsetInfo;

//===----------------------------------------------------------------------===//
// Implementation of constant buffer layout common between DirectX and
// SPIR/SPIR-V.
//===----------------------------------------------------------------------===//

class HLSLBufferLayoutBuilder {
private:
  CodeGenModule &CGM;

public:
  HLSLBufferLayoutBuilder(CodeGenModule &CGM) : CGM(CGM) {}

  /// Lays out a struct type following HLSL buffer rules and considering any
  /// explicit offset information. Previously created layout structs are cached
  /// by CGHLSLRuntime.
  ///
  /// The function iterates over all fields of the record type (including base
  /// classes) and works out a padded llvm type to represent the buffer layout.
  ///
  /// If a non-empty OffsetInfo is provided (ie, from `packoffset` annotations
  /// in the source), any provided offsets offsets will be respected. If the
  /// OffsetInfo is available but has empty entries, those will be layed out at
  /// the end of the structure.
  llvm::StructType *layOutStruct(const RecordType *StructType,
                                 const CGHLSLOffsetInfo &OffsetInfo);

  /// Lays out an array type following HLSL buffer rules.
  llvm::Type *layOutArray(const ConstantArrayType *AT);

  /// Lays out a type following HLSL buffer rules. Arrays and structures will be
  /// padded appropriately and nested objects will be converted as appropriate.
  llvm::Type *layOutType(QualType Type);
};

} // namespace CodeGen
} // namespace clang
