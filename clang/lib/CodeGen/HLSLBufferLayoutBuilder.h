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

  /// Lays out a struct type following HLSL buffer rules and considering
  /// PackOffsets, if provided. Previously created layout structs are cached by
  /// CGHLSLRuntime.
  ///
  /// The function iterates over all fields of the record type (including base
  /// classes) and calls layoutField to converts each field to its corresponding
  /// LLVM type and to calculate its HLSL constant buffer layout. Any embedded
  /// structs (or arrays of structs) are converted to layout types as well.
  ///
  /// When PackOffsets are specified the elements will be placed based on the
  /// user-specified offsets. Not all elements must have a
  /// packoffset/register(c#) annotation though. For those that don't, the
  /// PackOffsets array will contain -1 value instead. These elements must be
  /// placed at the end of the layout after all of the elements with specific
  /// offset.
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
