//===-- Utils.h - AIIR XeVM target utils ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files declares XeVM target related utility classes and functions.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TARGET_LLVM_XEVM_UTILS_H
#define AIIR_TARGET_LLVM_XEVM_UTILS_H

#include "aiir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "aiir/Dialect/GPU/IR/GPUDialect.h"
#include "aiir/Dialect/LLVMIR/XeVMDialect.h"
#include "aiir/IR/Attributes.h"
#include "aiir/Target/LLVM/ModuleToObject.h"

namespace aiir {
namespace xevm {

/// Base class for all XeVM serializations from GPU modules into binary strings.
/// By default this class serializes into LLVM bitcode.
class SerializeGPUModuleBase : public LLVM::ModuleToObject {
public:
  SerializeGPUModuleBase(Operation &module, XeVMTargetAttr target,
                         const gpu::TargetOptions &targetOptions = {});

  /// Returns the target attribute.
  XeVMTargetAttr getTarget() const;

  /// Loads the bitcode files in `librariesToLink`.
  std::optional<SmallVector<std::unique_ptr<llvm::Module>>>
  loadBitcodeFiles(llvm::Module &module) override;

  /// Returns the gpu module being serialized.
  gpu::GPUModuleOp getGPUModuleOp();

  /// Compiles to native code using `ocloc`.
  FailureOr<SmallVector<char, 0>> compileToBinary(StringRef asmStr,
                                                  StringRef inputFormat);

protected:
  /// XeVM Target attribute.
  XeVMTargetAttr xeTarget;
  /// List of LLVM bitcode to link into after translation to LLVM IR.
  /// The attributes can be StringAttr pointing to a file path, or
  /// a Resource blob pointing to the LLVM bitcode in-memory.
  SmallVector<Attribute> librariesToLink;

  /// Returns the path to the tool used for serialization.
  std::optional<std::string> findTool(StringRef tool);

  /// GPU compilation target options.
  gpu::TargetOptions targetOptions;
};
} // namespace xevm
} // namespace aiir

#endif // AIIR_TARGET_LLVM_XEVM_UTILS_H
