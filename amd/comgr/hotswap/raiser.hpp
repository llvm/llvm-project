#ifndef HOTSWAP_TRANSPILER_RAISER_HPP
#define HOTSWAP_TRANSPILER_RAISER_HPP

#include "code_object_utils.hpp"
#include "raise_failure.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>
#include <vector>

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace transpiler {

struct RaiseResult {
  std::unique_ptr<llvm::LLVMContext> ctx;
  std::unique_ptr<llvm::Module> module;
  int liftedCount = 0;
  int totalCount = 0;
  std::string irText;
  std::string disasmText;
  // Structured failure description. `failure.reason == None` iff `success`.
  RaiseFailure failure;
  bool usesScratchPrivateSegment = false;
  uint32_t sourcePrivateSegmentFixedSize = 0;
  bool success = false;
  bool hasDivergentExec = false;
};

RaiseResult raiseToIR(llvm::ArrayRef<uint8_t> textBytes,
                      llvm::StringRef sourceISA,
                      llvm::StringRef kernelName,
                      const KernelMeta &meta,
                      uint64_t kernelOffset = 0,
                      llvm::StringRef compilationTargetISA = "");

} // namespace transpiler

#endif
