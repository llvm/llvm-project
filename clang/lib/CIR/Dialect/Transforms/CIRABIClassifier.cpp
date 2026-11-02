//===- CIRABIClassifier.cpp - ABI classification for CIR ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the classification shared by the targets that classify
// through the LLVM ABI Lowering Library, and the two test drivers that stand
// in for a target.
//
//===----------------------------------------------------------------------===//

#include "CIRABIClassifier.h"
#include "TargetLowering/CIRABIRewriteContext.h"

#include "mlir/ABI/Targets/Test/TestTarget.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

using namespace mlir;
using namespace mlir::abi;
using namespace cir;

/// FIXME: Only the AMDGPU kernel convention is mapped. Every other convention
/// is classified as C.
llvm::CallingConv::ID cir::convertCallingConv(cir::CallingConv cc) {
  if (cc == cir::CallingConv::AMDGPUKernel)
    return llvm::CallingConv::AMDGPU_KERNEL;
  return llvm::CallingConv::C;
}

std::optional<FunctionClassification>
CIRABIClassifierCommon::classifyVariadicCall(cir::CIRCallOpInterface call,
                                             cir::FuncOp callee) {
  return classifyVariadicCall(call, callee.getFunctionType(),
                              getTargetInfo(callee),
                              convertCallingConv(callee.getCallingConv()));
}

std::optional<FunctionClassification>
CIRABIClassifierCommon::classifyIndirectCall(cir::CIRCallOpInterface call) {
  cir::FuncType funcTy = getIndirectCalleeType(call);
  const llvm::abi::TargetInfo &targetInfo =
      getTargetInfo(call->getParentOfType<cir::FuncOp>());
  // An indirect callee is an ordinary function pointer, so it uses the C
  // convention.
  if (call.getNumArgOperands() > funcTy.getNumInputs())
    return classifyVariadicCall(call, funcTy, targetInfo, llvm::CallingConv::C);
  return classifySignature(
      funcTy.getReturnType(), funcTy.getInputs(), requiredArgs(funcTy),
      targetInfo, llvm::CallingConv::C, [&]() { return call->emitOpError(); });
}

namespace {

/// Classifies with the MLIR test ABI target: predictable rules that
/// approximate x86_64 SysV, used by tests that need a target-independent
/// driver.
class TestTargetClassifier final : public CIRABIClassifier {
public:
  explicit TestTargetClassifier(const mlir::DataLayout &dl) : dl(dl) {}

  std::optional<FunctionClassification>
  classifyFunction(cir::FuncOp func) final {
    return cir::classifyFunction(func, dl, cir::CallConvTarget::Test,
                                 /*classificationAttrName=*/"");
  }

  /// The test target classifies only declared parameters.
  std::optional<FunctionClassification>
  classifyVariadicCall(cir::CIRCallOpInterface call, cir::FuncOp) final {
    call->emitOpError()
        << "variadic arguments not yet implemented in CallConvLowering";
    return std::nullopt;
  }

  std::optional<FunctionClassification>
  classifyIndirectCall(cir::CIRCallOpInterface call) final {
    cir::FuncType funcTy = getIndirectCalleeType(call);
    if (call.getNumArgOperands() > funcTy.getNumInputs()) {
      call->emitOpError()
          << "variadic arguments not yet implemented in CallConvLowering";
      return std::nullopt;
    }
    return withReturnVoidness(mlir::abi::test::classify(funcTy.getInputs(),
                                                        funcTy.getReturnType(),
                                                        dl),
                              funcTy.getReturnType());
  }

  std::optional<ArgClassification> classifyVAArg(cir::VAArgOp vaArg) final {
    vaArg->emitOpError()
        << "not yet implemented in CallConvLowering for the test target";
    return std::nullopt;
  }

private:
  const mlir::DataLayout &dl;
};

/// Reads a pre-built classification from a function attribute, so a test can
/// inject any classification (including shapes no target produces) without
/// depending on a real ABI target.
class InjectedClassifier final : public CIRABIClassifier {
public:
  InjectedClassifier(const mlir::DataLayout &dl, llvm::StringRef attrName)
      : dl(dl), attrName(attrName) {}

  std::optional<FunctionClassification>
  classifyFunction(cir::FuncOp func) final {
    return cir::classifyFunction(func, dl, cir::CallConvTarget::None, attrName);
  }

  /// The injected classification covers a function's declared parameters, so
  /// it cannot describe an operand passed through an ellipsis.
  std::optional<FunctionClassification>
  classifyVariadicCall(cir::CIRCallOpInterface call, cir::FuncOp) final {
    call->emitOpError()
        << "variadic arguments not yet implemented in CallConvLowering";
    return std::nullopt;
  }

  /// Nor can it describe a callee resolved at run time, which names no
  /// function to read the attribute from.
  std::optional<FunctionClassification>
  classifyIndirectCall(cir::CIRCallOpInterface call) final {
    call->emitOpError() << "indirect call cannot be classified in the "
                           "'classification-attr' driver mode";
    return std::nullopt;
  }

  std::optional<ArgClassification> classifyVAArg(cir::VAArgOp vaArg) final {
    vaArg->emitOpError() << "cannot be classified in the "
                            "'classification-attr' driver mode";
    return std::nullopt;
  }

private:
  const mlir::DataLayout &dl;
  std::string attrName;
};

} // namespace

std::unique_ptr<CIRABIClassifier>
cir::createTestTargetABIClassifier(const mlir::DataLayout &dl) {
  return std::make_unique<TestTargetClassifier>(dl);
}

std::unique_ptr<CIRABIClassifier>
cir::createInjectedABIClassifier(const mlir::DataLayout &dl,
                                 llvm::StringRef attrName) {
  return std::make_unique<InjectedClassifier>(dl, attrName);
}
