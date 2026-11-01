//===- CIRABIClassifier.h - ABI classification for CIR ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines CIRABIClassifier, the interface CallConvLowering uses to
// obtain a mlir::abi::FunctionClassification for a CIR signature.
//
// CIRABIClassifier is the producer half of CIR's calling-convention lowering.
// The consumer half is CIRABIRewriteContext, which turns a classification into
// IR and needs to know nothing about where it came from.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_CIR_DIALECT_TRANSFORMS_CIRABICLASSIFIER_H
#define CLANG_LIB_CIR_DIALECT_TRANSFORMS_CIRABICLASSIFIER_H

#include "mlir/ABI/ABIRewriteContext.h"
#include "mlir/ABI/ABITypeMapper.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/ABI/FunctionInfo.h"
#include "llvm/ABI/TargetInfo.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/CallingConv.h"

#include <memory>
#include <optional>

namespace cir {

/// The target-specific settings a classifier needs beyond what it can read
/// from the module.  Every field names the target it belongs to, because a
/// classifier for one target ignores the others' settings.
struct ABIClassifierOptions {
  // TODO: Make this class target-independent.

  /// The AVX level the x86_64 classifier sizes a native vector at.
  llvm::abi::X86AVXABILevel x86AvxAbiLevel = llvm::abi::X86AVXABILevel::None;

  /// Whether a function's own recorded target features may raise its x86_64
  /// AVX level above x86AvxAbiLevel.
  bool allowsX86TargetAttrAvx = true;

  /// The x86_64 flags whose value depends on the target and the requested ABI
  /// compatibility version.
  llvm::abi::X86ABICompatInfo x86AbiCompat;
};

/// Produces the classification CallConvLowering rewrites a signature from.
///
/// A subclass supplies one target's rules, or one of the two test drivers that
/// stand in for a target.  The pass holds a single instance for the whole
/// module.  Each method returns std::nullopt after emitting an error on the
/// operation it was given, including for an operation the subclass does not
/// support yet.
class CIRABIClassifier {
public:
  virtual ~CIRABIClassifier() = default;

  /// Classify \p func's declared signature.
  virtual std::optional<mlir::abi::FunctionClassification>
  classifyFunction(cir::FuncOp func) = 0;

  /// Classify a direct call to the variadic function \p callee that passes
  /// arguments through the ellipsis.  The classification covers every operand
  /// of \p call, and classifies the declared parameters the same way as
  /// classifyFunction(callee).
  virtual std::optional<mlir::abi::FunctionClassification>
  classifyVariadicCall(cir::CIRCallOpInterface call, cir::FuncOp callee) = 0;

  /// Classify \p call, whose callee is resolved at run time, from the
  /// signature its callee pointer points to.
  virtual std::optional<mlir::abi::FunctionClassification>
  classifyIndirectCall(cir::CIRCallOpInterface call) = 0;

  /// Classify the type fetched by \p vaArg as an unnamed argument with a full
  /// register budget.
  virtual std::optional<mlir::abi::ArgClassification>
  classifyVAArg(cir::VAArgOp vaArg) = 0;
};

/// Classifies through the LLVM ABI Lowering Library.  The signature's CIR types
/// are mapped to llvm::abi types, the target's TargetInfo classifies them, and
/// each llvm::abi::ArgInfo is converted to an ArgClassification.
///
/// A subclass supplies the TargetInfo and classifies `cir.va_arg`, whose
/// expansion is specific to each target.
class CIRABIClassifierCommon : public CIRABIClassifier {
public:
  std::optional<mlir::abi::FunctionClassification>
  classifyFunction(cir::FuncOp func) override;

  /// The call is classified at the callee's features, not the caller's,
  /// because the pass rewrites the definition and its call sites from one
  /// classification.  Classic instead arranges every call site from the
  /// caller and reports a caller whose features disagree with its callee in
  /// checkFunctionCallABI, which has no equivalent here yet.
  std::optional<mlir::abi::FunctionClassification>
  classifyVariadicCall(cir::CIRCallOpInterface call,
                       cir::FuncOp callee) override;

  /// A callee resolved at run time carries no features of its own, so the
  /// call is classified at the features of the function containing it, which
  /// is the declaration classic arranges every call site from.  A call that
  /// passes arguments through an ellipsis is classified from its own
  /// operands, as a direct one is.
  std::optional<mlir::abi::FunctionClassification>
  classifyIndirectCall(cir::CIRCallOpInterface call) override;

protected:
  CIRABIClassifierCommon(const mlir::DataLayout &dl, mlir::ModuleOp modOp)
      : dl(dl), modOp(modOp), typeMapper(dl) {}

  /// The library classifier for a function whose recorded target features
  /// may refine the ABI rules.
  virtual const llvm::abi::TargetInfo &
  getTargetInfo(cir::FuncOp featureSource) = 0;

  /// Classify \p ty, the type fetched by a `cir.va_arg`, as an unnamed
  /// argument with every argument register free.
  std::optional<mlir::abi::ArgClassification>
  classifyVarArgType(mlir::Type ty, const llvm::abi::TargetInfo &targetInfo,
                     llvm::function_ref<mlir::InFlightDiagnostic()> emitError);

  const mlir::DataLayout &dl;
  mlir::ModuleOp modOp;
  mlir::abi::ABITypeMapper typeMapper;

private:
  /// Classify a signature returning \p retCIR and taking \p inputs, whose
  /// first \p required entries are declared parameters, under \p callConv.
  std::optional<mlir::abi::FunctionClassification>
  classifySignature(mlir::Type retCIR, mlir::TypeRange inputs,
                    llvm::abi::RequiredArgs required,
                    const llvm::abi::TargetInfo &targetInfo,
                    llvm::CallingConv::ID callConv,
                    llvm::function_ref<mlir::InFlightDiagnostic()> emitError);

  /// Classify \p call, which passes more operands than \p calleeTy declares,
  /// from the call's operands.
  std::optional<mlir::abi::FunctionClassification>
  classifyVariadicCall(cir::CIRCallOpInterface call, cir::FuncType calleeTy,
                       const llvm::abi::TargetInfo &targetInfo,
                       llvm::CallingConv::ID callConv);
};

/// The MLIR test ABI target's classifier, whose predictable rules approximate
/// x86_64 SysV.
std::unique_ptr<CIRABIClassifier>
createTestTargetABIClassifier(const mlir::DataLayout &dl);

/// A classifier that reads each function's classification from the
/// DictionaryAttr named \p attrName.
std::unique_ptr<CIRABIClassifier>
createInjectedABIClassifier(const mlir::DataLayout &dl,
                            llvm::StringRef attrName);

/// The x86_64 System V classifier.
std::unique_ptr<CIRABIClassifier>
createX86_64ABIClassifier(const mlir::DataLayout &dl, mlir::ModuleOp modOp,
                          const ABIClassifierOptions &options);

/// The AMDGPU classifier.
std::unique_ptr<CIRABIClassifier>
createAMDGPUABIClassifier(const mlir::DataLayout &dl, mlir::ModuleOp modOp);

/// Map a CIR calling convention to the llvm::CallingConv the ABI classifier
/// uses.  Conventions the classifier does not distinguish fall back to C.
llvm::CallingConv::ID convertCallingConv(cir::CallingConv cc);

//===----------------------------------------------------------------------===//
// Classification helpers defined in CallConvLoweringPass.cpp
//===----------------------------------------------------------------------===//

/// Record on \p fc whether \p returnType is CIR's void.
std::optional<mlir::abi::FunctionClassification>
withReturnVoidness(std::optional<mlir::abi::FunctionClassification> fc,
                   mlir::Type returnType);

/// Classify \p func with the test target, or from the attribute named
/// \p classificationAttrName when that is non-empty.
std::optional<mlir::abi::FunctionClassification>
classifyFunction(cir::FuncOp func, const mlir::DataLayout &dl,
                 cir::CallConvTarget target,
                 llvm::StringRef classificationAttrName);

/// Where \p fnTy's declared parameters end and its ellipsis arguments begin.
llvm::abi::RequiredArgs requiredArgs(cir::FuncType fnTy);

/// The AVX level to classify \p func at: \p base, raised if the function's own
/// recorded feature list enables a wider vector.
llvm::abi::X86AVXABILevel funcAvxLevel(cir::FuncOp func,
                                       llvm::abi::X86AVXABILevel base);

} // namespace cir

#endif // CLANG_LIB_CIR_DIALECT_TRANSFORMS_CIRABICLASSIFIER_H
