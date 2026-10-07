//===- ACCEmitNYIFlang.cpp - Emit NYI diagnostics for OpenACC ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/OpenACC/Passes.h"
#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"

#include <optional>

namespace fir::acc {
#define GEN_PASS_DEF_ACCEMITNYIFLANG
#include "flang/Optimizer/OpenACC/Passes.h.inc"
} // namespace fir::acc

namespace {

static std::optional<llvm::StringRef>
getOpenACCDirectiveName(mlir::Operation *op) {
  return llvm::TypeSwitch<mlir::Operation *, std::optional<llvm::StringRef>>(op)
      .Case<mlir::acc::ParallelOp>(
          [](auto parallel) -> std::optional<llvm::StringRef> {
            return parallel.getCombined() ? "parallel loop" : "parallel";
          })
      .Case<mlir::acc::KernelsOp>(
          [](auto kernels) -> std::optional<llvm::StringRef> {
            return kernels.getCombined() ? "kernels loop" : "kernels";
          })
      .Case<mlir::acc::SerialOp>(
          [](auto serial) -> std::optional<llvm::StringRef> {
            return serial.getCombined() ? "serial loop" : "serial";
          })
      .Case<mlir::acc::DataOp>(
          [](auto) -> std::optional<llvm::StringRef> { return "data"; })
      .Case<mlir::acc::LoopOp>(
          [](auto) -> std::optional<llvm::StringRef> { return "loop"; })
      .Case<mlir::acc::EnterDataOp>(
          [](auto) -> std::optional<llvm::StringRef> { return "enter data"; })
      .Case<mlir::acc::ExitDataOp>(
          [](auto) -> std::optional<llvm::StringRef> { return "exit data"; })
      .Case<mlir::acc::HostDataOp>(
          [](auto) -> std::optional<llvm::StringRef> { return "host_data"; })
      .Case<mlir::acc::InitOp>(
          [](auto) -> std::optional<llvm::StringRef> { return "init"; })
      .Case<mlir::acc::ShutdownOp>(
          [](auto) -> std::optional<llvm::StringRef> { return "shutdown"; })
      .Case<mlir::acc::UpdateOp>(
          [](auto) -> std::optional<llvm::StringRef> { return "update"; })
      .Case<mlir::acc::SetOp>(
          [](auto) -> std::optional<llvm::StringRef> { return "set"; })
      .Case<mlir::acc::WaitOp>(
          [](auto) -> std::optional<llvm::StringRef> { return "wait"; })
      .Case<mlir::acc::AtomicReadOp, mlir::acc::AtomicWriteOp,
            mlir::acc::AtomicUpdateOp, mlir::acc::AtomicCaptureOp>(
          [](auto) -> std::optional<llvm::StringRef> { return "atomic"; })
      .Case<mlir::acc::RoutineOp>(
          [](auto) -> std::optional<llvm::StringRef> { return "routine"; })
      .Case<mlir::acc::DeclareEnterOp, mlir::acc::DeclareExitOp,
            mlir::acc::DeclareOp>(
          [](auto) -> std::optional<llvm::StringRef> { return "declare"; })
      .Case<mlir::acc::CacheOp>(
          [](auto) -> std::optional<llvm::StringRef> { return "cache"; })
      .Default([](mlir::Operation *) -> std::optional<llvm::StringRef> {
        return std::nullopt;
      });
}

class ACCEmitNYIFlang
    : public fir::acc::impl::ACCEmitNYIFlangBase<ACCEmitNYIFlang> {
public:
  void runOnOperation() override {
    mlir::acc::OpenACCSupport &accSupport =
        getAnalysis<mlir::acc::OpenACCSupport>();
    bool emittedError = false;

    // Check source-level directives first so that Flang's fatal TODO reports
    // the directive rather than one of its decomposed clause operations.
    // Pre-order ensures combined compute ops are seen before nested ops.
    getOperation().walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
      std::optional<llvm::StringRef> directiveName =
          getOpenACCDirectiveName(op);
      if (!directiveName)
        return;
      (void)accSupport.emitNYI(op->getLoc(), llvm::Twine("OpenACC ") +
                                                 *directiveName + " directive");
      emittedError = true;
    });

    // Diagnose any remaining OpenACC operation as a fallback.
    getOperation().walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
      if (getOpenACCDirectiveName(op) ||
          mlir::isa<mlir::acc::YieldOp, mlir::acc::TerminatorOp>(op) ||
          !mlir::isa<mlir::acc::OpenACCDialect>(op->getDialect()))
        return;
      (void)accSupport.emitNYI(op->getLoc(), llvm::Twine("OpenACC operation ") +
                                                 op->getName().getStringRef());
      emittedError = true;
    });

    if (emittedError)
      signalPassFailure();
  }
};

} // namespace
