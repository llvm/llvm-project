//===- Pass.cpp - C Interface for General Pass Management APIs ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Pass.h"

#include "aiir/CAPI/IR.h"
#include "aiir/CAPI/Pass.h"
#include "aiir/CAPI/Support.h"
#include "aiir/CAPI/Utils.h"
#include "aiir/Pass/PassManager.h"
#include "llvm/Support/ErrorHandling.h"
#include <optional>

using namespace aiir;

//===----------------------------------------------------------------------===//
// PassManager/OpPassManager APIs.
//===----------------------------------------------------------------------===//

AiirPassManager aiirPassManagerCreate(AiirContext ctx) {
  return wrap(new PassManager(unwrap(ctx)));
}

AiirPassManager aiirPassManagerCreateOnOperation(AiirContext ctx,
                                                 AiirStringRef anchorOp) {
  return wrap(new PassManager(unwrap(ctx), unwrap(anchorOp)));
}

void aiirPassManagerDestroy(AiirPassManager passManager) {
  delete unwrap(passManager);
}

AiirOpPassManager
aiirPassManagerGetAsOpPassManager(AiirPassManager passManager) {
  return wrap(static_cast<OpPassManager *>(unwrap(passManager)));
}

AiirLogicalResult aiirPassManagerRunOnOp(AiirPassManager passManager,
                                         AiirOperation op) {
  return wrap(unwrap(passManager)->run(unwrap(op)));
}

void aiirPassManagerEnableIRPrinting(AiirPassManager passManager,
                                     bool printBeforeAll, bool printAfterAll,
                                     bool printModuleScope,
                                     bool printAfterOnlyOnChange,
                                     bool printAfterOnlyOnFailure,
                                     AiirOpPrintingFlags flags,
                                     AiirStringRef treePrintingPath) {
  auto shouldPrintBeforePass = [printBeforeAll](Pass *, Operation *) {
    return printBeforeAll;
  };
  auto shouldPrintAfterPass = [printAfterAll](Pass *, Operation *) {
    return printAfterAll;
  };
  if (unwrap(treePrintingPath).empty())
    return unwrap(passManager)
        ->enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass,
                           printModuleScope, printAfterOnlyOnChange,
                           printAfterOnlyOnFailure, /*out=*/llvm::errs(),
                           *unwrap(flags));

  unwrap(passManager)
      ->enableIRPrintingToFileTree(shouldPrintBeforePass, shouldPrintAfterPass,
                                   printModuleScope, printAfterOnlyOnChange,
                                   printAfterOnlyOnFailure,
                                   unwrap(treePrintingPath), *unwrap(flags));
}

void aiirPassManagerEnableVerifier(AiirPassManager passManager, bool enable) {
  unwrap(passManager)->enableVerifier(enable);
}

void aiirPassManagerEnableTiming(AiirPassManager passManager) {
  unwrap(passManager)->enableTiming();
}

void aiirPassManagerEnableStatistics(AiirPassManager passManager,
                                     AiirPassDisplayMode displayMode) {
  PassDisplayMode mode;
  switch (displayMode) {
  case AIIR_PASS_DISPLAY_MODE_LIST:
    mode = PassDisplayMode::List;
    break;
  case AIIR_PASS_DISPLAY_MODE_PIPELINE:
    mode = PassDisplayMode::Pipeline;
    break;
  }
  unwrap(passManager)->enableStatistics(mode);
}

AiirOpPassManager aiirPassManagerGetNestedUnder(AiirPassManager passManager,
                                                AiirStringRef operationName) {
  return wrap(&unwrap(passManager)->nest(unwrap(operationName)));
}

AiirOpPassManager aiirOpPassManagerGetNestedUnder(AiirOpPassManager passManager,
                                                  AiirStringRef operationName) {
  return wrap(&unwrap(passManager)->nest(unwrap(operationName)));
}

void aiirPassManagerAddOwnedPass(AiirPassManager passManager, AiirPass pass) {
  unwrap(passManager)->addPass(std::unique_ptr<Pass>(unwrap(pass)));
}

void aiirOpPassManagerAddOwnedPass(AiirOpPassManager passManager,
                                   AiirPass pass) {
  unwrap(passManager)->addPass(std::unique_ptr<Pass>(unwrap(pass)));
}

AiirLogicalResult aiirOpPassManagerAddPipeline(AiirOpPassManager passManager,
                                               AiirStringRef pipelineElements,
                                               AiirStringCallback callback,
                                               void *userData) {
  detail::CallbackOstream stream(callback, userData);
  return wrap(parsePassPipeline(unwrap(pipelineElements), *unwrap(passManager),
                                stream));
}

void aiirPrintPassPipeline(AiirOpPassManager passManager,
                           AiirStringCallback callback, void *userData) {
  detail::CallbackOstream stream(callback, userData);
  unwrap(passManager)->printAsTextualPipeline(stream);
}

AiirLogicalResult aiirParsePassPipeline(AiirOpPassManager passManager,
                                        AiirStringRef pipeline,
                                        AiirStringCallback callback,
                                        void *userData) {
  detail::CallbackOstream stream(callback, userData);
  FailureOr<OpPassManager> pm = parsePassPipeline(unwrap(pipeline), stream);
  if (succeeded(pm))
    *unwrap(passManager) = std::move(*pm);
  return wrap(pm);
}

//===----------------------------------------------------------------------===//
// External Pass API.
//===----------------------------------------------------------------------===//

namespace aiir {
class ExternalPass;
} // namespace aiir
DEFINE_C_API_PTR_METHODS(AiirExternalPass, aiir::ExternalPass)

namespace aiir {
/// This pass class wraps external passes defined in other languages using the
/// AIIR C-interface
class ExternalPass : public Pass {
public:
  ExternalPass(TypeID passID, StringRef name, StringRef argument,
               StringRef description, std::optional<StringRef> opName,
               ArrayRef<AiirDialectHandle> dependentDialects,
               AiirExternalPassCallbacks callbacks, void *userData)
      : Pass(passID, opName), id(passID), name(name), argument(argument),
        description(description), dependentDialects(dependentDialects),
        callbacks(callbacks), userData(userData) {
    if (callbacks.construct)
      callbacks.construct(userData);
  }

  ~ExternalPass() override {
    if (callbacks.destruct)
      callbacks.destruct(userData);
  }

  StringRef getName() const override { return name; }
  StringRef getArgument() const override { return argument; }
  StringRef getDescription() const override { return description; }

  void getDependentDialects(DialectRegistry &registry) const override {
    AiirDialectRegistry cRegistry = wrap(&registry);
    for (AiirDialectHandle dialect : dependentDialects)
      aiirDialectHandleInsertDialect(dialect, cRegistry);
  }

  void signalPassFailure() { Pass::signalPassFailure(); }

protected:
  LogicalResult initialize(AIIRContext *ctx) override {
    if (callbacks.initialize)
      return unwrap(callbacks.initialize(wrap(ctx), userData));
    return success();
  }

  bool canScheduleOn(RegisteredOperationName opName) const override {
    if (std::optional<StringRef> specifiedOpName = getOpName())
      return opName.getStringRef() == specifiedOpName;
    return true;
  }

  void runOnOperation() override {
    callbacks.run(wrap(getOperation()), wrap(this), userData);
  }

  std::unique_ptr<Pass> clonePass() const override {
    void *clonedUserData = callbacks.clone(userData);
    return std::make_unique<ExternalPass>(id, name, argument, description,
                                          getOpName(), dependentDialects,
                                          callbacks, clonedUserData);
  }

private:
  TypeID id;
  std::string name;
  std::string argument;
  std::string description;
  std::vector<AiirDialectHandle> dependentDialects;
  AiirExternalPassCallbacks callbacks;
  void *userData;
};
} // namespace aiir

AiirPass aiirCreateExternalPass(AiirTypeID passID, AiirStringRef name,
                                AiirStringRef argument,
                                AiirStringRef description, AiirStringRef opName,
                                intptr_t nDependentDialects,
                                AiirDialectHandle *dependentDialects,
                                AiirExternalPassCallbacks callbacks,
                                void *userData) {
  return wrap(static_cast<aiir::Pass *>(new aiir::ExternalPass(
      unwrap(passID), unwrap(name), unwrap(argument), unwrap(description),
      opName.length > 0 ? std::optional<StringRef>(unwrap(opName))
                        : std::nullopt,
      {dependentDialects, static_cast<size_t>(nDependentDialects)}, callbacks,
      userData)));
}

void aiirExternalPassSignalFailure(AiirExternalPass pass) {
  unwrap(pass)->signalPassFailure();
}
