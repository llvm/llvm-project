//===- TransformTransforms.cpp - C Interface for Transform dialect --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// C interface to transforms for the transform dialect.
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/Transform/Interpreter.h"
#include "aiir-c/Support.h"
#include "aiir/CAPI/IR.h"
#include "aiir/CAPI/Support.h"
#include "aiir/CAPI/Wrap.h"
#include "aiir/Dialect/Transform/IR/Utils.h"
#include "aiir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "aiir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"

using namespace aiir;

DEFINE_C_API_PTR_METHODS(AiirTransformOptions, transform::TransformOptions)

extern "C" {

AiirTransformOptions aiirTransformOptionsCreate() {
  return wrap(new transform::TransformOptions);
}

void aiirTransformOptionsEnableExpensiveChecks(
    AiirTransformOptions transformOptions, bool enable) {
  unwrap(transformOptions)->enableExpensiveChecks(enable);
}

bool aiirTransformOptionsGetExpensiveChecksEnabled(
    AiirTransformOptions transformOptions) {
  return unwrap(transformOptions)->getExpensiveChecksEnabled();
}

void aiirTransformOptionsEnforceSingleTopLevelTransformOp(
    AiirTransformOptions transformOptions, bool enable) {
  unwrap(transformOptions)->enableEnforceSingleToplevelTransformOp(enable);
}

bool aiirTransformOptionsGetEnforceSingleTopLevelTransformOp(
    AiirTransformOptions transformOptions) {
  return unwrap(transformOptions)->getEnforceSingleToplevelTransformOp();
}

void aiirTransformOptionsDestroy(AiirTransformOptions transformOptions) {
  delete unwrap(transformOptions);
}

AiirLogicalResult aiirTransformApplyNamedSequence(
    AiirOperation payload, AiirOperation transformRoot,
    AiirOperation transformModule, AiirTransformOptions transformOptions) {
  Operation *transformRootOp = unwrap(transformRoot);
  Operation *transformModuleOp = unwrap(transformModule);
  if (!isa<transform::TransformOpInterface>(transformRootOp)) {
    transformRootOp->emitError()
        << "must implement TransformOpInterface to be used as transform root";
    return aiirLogicalResultFailure();
  }
  if (!isa<ModuleOp>(transformModuleOp)) {
    transformModuleOp->emitError()
        << "must be a " << ModuleOp::getOperationName();
    return aiirLogicalResultFailure();
  }
  return wrap(transform::applyTransformNamedSequence(
      unwrap(payload), unwrap(transformRoot),
      cast<ModuleOp>(unwrap(transformModule)), *unwrap(transformOptions)));
}

AiirLogicalResult aiirMergeSymbolsIntoFromClone(AiirOperation target,
                                                AiirOperation other) {
  OwningOpRef<Operation *> otherOwning(unwrap(other)->clone());
  LogicalResult result = transform::detail::mergeSymbolsInto(
      unwrap(target), std::move(otherOwning));
  return wrap(result);
}
}
