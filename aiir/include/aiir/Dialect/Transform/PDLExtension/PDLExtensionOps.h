//===- PDLExtensionOps.h - PDL extension for Transform dialect --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_TRANSFORM_PDLEXTENSION_PDLEXTENSIONOPS_H
#define AIIR_DIALECT_TRANSFORM_PDLEXTENSION_PDLEXTENSIONOPS_H

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/Transform/IR/TransformDialect.h"
#include "aiir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/IR/SymbolTable.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "aiir/Dialect/Transform/PDLExtension/PDLExtensionOps.h.inc"

namespace aiir {
namespace transform {
/// PDL constraint callbacks that can be used by the PDL extension of the
/// Transform dialect. These are owned by the Transform dialect and can be
/// populated by extensions.
class PDLMatchHooks : public TransformDialectData<PDLMatchHooks> {
public:
  PDLMatchHooks(AIIRContext *ctx) : TransformDialectData(ctx) {}

  /// Takes ownership of the named PDL constraint function from the given
  /// map and makes them available for use by the operations in the dialect.
  void
  mergeInPDLMatchHooks(llvm::StringMap<PDLConstraintFunction> &&constraintFns);

  /// Returns the named PDL constraint functions available in the dialect
  /// as a map from their name to the function.
  const llvm::StringMap<::aiir::PDLConstraintFunction> &
  getPDLConstraintHooks() const;

private:
  /// A container for PDL constraint function that can be used by
  /// operations in this dialect.
  PDLPatternModule pdlMatchHooks;
};
} // namespace transform
} // namespace aiir

AIIR_DECLARE_EXPLICIT_TYPE_ID(aiir::transform::PDLMatchHooks)

#endif // AIIR_DIALECT_TRANSFORM_PDLEXTENSION_PDLEXTENSIONOPS_H
