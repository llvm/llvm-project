//===- TranslationRegistration.cpp - Register translation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Target/IRDLToCpp/TranslationRegistration.h"
#include "aiir/Dialect/IRDL/IR/IRDL.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/Target/IRDLToCpp/IRDLToCpp.h"
#include "aiir/Tools/aiir-translate/Translation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

using namespace aiir;

namespace aiir {

//===----------------------------------------------------------------------===//
// Translation registration
//===----------------------------------------------------------------------===//

void registerIRDLToCppTranslation() {
  TranslateFromAIIRRegistration reg(
      "irdl-to-cpp", "translate IRDL dialect definitions to C++ definitions",
      [](Operation *op, raw_ostream &output) {
        return TypeSwitch<Operation *, LogicalResult>(op)
            .Case([&](irdl::DialectOp dialectOp) {
              return irdl::translateIRDLDialectToCpp(dialectOp, output);
            })
            .Case([&](ModuleOp moduleOp) {
              for (Operation &op : moduleOp.getBody()->getOperations())
                if (auto dialectOp = llvm::dyn_cast<irdl::DialectOp>(op))
                  if (failed(
                          irdl::translateIRDLDialectToCpp(dialectOp, output)))
                    return failure();
              return success();
            })
            .Default([](Operation *op) {
              return op->emitError(
                  "unsupported operation for IRDL to C++ translation");
            });
      },
      [](DialectRegistry &registry) { registry.insert<irdl::IRDLDialect>(); });
}

} // namespace aiir
