//===- AiirQueryMain.h - AIIR Query main ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for aiir-query for when built as standalone
// binary.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TOOLS_AIIRQUERY_AIIRQUERYMAIN_H
#define AIIR_TOOLS_AIIRQUERY_AIIRQUERYMAIN_H

#include "aiir/Query/Matcher/Registry.h"
#include "aiir/Support/LLVM.h"

namespace aiir {

class AIIRContext;

LogicalResult
aiirQueryMain(int argc, char **argv, AIIRContext &context,
              const aiir::query::matcher::Registry &matcherRegistry);

} // namespace aiir

#endif // AIIR_TOOLS_AIIRQUERY_AIIRQUERYMAIN_H
