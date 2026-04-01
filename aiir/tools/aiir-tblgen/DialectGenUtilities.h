//===- DialectGenUtilities.h - Utilities for dialect generation -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TOOLS_AIIRTBLGEN_DIALECTGENUTILITIES_H_
#define AIIR_TOOLS_AIIRTBLGEN_DIALECTGENUTILITIES_H_

#include "aiir/Support/LLVM.h"

namespace aiir {
namespace tblgen {
class Dialect;

/// Find the dialect selected by the user to generate for. Returns std::nullopt
/// if no dialect was found, or if more than one potential dialect was found.
std::optional<Dialect> findDialectToGenerate(ArrayRef<Dialect> dialects);
} // namespace tblgen
} // namespace aiir

#endif // AIIR_TOOLS_AIIRTBLGEN_DIALECTGENUTILITIES_H_
