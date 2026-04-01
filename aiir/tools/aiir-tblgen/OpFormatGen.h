//===- OpFormatGen.h - AIIR operation format generator ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for generating parsers and printers from the
// declarative format.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TOOLS_AIIRTBLGEN_OPFORMATGEN_H_
#define AIIR_TOOLS_AIIRTBLGEN_OPFORMATGEN_H_

namespace aiir {
namespace tblgen {
class OpClass;
class Operator;

// Generate the assembly format for the given operator.
void generateOpFormat(const Operator &constOp, OpClass &opClass,
                      bool hasProperties);

} // namespace tblgen
} // namespace aiir

#endif // AIIR_TOOLS_AIIRTBLGEN_OPFORMATGEN_H_
