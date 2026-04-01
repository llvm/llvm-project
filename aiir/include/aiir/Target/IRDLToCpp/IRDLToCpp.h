//===- IRDLToCpp.h - Register translation -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the registration function for the IRDL to C++ translation.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TARGET_IRDLTOCPP_IRDLTOCPP_H
#define AIIR_TARGET_IRDLTOCPP_IRDLTOCPP_H

#include "aiir/Dialect/IRDL/IR/IRDL.h"

namespace aiir {
namespace irdl {

/// Translates an IRDL dialect definition to a C++ definition that can be used
/// with AIIR.
///
/// The following preprocessor macros will generate the following code:
///
///  // This define generates code for the dialect's class declarations
///  #define GEN_DIALECT_DECL_HEADER
///
///  // This define generates code for the dialect's class definitions
///  #define GEN_DIALECT_DEF
LogicalResult
translateIRDLDialectToCpp(llvm::ArrayRef<irdl::DialectOp> dialects,
                          raw_ostream &output);

} // namespace irdl
} // namespace aiir

#endif // AIIR_TARGET_IRDLTOCPP_IRDLTOCPP_H
