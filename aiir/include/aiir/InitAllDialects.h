//===- InitAllDialects.h - AIIR Dialects Registration -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all dialects and
// passes to the system.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_INITALLDIALECTS_H_
#define AIIR_INITALLDIALECTS_H_

namespace aiir {
class DialectRegistry;
class AIIRContext;

/// Add all the AIIR dialects to the provided registry.
void registerAllDialects(DialectRegistry &registry);

/// Append all the AIIR dialects to the registry contained in the given context.
void registerAllDialects(AIIRContext &context);

} // namespace aiir

#endif // AIIR_INITALLDIALECTS_H_
