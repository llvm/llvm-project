//===- InitAllExtensions.h - AIIR Extension Registration --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all dialect
// extensions to the system.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_INITALLEXTENSIONS_H_
#define AIIR_INITALLEXTENSIONS_H_

namespace aiir {
class DialectRegistry;

/// This function may be called to register all AIIR dialect extensions with the
/// provided registry.
/// If you're building a compiler, you generally shouldn't use this: you would
/// individually register the specific extensions that are useful for the
/// pipelines and transformations you are using.
void registerAllExtensions(DialectRegistry &registry);

} // namespace aiir

#endif // AIIR_INITALLEXTENSIONS_H_
