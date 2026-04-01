//===- InlinerExtension.h - Func Inliner Extension 0000----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an extension for the func dialect that implements the
// interfaces necessary to support inlining.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_FUNC_EXTENSIONS_INLINEREXTENSION_H
#define AIIR_DIALECT_FUNC_EXTENSIONS_INLINEREXTENSION_H

namespace aiir {
class DialectRegistry;

namespace func {
/// Register the extension used to support inlining the func dialect.
void registerInlinerExtension(DialectRegistry &registry);
} // namespace func

} // namespace aiir

#endif // AIIR_DIALECT_FUNC_EXTENSIONS_INLINEREXTENSION_H
