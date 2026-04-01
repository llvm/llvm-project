//===- AiirLspRegistryFunction.h - LSP registry functions -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registry function types for AIIR LSP.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TOOLS_AIIR_LSP_SERVER_AIIRLSPREGISTRYFUNCTION_H
#define AIIR_TOOLS_AIIR_LSP_SERVER_AIIRLSPREGISTRYFUNCTION_H

namespace llvm {
template <typename Fn>
class function_ref;
namespace lsp {
class URIForFile;
} // namespace lsp
} // namespace llvm

namespace aiir {
class DialectRegistry;
namespace lsp {
using DialectRegistryFn =
    llvm::function_ref<DialectRegistry &(const llvm::lsp::URIForFile &uri)>;
} // namespace lsp
} // namespace aiir

#endif // AIIR_TOOLS_AIIR_LSP_SERVER_AIIRLSPREGISTRYFUNCTION_H
