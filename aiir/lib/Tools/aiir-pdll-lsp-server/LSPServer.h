//===- LSPServer.h - PDLL LSP Server ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIB_AIIR_TOOLS_AIIRPDLLLSPSERVER_LSPSERVER_H
#define LIB_AIIR_TOOLS_AIIRPDLLLSPSERVER_LSPSERVER_H

#include <memory>

namespace llvm {
struct LogicalResult;
namespace lsp {
class JSONTransport;
} // namespace lsp
} // namespace llvm

namespace aiir {
namespace lsp {
class PDLLServer;

/// Run the main loop of the LSP server using the given PDLL server and
/// transport.
llvm::LogicalResult runPdllLSPServer(PDLLServer &server,
                                     llvm::lsp::JSONTransport &transport);

} // namespace lsp
} // namespace aiir

#endif // LIB_AIIR_TOOLS_AIIRPDLLLSPSERVER_LSPSERVER_H
