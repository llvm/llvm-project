//===--- PathMapping.h - apply path mappings to LSP messages -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_PATHMAPPING_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_PATHMAPPING_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <string>
#include <vector>

namespace clang {
namespace clangd {

class Transport;

/// PathMappings are a collection of paired client and server paths.
/// These pairs are used to alter file:// URIs appearing in inbound and outbound
/// LSP messages, as the client's environment may have source files or
/// dependencies at different locations than the server. Therefore, both
/// paths are stored as they appear in file URI bodies, e.g. /usr/include or
/// /C:/config
///
/// For example, if the mappings were {{"/home/user", "/workarea"}}, then
/// a client-to-server LSP message would have file:///home/user/foo.cpp
/// remapped to file:///workarea/foo.cpp, and the same would happen for replies
/// (in the opposite order).
struct PathMapping {
  std::string ClientPath;
  std::string ServerPath;
  enum class Direction { ClientToServer, ServerToClient };
};
using PathMappings = std::vector<PathMapping>;

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const PathMapping &M);

/// Parse the command line \p RawPathMappings (e.g. "/client=/server") into
/// pairs. Returns an error if the mappings are malformed, i.e. not absolute or
/// not a proper pair.
llvm::Expected<PathMappings> parsePathMappings(llvm::StringRef RawPathMappings);

/// Returns a modified \p S with the first matching path in \p Mappings
/// substituted, if applicable
llvm::Optional<std::string> doPathMapping(llvm::StringRef S,
                                          PathMapping::Direction Dir,
                                          const PathMappings &Mappings);

/// Applies the \p Mappings to all the file:// URIs in \p Params.
/// NOTE: The first matching mapping will be applied, otherwise \p Params will
/// be untouched.
void applyPathMappings(llvm::json::Value &Params, PathMapping::Direction Dir,
                       const PathMappings &Mappings);

/// Creates a wrapping transport over \p Transp that applies the \p Mappings to
/// all inbound and outbound LSP messages. All calls are then delegated to the
/// regular transport (e.g. XPC, JSON).
std::unique_ptr<Transport>
createPathMappingTransport(std::unique_ptr<Transport> Transp,
                           PathMappings Mappings);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_PATHMAPPING_H
