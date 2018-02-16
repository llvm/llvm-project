//===--- Headers.h - Include headers -----------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_HEADERS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_HEADERS_H

#include "Path.h"
#include "clang/Basic/VirtualFileSystem.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace clang {
namespace clangd {
/// Determines the preferred way to #include a file, taking into account the
/// search path. Usually this will prefer a shorter representation like
/// 'Foo/Bar.h' over a longer one like 'Baz/include/Foo/Bar.h'.
///
/// \param Header is an absolute file path.
/// \return A quoted "path" or <path>. If \p Header is already (directly)
/// included in the file (including those included via different paths), this
/// returns an empty string.
llvm::Expected<std::string>
shortenIncludePath(PathRef File, llvm::StringRef Code, llvm::StringRef Header,
                   const tooling::CompileCommand &CompileCommand,
                   IntrusiveRefCntPtr<vfs::FileSystem> FS);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_HEADERS_H
