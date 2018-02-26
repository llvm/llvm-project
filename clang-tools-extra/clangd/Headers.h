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

/// Returns true if \p Include is literal include like "path" or <path>.
bool isLiteralInclude(llvm::StringRef Include);

/// Represents a header file to be #include'd.
struct HeaderFile {
  std::string File;
  /// If this is true, `File` is a literal string quoted with <> or "" that
  /// can be #included directly; otherwise, `File` is an absolute file path.
  bool Verbatim;

  bool valid() const;
};

/// Determines the preferred way to #include a file, taking into account the
/// search path. Usually this will prefer a shorter representation like
/// 'Foo/Bar.h' over a longer one like 'Baz/include/Foo/Bar.h'.
///
/// \param File is an absolute file path.
/// \param DeclaringHeader is the original header corresponding to \p
/// InsertedHeader e.g. the header that declares a symbol.
/// \param InsertedHeader The preferred header to be inserted. This could be the
/// same as DeclaringHeader but must be provided.
//  \return A quoted "path" or <path>. This returns an empty string if:
///   - Either \p DeclaringHeader or \p InsertedHeader is already (directly)
///   included in the file (including those included via different paths).
///   - \p DeclaringHeader or \p InsertedHeader is the same as \p File.
llvm::Expected<std::string>
calculateIncludePath(PathRef File, llvm::StringRef Code,
                     const HeaderFile &DeclaringHeader,
                     const HeaderFile &InsertedHeader,
                     const tooling::CompileCommand &CompileCommand,
                     IntrusiveRefCntPtr<vfs::FileSystem> FS);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_HEADERS_H
