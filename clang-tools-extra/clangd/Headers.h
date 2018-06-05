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
#include "Protocol.h"
#include "SourceCode.h"
#include "clang/Basic/VirtualFileSystem.h"
#include "clang/Format/Format.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Tooling/Inclusions/HeaderIncludes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
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

// An #include directive that we found in the main file.
struct Inclusion {
  Range R;             // Inclusion range.
  std::string Written; // Inclusion name as written e.g. <vector>.
  Path Resolved;       // Resolved path of included file. Empty if not resolved.
};

/// Returns a PPCallback that visits all inclusions in the main file.
std::unique_ptr<PPCallbacks>
collectInclusionsInMainFileCallback(const SourceManager &SM,
                                    std::function<void(Inclusion)> Callback);

/// Determines the preferred way to #include a file, taking into account the
/// search path. Usually this will prefer a shorter representation like
/// 'Foo/Bar.h' over a longer one like 'Baz/include/Foo/Bar.h'.
///
/// \param File is an absolute file path.
/// \param Inclusions Existing inclusions in the main file.
/// \param DeclaringHeader is the original header corresponding to \p
/// InsertedHeader e.g. the header that declares a symbol.
/// \param InsertedHeader The preferred header to be inserted. This could be the
/// same as DeclaringHeader but must be provided.
//  \return A quoted "path" or <path>. This returns an empty string if:
///   - Either \p DeclaringHeader or \p InsertedHeader is already (directly)
///   in \p Inclusions (including those included via different paths).
///   - \p DeclaringHeader or \p InsertedHeader is the same as \p File.
llvm::Expected<std::string> calculateIncludePath(
    PathRef File, StringRef BuildDir, HeaderSearch &HeaderSearchInfo,
    const std::vector<Inclusion> &Inclusions, const HeaderFile &DeclaringHeader,
    const HeaderFile &InsertedHeader);

// Calculates insertion edit for including a new header in a file.
class IncludeInserter {
public:
  IncludeInserter(StringRef FileName, StringRef Code,
                  const format::FormatStyle &Style, StringRef BuildDir,
                  HeaderSearch &HeaderSearchInfo)
      : FileName(FileName), Code(Code), BuildDir(BuildDir),
        HeaderSearchInfo(HeaderSearchInfo),
        Inserter(FileName, Code, Style.IncludeStyle) {}

  void addExisting(Inclusion Inc) { Inclusions.push_back(std::move(Inc)); }

  /// Returns a TextEdit that inserts a new header; if the header is not
  /// inserted e.g. it's an existing header, this returns None. If any header is
  /// invalid, this returns error.
  ///
  /// \param DeclaringHeader is the original header corresponding to \p
  /// InsertedHeader e.g. the header that declares a symbol.
  /// \param InsertedHeader The preferred header to be inserted. This could be
  /// the same as DeclaringHeader but must be provided.
  Expected<Optional<TextEdit>> insert(const HeaderFile &DeclaringHeader,
                                      const HeaderFile &InsertedHeader) const;

private:
  StringRef FileName;
  StringRef Code;
  StringRef BuildDir;
  HeaderSearch &HeaderSearchInfo;
  std::vector<Inclusion> Inclusions;
  tooling::HeaderIncludes Inserter; // Computers insertion replacement.
};

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_HEADERS_H
