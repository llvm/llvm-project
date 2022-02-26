//===-- CanonicalIncludes.h - remap #include header -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// At indexing time, we decide which file to #included for a symbol.
// Usually this is the file with the canonical decl, but there are exceptions:
// - private headers may have pragmas pointing to the matching public header.
//   (These are "IWYU" pragmas, named after the include-what-you-use tool).
// - the standard library is implemented in many files, without any pragmas.
//   We have a lookup table for common standard library implementations.
//   libstdc++ puts char_traits in bits/char_traits.h, but we #include <string>.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_CANONICALINCLUDES_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_CANONICALINCLUDES_H

#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <mutex>
#include <string>
#include <vector>

namespace clang {
namespace clangd {

/// Maps a definition location onto an #include file, based on a set of filename
/// rules.
/// Only const methods (i.e. mapHeader) in this class are thread safe.
class CanonicalIncludes {
public:
  /// Adds a string-to-string mapping from \p Path to \p CanonicalPath.
  void addMapping(llvm::StringRef Path, llvm::StringRef CanonicalPath);

  /// Returns the overridden include for symbol with \p QualifiedName, or "".
  llvm::StringRef mapSymbol(llvm::StringRef QualifiedName) const;

  /// Returns the overridden include for for files in \p Header, or "".
  llvm::StringRef mapHeader(llvm::StringRef Header) const;

  /// Adds mapping for system headers and some special symbols (e.g. STL symbols
  /// in <iosfwd> need to be mapped individually). Approximately, the following
  /// system headers are handled:
  ///   - C++ standard library e.g. bits/basic_string.h$ -> <string>
  ///   - Posix library e.g. bits/pthreadtypes.h$ -> <pthread.h>
  ///   - Compiler extensions, e.g. include/avx512bwintrin.h$ -> <immintrin.h>
  /// The mapping is hardcoded and hand-maintained, so it might not cover all
  /// headers.
  void addSystemHeadersMapping(const LangOptions &Language);

private:
  /// A map from full include path to a canonical path.
  llvm::StringMap<std::string> FullPathMapping;
  /// A map from a suffix (one or components of a path) to a canonical path.
  /// Used only for mapping standard headers.
  const llvm::StringMap<llvm::StringRef> *StdSuffixHeaderMapping = nullptr;
  /// A map from fully qualified symbol names to header names.
  /// Used only for mapping standard symbols.
  const llvm::StringMap<llvm::StringRef> *StdSymbolMapping = nullptr;
};

/// Returns a CommentHandler that parses pragma comment on include files to
/// determine when we should include a different header from the header that
/// directly defines a symbol. Mappinps are registered with \p Includes.
///
/// Currently it only supports IWYU private pragma:
/// https://github.com/include-what-you-use/include-what-you-use/blob/master/docs/IWYUPragmas.md#iwyu-pragma-private
///
/// We ignore other pragmas:
/// - keep: this is common but irrelevant: we do not currently remove includes
/// - export: this is common and potentially interesting, there are three cases:
///    * Points to a public header (common): we can suppress include2 if you
///      already have include1. Only marginally useful.
///    * Points to a private header annotated with `private` (somewhat commmon):
///      Not incrementally useful as we support private.
///    * Points to a private header without pragmas (rare). This is a reversed
///      private pragma, and is valuable but too rare to be worthwhile.
/// - no_include: this is about as common as private, but only affects the
///   current file, so the value is smaller. We could add support.
/// - friend: this is less common than private, has implementation difficulties,
///   and affects behavior in a limited scope.
/// - associated: extremely rare
std::unique_ptr<CommentHandler>
collectIWYUHeaderMaps(CanonicalIncludes *Includes);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_CANONICALINCLUDES_H
