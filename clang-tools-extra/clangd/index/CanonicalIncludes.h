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
#include "llvm/Support/Regex.h"
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

  /// Returns the canonical include for symbol with \p QualifiedName.
  /// \p Header is the file the declaration was reachable from.
  /// Header itself will be returned if there is no relevant mapping.
  llvm::StringRef mapHeader(llvm::StringRef Header,
                            llvm::StringRef QualifiedName) const;

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
std::unique_ptr<CommentHandler>
collectIWYUHeaderMaps(CanonicalIncludes *Includes);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_HEADERMAPCOLLECTOR_H
