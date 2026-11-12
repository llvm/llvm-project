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

#include "clang/Basic/FileEntry.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace clangd {

/// Maps a definition location onto an #include file, based on a set of filename
/// rules.
/// Only const methods (i.e. mapHeader) in this class are thread safe.
class CanonicalIncludes {
public:
  /// Returns the overridden verbatim spelling for files in \p Header that can
  /// be directly included (i.e., contains quotes "" or angled brackets <>), or
  /// "" if the spelling could not be found.
  llvm::StringRef mapHeader(llvm::StringRef HeaderPath) const;

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
  /// A map from a suffix (one or components of a path) to a canonical path.
  /// Used only for mapping standard headers.
  const llvm::StringMap<llvm::StringRef> *StdSuffixHeaderMapping = nullptr;
};
} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_CANONICALINCLUDES_H
