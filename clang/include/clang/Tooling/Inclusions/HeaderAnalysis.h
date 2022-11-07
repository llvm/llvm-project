//===--- HeaderAnalysis.h -----------------------------------------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_INCLUSIONS_HEADER_ANALYSIS_H
#define LLVM_CLANG_TOOLING_INCLUSIONS_HEADER_ANALYSIS_H

namespace clang {
class FileEntry;
class SourceManager;
class HeaderSearch;

namespace tooling {

/// Returns true if the given physical file is a self-contained header.
///
/// A header is considered self-contained if
//   - it has a proper header guard or has been #imported
//   - *and* it doesn't have a dont-include-me pattern.
///
/// This function can be expensive as it may scan the source code to find out
/// dont-include-me pattern heuristically.
bool isSelfContainedHeader(const FileEntry *FE, const SourceManager &SM,
                           HeaderSearch &HeaderInfo);

} // namespace tooling
} // namespace clang

#endif // LLVM_CLANG_TOOLING_INCLUSIONS_HEADER_ANALYSIS_H
