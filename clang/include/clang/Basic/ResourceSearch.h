//===--- ResourceSearch.h - Searching for Resources -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// User-provided filters include/exclude profile instrumentation in certain
// functions.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_BASIC_RESOURCESEARCH_H
#define LLVM_CLANG_BASIC_RESOURCESEARCH_H

#include "clang/Basic/FileEntry.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <optional>

namespace clang {

class FileManager;

OptionalFileEntryRef LookupFileWithStdVec(
    StringRef Filename, bool isAngled, bool OpenFile, FileManager &FM,
    const std::vector<std::string> &SearchEntries,
    OptionalFileEntryRef LookupFromFile, StringRef *FoundEntry = nullptr);
OptionalFileEntryRef LookupFileWith(StringRef Filename, bool isAngled,
                                    bool OpenFile, FileManager &FM,
                                    ArrayRef<StringRef> SearchEntries,
                                    OptionalFileEntryRef LookupFromFile,
                                    StringRef *FoundEntry = nullptr);
} // namespace clang

#endif
