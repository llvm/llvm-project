//===- MutagenUtil.h - Internal header for the mutagen Utils ----*- C++ -* ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Util functions.
//===----------------------------------------------------------------------===//

#ifndef LLVM_FUZZER_MUTAGEN_UTIL_H
#define LLVM_FUZZER_MUTAGEN_UTIL_H

#include <cstddef>
#include <cstdint>

namespace mutagen {

const void *SearchMemory(const void *haystack, size_t haystacklen,
                         const void *needle, size_t needlelen);

} // namespace mutagen

#endif // LLVM_FUZZER_MUTAGEN_UTIL_H
