//===- Filesystem.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_FILESYSTEM_H
#define LLD_ELF_FILESYSTEM_H

#include "lld/Common/LLVM.h"
#include <system_error>

namespace lld {
namespace elf {
void unlinkAsync(StringRef Path);
std::error_code tryCreateFile(StringRef Path);
} // namespace elf
} // namespace lld

#endif
