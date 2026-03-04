//===--- Definition of Linux stdin ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/stdin.h"

#include "hdr/types/FILE.h"
#include "src/__support/File/linux/file.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

constexpr size_t STDIN_BUFFER_SIZE = 512;
uint8_t stdin_buffer[STDIN_BUFFER_SIZE];
static LinuxFile StdIn(0, stdin_buffer, STDIN_BUFFER_SIZE, _IOFBF, false,
                       File::ModeFlags(File::OpenMode::READ));

LLVM_LIBC_VARIABLE(FILE *, stdin) = reinterpret_cast<FILE *>(&StdIn);

} // namespace LIBC_NAMESPACE_DECL
