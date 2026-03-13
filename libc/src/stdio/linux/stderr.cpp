//===--- Definition of Linux stderr ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/stderr.h"

#include "hdr/types/FILE.h"
#include "src/__support/File/linux/file.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

constexpr size_t STDERR_BUFFER_SIZE = 0;
static LinuxFile StdErr(2, nullptr, STDERR_BUFFER_SIZE, _IONBF, false,
                        File::ModeFlags(File::OpenMode::APPEND));

LLVM_LIBC_VARIABLE(FILE *, stderr) = reinterpret_cast<FILE *>(&StdErr);

} // namespace LIBC_NAMESPACE_DECL
