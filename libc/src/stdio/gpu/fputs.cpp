//===-- GPU Implementation of fputs ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fputs.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/macros/config.h"
#include "src/errno/libc_errno.h"
#include "src/stdio/gpu/file.h"

#include "hdr/stdio_macros.h" // for EOF.
#include "hdr/types/FILE.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, fputs,
                   (const char *__restrict str, ::FILE *__restrict stream)) {
  cpp::string_view str_view(str);
  auto written = file::write(stream, str, str_view.size());
  if (written != str_view.size())
    return EOF;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
