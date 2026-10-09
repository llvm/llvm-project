//===-- Implementation for getline -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/getline.h"

#include "hdr/types/FILE.h"
#include "hdr/types/size_t.h"
#include "hdr/types/ssize_t.h"
#include "src/__support/File/file.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/stdio/inline_getline.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(ssize_t, getline,
                   (char **__restrict lineptr, size_t *__restrict n,
                    ::FILE *__restrict stream)) {
  return LIBC_NAMESPACE::__getline(lineptr, n, '\n', stream);
}

} // namespace LIBC_NAMESPACE_DECL
