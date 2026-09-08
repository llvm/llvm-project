//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of fgetpos
///
//===----------------------------------------------------------------------===//

#include "src/stdio/fgetpos.h"
#include "hdr/types/FILE.h"
#include "hdr/types/fpos_t.h"
#include "src/__support/File/file.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/string/memory_utils/inline_memset.h"

namespace LIBC_NAMESPACE_DECL {

static_assert(sizeof(mbstate_t) >= sizeof(internal::mbstate),
              "mbstate_t is too small for internal::mbstate");
static_assert(alignof(mbstate_t) >= alignof(internal::mbstate),
              "mbstate_t has insufficient alignment for internal::mbstate");

LLVM_LIBC_FUNCTION(int, fgetpos,
                   (::FILE *__restrict stream, ::fpos_t *__restrict pos)) {
  LIBC_CRASH_ON_NULLPTR(stream);
  LIBC_CRASH_ON_NULLPTR(pos);
  auto result = reinterpret_cast<LIBC_NAMESPACE::File *>(stream)->get_pos();
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }
  File::Position fpos = result.value();
  pos->pos = fpos.offset;
  inline_memset(&pos->state, 0, sizeof(mbstate_t));
  inline_memcpy(&pos->state, &fpos.state, sizeof(internal::mbstate));
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
