//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation of the fgetws function, which reads a
/// wide-character string from a file.
///
//===----------------------------------------------------------------------===//

#include "src/wchar/fgetws.h"
#include "hdr/types/FILE.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/File/file.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(wchar_t *, fgetws,
                   (wchar_t *__restrict ws, int count,
                    ::FILE *__restrict stream)) {
  if (count <= 0)
    return nullptr;

  LIBC_CRASH_ON_NULLPTR(ws);
  LIBC_CRASH_ON_NULLPTR(stream);

  auto *f = reinterpret_cast<File *>(stream);

  if (count == 1) {
    ws[0] = L'\0';
    return ws;
  }

  f->lock();

  wchar_t *result = ws;
  int chars_read = 0;

  for (wchar_t c = L'\0'; chars_read < count - 1 && c != L'\n'; ++chars_read) {
    auto read_res = f->read_unlocked(&c, 1);
    if (read_res.has_error()) {
      libc_errno = read_res.error;
      result = nullptr;
      break;
    }
    if (read_res.value < 1) {
      if (chars_read == 0)
        result = nullptr;
      break;
    }
    ws[chars_read] = c;
  }

  if (result != nullptr)
    ws[chars_read] = L'\0';

  f->unlock();
  return result;
}

} // namespace LIBC_NAMESPACE_DECL
