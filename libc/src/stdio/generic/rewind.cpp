//===-- Implementation of rewind ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/rewind.h"
#include "src/__support/File/file.h"

#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, rewind, (::FILE * stream)) {
  LIBC_CRASH_ON_NULLPTR(stream);
  File *FilePtr = reinterpret_cast<File *>(stream);
  auto Result = FilePtr->seek(0, SEEK_SET);
  FilePtr->clearerr();

  if (!Result.has_value())
    libc_errno = Result.error();
}

} // namespace LIBC_NAMESPACE_DECL
