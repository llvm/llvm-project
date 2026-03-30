//===-- Implementation of fflush ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fflush.h"
#include "src/__support/File/file.h"
#include "src/stdio/stderr.h"
#include "src/stdio/stdin.h"
#include "src/stdio/stdout.h"

#include "hdr/types/FILE.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, fflush, (::FILE * stream)) {
  // If a non-null stream is specified, we only flush that single stream.
  if (stream != nullptr) {
    int result = reinterpret_cast<File *>(stream)->flush();
    if (result != 0) {
      libc_errno = result;
      return EOF;
    }
    return 0;
  }

  // If the stream is null, we flush all open streams as per C and POSIX
  // requirements.
  int total_error = 0;

  // We explicitly flush the standard streams as they may not be part of the
  // global file list if they are statically initialized.
  File *std_streams[] = {reinterpret_cast<File *>(stdin),
                         reinterpret_cast<File *>(stdout),
                         reinterpret_cast<File *>(stderr)};
  for (auto *s : std_streams) {
    if (s != nullptr) {
      int result = s->flush();
      if (result != 0)
        total_error = result;
    }
  }

  // We iterate over the global list of all open File objects to flush any
  // other streams that were opened via fopen.
  File::lock_list();
  for (File *f = File::get_first_file(); f != nullptr; f = f->get_next()) {
    int result = f->flush();
    if (result != 0)
      total_error = result;
  }
  File::unlock_list();

  if (total_error != 0) {
    libc_errno = total_error;
    return EOF;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
