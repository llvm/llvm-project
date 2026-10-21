//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Generic implementation of the internal flushing helpers.
///
//===----------------------------------------------------------------------===//

#include "src/stdio/fflush_internal.h"

#include "hdr/types/FILE.h"
#include "src/__support/File/file.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include "src/stdio/stderr.h"
#include "src/stdio/stdin.h"
#include "src/stdio/stdout.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

ErrorOr<int> flush_stream(::FILE *stream) {
  int result = reinterpret_cast<File *>(stream)->flush();
  if (result != 0)
    return Error(result);
  return 0;
}

ErrorOr<int> flush_all_streams() {
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

  if (total_error != 0)
    return Error(total_error);
  return 0;
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
