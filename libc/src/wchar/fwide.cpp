//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation of the fwide function, which sets and
/// gets the orientation of a stream.
///
//===----------------------------------------------------------------------===//

#include "src/wchar/fwide.h"
#include "hdr/types/FILE.h"
#include "src/__support/File/file.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, fwide, (::FILE * stream, int mode)) {
  LIBC_CRASH_ON_NULLPTR(stream);
  auto *f = reinterpret_cast<File *>(stream);

  File::Orientation orient;
  if (mode > 0) {
    orient = f->try_set_orientation(File::Orientation::WIDE);
  } else if (mode < 0) {
    orient = f->try_set_orientation(File::Orientation::BYTE);
  } else {
    orient = f->get_orientation();
  }

  if (orient == File::Orientation::WIDE)
    return 1;
  if (orient == File::Orientation::BYTE)
    return -1;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
