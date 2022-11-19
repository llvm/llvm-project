//===-- Internal implementation of vfscanf ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/scanf_core/vfscanf_internal.h"

#include "src/__support/arg_list.h"
#include "src/stdio/scanf_core/file_reader.h"
#include "src/stdio/scanf_core/reader.h"
#include "src/stdio/scanf_core/scanf_main.h"

#include <stdio.h>

namespace __llvm_libc {
namespace scanf_core {

int vfscanf_internal(::FILE *__restrict stream, const char *__restrict format,
                     internal::ArgList &args) {
  FileReader file_reader(stream);
  scanf_core::Reader reader(&file_reader);
  return scanf_core::scanf_main(&reader, format, args);
}

} // namespace scanf_core
} // namespace __llvm_libc
