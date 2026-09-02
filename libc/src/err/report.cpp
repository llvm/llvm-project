//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of internal error reporting helpers.
///
//===----------------------------------------------------------------------===//

#include "src/err/report.h"

#include "hdr/errno_macros.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/OSUtil/io.h"
#include "src/__support/StringUtil/error_to_string.h"
#include "src/__support/arg_list.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/printf_core/make_stderr_writer.h"
#include "src/__support/printf_core/printf_main.h"

#ifdef LIBC_FULL_BUILD
#include "src/errno/program_invocation_short_name.h"
#define PROGRAM_INVOCATION_SHORT_NAME                                          \
  LIBC_NAMESPACE::program_invocation_short_name
#else
extern "C" char *program_invocation_short_name;
#define PROGRAM_INVOCATION_SHORT_NAME ::program_invocation_short_name
#endif

namespace LIBC_NAMESPACE_DECL {
namespace err_reporting {

void report(bool show_err, int err_num, const char *fmt,
            internal::ArgList &args) {
  const char *progname = PROGRAM_INVOCATION_SHORT_NAME;
  if (!progname)
    progname = "";
  char buffer[1024];
  printf_core::Writer writer =
      printf_core::make_stderr_writer(buffer, sizeof(buffer));
  printf_core::WriteBuffer<char> &wb = writer.get_write_buffer();

  writer.write(progname);
  if (fmt != nullptr || show_err)
    writer.write(": ");

  if (fmt != nullptr) {
    if (!printf_core::printf_main(&writer, fmt, args)) {
      printf_core::flush_to_stderr(wb);
      return;
    }
    if (show_err)
      writer.write(": ");
  }

  if (show_err)
    writer.write(get_error_string(err_num));

  writer.write("\n");
  printf_core::flush_to_stderr(wb);
}

} // namespace err_reporting
} // namespace LIBC_NAMESPACE_DECL
