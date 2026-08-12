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
#include "src/__support/printf_core/printf_main.h"
#include "src/__support/printf_core/writer.h"

#ifdef __linux__
extern "C" char *program_invocation_short_name;
#endif

namespace LIBC_NAMESPACE_DECL {
namespace err_reporting {

void report(bool show_err, int err_num, const char *fmt,
            internal::ArgList &args) {
  const char *progname = "libllvmlibc";
  // TODO: Use a proper way to get progname if available.
#ifdef __linux__
  progname = program_invocation_short_name;
#endif

  char buffer[1024];
  printf_core::FlushingBuffer wb(
      buffer, sizeof(buffer),
      [](cpp::string_view str, [[maybe_unused]] void *raw_stream) -> int {
        write_to_stderr(str);
        return static_cast<int>(str.size());
      },
      nullptr);
  printf_core::Writer writer(wb);

  writer.write(progname);
  if (fmt != nullptr || show_err)
    writer.write(": ");

  if (fmt != nullptr) {
    if (!printf_core::printf_main(&writer, fmt, args)) {
      wb.flush_to_stream();
      return;
    }
    if (show_err)
      writer.write(": ");
  }

  if (show_err)
    writer.write(get_error_string(err_num));

  writer.write("\n");
  wb.flush_to_stream();
}

} // namespace err_reporting
} // namespace LIBC_NAMESPACE_DECL
