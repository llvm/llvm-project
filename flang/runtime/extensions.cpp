//===-- runtime/extensions.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// These C-coded entry points with Fortran-mangled names implement legacy
// extensions that will eventually be implemented in Fortran.

#include "flang/Runtime/extensions.h"
#include "terminator.h"
#include "flang/Runtime/command.h"
#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/io-api.h"
#include <ctime>

#ifdef _WIN32
inline void ctime_alloc(char *buffer, size_t bufsize, const time_t cur_time,
    Fortran::runtime::Terminator terminator) {
  int error = ctime_s(buffer, bufsize, &cur_time);
  RUNTIME_CHECK(terminator, error == 0);
}
#else
inline void ctime_alloc(char *buffer, size_t bufsize, const time_t cur_time,
    Fortran::runtime::Terminator terminator) {
  const char *res = ctime_r(&cur_time, buffer);
  RUNTIME_CHECK(terminator, res != nullptr);
}
#endif

extern "C" {

namespace Fortran::runtime {
namespace io {
// SUBROUTINE FLUSH(N)
//   FLUSH N
// END
void FORTRAN_PROCEDURE_NAME(flush)(const int &unit) {
  Cookie cookie{IONAME(BeginFlush)(unit, __FILE__, __LINE__)};
  IONAME(EndIoStatement)(cookie);
}
} // namespace io

// RESULT = IARGC()
std::int32_t FORTRAN_PROCEDURE_NAME(iargc)() { return RTNAME(ArgumentCount)(); }

void FORTRAN_PROCEDURE_NAME(fdate)(std::int8_t *arg, std::int64_t length) {
  std::array<char, 26> str;
  // If the length is too short to fit completely, blank return.
  if (length < 24) {
    str.fill(' ');
    strncpy(reinterpret_cast<char *>(arg), str.data(), length);
    return;
  }

  Terminator terminator{__FILE__, __LINE__};
  std::time_t current_time;
  std::time(&current_time);
  // Day Mon dd hh:mm:ss yyyy\n\0 is 26 characters, e.g.
  // Tue May 26 21:51:03 2015\n\0

  ctime_alloc(str.data(), str.size(), current_time, terminator);

  std::fill(str.begin() + 24, str.begin() + length, ' ');
  strncpy(reinterpret_cast<char *>(arg), str.data(), length);
}

// CALL GETARG(N, ARG)
void FORTRAN_PROCEDURE_NAME(getarg)(
    std::int32_t &n, std::int8_t *arg, std::int64_t length) {
  Descriptor value{*Descriptor::Create(1, length, arg, 0)};
  (void)RTNAME(GetCommandArgument)(
      n, &value, nullptr, nullptr, __FILE__, __LINE__);
}
} // namespace Fortran::runtime
} // extern "C"
