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
#include "tools.h"
#include "flang/Runtime/command.h"
#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/io-api.h"
#include <ctime>

#ifdef _WIN32
inline void CtimeBuffer(char *buffer, size_t bufsize, const time_t cur_time,
    Fortran::runtime::Terminator terminator) {
  int error{ctime_s(buffer, bufsize, &cur_time)};
  RUNTIME_CHECK(terminator, error == 0);
}
#elif _POSIX_C_SOURCE >= 1 || _XOPEN_SOURCE || _BSD_SOURCE || _SVID_SOURCE || \
    defined(_POSIX_SOURCE)
inline void CtimeBuffer(char *buffer, size_t bufsize, const time_t cur_time,
    Fortran::runtime::Terminator terminator) {
  const char *res{ctime_r(&cur_time, buffer)};
  RUNTIME_CHECK(terminator, res != nullptr);
}
#else
inline void CtimeBuffer(char *buffer, size_t bufsize, const time_t cur_time,
    Fortran::runtime::Terminator terminator) {
  buffer[0] = '\0';
  terminator.Crash("fdate is not supported.");
}
#endif

#if _REENTRANT || _POSIX_C_SOURCE >= 199506L
// System is posix-compliant and has getlogin_r
#include <unistd.h>
#endif

extern "C" {

namespace Fortran::runtime {

void GetUsernameEnvVar(const char *envName, char *arg, std::int64_t length) {
  Descriptor name{*Descriptor::Create(
      1, std::strlen(envName) + 1, const_cast<char *>(envName), 0)};
  Descriptor value{*Descriptor::Create(1, length, arg, 0)};

  RTNAME(GetEnvVariable)
  (name, &value, nullptr, false, nullptr, __FILE__, __LINE__);
}
namespace io {
// SUBROUTINE FLUSH(N)
//   FLUSH N
// END
void FORTRAN_PROCEDURE_NAME(flush)(const int &unit) {
  Cookie cookie{IONAME(BeginFlush)(unit, __FILE__, __LINE__)};
  IONAME(EndIoStatement)(cookie);
}
} // namespace io

// CALL FDATE(DATE)
void FORTRAN_PROCEDURE_NAME(fdate)(char *arg, std::int64_t length) {
  // Day Mon dd hh:mm:ss yyyy\n\0 is 26 characters, e.g.
  // Tue May 26 21:51:03 2015\n\0
  char str[26];
  // Insufficient space, fill with spaces and return.
  if (length < 24) {
    std::memset(arg, ' ', length);
    return;
  }

  Terminator terminator{__FILE__, __LINE__};
  std::time_t current_time;
  std::time(&current_time);
  CtimeBuffer(str, sizeof(str), current_time, terminator);

  // Pad space on the last two byte `\n\0`, start at index 24 included.
  CopyAndPad(arg, str, length, 24);
}

// RESULT = IARGC()
std::int32_t FORTRAN_PROCEDURE_NAME(iargc)() { return RTNAME(ArgumentCount)(); }

// CALL GETARG(N, ARG)
void FORTRAN_PROCEDURE_NAME(getarg)(
    std::int32_t &n, char *arg, std::int64_t length) {
  Descriptor value{*Descriptor::Create(1, length, arg, 0)};
  (void)RTNAME(GetCommandArgument)(
      n, &value, nullptr, nullptr, __FILE__, __LINE__);
}

// CALL GETLOG(USRNAME)
void FORTRAN_PROCEDURE_NAME(getlog)(char *arg, std::int64_t length) {
#if _REENTRANT || _POSIX_C_SOURCE >= 199506L
  if (length >= 1 && getlogin_r(arg, length) == 0) {
    auto loginLen{std::strlen(arg)};
    std::memset(
        arg + loginLen, ' ', static_cast<std::size_t>(length) - loginLen);
    return;
  }
#endif
#if _WIN32
  GetUsernameEnvVar("USERNAME", arg, length);
#else
  GetUsernameEnvVar("LOGNAME", arg, length);
#endif
}

} // namespace Fortran::runtime
} // extern "C"
