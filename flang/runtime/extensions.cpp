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
#include "tools.h"
#include "flang/Runtime/command.h"
#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/io-api.h"

#if _REENTRANT || _POSIX_C_SOURCE >= 199506L
// System is posix-compliant and has getlogin_r
#include <unistd.h>
#endif

extern "C" {

namespace Fortran::runtime {

void GetUsernameEnvVar(
    const char *envName, std::byte *arg, std::int64_t length) {
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

// RESULT = IARGC()
std::int32_t FORTRAN_PROCEDURE_NAME(iargc)() { return RTNAME(ArgumentCount)(); }

// CALL GETARG(N, ARG)
void FORTRAN_PROCEDURE_NAME(getarg)(
    std::int32_t &n, std::int8_t *arg, std::int64_t length) {
  Descriptor value{*Descriptor::Create(1, length, arg, 0)};
  (void)RTNAME(GetCommandArgument)(
      n, &value, nullptr, nullptr, __FILE__, __LINE__);
}

// CALL GETLOG(USRNAME)
void FORTRAN_PROCEDURE_NAME(getlog)(std::byte *arg, std::int64_t length) {
#if _REENTRANT || _POSIX_C_SOURCE >= 199506L
  const int nameMaxLen{LOGIN_NAME_MAX + 1};
  char str[nameMaxLen];

  int error{getlogin_r(str, nameMaxLen)};
  if (error == 0) {
    // no error: find first \0 in string then pad from there
    CopyAndPad(reinterpret_cast<char *>(arg), str, length, std::strlen(str));
  } else {
    // error occur: get username from environment variable
    GetUsernameEnvVar("LOGNAME", arg, length);
  }
#elif _WIN32
  // Get username from environment to avoid link to Advapi32.lib
  GetUsernameEnvVar("USERNAME", arg, length);
#else
  GetUsernameEnvVar("LOGNAME", arg, length);
#endif
}

} // namespace Fortran::runtime
} // extern "C"
