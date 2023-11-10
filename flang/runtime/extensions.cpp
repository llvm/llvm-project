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
#include "flang/Runtime/command.h"
#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/io-api.h"
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <lmcons.h> // UNLEN=256
#include <wchar.h> // wchar_t cast to LPWSTR
#pragma comment(lib, "Advapi32.lib") // Link Advapi32.lib for GetUserName
#define LOGIN_NAME_MAX UNLEN

inline int getlogin_r(char *buf, size_t bufSize) {
  wchar_t w_username[UNLEN + 1];
  DWORD namelen = sizeof(w_username) / sizeof(w_username[0]);

  if (GetUserName(w_username, &namelen)) {
    // Convert the wchar_t string to a regular C string
    if (wcstombs(buf, w_username, UNLEN + 1) == -1) {
      // Conversion failed
      return -1;
    }
    return (buf[0] == 0 ? -1 : 0);
  } else {
    return -1;
  }
  return -1;
}
#elif _REENTRANT || _POSIX_C_SOURCE >= 199506L
// System is posix-compliant and has getlogin_r
#include <unistd.h>
#else
// System is not posix-compliant
inline int getlogin_r(char *buf, size_t bufsize) { return -1; }
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

// CALL GETARG(N, ARG)
void FORTRAN_PROCEDURE_NAME(getarg)(
    std::int32_t &n, std::int8_t *arg, std::int64_t length) {
  Descriptor value{*Descriptor::Create(1, length, arg, 0)};
  (void)RTNAME(GetCommandArgument)(
      n, &value, nullptr, nullptr, __FILE__, __LINE__);
}

void FORTRAN_PROCEDURE_NAME(getlog)(std::int8_t *arg, std::int64_t length) {
  std::array<char, LOGIN_NAME_MAX + 1> str;
  int error = getlogin_r(str.data(), str.size());
  assert(error == 0 && "getlogin_r returned an error");

  // Trim space from right/end
  int i = str.size();
  while (' ' == str[--i]) {
    str[i] = 0;
  }
  strncpy(reinterpret_cast<char *>(arg), str.data(), length);
}

} // namespace Fortran::runtime
} // extern "C"
