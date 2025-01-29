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
#include "flang/Runtime/entry-names.h"
#include "flang/Runtime/io-api.h"
#include <chrono>
#include <cstring>
#include <ctime>
#include <signal.h>
#include <thread>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <synchapi.h>

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

#ifndef _WIN32
// posix-compliant and has getlogin_r and F_OK
#include <unistd.h>
#else
#include <direct.h>
#endif

extern "C" {

namespace Fortran::runtime {

gid_t RTNAME(GetGID)() {
#ifdef _WIN32
  // Group IDs don't exist on Windows, return 1 to avoid errors
  return 1;
#else
  return getgid();
#endif
}

uid_t RTNAME(GetUID)() {
#ifdef _WIN32
  // User IDs don't exist on Windows, return 1 to avoid errors
  return 1;
#else
  return getuid();
#endif
}

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

std::intptr_t RTNAME(Malloc)(std::size_t size) {
  return reinterpret_cast<std::intptr_t>(std::malloc(size));
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

void RTNAME(Free)(std::intptr_t ptr) {
  std::free(reinterpret_cast<void *>(ptr));
}

std::int64_t RTNAME(Signal)(std::int64_t number, void (*handler)(int)) {
  // using auto for portability:
  // on Windows, this is a void *
  // on POSIX, this has the same type as handler
  auto result = signal(number, handler);

  // GNU defines the intrinsic as returning an integer, not a pointer. So we
  // have to reinterpret_cast
  return static_cast<int64_t>(reinterpret_cast<std::uintptr_t>(result));
}

// CALL SLEEP(SECONDS)
void RTNAME(Sleep)(std::int64_t seconds) {
  // ensure that conversion to unsigned makes sense,
  // sleep(0) is an immidiate return anyway
  if (seconds < 1) {
    return;
  }
#if _WIN32
  Sleep(seconds * 1000);
#else
  sleep(seconds);
#endif
}

// TODO: not supported on Windows
#ifndef _WIN32
std::int64_t FORTRAN_PROCEDURE_NAME(access)(const char *name,
    std::int64_t nameLength, const char *mode, std::int64_t modeLength) {
  std::int64_t ret{-1};
  if (nameLength <= 0 || modeLength <= 0 || !name || !mode) {
    return ret;
  }

  // ensure name is null terminated
  char *newName{nullptr};
  if (name[nameLength - 1] != '\0') {
    newName = static_cast<char *>(std::malloc(nameLength + 1));
    std::memcpy(newName, name, nameLength);
    newName[nameLength] = '\0';
    name = newName;
  }

  // calculate mode
  bool read{false};
  bool write{false};
  bool execute{false};
  bool exists{false};
  int imode{0};

  for (std::int64_t i = 0; i < modeLength; ++i) {
    switch (mode[i]) {
    case 'r':
      read = true;
      break;
    case 'w':
      write = true;
      break;
    case 'x':
      execute = true;
      break;
    case ' ':
      exists = true;
      break;
    default:
      // invalid mode
      goto cleanup;
    }
  }
  if (!read && !write && !execute && !exists) {
    // invalid mode
    goto cleanup;
  }

  if (!read && !write && !execute) {
    imode = F_OK;
  } else {
    if (read) {
      imode |= R_OK;
    }
    if (write) {
      imode |= W_OK;
    }
    if (execute) {
      imode |= X_OK;
    }
  }
  ret = access(name, imode);

cleanup:
  if (newName) {
    free(newName);
  }
  return ret;
}
#endif

// CHDIR(DIR)
int RTNAME(Chdir)(const char *name) {
// chdir alias seems to be deprecated on Windows.
#ifndef _WIN32
  return chdir(name);
#else
  return _chdir(name);
#endif
}

} // namespace Fortran::runtime
} // extern "C"
