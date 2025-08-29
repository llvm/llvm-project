//===-- lib/runtime/command.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/command.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang-rt/runtime/environment.h"
#include "flang-rt/runtime/stat.h"
#include "flang-rt/runtime/terminator.h"
#include "flang-rt/runtime/tools.h"
#include <cerrno>
#include <cstdlib>
#include <limits>

#ifdef _WIN32
#include "flang/Common/windows-include.h"
#include <direct.h>
#define getcwd _getcwd
#define unlink _unlink
#define PATH_MAX MAX_PATH

#ifdef _MSC_VER
// On Windows GetCurrentProcessId returns a DWORD aka uint32_t
#include <processthreadsapi.h>
inline pid_t getpid() { return GetCurrentProcessId(); }
#endif
#else
#include <unistd.h> //getpid() unlink()

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif
#endif

namespace Fortran::runtime {
std::int32_t RTNAME(ArgumentCount)() {
  int argc{executionEnvironment.argc};
  if (argc > 1) {
    // C counts the command name as one of the arguments, but Fortran doesn't.
    return argc - 1;
  }
  return 0;
}

pid_t RTNAME(GetPID)() { return getpid(); }

// Returns the length of the \p string. Assumes \p string is valid.
static std::int64_t StringLength(const char *string) {
  std::size_t length{std::strlen(string)};
  if (length <= std::numeric_limits<std::int64_t>::max())
    return static_cast<std::int64_t>(length);
  return 0;
}

static void FillWithSpaces(const Descriptor &value, std::size_t offset = 0) {
  if (offset < value.ElementBytes()) {
    runtime::memset(
        value.OffsetElement(offset), ' ', value.ElementBytes() - offset);
  }
}

static std::int32_t CheckAndCopyCharsToDescriptor(const Descriptor *value,
    const char *rawValue, const Descriptor *errmsg, std::size_t &offset) {
  bool haveValue{IsValidCharDescriptor(value)};

  std::int64_t len{StringLength(rawValue)};
  if (len <= 0) {
    if (haveValue) {
      FillWithSpaces(*value);
    }
    return ToErrmsg(errmsg, StatMissingArgument);
  }

  std::int32_t stat{StatOk};
  if (haveValue) {
    stat = CopyCharsToDescriptor(*value, rawValue, len, errmsg, offset);
  }

  offset += len;
  return stat;
}

template <int KIND> struct FitsInIntegerKind {
  bool operator()([[maybe_unused]] std::int64_t value) {
    if constexpr (KIND >= 8) {
      return true;
    } else {
      return value <= std::numeric_limits<Fortran::runtime::CppTypeFor<
                          Fortran::common::TypeCategory::Integer, KIND>>::max();
    }
  }
};

static bool FitsInDescriptor(
    const Descriptor *length, std::int64_t value, Terminator &terminator) {
  auto typeCode{length->type().GetCategoryAndKind()};
  int kind{typeCode->second};
  return Fortran::runtime::ApplyIntegerKind<FitsInIntegerKind, bool>(
      kind, terminator, value);
}

std::int32_t RTNAME(GetCommandArgument)(std::int32_t n, const Descriptor *value,
    const Descriptor *length, const Descriptor *errmsg, const char *sourceFile,
    int line) {
  Terminator terminator{sourceFile, line};

  if (value) {
    RUNTIME_CHECK(terminator, IsValidCharDescriptor(value));
    FillWithSpaces(*value);
  }

  // Store 0 in case we error out later on.
  if (length) {
    RUNTIME_CHECK(terminator, IsValidIntDescriptor(length));
    StoreIntToDescriptor(length, 0, terminator);
  }

  if (n < 0 || n >= executionEnvironment.argc) {
    return ToErrmsg(errmsg, StatInvalidArgumentNumber);
  }

  const char *arg{executionEnvironment.argv[n]};
  std::int64_t argLen{StringLength(arg)};
  if (argLen <= 0) {
    return ToErrmsg(errmsg, StatMissingArgument);
  }

  if (length && FitsInDescriptor(length, argLen, terminator)) {
    StoreIntToDescriptor(length, argLen, terminator);
  }

  if (value) {
    return CopyCharsToDescriptor(*value, arg, argLen, errmsg);
  }

  return StatOk;
}

std::int32_t RTNAME(GetCommand)(const Descriptor *value,
    const Descriptor *length, const Descriptor *errmsg, const char *sourceFile,
    int line) {
  Terminator terminator{sourceFile, line};

  if (value) {
    RUNTIME_CHECK(terminator, IsValidCharDescriptor(value));
  }

  // Store 0 in case we error out later on.
  if (length) {
    RUNTIME_CHECK(terminator, IsValidIntDescriptor(length));
    StoreIntToDescriptor(length, 0, terminator);
  }

  auto shouldContinue = [&](std::int32_t stat) -> bool {
    // We continue as long as everything is ok OR the value descriptor is
    // too short, but we still need to compute the length.
    return stat == StatOk || (length && stat == StatValueTooShort);
  };

  std::size_t offset{0};

  if (executionEnvironment.argc == 0) {
    return CheckAndCopyCharsToDescriptor(value, "", errmsg, offset);
  }

  // value = argv[0]
  std::int32_t stat{CheckAndCopyCharsToDescriptor(
      value, executionEnvironment.argv[0], errmsg, offset)};
  if (!shouldContinue(stat)) {
    return stat;
  }

  // value += " " + argv[1:n]
  for (std::int32_t i{1}; i < executionEnvironment.argc; ++i) {
    stat = CheckAndCopyCharsToDescriptor(value, " ", errmsg, offset);
    if (!shouldContinue(stat)) {
      return stat;
    }

    stat = CheckAndCopyCharsToDescriptor(
        value, executionEnvironment.argv[i], errmsg, offset);
    if (!shouldContinue(stat)) {
      return stat;
    }
  }

  if (length && FitsInDescriptor(length, offset, terminator)) {
    StoreIntToDescriptor(length, offset, terminator);
  }

  // value += spaces for padding
  if (value) {
    FillWithSpaces(*value, offset);
  }

  return stat;
}

static std::size_t LengthWithoutTrailingSpaces(const Descriptor &d) {
  std::size_t s{d.ElementBytes()}; // This can be 0.
  while (s != 0 && *d.OffsetElement(s - 1) == ' ') {
    --s;
  }
  return s;
}

std::int32_t RTNAME(GetEnvVariable)(const Descriptor &name,
    const Descriptor *value, const Descriptor *length, bool trim_name,
    const Descriptor *errmsg, const char *sourceFile, int line) {
  Terminator terminator{sourceFile, line};

  if (value) {
    RUNTIME_CHECK(terminator, IsValidCharDescriptor(value));
    FillWithSpaces(*value);
  }

  // Store 0 in case we error out later on.
  if (length) {
    RUNTIME_CHECK(terminator, IsValidIntDescriptor(length));
    StoreIntToDescriptor(length, 0, terminator);
  }

  const char *rawValue{nullptr};
  std::size_t nameLength{
      trim_name ? LengthWithoutTrailingSpaces(name) : name.ElementBytes()};
  if (nameLength != 0) {
    rawValue = executionEnvironment.GetEnv(
        name.OffsetElement(), nameLength, terminator);
  }
  if (!rawValue) {
    return ToErrmsg(errmsg, StatMissingEnvVariable);
  }

  std::int64_t varLen{StringLength(rawValue)};
  if (length && FitsInDescriptor(length, varLen, terminator)) {
    StoreIntToDescriptor(length, varLen, terminator);
  }

  if (value) {
    return CopyCharsToDescriptor(*value, rawValue, varLen, errmsg);
  }
  return StatOk;
}

std::int32_t RTNAME(GetCwd)(
    const Descriptor &cwd, const char *sourceFile, int line) {
  Terminator terminator{sourceFile, line};

  RUNTIME_CHECK(terminator, IsValidCharDescriptor(&cwd));

  char *buf{(char *)AllocateMemoryOrCrash(terminator, PATH_MAX)};

  if (!getcwd(buf, PATH_MAX)) {
    return StatMissingCurrentWorkDirectory;
  }

  std::int64_t strLen{StringLength(buf)};
  std::int32_t status{CopyCharsToDescriptor(cwd, buf, strLen)};

  std::free(buf);
  return status;
}

std::int32_t RTNAME(Hostnm)(
    const Descriptor &res, const char *sourceFile, int line) {
  Terminator terminator{sourceFile, line};

  RUNTIME_CHECK(terminator, IsValidCharDescriptor(&res));

  char buf[256];
  std::int32_t status{0};

  // Fill the output with spaces. Upon success, CopyCharsToDescriptor()
  // will overwrite part of the string with the result, so we'll end up
  // with a padded string. If we fail to obtain the host name, we return
  // the string of all spaces, which is the original gfortran behavior.
  FillWithSpaces(res);

#ifdef _WIN32

  DWORD dwSize{sizeof(buf)};

  // Note: Winsock has gethostname(), but use Win32 API GetComputerNameEx(),
  // in order to avoid adding dependency on Winsock.
  if (!GetComputerNameExA(ComputerNameDnsHostname, buf, &dwSize)) {
    status = GetLastError();
  }

#else

  if (gethostname(buf, sizeof(buf)) < 0) {
    status = errno;
  }

#endif

  if (status == 0) {
    std::int64_t strLen{StringLength(buf)};
    status = CopyCharsToDescriptor(res, buf, strLen);

    // Note: if the result string is too short, then we'll return partial
    // host name with "too short" error status.
  }

  return status;
}

std::int32_t RTNAME(PutEnv)(
    const char *str, size_t str_length, const char *sourceFile, int line) {
  Terminator terminator{sourceFile, line};

  RUNTIME_CHECK(terminator, str && str_length);

  // Note: don't trim the input string, because the user should be able
  // to set the value to all spaces if necessary.

  // While Fortran's putenv() extended intrinsic sementics loosly follow
  // Linux C library putenv(), don't actually use putenv() on Linux, because
  // it takes the passed string pointer and incorporates it into the
  // environment without copy. To make this safe, one would have to copy
  // the passed string into some allocated memory, but then there's no good
  // way to deallocate it. Instead, use the implementation from
  // ExecutionEnvironment, which does the right thing for both Windows and
  // Linux.

  std::int32_t status{0};

  // Split the input string into name and value substrings. Note:
  // if input string is in "name=value" form, then we set variable "name" with
  // value "value". If the input string is in "name=" form, then we delete
  // the variable "name".

  const char *str_end = str + str_length;
  const char *str_sep = std::find(str, str_end, '=');
  if (str_sep == str_end) {
    // No separator, invalid input string
    status = EINVAL;
  } else if ((str_sep + 1) == str_end) {
    // "name=" form, which means we need to delete this variable
    status = executionEnvironment.UnsetEnv(str, str_sep - str, terminator);
  } else {
    // Example: consider str "abc=defg", str_length = 8
    //
    // addr:     05 06 07 08 09 10 11 12 13
    // str@addr:  a  b  c  =  d  e  f  g ??
    //
    // str = 5, str_end = 13, str_sep = 8, name length: str_sep - str = 3
    // value ptr: str_sep + 1 = 9, value length: 4
    //
    status = executionEnvironment.SetEnv(
        str, str_sep - str, str_sep + 1, str_end - str_sep - 1, terminator);
  }

  return status;
}

std::int32_t RTNAME(Unlink)(
    const char *str, size_t strLength, const char *sourceFile, int line) {
  Terminator terminator{sourceFile, line};

  auto pathLength = TrimTrailingSpaces(str, strLength);
  auto path = SaveDefaultCharacter(str, pathLength, terminator);

  std::int32_t status{0};

  if (unlink(path.get()) != 0) {
    status = errno;
  }

  return status;
}

} // namespace Fortran::runtime
