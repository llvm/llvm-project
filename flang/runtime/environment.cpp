//===-- runtime/environment.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "environment.h"
#include "environment-default-list.h"
#include "memory.h"
#include "tools.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#ifdef _WIN32
extern char **_environ;
#else
extern char **environ;
#endif

namespace Fortran::runtime {

#ifndef FLANG_RUNTIME_NO_GLOBAL_VAR_DEFS
RT_OFFLOAD_VAR_GROUP_BEGIN
RT_VAR_ATTRS ExecutionEnvironment executionEnvironment;
RT_OFFLOAD_VAR_GROUP_END
#endif // FLANG_RUNTIME_NO_GLOBAL_VAR_DEFS

static void SetEnvironmentDefaults(const EnvironmentDefaultList *envDefaults) {
  if (!envDefaults) {
    return;
  }

  for (int itemIndex = 0; itemIndex < envDefaults->numItems; ++itemIndex) {
    const char *name = envDefaults->item[itemIndex].name;
    const char *value = envDefaults->item[itemIndex].value;
#ifdef _WIN32
    if (auto *x{std::getenv(name)}) {
      continue;
    }
    if (_putenv_s(name, value) != 0) {
#else
    if (setenv(name, value, /*overwrite=*/0) == -1) {
#endif
      Fortran::runtime::Terminator{__FILE__, __LINE__}.Crash(
          std::strerror(errno));
    }
  }
}

RT_OFFLOAD_API_GROUP_BEGIN
Fortran::common::optional<Convert> GetConvertFromString(
    const char *x, std::size_t n) {
  static const char *keywords[]{
      "UNKNOWN", "NATIVE", "LITTLE_ENDIAN", "BIG_ENDIAN", "SWAP", nullptr};
  switch (IdentifyValue(x, n, keywords)) {
  case 0:
    return Convert::Unknown;
  case 1:
    return Convert::Native;
  case 2:
    return Convert::LittleEndian;
  case 3:
    return Convert::BigEndian;
  case 4:
    return Convert::Swap;
  default:
    return Fortran::common::nullopt;
  }
}
RT_OFFLOAD_API_GROUP_END

void ExecutionEnvironment::Configure(int ac, const char *av[],
    const char *env[], const EnvironmentDefaultList *envDefaults) {
  argc = ac;
  argv = av;
  SetEnvironmentDefaults(envDefaults);
#ifdef _WIN32
  envp = _environ;
#else
  envp = environ;
#endif
  listDirectedOutputLineLengthLimit = 79; // PGI default
  defaultOutputRoundingMode =
      decimal::FortranRounding::RoundNearest; // RP(==RN)
  conversion = Convert::Unknown;

  if (auto *x{std::getenv("FORT_FMT_RECL")}) {
    char *end;
    auto n{std::strtol(x, &end, 10)};
    if (n > 0 && n < std::numeric_limits<int>::max() && *end == '\0') {
      listDirectedOutputLineLengthLimit = n;
    } else {
      std::fprintf(
          stderr, "Fortran runtime: FORT_FMT_RECL=%s is invalid; ignored\n", x);
    }
  }

  if (auto *x{std::getenv("FORT_CONVERT")}) {
    if (auto convert{GetConvertFromString(x, std::strlen(x))}) {
      conversion = *convert;
    } else {
      std::fprintf(
          stderr, "Fortran runtime: FORT_CONVERT=%s is invalid; ignored\n", x);
    }
  }

  if (auto *x{std::getenv("NO_STOP_MESSAGE")}) {
    char *end;
    auto n{std::strtol(x, &end, 10)};
    if (n >= 0 && n <= 1 && *end == '\0') {
      noStopMessage = n != 0;
    } else {
      std::fprintf(stderr,
          "Fortran runtime: NO_STOP_MESSAGE=%s is invalid; ignored\n", x);
    }
  }

  if (auto *x{std::getenv("DEFAULT_UTF8")}) {
    char *end;
    auto n{std::strtol(x, &end, 10)};
    if (n >= 0 && n <= 1 && *end == '\0') {
      defaultUTF8 = n != 0;
    } else {
      std::fprintf(
          stderr, "Fortran runtime: DEFAULT_UTF8=%s is invalid; ignored\n", x);
    }
  }

  if (auto *x{std::getenv("FORT_CHECK_POINTER_DEALLOCATION")}) {
    char *end;
    auto n{std::strtol(x, &end, 10)};
    if (n >= 0 && n <= 1 && *end == '\0') {
      checkPointerDeallocation = n != 0;
    } else {
      std::fprintf(stderr,
          "Fortran runtime: FORT_CHECK_POINTER_DEALLOCATION=%s is invalid; "
          "ignored\n",
          x);
    }
  }

  if (auto *x{std::getenv("ACC_OFFLOAD_STACK_SIZE")}) {
    char *end;
    auto n{std::strtoul(x, &end, 10)};
    if (n > 0 && n < std::numeric_limits<std::size_t>::max() && *end == '\0') {
      cudaStackLimit = n;
    } else {
      std::fprintf(stderr,
          "Fortran runtime: ACC_OFFLOAD_STACK_SIZE=%s is invalid; ignored\n",
          x);
    }
  }

  // TODO: Set RP/ROUND='PROCESSOR_DEFINED' from environment
}

const char *ExecutionEnvironment::GetEnv(
    const char *name, std::size_t name_length, const Terminator &terminator) {
  RUNTIME_CHECK(terminator, name && name_length);

  OwningPtr<char> cStyleName{
      SaveDefaultCharacter(name, name_length, terminator)};
  RUNTIME_CHECK(terminator, cStyleName);

  return std::getenv(cStyleName.get());
}
} // namespace Fortran::runtime
