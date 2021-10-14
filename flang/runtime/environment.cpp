//===-- runtime/environment.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "environment.h"
#include "tools.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace Fortran::runtime {

ExecutionEnvironment executionEnvironment;

std::optional<Convert> GetConvertFromString(const char *x, std::size_t n) {
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
    return std::nullopt;
  }
}

void ExecutionEnvironment::Configure(
    int ac, const char *av[], const char *env[]) {
  argc = ac;
  argv = av;
  envp = env;
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

  // TODO: Set RP/ROUND='PROCESSOR_DEFINED' from environment
}

const char *ExecutionEnvironment::GetEnv(
    const char *name, std::size_t name_length) {
  if (!envp) {
    // TODO: Ask std::getenv.
    return nullptr;
  }

  // envp is an array of strings of the form "name=value".
  for (const char **var{envp}; *var != nullptr; ++var) {
    const char *eq{std::strchr(*var, '=')};
    if (!eq) {
      // Found a malformed environment string, just ignore it.
      continue;
    }
    if (static_cast<std::size_t>(eq - *var) != name_length) {
      continue;
    }
    if (std::memcmp(*var, name, name_length) == 0) {
      return eq + 1;
    }
  }
  return nullptr;
}
} // namespace Fortran::runtime
