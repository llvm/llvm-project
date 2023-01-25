//===-- runtime/environment.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_ENVIRONMENT_H_
#define FORTRAN_RUNTIME_ENVIRONMENT_H_

#include "flang/Decimal/decimal.h"
#include <optional>

struct EnvironmentDefaultList;

namespace Fortran::runtime {

class Terminator;

#if FLANG_BIG_ENDIAN
constexpr bool isHostLittleEndian{false};
#elif FLANG_LITTLE_ENDIAN
constexpr bool isHostLittleEndian{true};
#else
#error host endianness is not known
#endif

// External unformatted I/O data conversions
enum class Convert { Unknown, Native, LittleEndian, BigEndian, Swap };

std::optional<Convert> GetConvertFromString(const char *, std::size_t);

struct ExecutionEnvironment {
  constexpr ExecutionEnvironment(){};
  void Configure(int argc, const char *argv[], const char *envp[],
      const EnvironmentDefaultList *envDefaults);
  const char *GetEnv(
      const char *name, std::size_t name_length, const Terminator &terminator);

  int argc{0};
  const char **argv{nullptr};
  char **envp{nullptr};

  int listDirectedOutputLineLengthLimit{79}; // FORT_FMT_RECL
  enum decimal::FortranRounding defaultOutputRoundingMode{
      decimal::FortranRounding::RoundNearest}; // RP(==PN)
  Convert conversion{Convert::Unknown}; // FORT_CONVERT
  bool noStopMessage{false}; // NO_STOP_MESSAGE=1 inhibits "Fortran STOP"
  bool defaultUTF8{false}; // DEFAULT_UTF8
};

extern ExecutionEnvironment executionEnvironment;
} // namespace Fortran::runtime

#endif // FORTRAN_RUNTIME_ENVIRONMENT_H_
