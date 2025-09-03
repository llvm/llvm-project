//===-- include/flang-rt/runtime/environment.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FLANG_RT_RUNTIME_ENVIRONMENT_H_
#define FLANG_RT_RUNTIME_ENVIRONMENT_H_

#include "flang/Common/optional.h"
#include "flang/Decimal/decimal.h"
#include "flang/Runtime/entry-names.h"

struct EnvironmentDefaultList;

namespace Fortran::runtime {

class Terminator;

RT_OFFLOAD_VAR_GROUP_BEGIN
#if FLANG_BIG_ENDIAN
constexpr bool isHostLittleEndian{false};
#elif FLANG_LITTLE_ENDIAN
constexpr bool isHostLittleEndian{true};
#else
#error host endianness is not known
#endif
RT_OFFLOAD_VAR_GROUP_END

// External unformatted I/O data conversions
enum class Convert { Unknown, Native, LittleEndian, BigEndian, Swap };

RT_API_ATTRS common::optional<Convert> GetConvertFromString(
    const char *, std::size_t);

struct ExecutionEnvironment {

  typedef void (*ConfigEnvCallbackPtr)(
      int, const char *[], const char *[], const EnvironmentDefaultList *);

#if !defined(_OPENMP)
  // FIXME: https://github.com/llvm/llvm-project/issues/84942
  constexpr
#endif
      ExecutionEnvironment(){};
  void Configure(int argc, const char *argv[], const char *envp[],
      const EnvironmentDefaultList *envDefaults);

  // Maximum number of registered pre and post ExecutionEnvironment::Configure()
  // callback functions.
  static constexpr int nConfigEnvCallback{8};

  const char *GetEnv(
      const char *name, std::size_t name_length, const Terminator &terminator);

  std::int32_t SetEnv(const char *name, std::size_t name_length,
      const char *value, std::size_t value_length,
      const Terminator &terminator);

  std::int32_t UnsetEnv(
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
  bool checkPointerDeallocation{true}; // FORT_CHECK_POINTER_DEALLOCATION

  enum InternalDebugging { WorkQueue = 1 };
  int internalDebugging{0}; // FLANG_RT_DEBUG

  // CUDA related variables
  std::size_t cudaStackLimit{0}; // ACC_OFFLOAD_STACK_SIZE
  bool cudaDeviceIsManaged{false}; // NV_CUDAFOR_DEVICE_IS_MANAGED
};

RT_OFFLOAD_VAR_GROUP_BEGIN
extern RT_VAR_ATTRS ExecutionEnvironment executionEnvironment;
RT_OFFLOAD_VAR_GROUP_END

// ExecutionEnvironment::Configure() allows for optional callback functions
// to be run pre and post the core logic.
// Most likely scenario is when a user supplied constructor function is
// run prior to _QQmain calling RTNAME(ProgramStart)().

extern "C" {
bool RTNAME(RegisterConfigureEnv)(ExecutionEnvironment::ConfigEnvCallbackPtr,
    ExecutionEnvironment::ConfigEnvCallbackPtr);
}

} // namespace Fortran::runtime
#endif // FLANG_RT_RUNTIME_ENVIRONMENT_H_
