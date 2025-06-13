//===-- lib/runtime/assign_omp.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/assign.h"
#include "flang-rt/runtime/assign-impl.h"
#include "flang-rt/runtime/derived.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang-rt/runtime/stat.h"
#include "flang-rt/runtime/terminator.h"
#include "flang-rt/runtime/tools.h"
#include "flang-rt/runtime/type-info.h"

#include <omp.h>

namespace Fortran::runtime {

RT_API_ATTRS static void Assign(Descriptor &to, const Descriptor &from,
    Terminator &terminator, int flags, int32_t omp_device) {
  std::size_t toElementBytes{to.ElementBytes()};
  std::size_t fromElementBytes{from.ElementBytes()};
  std::size_t toElements{to.Elements()};
  std::size_t fromElements{from.Elements()};

  if (toElementBytes != fromElementBytes)
    terminator.Crash("Assign: toElementBytes != fromElementBytes");
  if (toElements != fromElements)
    terminator.Crash("Assign: toElements != fromElements");

  void *host_to_ptr = to.raw().base_addr;
  void *host_from_ptr = from.raw().base_addr;
  size_t length = toElements * toElementBytes;

  printf("assign length: %zu\n", length);

  if (!omp_target_is_present(host_to_ptr, omp_device))
    terminator.Crash("Assign: !omp_target_is_present(host_to_ptr, omp_device)");
  if (!omp_target_is_present(host_from_ptr, omp_device))
    terminator.Crash(
        "Assign: !omp_target_is_present(host_from_ptr, omp_device)");

  printf("host_to_ptr: %p\n", host_to_ptr);
#pragma omp target data use_device_ptr(host_to_ptr, host_from_ptr) device(omp_device)
  {
    printf("device_to_ptr: %p\n", host_to_ptr);
    // TODO do we need to handle overlapping memory? does this function do that?
    omp_target_memcpy(host_to_ptr, host_from_ptr, length, /*dst_offset*/ 0,
        /*src_offset*/ 0, /*dst*/ omp_device, /*src*/ omp_device);
  }

  return;
}

extern "C" {
RT_EXT_API_GROUP_BEGIN
void RTDEF(Assign_omp)(Descriptor &to, const Descriptor &from,
    const char *sourceFile, int sourceLine, int32_t omp_device) {
  Terminator terminator{sourceFile, sourceLine};
  // All top-level defined assignments can be recognized in semantics and
  // will have been already been converted to calls, so don't check for
  // defined assignment apart from components.
  Assign(to, from, terminator,
      MaybeReallocate | NeedFinalization | ComponentCanBeDefinedAssignment,
      omp_device);
}
} // extern "C"

}