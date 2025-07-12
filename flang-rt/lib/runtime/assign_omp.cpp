//===-- lib/runtime/assign_omp.cpp ----------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang-rt/runtime/assign-impl.h"
#include "flang-rt/runtime/derived.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang-rt/runtime/stat.h"
#include "flang-rt/runtime/terminator.h"
#include "flang-rt/runtime/tools.h"
#include "flang-rt/runtime/type-info.h"
#include "flang/Runtime/assign.h"

#include <omp.h>

namespace Fortran::runtime {
namespace omp {

typedef int32_t OMPDeviceTy;

template <typename T> static T *getDevicePtr(T *anyPtr, OMPDeviceTy ompDevice) {
  auto voidAnyPtr = reinterpret_cast<void *>(anyPtr);
  // If not present on the device it should already be a device ptr
  if (!omp_target_is_present(voidAnyPtr, ompDevice))
    return anyPtr;
  T *device_ptr = omp_get_mapped_ptr(anyPtr, ompDevice);
  return device_ptr;
}

RT_API_ATTRS static void Assign(Descriptor &to, const Descriptor &from,
    Terminator &terminator, int flags, OMPDeviceTy omp_device) {
  std::size_t toElementBytes{to.ElementBytes()};
  std::size_t fromElementBytes{from.ElementBytes()};
  std::size_t toElements{to.Elements()};
  std::size_t fromElements{from.Elements()};

  if (toElementBytes != fromElementBytes)
    terminator.Crash("Assign: toElementBytes != fromElementBytes");
  if (toElements != fromElements)
    terminator.Crash("Assign: toElements != fromElements");

  // Get base addresses and calculate length
  void *to_base = to.raw().base_addr;
  void *from_base = from.raw().base_addr;
  size_t length = toElements * toElementBytes;

  // Get device pointers after ensuring data is on device
  void *to_ptr = getDevicePtr(to_base, omp_device);
  void *from_ptr = getDevicePtr(from_base, omp_device);

  // Perform copy between device pointers
  int result = omp_target_memcpy(to_ptr, from_ptr, length,
      /*dst_offset*/ 0, /*src_offset*/ 0, omp_device, omp_device);

  if (result != 0)
    terminator.Crash("Assign: omp_target_memcpy failed");
  return;
}

extern "C" {
RT_EXT_API_GROUP_BEGIN
void RTDEF(Assign_omp)(Descriptor &to, const Descriptor &from,
    const char *sourceFile, int sourceLine, omp::OMPDeviceTy omp_device) {
  Terminator terminator{sourceFile, sourceLine};
  Fortran::runtime::omp::Assign(to, from, terminator,
      MaybeReallocate | NeedFinalization | ComponentCanBeDefinedAssignment,
      omp_device);
}

} // extern "C"
} // namespace omp
} // namespace Fortran::runtime
