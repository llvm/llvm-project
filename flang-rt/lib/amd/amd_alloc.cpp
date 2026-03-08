//===-- lib/amd/amd_alloc.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#warning "amd_alloc.cpp was part of the build"

#define ALLOC_INITIAL_SIZE (128 * 1024 * 1024)
#define ALLOC_BLOCK_SIZE (512)

#define ALLOC_DEBUG 1

#include "flang/Runtime/AMD/amd_alloc.h"
#include "flang-rt/runtime/allocator-registry.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang/Support/Fortran.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>

namespace Fortran::runtime::amd {

static bool debugEnabled;

// ====================== OPENMP ======================

// Declare OpenMP memory management routines to avoid importing
// definitions via "omp.h" (and thus create a dependency to the
// OpenMP runtime library code).
extern "C" int omp_get_default_device(void);
extern "C" void *omp_target_alloc(std::size_t, int);
extern "C" void omp_target_free(void *, int);

static void *OpenMPAlloc(std::size_t AllocationSize, std::int64_t *) {
#if ALLOC_DEBUG
  if (debugEnabled) {
    std::fprintf(stderr, "[AMD_ALLOC] %s(%zu) (%s:%d)\n", __PRETTY_FUNCTION__,
        AllocationSize, __FILE__, __LINE__);
  }
#endif
  int device{omp_get_default_device()};
  void *pointer{omp_target_alloc(AllocationSize, device)};
#if ALLOC_DEBUG
  if (debugEnabled) {
    std::fprintf(stderr,
        "[AMD_ALLOC] pointer of size %zu allocated at %p"
        " on device %d.\n",
        AllocationSize, pointer, device);
  }
#endif
  return pointer;
}

void OpenMPFree(void *pointer) {
#if ALLOC_DEBUG
  if (debugEnabled) {
    std::fprintf(stderr, "[AMD_ALLOC] %s(%p) (%s:%d)\n", __PRETTY_FUNCTION__,
        pointer, __FILE__, __LINE__);
  }
#endif
  omp_target_free(pointer, omp_get_default_device());
}

static void registerOpenMPAllocator() {
#if ALLOC_DEBUG
  if (debugEnabled) {
    std::fprintf(
        stderr, "[AMD_ALLOC] registering OpenMP device memory allocator\n");
  }
#endif // ALLOC_DEBUG
  allocatorRegistry.Register(1, {&OpenMPAlloc, &OpenMPFree});
}

static const char *getStringFromEnvironment(
    const char *envirable, const char *defaultValue = "") {
  if (auto value{std::getenv(envirable)}) {
    return value;
  }
  return defaultValue;
}

static int getIntFromEnvironment(
    const char *envirable, const int defaultValue = 0) {
  int result = defaultValue;
  char *end;
  if (auto value{std::getenv(envirable)}) {
    auto number{std::strtoul(value, &end, 10)};
    if (number > 0 && number < std::numeric_limits<int>::max() &&
        *end == '\0') {
      result = number;
    } else {
      std::fprintf(stderr, "Fortran runtime: %s=%s is invalid; ignored\n",
          envirable, value);
    }
  }
  return result;
}

static std::pair<std::string_view, std::string_view> splitAtColon(
    std::string_view str) {
  const char *data = str.data();
  size_t len = str.size();
  const char *colon = static_cast<const char *>(std::memchr(data, ':', len));
  if (!colon) {
    return {str, std::string_view()};
  }
  size_t colon_pos = colon - data;
  return {std::string_view(data, colon_pos),
      std::string_view(colon + 1, len - colon_pos - 1)};
}

extern "C" {
void RTDEF(AMDRegisterAllocator)() {
#if ALLOC_DEBUG
  debugEnabled = false;
  if (getIntFromEnvironment("AMD_ALLOC_DEBUG", 0) != 0) {
    debugEnabled = true;
  }
  if (debugEnabled) {
    std::fprintf(stderr, "[AMD_ALLOC] %s (%s:%d)\n", __PRETTY_FUNCTION__,
        __FILE__, __LINE__);
  }
#endif

  // Determine what allocator to register via very simplistic parsing of syntax
  // ALLOCATOR:MEMORY_KIND.  Proper values are: OPENMP
  const char *allocator_env = getStringFromEnvironment("AMD_ALLOC", "openmp");
  char allocator[256];
  std::strncpy(allocator, allocator_env, sizeof(allocator) - 1);
  allocator[sizeof(allocator) - 1] = '\0';
  for (char *p = allocator; *p; ++p)
    *p = ::toupper(*p);
#if ALLOC_DEBUG
  if (debugEnabled) {
    std::fprintf(stderr, "[AMD_ALLOC] requesting allocator: %s\n", allocator);
  }
#endif // ALLOC_DEBUG
  std::pair<std::string_view, std::string_view> allocSpec{
      splitAtColon(allocator)};
  if (allocSpec.first != "OPENMP") {
    std::fprintf(stderr,
        "[AMD_ALLOC] warning: wrong allocator ('%.*s') specified for AMD "
        "allocator, using 'OPENMP' instead.\n",
        static_cast<int>(allocSpec.first.size()), allocSpec.first.data());
    allocSpec.first = "OPENMP";
  }
  if (allocSpec.first == "OPENMP") {
    if (!allocSpec.second.empty()) {
      std::fprintf(stderr,
          "[AMD_ALLOC] warning: OpenMP allocator does not "
          "accept allocator option type '%.*s'.\n",
          static_cast<int>(allocSpec.second.size()), allocSpec.second.data());
    }
    registerOpenMPAllocator();
  }
}

void RTDEF(AMDAllocatableSetAllocIdx)(Descriptor &descriptor, int pos) {
  if (descriptor.IsAllocatable() && !descriptor.IsAllocated()) {
#if ALLOC_DEBUG
    if (debugEnabled) {
      std::fprintf(
          stderr, "[AMD_ALLOC] AMDAllocatableSetAllocIdx = %d \n", pos);
    }
#endif
    descriptor.SetAllocIdx(pos);
  }
}
} // extern "C"

} // namespace Fortran::runtime::amd
