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

#include "flang-rt/runtime/allocator-registry.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang/Runtime/AMD/amd_alloc.h"
#include "flang/Support/Fortran.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

// Deliberately use the C interface of Umpire, as it does not require
// support for exceptions and RTTI, which are not avilable in the
// Fortran runtime build.
// TODO: go back to the correct header that is imported from an
//       installation of Umpire
// #include "umpire/interface/c_fortran/umpire.h"
#include "flang-rt/runtime/amd/umpire/interface/c_fortran/umpire.h"

namespace Fortran::runtime::amd {

extern bool amdAllocatorEnabled; // connects to descriptor.h
static bool debugEnabled;
static umpire_resourcemanager resourceManager;
static umpire_allocator memoryPool;

void *UmpireAlloc(std::size_t AllocationSize, std::int64_t) {
#if ALLOC_DEBUG
  if (debugEnabled) {
    std::fprintf(stderr, "[AMD_ALLOC] %s(%zu) (%s:%d)\n", __PRETTY_FUNCTION__,
        AllocationSize, __FILE__, __LINE__);
  }
#endif
  void *pointer{umpire_allocator_allocate(&memoryPool, AllocationSize)};
#if ALLOC_DEBUG
  if (debugEnabled) {
    std::fprintf(stderr, "[AMD_ALLOC] pointer of size %zu allocated at %p\n",
        AllocationSize, pointer);
  }
#endif
  return pointer;
}

void UmpireFree(void *pointer) {
#if ALLOC_DEBUG
  if (debugEnabled) {
    std::fprintf(stderr, "[AMD_ALLOC] %s(%p) (%s:%d)\n", __PRETTY_FUNCTION__,
        pointer, __FILE__, __LINE__);
  }
#endif
  umpire_allocator_deallocate(&memoryPool, pointer);
}

void registerUmpireAllocator(
    const std::string &pool, const int initialSize, const int blockSize) {
#if ALLOC_DEBUG
  if (debugEnabled) {
    std::fprintf(stderr,
        "[AMD_ALLOC] registering Umpire dynamically growing allocator for "
        "'%s'\n",
        pool.c_str());
  }
#endif // ALLOC_DEBUG
  // Configure a dynanmically growing memory pool.
  umpire_allocator allocator;
  umpire_resourcemanager_get_instance(&resourceManager);
  umpire_resourcemanager_get_allocator_by_name(
      &resourceManager, pool.c_str(), &allocator);
  umpire_resourcemanager_make_allocator_list_pool(
      &resourceManager, "pool", allocator, initialSize, blockSize, &memoryPool);

  allocatorRegistry.Register(1, {&UmpireAlloc, &UmpireFree});
}

static std::string getStringFromEnvironment(
    const char *envirable, const std::string defaultValue = "") {
  if (auto value{std::getenv(envirable)}) {
    return std::string{value};
  }
  return std::string{defaultValue};
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

static std::pair<std::string, std::string> splitAtColon(
    const std::string &str) {
  size_t colon = str.find(':');
  if (colon == std::string::npos) {
    return {str, ""};
  }
  return {str.substr(0, colon), str.substr(colon + 1)};
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

  // Get some basic values from the environment about initial pool size,
  // allocation block size, etc.
  auto initialSize{
      getIntFromEnvironment("AMD_ALLOC_INITIAL_SIZE", ALLOC_INITIAL_SIZE)};
  auto blockSize{
      getIntFromEnvironment("AMD_ALLOC_BLOCK_SIZE", ALLOC_BLOCK_SIZE)};
#if ALLOC_DEBUG
  if (debugEnabled) {
    std::fprintf(stderr,
        "[AMD_ALLOC] initial pool size = %d (%.2f MB), block size = %d (%f.2 "
        "kB)\n",
        initialSize, initialSize / 1048576.0f, blockSize, blockSize / 1024.0f);
  }
#endif

  // Determine what allocator to register via very simplistic parsing of syntax
  // ALLOCATOR:MEMORY_KIND.  Proper values are: Umpire:host, Umpire:device.
  std::string allocator = getStringFromEnvironment("AMD_ALLOC", "umpire:host");
  std::transform(
      allocator.begin(), allocator.end(), allocator.begin(), ::toupper);
#if ALLOC_DEBUG
  if (debugEnabled) {
    std::fprintf(
        stderr, "[AMD_ALLOC] requesting allocator: %s\n", allocator.c_str());
  }
#endif // ALLOC_DEBUG
  std::pair<std::string, std::string> allocSpec{splitAtColon(allocator)};
  if (allocSpec.first != "UMPIRE") {
    std::fprintf(stderr,
        "[AMD_ALLOC] warning: wrong allocator ('%s') specified for Umpire "
        "allocator, using 'UMPIRE' instead\n",
        allocSpec.first.c_str());
    allocSpec.first = std::string("UMPIRE");
  }
  if (allocSpec.first == "UMPIRE") {
    // Register this allocator in the infrastructure as allocator 1.
    // This has a counter part in descriptor.cpp, where (right now)
    // the allocator is hard-coded to be allocator 1 (and the default
    // allocator has been disabled).
    if (allocSpec.second != "HOST" && allocSpec.second != "DEVICE") {
      std::fprintf(stderr,
          "[AMD_ALLOC] warning: wrong pool ('%s') specified for Umpire "
          "allocator, using 'HOST' instead\n",
          allocSpec.second.c_str());
      allocSpec.second = std::string{"HOST"};
    }
    registerUmpireAllocator(allocSpec.second, initialSize, blockSize);
  }

  // enable the allocator in the allocator infrastructure
  amdAllocatorEnabled = true;
}

void RTDEF(AMDAllocatableSetAllocIdx)(Descriptor &descriptor, int pos) {
  if (descriptor.IsAllocatable() && !descriptor.IsAllocated()) {
    if (debugEnabled) {
      std::fprintf(
          stderr, "[AMD_ALLOC] AMDAllocatableSetAllocIdx = %d \n", pos);
    }
    descriptor.SetAllocIdx(pos);
  }
}
} // extern "C"

} // namespace Fortran::runtime::amd
