//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/usm_functions.hpp>

#include <detail/device_impl.hpp>
#include <detail/offload/offload_utils.hpp>

#include <OffloadAPI.h>

#include <algorithm>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

// SYCL 2020 4.8.3.2. Device allocation functions.

void *aligned_alloc_device(size_t alignment, size_t numBytes,
                           const device &syclDevice, const context &syclContext,
                           const property_list &propList) {
  return aligned_alloc(alignment, numBytes, syclDevice, syclContext,
                       usm::alloc::device, propList);
}

void *aligned_alloc_device(size_t alignment, size_t numBytes,
                           const queue &syclQueue,
                           const property_list &propList) {
  return aligned_alloc_device(alignment, numBytes, syclQueue.get_device(),
                              syclQueue.get_context(), propList);
}

void *malloc_device(std::size_t numBytes, const device &syclDevice,
                    const context &syclContext, const property_list &propList) {
  return malloc(numBytes, syclDevice, syclContext, usm::alloc::device,
                propList);
}

void *malloc_device(std::size_t numBytes, const queue &syclQueue,
                    const property_list &propList) {
  return malloc_device(numBytes, syclQueue.get_device(),
                       syclQueue.get_context(), propList);
}

// SYCL 2020 4.8.3.3. Host allocation functions.

static device getHostAllocDevice(const context &syclContext) {
  auto ContextDevices = syclContext.get_devices();

  auto It = std::find_if(
      ContextDevices.begin(), ContextDevices.end(),
      [](const device &Dev) { return Dev.has(aspect::usm_host_allocations); });

  if (It == ContextDevices.end()) {
    throw sycl::exception(
        sycl::errc::feature_not_supported,
        "None of the context's devices support host USM allocations.");
  }
  return *It;
}

void *aligned_alloc_host(size_t alignment, size_t numBytes,
                         const context &syclContext,
                         const property_list &propList) {
  auto device = getHostAllocDevice(syclContext);
  return aligned_alloc(alignment, numBytes, device, syclContext,
                       usm::alloc::host, propList);
}

void *aligned_alloc_host(size_t alignment, size_t numBytes,
                         const queue &syclQueue,
                         const property_list &propList) {
  return aligned_alloc_host(alignment, numBytes, syclQueue.get_context(),
                            propList);
}

void *malloc_host(std::size_t numBytes, const context &syclContext,
                  const property_list &propList) {
  return aligned_alloc_host(0, numBytes, syclContext, propList);
}

void *malloc_host(std::size_t numBytes, const queue &syclQueue,
                  const property_list &propList) {
  return malloc_host(numBytes, syclQueue.get_context(), propList);
}

// SYCL 2020 4.8.3.4. Shared allocation functions.

void *aligned_alloc_shared(size_t alignment, size_t numBytes,
                           const device &syclDevice, const context &syclContext,
                           const property_list &propList) {
  return aligned_alloc(alignment, numBytes, syclDevice, syclContext,
                       usm::alloc::shared, propList);
}

void *aligned_alloc_shared(size_t alignment, size_t numBytes,
                           const queue &syclQueue,
                           const property_list &propList) {
  return aligned_alloc_shared(alignment, numBytes, syclQueue.get_device(),
                              syclQueue.get_context(), propList);
}

void *malloc_shared(std::size_t numBytes, const device &syclDevice,
                    const context &syclContext, const property_list &propList) {
  return malloc(numBytes, syclDevice, syclContext, usm::alloc::shared,
                propList);
}

void *malloc_shared(std::size_t numBytes, const queue &syclQueue,
                    const property_list &propList) {
  return malloc_shared(numBytes, syclQueue.get_device(),
                       syclQueue.get_context(), propList);
}

// SYCL 2020 4.8.3.5. Parameterized allocation functions.

static aspect getAspectByAllocationKind(usm::alloc kind) {
  switch (kind) {
  case usm::alloc::host:
    return aspect::usm_host_allocations;
  case usm::alloc::device:
    return aspect::usm_device_allocations;
  case usm::alloc::shared:
    return aspect::usm_shared_allocations;
  case usm::alloc::unknown:
    // usm::alloc::unknown can be returned to user from get_pointer_type but
    // it can't be converted to a valid backend type.
    throw exception(sycl::make_error_code(sycl::errc::invalid),
                    "Invalid USM allocation kind requested");
  }
}

void *aligned_alloc(std::size_t alignment, std::size_t numBytes,
                    const device &syclDevice, const context &syclContext,
                    usm::alloc kind, const property_list &propList) {

  auto ContextDevices = syclContext.get_devices();
  if (std::none_of(ContextDevices.begin(), ContextDevices.end(),
                   [&syclDevice](device Dev) { return Dev == syclDevice; }))
    throw exception(make_error_code(errc::invalid),
                    "Specified device is not contained by specified context.");

  if (!syclDevice.has(getAspectByAllocationKind(kind)))
    throw sycl::exception(
        sycl::errc::feature_not_supported,
        "Device doesn't support requested kind of USM allocation");

  if (!numBytes)
    return nullptr;

  void *Ptr{};
  auto OLDevice = detail::getSyclObjImpl(syclDevice)->getOLHandle();

  ol_result_t Result;
  if (alignment == 0) {
    Result =
        kind == usm::alloc::host
            ? detail::callNoCheck(olMemAllocHost, OLDevice, numBytes, &Ptr)
            : detail::callNoCheck(olMemAlloc, OLDevice,
                                  detail::getOlAllocType(kind), numBytes, &Ptr);
  } else {
    Result = kind == usm::alloc::host
                 ? detail::callNoCheck(olMemAllocAlignedHost, OLDevice,
                                       numBytes, alignment, &Ptr)
                 : detail::callNoCheck(olMemAllocAligned, OLDevice,
                                       detail::getOlAllocType(kind), numBytes,
                                       alignment, &Ptr);
  }
  return detail::isFailed(Result) ? nullptr : Ptr;
}

void *aligned_alloc(std::size_t alignment, std::size_t numBytes,
                    const queue &syclQueue, usm::alloc kind,
                    const property_list &propList) {
  return aligned_alloc(alignment, numBytes, syclQueue.get_device(),
                       syclQueue.get_context(), kind, propList);
}

void *malloc(std::size_t numBytes, const device &syclDevice,
             const context &syclContext, usm::alloc kind,
             const property_list &propList) {
  return aligned_alloc(0, numBytes, syclDevice, syclContext, kind, propList);
}

void *malloc(std::size_t numBytes, const queue &syclQueue, usm::alloc kind,
             const property_list &propList) {
  return malloc(numBytes, syclQueue.get_device(), syclQueue.get_context(), kind,
                propList);
}

// SYCL 2020 4.8.3.6. Memory deallocation functions.

void free(void *ptr, const context &ctxt) {
  std::ignore = ctxt;
  detail::callAndThrow(olMemFree, ptr);
}

void free(void *ptr, const queue &q) { return free(ptr, q.get_context()); }

_LIBSYCL_END_NAMESPACE_SYCL
