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

void *malloc_host(std::size_t numBytes, const context &syclContext,
                  const property_list &propList) {
  auto ContextDevices = syclContext.get_devices();
  assert(!ContextDevices.empty() && "Context can't be created without device");
  if (std::none_of(
          ContextDevices.begin(), ContextDevices.end(),
          [](device Dev) { return Dev.has(aspect::usm_host_allocations); }))
    throw sycl::exception(
        sycl::errc::feature_not_supported,
        "All devices of context do not support host USM allocations.");
  return malloc(numBytes, ContextDevices[0], syclContext, usm::alloc::host,
                propList);
}

void *malloc_host(std::size_t numBytes, const queue &syclQueue,
                  const property_list &propList) {
  return malloc_host(numBytes, syclQueue.get_context(), propList);
}

// SYCL 2020 4.8.3.4. Shared allocation functions.

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

// SYCL 2020 4.8.3.5. Parameterized allocation functions

static aspect getAspectByAllocationKind(usm::alloc kind) {
  switch (kind) {
  case usm::alloc::host:
    return aspect::usm_host_allocations;
  case usm::alloc::device:
    return aspect::usm_device_allocations;
  case usm::alloc::shared:
    return aspect::usm_shared_allocations;
  default:
    assert(false &&
           "Must be unreachable, usm::unknown allocation can't be requested");
    // usm::alloc::unknown can be returned to user from get_pointer_type but
    // it can't be converted to a valid backend type and there is no need to
    // do that.
    throw exception(sycl::make_error_code(sycl::errc::runtime),
                    "USM type is not supported");
  }
}

void *malloc(std::size_t numBytes, const device &syclDevice,
             const context &syclContext, usm::alloc kind,
             const property_list &propList) {
  auto ContextDevices = syclContext.get_devices();
  assert(!ContextDevices.empty() && "Context can't be created without device");
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
  auto Result = detail::callNoCheck(
      olMemAlloc, detail::getSyclObjImpl(syclDevice)->getOLHandle(),
      detail::convertUSMTypeToOL(kind), numBytes, &Ptr);
  assert(!!Result != !!Ptr && "Successful USM allocation can't return nullptr");
  return detail::isSuccess(Result) ? Ptr : nullptr;
}

void *malloc(std::size_t numBytes, const queue &syclQueue, usm::alloc kind,
             const property_list &propList) {
  return malloc(numBytes, syclQueue.get_device(), syclQueue.get_context(), kind,
                propList);
}

// SYCL 2020 4.8.3.6. Memory deallocation functions

void free(void *ptr, const context &ctxt) {
  std::ignore = ctxt;
  detail::callAndThrow(olMemFree, ptr);
}

void free(void *ptr, const queue &q) { return free(ptr, q.get_context()); }

_LIBSYCL_END_NAMESPACE_SYCL
