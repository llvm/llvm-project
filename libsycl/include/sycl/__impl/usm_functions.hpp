//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of USM allocation functions.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_USM_FUNCTIONS_HPP
#define _LIBSYCL___IMPL_USM_FUNCTIONS_HPP

#include <sycl/__impl/context.hpp>
#include <sycl/__impl/queue.hpp>
#include <sycl/__impl/usm_alloc_type.hpp>

#include <sycl/__impl/detail/config.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

/// \name  SYCL 2020 4.8.3.2. Device allocation functions.
/// \brief Allocations in device memory are not accessible by the host.
/// @{
/// Allocates device USM.
///
/// \param numBytes the number of bytes to allocate.
/// \param syclDevice the device to use for the allocation.
/// \param syclContext a context containing syclDevice or its parent device if
/// syclDevice is a subdevice.
/// \param propList the list of properties for the allocation.
/// \return a pointer to the newly allocated memory, which is allocated on
/// syclDevice and which must eventually be deallocated with sycl::free in order
/// to avoid a memory leak.
_LIBSYCL_EXPORT void *malloc_device(std::size_t numBytes,
                                    const device &syclDevice,
                                    const context &syclContext,
                                    const property_list &propList = {});

/// Allocates device USM.
///
/// \param count the number of elements of type T to allocate.
/// \param syclDevice the device to use for the allocation.
/// \param syclContext a context containing syclDevice or its parent device if
/// syclDevice is a subdevice.
/// \param propList the list of properties for the allocation.
/// \return a pointer to the newly allocated memory, which is allocated on
/// syclDevice and which must eventually be deallocated with sycl::free in order
/// to avoid a memory leak.
template <typename T>
T *malloc_device(std::size_t count, const device &syclDevice,
                 const context &syclContext,
                 const property_list &propList = {}) {
  // TODO: to rewrite with aligned_malloc_device once it's supported in
  // liboffload.
  return static_cast<T *>(
      malloc_device(count * sizeof(T), syclDevice, syclContext, propList));
}

/// Allocates device USM.
///
/// \param numBytes the number of bytes to allocate.
/// \param syclQueue a queue that provides the device and context.
/// \param propList the list of properties for the allocation.
/// \return a pointer to the newly allocated memory, which is allocated on
/// syclDevice and which must eventually be deallocated with sycl::free in order
/// to avoid a memory leak.
_LIBSYCL_EXPORT void *malloc_device(std::size_t numBytes,
                                    const queue &syclQueue,
                                    const property_list &propList = {});

/// Allocates device USM.
///
/// \param count the number of elements of type T to allocate.
/// \param syclQueue a queue that provides the device and context.
/// \param propList the list of properties for the allocation.
/// \return a pointer to the newly allocated memory, which is allocated on
/// syclDevice and which must eventually be deallocated with sycl::free in order
/// to avoid a memory leak.
template <typename T>
T *malloc_device(std::size_t count, const queue &syclQueue,
                 const property_list &propList = {}) {
  return malloc_device<T>(count, syclQueue.get_device(),
                          syclQueue.get_context(), propList);
}
/// @}

/// \name SYCL 2020 4.8.3.3. Host allocation functions.
/// \brief Allocations in host memory are accessible by a device.
/// @{
/// Allocates host USM.
///
/// \param numBytes the number of bytes to allocate.
/// \param syclContext the context that should have access to the allocated
/// memory.
/// \param propList the list of properties for the allocation.
/// \return a pointer to the newly allocated memory, which must eventually be
/// deallocated with sycl::free in order to avoid a memory leak.
_LIBSYCL_EXPORT void *malloc_host(std::size_t numBytes,
                                  const context &syclContext,
                                  const property_list &propList = {});

/// Allocates host USM.
///
/// \param count the number of elements of type T to allocate.
/// \param syclContext the context that should have access to the allocated
/// memory.
/// \param propList the list of properties for the allocation.
/// \return a pointer to the newly allocated memory, which must eventually be
/// deallocated with sycl::free in order to avoid a memory leak.
template <typename T>
T *malloc_host(std::size_t count, const context &syclContext,
               const property_list &propList = {}) {
  // TODO: to rewrite with aligned_malloc_host once it's supported in
  // liboffload.
  return static_cast<T *>(
      malloc_host(count * sizeof(T), syclContext, propList));
}

/// Allocates host USM.
///
/// \param numBytes the number of bytes to allocate.
/// \param syclQueue queue that provides the context.
/// \param propList the list of properties for the allocation.
/// \return a pointer to the newly allocated memory, which must eventually be
/// deallocated with sycl::free in order to avoid a memory leak.
_LIBSYCL_EXPORT void *malloc_host(std::size_t numBytes, const queue &syclQueue,
                                  const property_list &propList = {});

/// Allocates host USM.
///
/// \param count the number of elements of type T to allocate.
/// \param syclQueue queue that provides the context.
/// \param propList the list of properties for the allocation.
/// \return a pointer to the newly allocated memory, which must eventually be
/// deallocated with sycl::free in order to avoid a memory leak.
template <typename T>
T *malloc_host(std::size_t count, const queue &syclQueue,
               const property_list &propList = {}) {
  return malloc_host<T>(count, syclQueue.get_context(), propList);
}
/// @}

/// \name SYCL 2020 4.8.3.4. Shared allocation functions.
/// \brief Allocations in shared memory are accessible by both host and device.
/// @{
/// Allocates shared USM.
///
/// \param numBytes the number of bytes to allocate.
/// \param syclDevice the device to use for the allocation.
/// \param syclContext a context containing syclDevice or its parent device if
/// syclDevice is a subdevice.
/// \param propList the list of properties for the allocation.
/// \return a pointer to the newly allocated memory, which must eventually be
/// deallocated with sycl::free in order to avoid a memory leak.
_LIBSYCL_EXPORT void *malloc_shared(std::size_t numBytes,
                                    const device &syclDevice,
                                    const context &syclContext,
                                    const property_list &propList = {});

/// Allocates shared USM.
///
/// \param count the number of elements of type T to allocate.
/// \param syclDevice the device to use for the allocation.
/// \param syclContext a context containing syclDevice or its parent device if
/// syclDevice is a subdevice.
/// \param propList the list of properties for the allocation.
/// \return a pointer to the newly allocated memory, which must eventually be
/// deallocated with sycl::free in order to avoid a memory leak.
template <typename T>
T *malloc_shared(std::size_t count, const device &syclDevice,
                 const context &syclContext,
                 const property_list &propList = {}) {
  // TODO: to rewrite with aligned_malloc_shared once it's supported in
  // liboffload.
  return static_cast<T *>(
      malloc_shared(count * sizeof(T), syclDevice, syclContext, propList));
}

/// Allocates shared USM.
///
/// \param numBytes the number of bytes to allocate.
/// \param syclQueue a queue that provides the device and context.
/// \param propList the list of properties for the allocation.
/// \return a pointer to the newly allocated memory, which must eventually be
/// deallocated with sycl::free in order to avoid a memory leak.
_LIBSYCL_EXPORT void *malloc_shared(std::size_t numBytes,
                                    const queue &syclQueue,
                                    const property_list &propList = {});

/// Allocates shared USM.
///
/// \param count the number of elements of type T to allocate.
/// \param syclQueue a queue that provides the device and context.
/// \param propList the list of properties for the allocation.
/// \return a pointer to the newly allocated memory, which must eventually be
/// deallocated with sycl::free in order to avoid a memory leak.
template <typename T>
T *malloc_shared(std::size_t count, const queue &syclQueue,
                 const property_list &propList = {}) {
  return malloc_shared<T>(count, syclQueue.get_device(),
                          syclQueue.get_context(), propList);
}
/// @}

/// \name  SYCL 2020 4.8.3.5. Parameterized allocation functions.
/// @{
/// Allocates USM of type `kind`.
///
/// \param numBytes the number of bytes to allocate.
/// \param syclDevice the device to use for the allocation. The syclDevice
/// parameter is ignored if kind is usm::alloc::host.
/// \param syclContext a context containing syclDevice or its parent device if
/// syclDevice is a subdevice.
/// \param kind the type of memory to allocate.
/// \param propList the list of properties for the allocation.
/// \return a pointer to the newly allocated memory, which must eventually be
/// deallocated with sycl::free in order to avoid a memory leak. If there are
/// not enough resources to allocate the requested memory, these functions
/// return nullptr.
_LIBSYCL_EXPORT void *malloc(std::size_t numBytes, const device &syclDevice,
                             const context &syclContext, usm::alloc kind,
                             const property_list &propList = {});

/// Allocates USM of type `kind`.
///
/// \param count the number of elements of type T to allocate.
/// \param syclDevice the device to use for the allocation. The syclDevice
/// parameter is ignored if kind is usm::alloc::host.
/// \param syclContext a context containing syclDevice or its parent device if
/// syclDevice is a subdevice.
/// \param kind the type of memory to allocate.
/// \param propList the list of properties for the allocation.
/// \return a pointer to the newly allocated memory, which must eventually be
/// deallocated with sycl::free in order to avoid a memory leak. If there are
/// not enough resources to allocate the requested memory, these functions
/// return nullptr.
template <typename T>
T *malloc(std::size_t count, const device &syclDevice,
          const context &syclContext, usm::alloc kind,
          const property_list &propList = {}) {
  // TODO: to rewrite with aligned_malloc once it's supported in liboffload.
  return static_cast<T *>(
      malloc(count * sizeof(T), syclDevice, syclContext, kind, propList));
}

/// Allocates USM of type `kind`.
///
/// \param numBytes the number of bytes to allocate.
/// \param syclQueue a queue that provides the device and context.
/// \param kind the type of memory to allocate.
/// \param propList the list of properties for the allocation.
/// \return a pointer to the newly allocated memory, which must eventually be
/// deallocated with sycl::free in order to avoid a memory leak. If there are
/// not enough resources to allocate the requested memory, these functions
/// return nullptr.
_LIBSYCL_EXPORT void *malloc(std::size_t numBytes, const queue &syclQueue,
                             usm::alloc kind,
                             const property_list &propList = {});

/// Allocates USM of type `kind`.
///
/// \param count the number of elements of type T to allocate.
/// \param syclQueue a queue that provides the device and context.
/// \param kind the type of memory to allocate.
/// \param propList the list of properties for the allocation.
/// \return a pointer to the newly allocated memory, which must eventually be
/// deallocated with sycl::free in order to avoid a memory leak. If there are
/// not enough resources to allocate the requested memory, these functions
/// return nullptr.
template <typename T>
T *malloc(std::size_t count, const queue &syclQueue, usm::alloc kind,
          const property_list &propList = {}) {
  return malloc<T>(count, syclQueue.get_device(), syclQueue.get_context(), kind,
                   propList);
}
/// @}

/// \name  SYCL 2020 4.8.3.6. Memory deallocation functions.
/// @{
/// Deallocate USM of any kind.
///
/// \param ptr a pointer that satisfies the following preconditions: points to
/// memory allocated against ctxt using one of the USM allocation routines, or
/// is a null pointer; ptr has not previously been deallocated; there are no
/// in-progress or enqueued commands using the memory pointed to by ptr.
/// \param ctxt the context that is associated with ptr.
_LIBSYCL_EXPORT void free(void *ptr, const context &ctxt);

/// Deallocate USM of any kind.
///
/// Equivalent to free(ptr, q.get_context()).
///
/// \param ptr a pointer that satisfies the following preconditions: points to
/// memory allocated against ctxt using one of the USM allocation routines, or
/// is a null pointer; ptr has not previously been deallocated; there are no
/// in-progress or enqueued commands using the memory pointed to by ptr.
/// \param q a queue to determine the context associated with ptr.
_LIBSYCL_EXPORT void free(void *ptr, const queue &q);
/// @}

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_USM_FUNCTIONS_HPP
