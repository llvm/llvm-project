//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_USM_FUNCTIONS_HPP
#define _LIBSYCL___IMPL_USM_FUNCTIONS_HPP

#include <sycl/__impl/detail/config.hpp>

#include <sycl/__impl/context.hpp>
#include <sycl/__impl/queue.hpp>
#include <sycl/__impl/usm_alloc_type.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

// SYCL 2020 4.8.3.2. Device allocation functions.

/// Allocates device USM.
///
/// \param numBytes  allocation size that is specified in bytes.
/// \param syclDevice device that is used for allocation.
/// \param syclContext context that contains syclDevice or its parent device if
/// syclDevice is a subdevice.
/// \param propList properties for the memory allocation.
/// \return a pointer to the newly allocated memory, which is allocated on
/// syclDevice and which must eventually be deallocated with sycl::free in order
/// to avoid a memory leak.
void *_LIBSYCL_EXPORT malloc_device(std::size_t numBytes,
                                    const device &syclDevice,
                                    const context &syclContext,
                                    const property_list &propList = {});

/// Allocates device USM.
///
/// \param count  allocation size that is specified in number of elements of
/// type T.
/// \param syclDevice device that is used for allocation.
/// \param syclContext context that contains syclDevice or its parent device if
/// syclDevice is a subdevice.
/// \param propList properties for the memory allocation.
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
/// \param numBytes  allocation size that is specified in bytes.
/// \param syclQueue queue that provides the device and context.
/// \param propList properties for the memory allocation.
/// \return a pointer to the newly allocated memory, which is allocated on
/// syclDevice and which must eventually be deallocated with sycl::free in order
/// to avoid a memory leak.
void *_LIBSYCL_EXPORT malloc_device(std::size_t numBytes,
                                    const queue &syclQueue,
                                    const property_list &propList = {});

/// Allocates device USM.
///
/// \param count  allocation size that is specified in number of elements of
/// type T.
/// \param syclQueue queue that provides the device and context.
/// \param propList properties for the memory allocation.
/// \return a pointer to the newly allocated memory, which is allocated on
/// syclDevice and which must eventually be deallocated with sycl::free in order
/// to avoid a memory leak.
template <typename T>
T *malloc_device(std::size_t count, const queue &syclQueue,
                 const property_list &propList = {}) {
  return malloc_device<T>(count, syclQueue.get_device(),
                          syclQueue.get_context(), propList);
}

// SYCL 2020 4.8.3.3. Host allocation functions.

/// Allocates host USM.
///
/// \param numBytes  allocation size that is specified in bytes.
/// \param syclContext context that should have access to the allocated memory.
/// \param propList properties for the memory allocation.
/// \return a pointer to the newly allocated memory, which must eventually be
/// deallocated with sycl::free in order to avoid a memory leak.
void *_LIBSYCL_EXPORT malloc_host(std::size_t numBytes,
                                  const context &syclContext,
                                  const property_list &propList = {});

/// Allocates host USM.
///
/// \param count  allocation size that is specified in number of elements of
/// type T.
/// \param syclContext context that should have access to the allocated memory.
/// \param propList properties for the memory allocation.
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
/// \param numBytes  allocation size that is specified in bytes.
/// \param syclQueue queue that provides the context.
/// \param propList properties for the memory allocation.
/// \return a pointer to the newly allocated memory, which must eventually be
/// deallocated with sycl::free in order to avoid a memory leak.
void *_LIBSYCL_EXPORT malloc_host(std::size_t numBytes, const queue &syclQueue,
                                  const property_list &propList = {});

/// Allocates host USM.
///
/// \param count  allocation size that is specified in number of elements of
/// type T.
/// \param syclQueue queue that provides the context.
/// \param propList properties for the memory allocation.
/// \return a pointer to the newly allocated memory, which must eventually be
/// deallocated with sycl::free in order to avoid a memory leak.
template <typename T>
T *malloc_host(std::size_t count, const queue &syclQueue,
               const property_list &propList = {}) {
  return malloc_host<T>(count, syclQueue.get_context(), propList);
}

// SYCL 2020 4.8.3.4. Shared allocation functions.

/// Allocates shared  USM.
///
/// \param numBytes  allocation size that is specified in bytes.
/// \param syclDevice device that is used for allocation.
/// \param syclContext context that contains syclDevice or its parent device if
/// syclDevice is a subdevice.
/// \param propList properties for the memory allocation.
/// \return a pointer to the newly allocated memory, which must eventually be
/// deallocated with sycl::free in order to avoid a memory leak.
void *_LIBSYCL_EXPORT malloc_shared(std::size_t numBytes,
                                    const device &syclDevice,
                                    const context &syclContext,
                                    const property_list &propList = {});

/// Allocates shared  USM.
///
/// \param count  allocation size that is specified in number of elements of
/// type T.
/// \param syclDevice device that is used for allocation.
/// \param syclContext context that contains syclDevice or its parent device if
/// syclDevice is a subdevice.
/// \param propList properties for the memory allocation.
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

/// Allocates shared  USM.
///
/// \param numBytes  allocation size that is specified in bytes.
/// \param syclQueue queue that provides the device and context.
/// \param propList properties for the memory allocation.
/// \return a pointer to the newly allocated memory, which must eventually be
/// deallocated with sycl::free in order to avoid a memory leak.
void *_LIBSYCL_EXPORT malloc_shared(std::size_t numBytes,
                                    const queue &syclQueue,
                                    const property_list &propList = {});

/// Allocates shared  USM.
///
/// \param count  allocation size that is specified in number of elements of
/// type T.
/// \param syclQueue queue that provides the device and context.
/// \param propList properties for the memory allocation.
/// \return a pointer to the newly allocated memory, which must eventually be
/// deallocated with sycl::free in order to avoid a memory leak.
template <typename T>
T *malloc_shared(std::size_t count, const queue &syclQueue,
                 const property_list &propList = {}) {
  return malloc_shared<T>(count, syclQueue.get_device(),
                          syclQueue.get_context(), propList);
}

// SYCL 2020 4.8.3.5. Parameterized allocation functions

/// Allocates USM of type `kind`.
///
/// \param numBytes  allocation size that is specified in bytes.
/// \param syclDevice device that is used for allocation. The syclDevice
/// parameter is ignored if kind is usm::alloc::host.
/// \param syclContext context that contains syclDevice or its parent device if
/// syclDevice is a subdevice.
/// \param kind type of memory to allocate.
/// \param propList properties for the memory allocation.
/// \return a pointer to the newly allocated memory, which must eventually be
/// deallocated with sycl::free in order to avoid a memory leak. If there are
/// not enough resources to allocate the requested memory, these functions
/// return nullptr.
void *_LIBSYCL_EXPORT malloc(std::size_t numBytes, const device &syclDevice,
                             const context &syclContext, usm::alloc kind,
                             const property_list &propList = {});

/// Allocates USM of type `kind`.
///
/// \param count  allocation size that is specified in number of elements of
/// type T.
/// \param syclDevice device that is used for allocation. The syclDevice
/// parameter is ignored if kind is usm::alloc::host.
/// \param syclContext context that contains syclDevice or its parent device if
/// syclDevice is a subdevice.
/// \param kind type of memory to allocate.
/// \param propList properties for the memory allocation.
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
/// \param numBytes  allocation size that is specified in bytes.
/// \param syclQueue queue that provides the device and context.
/// \param kind type of memory to allocate.
/// \param propList properties for the memory allocation.
/// \return a pointer to the newly allocated memory, which must eventually be
/// deallocated with sycl::free in order to avoid a memory leak. If there are
/// not enough resources to allocate the requested memory, these functions
/// return nullptr.
void *_LIBSYCL_EXPORT malloc(std::size_t numBytes, const queue &syclQueue,
                             usm::alloc kind,
                             const property_list &propList = {});

/// Allocates USM of type `kind`.
///
/// \param count  allocation size that is specified in number of elements of
/// type T.
/// \param syclQueue queue that provides the device and context.
/// \param kind type of memory to allocate.
/// \param propList properties for the memory allocation.
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

// SYCL 2020 4.8.3.6. Memory deallocation functions

/// Deallocate USM of any kind.
///
/// \param ptr pointer that satisfies the following preconditions: points to
/// memory allocated against ctxt using one of the USM allocation routines, or
/// is a null pointer, ptr has not previously been deallocated; there are no
/// in-progress or enqueued commands using the memory pointed to by ptr.
/// \param ctxt context that is associated with ptr.
void _LIBSYCL_EXPORT free(void *ptr, const context &ctxt);

/// Deallocate USM of any kind.
///
/// Equivalent to free(ptr, q.get_context()).
///
/// \param ptr pointer that satisfies the following preconditions: points to
/// memory allocated against ctxt using one of the USM allocation routines, or
/// is a null pointer, ptr has not previously been deallocated; there are no
/// in-progress or enqueued commands using the memory pointed to by ptr.
/// \param q queue to determine the context associated with ptr.
void _LIBSYCL_EXPORT free(void *ptr, const queue &q);

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_USM_FUNCTIONS_HPP
