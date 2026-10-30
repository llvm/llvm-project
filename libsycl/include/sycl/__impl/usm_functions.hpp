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

#include <algorithm>
#include <cstddef>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

/// \defgroup usm_device_alloc SYCL 2020 4.8.3.2. Device allocation functions.
/// Allocations in device memory are not accessible by the host.

/// \defgroup usm_aligned_alloc_device aligned_alloc_device
/// \ingroup usm_device_alloc
/// Allocates device USM with specified alignment.
/// @{

/// \copybrief usm_aligned_alloc_device
///
/// \param alignment the minimum alignment of the allocated memory.
/// \param numBytes the number of bytes to allocate.
/// \param syclDevice the device to use for the allocation.
/// \param syclContext a context containing syclDevice or its parent device if
/// syclDevice is a subdevice.
/// \param propList the list of properties for the allocation.
/// \return a pointer to the newly allocated memory, which is allocated on the
/// device and which must eventually be deallocated with sycl::free in order to
/// avoid a memory leak. Returns nullptr if the requested alignment is not
/// supported by the implementation.
_LIBSYCL_EXPORT
void *aligned_alloc_device(std::size_t alignment, std::size_t numBytes,
                           const device &syclDevice, const context &syclContext,
                           const property_list &propList = {});

/// \overload
///
/// \param alignment the minimum alignment of the allocated memory.
/// \param count the number of elements of type T to allocate.
/// \param syclDevice the device to use for the allocation.
/// \param syclContext a context containing syclDevice or its parent device if
/// syclDevice is a subdevice.
/// \param propList the list of properties for the allocation.
template <typename T>
T *aligned_alloc_device(std::size_t alignment, std::size_t count,
                        const device &syclDevice, const context &syclContext,
                        const property_list &propList = {}) {
  return static_cast<T *>(aligned_alloc_device(std::max(alignment, alignof(T)),
                                               count * sizeof(T), syclDevice,
                                               syclContext, propList));
}

/// \overload
///
/// \param alignment the minimum alignment of the allocated memory.
/// \param numBytes the number of bytes to allocate.
/// \param syclQueue a queue that provides the device and context.
/// \param propList the list of properties for the allocation.
_LIBSYCL_EXPORT
void *aligned_alloc_device(std::size_t alignment, std::size_t numBytes,
                           const queue &syclQueue,
                           const property_list &propList = {});

/// \overload
///
/// \param alignment the minimum alignment of the allocated memory.
/// \param count the number of elements of type T to allocate.
/// \param syclQueue a queue that provides the device and context.
/// \param propList the list of properties for the allocation.
template <typename T>
T *aligned_alloc_device(std::size_t alignment, std::size_t count,
                        const queue &syclQueue,
                        const property_list &propList = {}) {
  return aligned_alloc_device<T>(alignment, count, syclQueue.get_device(),
                                 syclQueue.get_context(), propList);
}
/// @}

/// \defgroup usm_malloc_device malloc_device
/// \ingroup usm_device_alloc
/// Allocates device USM.
/// @{

/// \copybrief usm_malloc_device
///
/// \param numBytes the number of bytes to allocate.
/// \param syclDevice the device to use for the allocation.
/// \param syclContext a context containing syclDevice or its parent device if
/// syclDevice is a subdevice.
/// \param propList the list of properties for the allocation.
/// \return a pointer to the newly allocated memory, which is allocated on the
/// device and which must eventually be deallocated with sycl::free in order to
/// avoid a memory leak.
_LIBSYCL_EXPORT void *malloc_device(std::size_t numBytes,
                                    const device &syclDevice,
                                    const context &syclContext,
                                    const property_list &propList = {});

/// \overload
///
/// \param count the number of elements of type T to allocate.
/// \param syclDevice the device to use for the allocation.
/// \param syclContext a context containing syclDevice or its parent device if
/// syclDevice is a subdevice.
/// \param propList the list of properties for the allocation.
template <typename T>
T *malloc_device(std::size_t count, const device &syclDevice,
                 const context &syclContext,
                 const property_list &propList = {}) {
  return aligned_alloc_device<T>(alignof(T), count, syclDevice, syclContext,
                                 propList);
}

/// \overload
///
/// \param numBytes the number of bytes to allocate.
/// \param syclQueue a queue that provides the device and context.
/// \param propList the list of properties for the allocation.
_LIBSYCL_EXPORT void *malloc_device(std::size_t numBytes,
                                    const queue &syclQueue,
                                    const property_list &propList = {});

/// \overload
///
/// \param count the number of elements of type T to allocate.
/// \param syclQueue a queue that provides the device and context.
/// \param propList the list of properties for the allocation.
template <typename T>
T *malloc_device(std::size_t count, const queue &syclQueue,
                 const property_list &propList = {}) {
  return malloc_device<T>(count, syclQueue.get_device(),
                          syclQueue.get_context(), propList);
}
/// @}

/// \defgroup usm_host_alloc SYCL 2020 4.8.3.3. Host allocation functions.
/// Allocations in host memory are accessible by a device.

/// \defgroup usm_aligned_alloc_host aligned_alloc_host
/// \ingroup usm_host_alloc
/// Allocates host USM with specified alignment.
/// @{

/// \copybrief usm_aligned_alloc_host
///
/// \param alignment the minimum alignment of the allocated memory.
/// \param numBytes the number of bytes to allocate.
/// \param syclContext the context that should have access to the allocated
/// memory.
/// \param propList the list of properties for the allocation.
/// \return a pointer to the newly allocated memory, which must eventually be
/// deallocated with sycl::free in order to avoid a memory leak. Returns nullptr
/// if the requested alignment is not supported by the implementation.
_LIBSYCL_EXPORT
void *aligned_alloc_host(std::size_t alignment, std::size_t numBytes,
                         const context &syclContext,
                         const property_list &propList = {});

/// \overload
///
/// \param alignment the minimum alignment of the allocated memory.
/// \param count the number of elements of type T to allocate.
/// \param syclContext the context that should have access to the allocated
/// memory.
/// \param propList the list of properties for the allocation.
template <typename T>
T *aligned_alloc_host(std::size_t alignment, std::size_t count,
                      const context &syclContext,
                      const property_list &propList = {}) {
  return static_cast<T *>(aligned_alloc_host(std::max(alignment, alignof(T)),
                                             count * sizeof(T), syclContext,
                                             propList));
}

/// \overload
///
/// \param alignment the minimum alignment of the allocated memory.
/// \param numBytes the number of bytes to allocate.
/// \param syclQueue a queue that provides the context.
/// \param propList the list of properties for the allocation.
_LIBSYCL_EXPORT
void *aligned_alloc_host(std::size_t alignment, std::size_t numBytes,
                         const queue &syclQueue,
                         const property_list &propList = {});

/// \overload
///
/// \param alignment the minimum alignment of the allocated memory.
/// \param count the number of elements of type T to allocate.
/// \param syclQueue a queue that provides the context.
/// \param propList the list of properties for the allocation.
template <typename T>
T *aligned_alloc_host(std::size_t alignment, std::size_t count,
                      const queue &syclQueue,
                      const property_list &propList = {}) {
  return aligned_alloc_host<T>(alignment, count, syclQueue.get_context(),
                               propList);
}
/// @}

/// \defgroup usm_malloc_host malloc_host
/// \ingroup usm_host_alloc
/// Allocates host USM.
/// @{

/// \copybrief usm_malloc_host
///
/// \param numBytes the number of bytes to allocate.
/// \param syclContext the context that should have access to the allocated
/// memory.
/// \param propList the list of properties for the allocation.
/// \return a pointer to the newly allocated memory, which must eventually be
/// deallocated with sycl::free in order to avoid a memory leak.
_LIBSYCL_EXPORT
void *malloc_host(std::size_t numBytes, const context &syclContext,
                  const property_list &propList = {});

/// \overload
///
/// \param count the number of elements of type T to allocate.
/// \param syclContext the context that should have access to the allocated
/// memory.
/// \param propList the list of properties for the allocation.
template <typename T>
T *malloc_host(std::size_t count, const context &syclContext,
               const property_list &propList = {}) {
  return aligned_alloc_host<T>(alignof(T), count, syclContext, propList);
}

/// \overload
///
/// \param numBytes the number of bytes to allocate.
/// \param syclQueue a queue that provides the context.
/// \param propList the list of properties for the allocation.
_LIBSYCL_EXPORT void *malloc_host(std::size_t numBytes, const queue &syclQueue,
                                  const property_list &propList = {});

/// \overload
///
/// \param count the number of elements of type T to allocate.
/// \param syclQueue a queue that provides the context.
/// \param propList the list of properties for the allocation.
template <typename T>
T *malloc_host(std::size_t count, const queue &syclQueue,
               const property_list &propList = {}) {
  return malloc_host<T>(count, syclQueue.get_context(), propList);
}
/// @}

/// \defgroup usm_shared_alloc SYCL 2020 4.8.3.4. Shared allocation functions.
/// Allocations in shared memory are accessible by both host and device.

/// \defgroup usm_aligned_alloc_shared aligned_alloc_shared
/// \ingroup usm_shared_alloc
/// Allocates shared USM with specified alignment.
/// @{

/// \copybrief usm_aligned_alloc_shared
///
/// \param alignment the minimum alignment of the allocated memory.
/// \param numBytes the number of bytes to allocate.
/// \param syclDevice the device to use for the allocation.
/// \param syclContext a context containing syclDevice or its parent device if
/// syclDevice is a subdevice.
/// \param propList the list of properties for the allocation.
/// \return a pointer to the newly allocated memory, which must eventually be
/// deallocated with sycl::free in order to avoid a memory leak. Returns nullptr
/// if the requested alignment is not supported by the implementation.
_LIBSYCL_EXPORT
void *aligned_alloc_shared(std::size_t alignment, std::size_t numBytes,
                           const device &syclDevice, const context &syclContext,
                           const property_list &propList = {});

/// \overload
///
/// \param alignment the minimum alignment of the allocated memory.
/// \param count the number of elements of type T to allocate.
/// \param syclDevice the device to use for the allocation.
/// \param syclContext a context containing syclDevice or its parent device if
/// syclDevice is a subdevice.
/// \param propList the list of properties for the allocation.
template <typename T>
T *aligned_alloc_shared(std::size_t alignment, std::size_t count,
                        const device &syclDevice, const context &syclContext,
                        const property_list &propList = {}) {
  return static_cast<T *>(aligned_alloc_shared(std::max(alignment, alignof(T)),
                                               count * sizeof(T), syclDevice,
                                               syclContext, propList));
}

/// \overload
///
/// \param alignment the minimum alignment of the allocated memory.
/// \param numBytes the number of bytes to allocate.
/// \param syclQueue a queue that provides the device and context.
/// \param propList the list of properties for the allocation.
_LIBSYCL_EXPORT
void *aligned_alloc_shared(std::size_t alignment, std::size_t numBytes,
                           const queue &syclQueue,
                           const property_list &propList = {});

/// \overload
///
/// \param alignment the minimum alignment of the allocated memory.
/// \param count the number of elements of type T to allocate.
/// \param syclQueue a queue that provides the device and context.
/// \param propList the list of properties for the allocation.
template <typename T>
T *aligned_alloc_shared(std::size_t alignment, std::size_t count,
                        const queue &syclQueue,
                        const property_list &propList = {}) {
  return aligned_alloc_shared<T>(alignment, count, syclQueue.get_device(),
                                 syclQueue.get_context(), propList);
}
/// @}

/// \defgroup usm_malloc_shared malloc_shared
/// \ingroup usm_shared_alloc
/// Allocates shared USM.
/// @{

/// \copybrief usm_malloc_shared
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

/// \overload
///
/// \param count the number of elements of type T to allocate.
/// \param syclDevice the device to use for the allocation.
/// \param syclContext a context containing syclDevice or its parent device if
/// syclDevice is a subdevice.
/// \param propList the list of properties for the allocation.
template <typename T>
T *malloc_shared(std::size_t count, const device &syclDevice,
                 const context &syclContext,
                 const property_list &propList = {}) {
  return aligned_alloc_shared<T>(alignof(T), count, syclDevice, syclContext,
                                 propList);
}

/// \overload
///
/// \param numBytes the number of bytes to allocate.
/// \param syclQueue a queue that provides the device and context.
/// \param propList the list of properties for the allocation.
_LIBSYCL_EXPORT void *malloc_shared(std::size_t numBytes,
                                    const queue &syclQueue,
                                    const property_list &propList = {});

/// \overload
///
/// \param count the number of elements of type T to allocate.
/// \param syclQueue a queue that provides the device and context.
/// \param propList the list of properties for the allocation.
template <typename T>
T *malloc_shared(std::size_t count, const queue &syclQueue,
                 const property_list &propList = {}) {
  return malloc_shared<T>(count, syclQueue.get_device(),
                          syclQueue.get_context(), propList);
}
/// @}

/// \defgroup usm_param SYCL 2020 4.8.3.5. Parameterized allocation functions.

/// \defgroup usm_aligned_alloc aligned_alloc
/// \ingroup usm_param
/// Allocates USM of type `kind` with specified alignment.
/// @{

/// \copybrief usm_aligned_alloc
///
/// \param alignment the minimum alignment of the allocated memory.
/// \param numBytes the number of bytes to allocate.
/// \param syclDevice the device to use for the allocation. The syclDevice
/// parameter is ignored if kind is usm::alloc::host.
/// \param syclContext a context containing syclDevice or its parent device if
/// syclDevice is a subdevice.
/// \param kind the type of memory to allocate.
/// \param propList the list of properties for the allocation.
/// \return a pointer to the newly allocated memory, which must eventually be
/// deallocated with sycl::free in order to avoid a memory leak. Returns nullptr
/// if there are not enough resources to allocate the requested memory, or if
/// the requested alignment is not supported by the implementation.
_LIBSYCL_EXPORT
void *aligned_alloc(std::size_t alignment, std::size_t numBytes,
                    const device &syclDevice, const context &syclContext,
                    usm::alloc kind, const property_list &propList = {});

/// \overload
///
/// \param alignment the minimum alignment of the allocated memory.
/// \param count the number of elements of type T to allocate.
/// \param syclDevice the device to use for the allocation. The syclDevice
/// parameter is ignored if kind is usm::alloc::host.
/// \param syclContext a context containing syclDevice or its parent device if
/// syclDevice is a subdevice.
/// \param kind the type of memory to allocate.
/// \param propList the list of properties for the allocation.
template <typename T>
T *aligned_alloc(std::size_t alignment, std::size_t count,
                 const device &syclDevice, const context &syclContext,
                 usm::alloc kind, const property_list &propList = {}) {
  return static_cast<T *>(aligned_alloc(std::max(alignment, alignof(T)),
                                        count * sizeof(T), syclDevice,
                                        syclContext, kind, propList));
}

/// \overload
///
/// \param alignment the minimum alignment of the allocated memory.
/// \param numBytes the number of bytes to allocate.
/// \param syclQueue a queue that provides the device and context.
/// \param kind the type of memory to allocate.
/// \param propList the list of properties for the allocation.
_LIBSYCL_EXPORT
void *aligned_alloc(std::size_t alignment, std::size_t numBytes,
                    const queue &syclQueue, usm::alloc kind,
                    const property_list &propList = {});

/// \overload
///
/// \param alignment the minimum alignment of the allocated memory.
/// \param count the number of elements of type T to allocate.
/// \param syclQueue a queue that provides the device and context.
/// \param kind the type of memory to allocate.
/// \param propList the list of properties for the allocation.
template <typename T>
T *aligned_alloc(std::size_t alignment, std::size_t count,
                 const queue &syclQueue, usm::alloc kind,
                 const property_list &propList = {}) {
  return aligned_alloc<T>(alignment, count, syclQueue.get_device(),
                          syclQueue.get_context(), kind, propList);
}
/// @}

/// \defgroup usm_malloc malloc
/// \ingroup usm_param
/// Allocates USM of type `kind`.
/// @{

/// \copybrief usm_malloc
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

/// \overload
///
/// \param count the number of elements of type T to allocate.
/// \param syclDevice the device to use for the allocation. The syclDevice
/// parameter is ignored if kind is usm::alloc::host.
/// \param syclContext a context containing syclDevice or its parent device if
/// syclDevice is a subdevice.
/// \param kind the type of memory to allocate.
/// \param propList the list of properties for the allocation.
template <typename T>
T *malloc(std::size_t count, const device &syclDevice,
          const context &syclContext, usm::alloc kind,
          const property_list &propList = {}) {
  return aligned_alloc<T>(alignof(T), count, syclDevice, syclContext, kind,
                          propList);
}

/// \overload
///
/// \param numBytes the number of bytes to allocate.
/// \param syclQueue a queue that provides the device and context.
/// \param kind the type of memory to allocate.
/// \param propList the list of properties for the allocation.
_LIBSYCL_EXPORT void *malloc(std::size_t numBytes, const queue &syclQueue,
                             usm::alloc kind,
                             const property_list &propList = {});

/// \overload
///
/// \param count the number of elements of type T to allocate.
/// \param syclQueue a queue that provides the device and context.
/// \param kind the type of memory to allocate.
/// \param propList the list of properties for the allocation.
template <typename T>
T *malloc(std::size_t count, const queue &syclQueue, usm::alloc kind,
          const property_list &propList = {}) {
  return malloc<T>(count, syclQueue.get_device(), syclQueue.get_context(), kind,
                   propList);
}
/// @}

/// \defgroup usm_dealloc SYCL 2020 4.8.3.6. Memory deallocation functions.

/// \defgroup usm_free free
/// \ingroup usm_dealloc
/// Deallocates USM of any kind.
/// @{

/// \copybrief usm_free
///
/// \param ptr a pointer that satisfies the following preconditions: points to
/// memory allocated against the context using one of the USM allocation
/// routines, or is a null pointer; ptr has not previously been deallocated;
/// there are no in-progress or enqueued commands using the memory pointed to by
/// ptr.
/// \param ctxt the context that is associated with ptr.
_LIBSYCL_EXPORT void free(void *ptr, const context &ctxt);

/// \overload
///
/// \param ptr a pointer that satisfies the following preconditions: points to
/// memory allocated against the context using one of the USM allocation
/// routines, or is a null pointer; ptr has not previously been deallocated;
/// there are no in-progress or enqueued commands using the memory pointed to by
/// ptr.
/// \param q a queue to determine the context associated with ptr. Equivalent
/// to free(ptr, q.get_context()).
_LIBSYCL_EXPORT void free(void *ptr, const queue &q);
/// @}

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_USM_FUNCTIONS_HPP
