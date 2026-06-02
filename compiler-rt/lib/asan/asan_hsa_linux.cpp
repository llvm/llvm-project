//===-- asan_hsa_linux.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Linux host HSA API interception for AddressSanitizer (SANITIZER_AMDHSA).
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"

#if SANITIZER_AMDHSA

#  include "asan_hsa_linux.h"
#  include "asan_interceptors.h"
#  include "asan_interceptors_memintrinsics.h"
#  include "asan_internal.h"
#  include "asan_report.h"
#  include "asan_stack.h"
#  include "asan_suppressions.h"

using namespace __asan;

// This TU provides the HSA interceptors, so it must also define the underlying
// REAL() slots (e.g. __interception::real_hsa_init). Otherwise the shared
// runtime fails to link when any interceptor references REAL(hsa_*).
DEFINE_REAL(hsa_status_t, hsa_init, );
DEFINE_REAL(hsa_status_t, hsa_memory_copy, void*, const void*, size_t)
DEFINE_REAL(hsa_status_t, hsa_amd_agents_allow_access, uint32_t num_agents,
            const hsa_agent_t* agents, const uint32_t* flags, const void* ptr)
DEFINE_REAL(hsa_status_t, hsa_amd_memory_pool_allocate,
            hsa_amd_memory_pool_t memory_pool, size_t size, uint32_t flags,
            void** ptr)
DEFINE_REAL(hsa_status_t, hsa_amd_memory_pool_free, void* ptr)
DEFINE_REAL(hsa_status_t, hsa_amd_memory_async_copy, void*, hsa_agent_t,
            const void*, hsa_agent_t, size_t, uint32_t, const hsa_signal_t*,
            hsa_signal_t)
#  if HSA_AMD_INTERFACE_VERSION_MINOR >= 1
DEFINE_REAL(hsa_status_t, hsa_amd_memory_async_copy_on_engine, void*,
            hsa_agent_t, const void*, hsa_agent_t, size_t, uint32_t,
            const hsa_signal_t*, hsa_signal_t, hsa_amd_sdma_engine_id_t, bool)
#  endif
DEFINE_REAL(hsa_status_t, hsa_amd_ipc_memory_create, void* ptr, size_t len,
            hsa_amd_ipc_memory_t* handle)
DEFINE_REAL(hsa_status_t, hsa_amd_ipc_memory_attach,
            const hsa_amd_ipc_memory_t* handle, size_t len, uint32_t num_agents,
            const hsa_agent_t* mapping_agents, void** mapped_ptr)
DEFINE_REAL(hsa_status_t, hsa_amd_ipc_memory_detach, void* mapped_ptr)
DEFINE_REAL(hsa_status_t, hsa_amd_vmem_address_reserve_align, void** ptr,
            size_t size, uint64_t address, uint64_t alignment, uint64_t flags)
DEFINE_REAL(hsa_status_t, hsa_amd_vmem_address_free, void* ptr, size_t size)
DEFINE_REAL(hsa_status_t, hsa_amd_pointer_info, const void* ptr,
            hsa_amd_pointer_info_t* info, void* (*alloc)(size_t),
            uint32_t* num_agents_accessible, hsa_agent_t** accessible)

namespace __asan {

static void ENSURE_HSA_INITED() {
  // The HSA interceptors are initialized lazily: if the program calls into HSA
  // before ASan's global interceptor init runs, we still need the REAL() slots
  // to be populated so we can call through to ROCr.
  if (!REAL(hsa_init))
    InitializeAmdgpuInterceptors();
}

INTERCEPTOR(hsa_status_t, hsa_amd_memory_pool_allocate,
            hsa_amd_memory_pool_t memory_pool, size_t size, uint32_t flags,
            void** ptr) {
  AsanInitFromRtl();
  ENSURE_HSA_INITED();
  GET_STACK_TRACE_MALLOC;
  return asan_hsa_amd_memory_pool_allocate(memory_pool, size, flags, ptr,
                                           &stack);
}

INTERCEPTOR(hsa_status_t, hsa_amd_memory_pool_free, void* ptr) {
  AsanInitFromRtl();
  ENSURE_HSA_INITED();
  GET_STACK_TRACE_FREE;
  return asan_hsa_amd_memory_pool_free(ptr, &stack);
}

INTERCEPTOR(hsa_status_t, hsa_amd_agents_allow_access, uint32_t num_agents,
            const hsa_agent_t* agents, const uint32_t* flags, const void* ptr) {
  AsanInitFromRtl();
  ENSURE_HSA_INITED();
  GET_STACK_TRACE_FREE;
  return asan_hsa_amd_agents_allow_access(num_agents, agents, flags, ptr,
                                          &stack);
}

INTERCEPTOR(hsa_status_t, hsa_memory_copy, void* dst, const void* src,
            size_t size) {
  AsanInitFromRtl();
  ENSURE_HSA_INITED();
  if (flags()->replace_intrin) {
    if (dst != src) {
      CHECK_RANGES_OVERLAP("hsa_memory_copy", dst, size, src, size);
    }
    ASAN_READ_RANGE(nullptr, src, size);
    ASAN_WRITE_RANGE(nullptr, dst, size);
  }
  return REAL(hsa_memory_copy)(dst, src, size);
}

INTERCEPTOR(hsa_status_t, hsa_amd_memory_async_copy, void* dst,
            hsa_agent_t dst_agent, const void* src, hsa_agent_t src_agent,
            size_t size, uint32_t num_dep_signals,
            const hsa_signal_t* dep_signals, hsa_signal_t completion_signal) {
  AsanInitFromRtl();
  ENSURE_HSA_INITED();
  if (flags()->replace_intrin) {
    if (dst != src) {
      CHECK_RANGES_OVERLAP("hsa_amd_memory_async_copy", dst, size, src, size);
    }
    ASAN_READ_RANGE(nullptr, src, size);
    ASAN_WRITE_RANGE(nullptr, dst, size);
  }
  return REAL(hsa_amd_memory_async_copy)(dst, dst_agent, src, src_agent, size,
                                         num_dep_signals, dep_signals,
                                         completion_signal);
}

#  if HSA_AMD_INTERFACE_VERSION_MINOR >= 1
INTERCEPTOR(hsa_status_t, hsa_amd_memory_async_copy_on_engine, void* dst,
            hsa_agent_t dst_agent, const void* src, hsa_agent_t src_agent,
            size_t size, uint32_t num_dep_signals,
            const hsa_signal_t* dep_signals, hsa_signal_t completion_signal,
            hsa_amd_sdma_engine_id_t engine_id, bool force_copy_on_sdma) {
  AsanInitFromRtl();
  ENSURE_HSA_INITED();
  if (flags()->replace_intrin) {
    if (dst != src) {
      CHECK_RANGES_OVERLAP("hsa_amd_memory_async_copy_on_engine", dst, size,
                           src, size);
    }
    ASAN_READ_RANGE(nullptr, src, size);
    ASAN_WRITE_RANGE(nullptr, dst, size);
  }
  return REAL(hsa_amd_memory_async_copy_on_engine)(
      dst, dst_agent, src, src_agent, size, num_dep_signals, dep_signals,
      completion_signal, engine_id, force_copy_on_sdma);
}
#  endif

INTERCEPTOR(hsa_status_t, hsa_amd_ipc_memory_create, void* ptr, size_t len,
            hsa_amd_ipc_memory_t* handle) {
  AsanInitFromRtl();
  ENSURE_HSA_INITED();
  return asan_hsa_amd_ipc_memory_create(ptr, len, handle);
}

INTERCEPTOR(hsa_status_t, hsa_amd_ipc_memory_attach,
            const hsa_amd_ipc_memory_t* handle, size_t len, uint32_t num_agents,
            const hsa_agent_t* mapping_agents, void** mapped_ptr) {
  AsanInitFromRtl();
  ENSURE_HSA_INITED();
  return asan_hsa_amd_ipc_memory_attach(handle, len, num_agents, mapping_agents,
                                        mapped_ptr);
}

INTERCEPTOR(hsa_status_t, hsa_amd_ipc_memory_detach, void* mapped_ptr) {
  AsanInitFromRtl();
  ENSURE_HSA_INITED();
  return asan_hsa_amd_ipc_memory_detach(mapped_ptr);
}

INTERCEPTOR(hsa_status_t, hsa_amd_vmem_address_reserve_align, void** ptr,
            size_t size, uint64_t address, uint64_t alignment, uint64_t flags) {
  AsanInitFromRtl();
  ENSURE_HSA_INITED();
  GET_STACK_TRACE_MALLOC;
  return asan_hsa_amd_vmem_address_reserve_align(ptr, size, address, alignment,
                                                 flags, &stack);
}

INTERCEPTOR(hsa_status_t, hsa_amd_vmem_address_free, void* ptr, size_t size) {
  AsanInitFromRtl();
  ENSURE_HSA_INITED();
  GET_STACK_TRACE_FREE;
  return asan_hsa_amd_vmem_address_free(ptr, size, &stack);
}

INTERCEPTOR(hsa_status_t, hsa_amd_pointer_info, const void* ptr,
            hsa_amd_pointer_info_t* info, void* (*alloc)(size_t),
            uint32_t* num_agents_accessible, hsa_agent_t** accessible) {
  AsanInitFromRtl();
  ENSURE_HSA_INITED();
  return asan_hsa_amd_pointer_info(ptr, info, alloc, num_agents_accessible,
                                   accessible);
}

INTERCEPTOR(hsa_status_t, hsa_init) {
  AsanInitFromRtl();
  ENSURE_HSA_INITED();
  return asan_hsa_init();
}

void InitializeAmdgpuInterceptors() {
  ASAN_INTERCEPT_FUNC(hsa_init);
  ASAN_INTERCEPT_FUNC(hsa_memory_copy);
  ASAN_INTERCEPT_FUNC(hsa_amd_memory_pool_allocate);
  ASAN_INTERCEPT_FUNC(hsa_amd_memory_pool_free);
  ASAN_INTERCEPT_FUNC(hsa_amd_agents_allow_access);
  ASAN_INTERCEPT_FUNC(hsa_amd_memory_async_copy);
#  if HSA_AMD_INTERFACE_VERSION_MINOR >= 1
  ASAN_INTERCEPT_FUNC(hsa_amd_memory_async_copy_on_engine);
#  endif
  ASAN_INTERCEPT_FUNC(hsa_amd_ipc_memory_create);
  ASAN_INTERCEPT_FUNC(hsa_amd_ipc_memory_attach);
  ASAN_INTERCEPT_FUNC(hsa_amd_ipc_memory_detach);
  ASAN_INTERCEPT_FUNC(hsa_amd_vmem_address_reserve_align);
  ASAN_INTERCEPT_FUNC(hsa_amd_vmem_address_free);
  ASAN_INTERCEPT_FUNC(hsa_amd_pointer_info);
}

}  // namespace __asan

#endif  // SANITIZER_AMDHSA
