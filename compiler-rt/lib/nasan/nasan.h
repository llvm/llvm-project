//===-- nasan.h - NoAliasSanitizer Public Interface ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Public interface for NoAliasSanitizer.
//
//===----------------------------------------------------------------------===//

#ifndef NASAN_H
#define NASAN_H

#include "sanitizer_common/sanitizer_internal_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Provenance ID type
typedef __sanitizer::u64 ProvenanceID;

// Scope management
SANITIZER_INTERFACE_ATTRIBUTE
ProvenanceID __nasan_create_provenance(void *param, const char *func_name,
                                       const char *file, int line);
SANITIZER_INTERFACE_ATTRIBUTE
void __nasan_destroy_provenance(ProvenanceID prov);

// Provenance tracking (pointer-level)
SANITIZER_INTERFACE_ATTRIBUTE
void __nasan_set_pointer_provenance(void *ptr, ProvenanceID prov);
SANITIZER_INTERFACE_ATTRIBUTE
ProvenanceID __nasan_get_pointer_provenance(void *ptr);
SANITIZER_INTERFACE_ATTRIBUTE
void __nasan_inherit_provenance(void *dst_ptr, void *src_ptr);

// Provenance merging (for PHI nodes, select instructions)
SANITIZER_INTERFACE_ATTRIBUTE
void __nasan_merge_provenance(void *dst_ptr, void **src_ptrs,
                              __sanitizer::uptr count);

// Memory store/load tracking (for propagating provenance through memory)
SANITIZER_INTERFACE_ATTRIBUTE
void __nasan_record_pointer_store(void *addr, ProvenanceID prov);
SANITIZER_INTERFACE_ATTRIBUTE
void __nasan_propagate_through_load(void *dst_ptr, void *src_addr);

// Access validation - takes provenance ID directly to avoid shadow memory race
SANITIZER_INTERFACE_ATTRIBUTE
void __nasan_check_load(__sanitizer::u64 addr, __sanitizer::u64 size,
                        ProvenanceID prov);
SANITIZER_INTERFACE_ATTRIBUTE
void __nasan_check_store(__sanitizer::u64 addr, __sanitizer::u64 size,
                         ProvenanceID prov);

// Function calls
SANITIZER_INTERFACE_ATTRIBUTE
void __nasan_function_entry();
SANITIZER_INTERFACE_ATTRIBUTE
void __nasan_function_exit();

// Allocations (create new provenance roots)
SANITIZER_INTERFACE_ATTRIBUTE
ProvenanceID __nasan_create_allocation_provenance(void *ptr,
                                                  __sanitizer::uptr size);
SANITIZER_INTERFACE_ATTRIBUTE
void __nasan_destroy_allocation_provenance(void *ptr);

// Exception handling
SANITIZER_INTERFACE_ATTRIBUTE
void __nasan_handle_exception_cleanup(ProvenanceID prov);

// Debugging
SANITIZER_INTERFACE_ATTRIBUTE
void __nasan_print_pointer_info(void *ptr);
SANITIZER_INTERFACE_ATTRIBUTE
void __nasan_dump_state();

// Initialization
SANITIZER_INTERFACE_ATTRIBUTE
void __nasan_init();

#ifdef __cplusplus
}
#endif

#endif // NASAN_H
