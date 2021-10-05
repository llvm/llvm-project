/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

typedef ulong uptr;


void* __asan_memmove(void* to, void* from, uptr size) { return to; }

void* __asan_memcpy(void* to, void* from, uptr size) { return to; }

void* __asan_memset(void* s, int c, uptr n) { return s; }

void __asan_handle_no_return(void) {}

void __sanitizer_ptr_cmp(uptr a, uptr b) {}

void __sanitizer_ptr_sub(uptr a, uptr b) {}

void __asan_before_dynamic_init(uptr addr) {}

void __asan_after_dynamic_init(void) {}

void __asan_register_image_globals(uptr flag) {}

void __asan_unregister_image_globals(uptr flag) {}

void __asan_register_elf_globals(uptr flag, uptr start, uptr stop) {}

void __asan_unregister_elf_globals(uptr flag, uptr start, uptr stop) {}

void __asan_init(void) {}

void __asan_version_mismatch_check_v8(void) {}

