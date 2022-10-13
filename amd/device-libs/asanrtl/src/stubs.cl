/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "asan_util.h"

OPT_NONE NO_SANITIZE_ADDR void __asan_handle_no_return(void) {}

OPT_NONE NO_SANITIZE_ADDR void __sanitizer_ptr_cmp(uptr a, uptr b) {}

OPT_NONE NO_SANITIZE_ADDR void __sanitizer_ptr_sub(uptr a, uptr b) {}

OPT_NONE NO_SANITIZE_ADDR void __asan_before_dynamic_init(uptr addr) {}

OPT_NONE NO_SANITIZE_ADDR void __asan_after_dynamic_init(void) {}

OPT_NONE NO_SANITIZE_ADDR void __asan_register_image_globals(uptr flag) {}

OPT_NONE NO_SANITIZE_ADDR void __asan_unregister_image_globals(uptr flag) {}

OPT_NONE NO_SANITIZE_ADDR void __asan_register_elf_globals(uptr flag, uptr start, uptr stop) {}

OPT_NONE NO_SANITIZE_ADDR void __asan_unregister_elf_globals(uptr flag, uptr start, uptr stop) {}

OPT_NONE NO_SANITIZE_ADDR void __asan_init(void) {}

OPT_NONE NO_SANITIZE_ADDR void __asan_version_mismatch_check_v8(void) {}

