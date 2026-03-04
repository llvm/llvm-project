/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "asan_util.h"

USED NO_SANITIZE_ADDR void __asan_handle_no_return(void) {}

USED NO_SANITIZE_ADDR void __sanitizer_ptr_cmp(uptr a, uptr b) {}

USED NO_SANITIZE_ADDR void __sanitizer_ptr_sub(uptr a, uptr b) {}

USED NO_SANITIZE_ADDR void __asan_before_dynamic_init(uptr addr) {}

USED NO_SANITIZE_ADDR void __asan_after_dynamic_init(void) {}

USED NO_SANITIZE_ADDR void __asan_register_image_globals(uptr flag) {}

USED NO_SANITIZE_ADDR void __asan_unregister_image_globals(uptr flag) {}

USED NO_SANITIZE_ADDR void __asan_init(void) {}

USED NO_SANITIZE_ADDR void __asan_version_mismatch_check_v8(void) {}

