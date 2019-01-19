//===------------------------ libunwind_ext.h -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
//  Extensions to libunwind API.
//
//===----------------------------------------------------------------------===//

#ifndef __LIBUNWIND_EXT__
#define __LIBUNWIND_EXT__

#include "config.h"
#include <libunwind.h>
#include <unwind.h>

#define UNW_STEP_SUCCESS 1
#define UNW_STEP_END     0

#ifdef __cplusplus
extern "C" {
#endif
// SPI
extern void unw_iterate_dwarf_unwind_cache(void (*func)(unw_word_t ip_start,
                                                        unw_word_t ip_end,
                                                        unw_word_t fde,
                                                        unw_word_t mh));

// IPI
extern void _unw_add_dynamic_fde(unw_word_t fde);
extern void _unw_remove_dynamic_fde(unw_word_t fde);

#if defined(_LIBUNWIND_ARM_EHABI)
extern const uint32_t* decode_eht_entry(const uint32_t*, size_t*, size_t*);
extern _Unwind_Reason_Code _Unwind_VRS_Interpret(_Unwind_Context *context,
                                                 const uint32_t *data,
                                                 size_t offset, size_t len);
#endif

#ifdef __cplusplus
}
#endif

#endif // __LIBUNWIND_EXT__
