//===------------------------ libunwind_ext.h -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//
//  Extensions to libunwind API.
//
//===----------------------------------------------------------------------===//

#ifndef __LIBUNWIND_EXT__
#define __LIBUNWIND_EXT__

#include <libunwind.h>

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

#ifdef __cplusplus
}
#endif

#endif // __LIBUNWIND_EXT__
