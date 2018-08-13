//===-- hwasan_interface_internal.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of HWAddressSanitizer.
//
// Private Hwasan interface header.
//===----------------------------------------------------------------------===//

#ifndef HWASAN_INTERFACE_INTERNAL_H
#define HWASAN_INTERFACE_INTERNAL_H

#include "sanitizer_common/sanitizer_internal_defs.h"

extern "C" {

SANITIZER_INTERFACE_ATTRIBUTE
void __hwasan_shadow_init();

SANITIZER_INTERFACE_ATTRIBUTE
void __hwasan_init();

using __sanitizer::uptr;
using __sanitizer::sptr;
using __sanitizer::uu64;
using __sanitizer::uu32;
using __sanitizer::uu16;
using __sanitizer::u64;
using __sanitizer::u32;
using __sanitizer::u16;
using __sanitizer::u8;

SANITIZER_INTERFACE_ATTRIBUTE
extern uptr __hwasan_shadow_memory_dynamic_address;

SANITIZER_INTERFACE_ATTRIBUTE
void __hwasan_loadN(uptr, uptr);
SANITIZER_INTERFACE_ATTRIBUTE
void __hwasan_load1(uptr);
SANITIZER_INTERFACE_ATTRIBUTE
void __hwasan_load2(uptr);
SANITIZER_INTERFACE_ATTRIBUTE
void __hwasan_load4(uptr);
SANITIZER_INTERFACE_ATTRIBUTE
void __hwasan_load8(uptr);
SANITIZER_INTERFACE_ATTRIBUTE
void __hwasan_load16(uptr);

SANITIZER_INTERFACE_ATTRIBUTE
void __hwasan_loadN_noabort(uptr, uptr);
SANITIZER_INTERFACE_ATTRIBUTE
void __hwasan_load1_noabort(uptr);
SANITIZER_INTERFACE_ATTRIBUTE
void __hwasan_load2_noabort(uptr);
SANITIZER_INTERFACE_ATTRIBUTE
void __hwasan_load4_noabort(uptr);
SANITIZER_INTERFACE_ATTRIBUTE
void __hwasan_load8_noabort(uptr);
SANITIZER_INTERFACE_ATTRIBUTE
void __hwasan_load16_noabort(uptr);

SANITIZER_INTERFACE_ATTRIBUTE
void __hwasan_storeN(uptr, uptr);
SANITIZER_INTERFACE_ATTRIBUTE
void __hwasan_store1(uptr);
SANITIZER_INTERFACE_ATTRIBUTE
void __hwasan_store2(uptr);
SANITIZER_INTERFACE_ATTRIBUTE
void __hwasan_store4(uptr);
SANITIZER_INTERFACE_ATTRIBUTE
void __hwasan_store8(uptr);
SANITIZER_INTERFACE_ATTRIBUTE
void __hwasan_store16(uptr);

SANITIZER_INTERFACE_ATTRIBUTE
void __hwasan_storeN_noabort(uptr, uptr);
SANITIZER_INTERFACE_ATTRIBUTE
void __hwasan_store1_noabort(uptr);
SANITIZER_INTERFACE_ATTRIBUTE
void __hwasan_store2_noabort(uptr);
SANITIZER_INTERFACE_ATTRIBUTE
void __hwasan_store4_noabort(uptr);
SANITIZER_INTERFACE_ATTRIBUTE
void __hwasan_store8_noabort(uptr);
SANITIZER_INTERFACE_ATTRIBUTE
void __hwasan_store16_noabort(uptr);

SANITIZER_INTERFACE_ATTRIBUTE
void __hwasan_tag_memory(uptr p, u8 tag, uptr sz);

SANITIZER_INTERFACE_ATTRIBUTE
u8 __hwasan_generate_tag();

// Returns the offset of the first tag mismatch or -1 if the whole range is
// good.
SANITIZER_INTERFACE_ATTRIBUTE
sptr __hwasan_test_shadow(const void *x, uptr size);

SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
/* OPTIONAL */ const char* __hwasan_default_options();

SANITIZER_INTERFACE_ATTRIBUTE
void __hwasan_print_shadow(const void *x, uptr size);

SANITIZER_INTERFACE_ATTRIBUTE
u16 __sanitizer_unaligned_load16(const uu16 *p);

SANITIZER_INTERFACE_ATTRIBUTE
u32 __sanitizer_unaligned_load32(const uu32 *p);

SANITIZER_INTERFACE_ATTRIBUTE
u64 __sanitizer_unaligned_load64(const uu64 *p);

SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_unaligned_store16(uu16 *p, u16 x);

SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_unaligned_store32(uu32 *p, u32 x);

SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_unaligned_store64(uu64 *p, u64 x);

SANITIZER_INTERFACE_ATTRIBUTE
void __hwasan_enable_allocator_tagging();

SANITIZER_INTERFACE_ATTRIBUTE
void __hwasan_disable_allocator_tagging();

}  // extern "C"

#endif  // HWASAN_INTERFACE_INTERNAL_H
