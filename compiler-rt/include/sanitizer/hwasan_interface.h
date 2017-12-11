//===-- sanitizer/asan_interface.h ------------------------------*- C++ -*-===//
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
// Public interface header.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_HWASAN_INTERFACE_H
#define SANITIZER_HWASAN_INTERFACE_H

#include <sanitizer/common_interface_defs.h>

#ifdef __cplusplus
extern "C" {
#endif
  // This function may be optionally provided by user and should return
  // a string containing HWASan runtime options. See asan_flags.h for details.
  const char* __hwasan_default_options();

  void __hwasan_enable_allocator_tagging();
  void __hwasan_disable_allocator_tagging();

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SANITIZER_HWASAN_INTERFACE_H
