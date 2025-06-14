//===-- tsan_interface_avx2.h ------------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
// The functions declared in this header will be inserted by the instrumentation
// module.
// This header can be included by the instrumented program or by TSan tests.
//===----------------------------------------------------------------------===//
#ifndef TSAN_INTERFACE_AVX2_H
#define TSAN_INTERFACE_AVX2_H

#include <immintrin.h>
#include <sanitizer_common/sanitizer_internal_defs.h>
#include <stdint.h>

// This header should NOT include any other headers.
// All functions in this header are extern "C" and start with __tsan_.

#ifdef __cplusplus
extern "C" {
#endif

#if !SANITIZER_GO
#  ifdef __AVX2__
SANITIZER_INTERFACE_ATTRIBUTE void __tsan_scatter_vector4(__m256i vaddr,
                                                          int width,
                                                          uint8_t mask);
SANITIZER_INTERFACE_ATTRIBUTE void __tsan_gather_vector4(__m256i vaddr,
                                                         int width,
                                                         uint8_t mask);
#  endif /*__AVX2__*/
#endif   // SANITIZER_GO

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /*TSAN_INTERFACE_AVX2_H*/
