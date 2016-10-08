//===------------------------- fallback_malloc.h --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef _FALLBACK_MALLOC_H
#define _FALLBACK_MALLOC_H

#include <cstddef> // for size_t

namespace __cxxabiv1 {

#pragma GCC visibility push(hidden)

// Allocate some memory from _somewhere_
void * __malloc_with_fallback(size_t size);

// Allocate and zero-initialize memory from _somewhere_
void * __calloc_with_fallback(size_t count, size_t size);

void __free_with_fallback(void *ptr);

#pragma GCC visibility pop

} // namespace __cxxabiv1

#endif
