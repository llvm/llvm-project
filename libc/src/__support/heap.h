//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Provide a generic Heap interface to switch between allocator
/// implementations.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_HEAP_H
#define LLVM_LIBC_SRC___SUPPORT_HEAP_H

#include "src/__support/macros/config.h"

#if defined(LIBC_HEAP_TYPE_FLAT_TLSF)
#include "src/__support/flat_tlsf/global.h"
#else
#include "src/__support/freelist_heap.h"
#endif

namespace LIBC_NAMESPACE_DECL {

#if defined(LIBC_HEAP_TYPE_FLAT_TLSF)
using GlobalHeap = flat_tlsf::FlatTlsfHeap;
extern flat_tlsf::FlatTlsfHeap *global_heap;
#else
using GlobalHeap = FreeListHeap;
extern FreeListHeap *global_heap;
#endif

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_HEAP_H
