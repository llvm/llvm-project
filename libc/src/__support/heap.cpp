//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Provide global_heap pointer definition and initialization.
///
//===----------------------------------------------------------------------===//

#include "src/__support/heap.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

#if defined(LIBC_HEAP_TYPE_FLAT_TLSF)
namespace flat_tlsf {
extern FlatTlsfHeap flat_tlsf_heap_symbols;
} // namespace flat_tlsf
flat_tlsf::FlatTlsfHeap *global_heap = &flat_tlsf::flat_tlsf_heap_symbols;
#else
extern FreeListHeap freelist_heap_symbols;
FreeListHeap *global_heap = &freelist_heap_symbols;
#endif

} // namespace LIBC_NAMESPACE_DECL
