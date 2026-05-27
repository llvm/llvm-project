//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Provide global FlatTlsfHeap instance.
///
//===----------------------------------------------------------------------===//

#include "src/__support/flat_tlsf/global.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace flat_tlsf {

LIBC_CONSTINIT FlatTlsfHeap flat_tlsf_heap_symbols;
FlatTlsfHeap *flat_tlsf_heap = &flat_tlsf_heap_symbols;

} // namespace flat_tlsf
} // namespace LIBC_NAMESPACE_DECL
