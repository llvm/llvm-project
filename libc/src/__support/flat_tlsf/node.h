//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Provide Node struct for the flat_tlsf allocator.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FLAT_TLSF_NODE_H
#define LLVM_LIBC_SRC___SUPPORT_FLAT_TLSF_NODE_H

#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace flat_tlsf {

struct Node {
  Node *next;
  // Use next_of_prev to avoid branches on special guardian pointer.
  Node **next_of_prev;

  LIBC_INLINE Node **addr_of_next() { return &next; }

  LIBC_INLINE void link_at(Node data) {
    *this = data;
    *data.next_of_prev = this;
    if (data.next)
      data.next->next_of_prev = addr_of_next();
  }

  LIBC_INLINE void unlink() {
    *next_of_prev = next;
    if (next)
      next->next_of_prev = next_of_prev;
  }
};

} // namespace flat_tlsf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FLAT_TLSF_NODE_H
