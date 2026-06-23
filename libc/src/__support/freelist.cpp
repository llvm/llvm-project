//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation for freelist.
///
//===----------------------------------------------------------------------===//

#include "freelist.h"

namespace LIBC_NAMESPACE_DECL {

void FreeList::push(Node *node, const FreeListSecrets &secrets) {
  if (begin_) {
    Node *begin_prev = secrets.decrypt_prev(begin_, begin_->prev);

    LIBC_HARDENING_ASSERT(secrets.decrypt_next(begin_prev->next) == begin_ &&
                          "Corrupted free list links (push check)");

    node->prev = secrets.encrypt_prev(node, begin_prev);
    node->next = secrets.encrypt_next(begin_);
    begin_prev->next = secrets.encrypt_next(node);
    begin_->prev = secrets.encrypt_prev(begin_, node);
  } else {
    begin_ = node;
    node->next = secrets.encrypt_next(node);
    node->prev = secrets.encrypt_prev(node, node);
  }
}

void FreeList::remove(Node *node, const FreeListSecrets &secrets) {
  LIBC_ASSERT(begin_ && "cannot remove from empty list");
  Node *node_next = secrets.decrypt_next(node->next);
  if (node == node_next) {
    LIBC_ASSERT(node == begin_ &&
                "a self-referential node must be the only element");
    begin_ = nullptr;
  } else {
    Node *node_prev = secrets.decrypt_prev(node, node->prev);

    LIBC_HARDENING_ASSERT(
        secrets.decrypt_next(node_prev->next) == node &&
        "Corrupted free list links (remove check prev->next)");
    LIBC_HARDENING_ASSERT(
        secrets.decrypt_prev(node_next, node_next->prev) == node &&
        "Corrupted free list links (remove check next->prev)");

    node_prev->next = secrets.encrypt_next(node_next);
    node_next->prev = secrets.encrypt_prev(node_next, node_prev);
    if (begin_ == node)
      begin_ = node_next;
  }
}

} // namespace LIBC_NAMESPACE_DECL
