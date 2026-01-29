//===-- Implementation of tdestroy ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/search/tdestroy.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/weak_avl.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, tdestroy,
                   (__llvm_libc_tnode * root, void (*free_node)(void *))) {
  if (!root)
    return;
  using Node = WeakAVLNode<const void *>;
  Node *node = reinterpret_cast<Node *>(root);
  Node::destroy(node, [free_node](const void *&data) {
    free_node(const_cast<void *>(data));
  });
}

} // namespace LIBC_NAMESPACE_DECL
