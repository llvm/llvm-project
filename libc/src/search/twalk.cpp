//===-- Implementation of twalk --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/search/twalk.h"
#include "hdr/types/posix_tnode.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/weak_avl.h"

namespace LIBC_NAMESPACE_DECL {

using Node = WeakAVLNode<const void *>;

LLVM_LIBC_FUNCTION(void, twalk,
                   (const __llvm_libc_tnode *root,
                    void (*action)(const __llvm_libc_tnode *, VISIT, int))) {
  if (!root)
    return;
  const Node *node = reinterpret_cast<const Node *>(root);
  Node::walk(node, [action](const Node *n, Node::WalkType type, int depth) {
    VISIT v = (type == Node::WalkType::PreOrder)    ? preorder
              : (type == Node::WalkType::InOrder)   ? postorder
              : (type == Node::WalkType::PostOrder) ? endorder
                                                    : leaf;
    action(reinterpret_cast<const __llvm_libc_tnode *>(n), v, depth);
  });
}

} // namespace LIBC_NAMESPACE_DECL
