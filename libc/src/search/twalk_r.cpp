//===-- Implementation of twalk_r ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/search/twalk_r.h"
#include "hdr/types/posix_tnode.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/weak_avl.h"

namespace LIBC_NAMESPACE_DECL {

using Node = WeakAVLNode<const void *>;

LLVM_LIBC_FUNCTION(void, twalk_r,
                   (const __llvm_libc_tnode *root,
                    void (*action)(const __llvm_libc_tnode *, VISIT, void *),
                    void *closure)) {
  if (!root)
    return;
  const Node *node = reinterpret_cast<const Node *>(root);
  Node::walk(node, [action, closure](const Node *n, Node::WalkType type, int) {
    VISIT v = (type == Node::WalkType::PreOrder)    ? preorder
              : (type == Node::WalkType::InOrder)   ? postorder
              : (type == Node::WalkType::PostOrder) ? endorder
                                                    : leaf;
    action(reinterpret_cast<const __llvm_libc_tnode *>(n), v, closure);
  });
}

} // namespace LIBC_NAMESPACE_DECL
