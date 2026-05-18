//===-- Implementation of tfind ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/search/tfind.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/weak_avl.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(__llvm_libc_tnode *, tfind,
                   (const void *key, __llvm_libc_tnode *const *rootp,
                    int (*compar)(const void *, const void *))) {
  if (!rootp)
    return nullptr;
  using Node = WeakAVLNode<const void *>;
  Node *root = reinterpret_cast<Node *>(*rootp);
  Node::OptionalNodePtr node = Node::find(root, key, compar);
  return node ? reinterpret_cast<__llvm_libc_tnode *>(*node) : nullptr;
}

} // namespace LIBC_NAMESPACE_DECL
