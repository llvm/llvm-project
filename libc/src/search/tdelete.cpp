//===-- Implementation of tdelete -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/search/tdelete.h"
#include "hdr/types/posix_tnode.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/weak_avl.h"

namespace LIBC_NAMESPACE_DECL {

// The tdelete() function shall return a pointer to the parent of the deleted
// node, or an unspecified non-null pointer if the deleted node was the root
// node, or a null pointer if the node is not found.
LLVM_LIBC_FUNCTION(void *, tdelete,
                   (const void *key, __llvm_libc_tnode **rootp,
                    int (*compar)(const void *, const void *))) {
  if (!rootp)
    return nullptr;
  using Node = WeakAVLNode<const void *>;
  Node *&root = *reinterpret_cast<Node **>(rootp);
  Node::OptionalNodePtr node = Node::find(root, key, compar);
  if (!node)
    return nullptr;
  void *result = const_cast<Node *>(node.value()->get_parent());
  if (!result)
    result = cpp::bit_cast<void *>(cpp::numeric_limits<uintptr_t>::max());
  Node::erase(root, *node);
  return result;
}

} // namespace LIBC_NAMESPACE_DECL
