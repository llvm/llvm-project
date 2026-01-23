//===-- weak_avl_fuzz.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Fuzzing test for llvm-libc weak AVL implementations.
///
//===----------------------------------------------------------------------===//
#include "hdr/types/ENTRY.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/macros/config.h"
#include "src/__support/weak_avl.h"

namespace LIBC_NAMESPACE_DECL {

// A sequence of actions:
// - Erase: a single byte valued (5, 6 mod 7) followed by an int
// - Find: a single byte valued (4 mod 7) followed by an int
// - FindOrInsert: a single byte valued (0,1,2,3 mod 7) followed by an int
extern "C" size_t LLVMFuzzerMutate(uint8_t *data, size_t size, size_t max_size);
extern "C" size_t LLVMFuzzerCustomMutator(uint8_t *data, size_t size,
                                          size_t max_size, unsigned int seed) {
  size = LLVMFuzzerMutate(data, size, max_size);
  return size / (1 + sizeof(int)) * (1 + sizeof(int));
}

class AVLTree {
  using Node = WeakAVLNode<int>;
  Node *root = nullptr;
  bool reversed = false;
  static int compare(int a, int b) { return (a > b) - (a < b); }
  static int reverse_compare(int a, int b) { return (b > a) - (b < a); }

public:
  AVLTree(bool reversed = false) : reversed(reversed) {}
  bool find(int key) {
    return Node::find(root, key, reversed ? reverse_compare : compare)
        .has_value();
  }
  bool find_or_insert(int key) {
    return Node::find_or_insert(root, key, reversed ? reverse_compare : compare)
        .has_value();
  }
  bool erase(int key) {
    if (cpp::optional<Node *> node =
            Node::find(root, key, reversed ? reverse_compare : compare)) {
      Node::erase(root, node.value());
      return true;
    }
    return false;
  }
  ~AVLTree() { Node::destroy(root); }
};

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  AVLTree tree1;
  AVLTree tree2(true);
  for (size_t i = 0; i + (1 + sizeof(int)) <= size; i += 1 + sizeof(int)) {
    uint8_t action = data[i];
    int key;
    __builtin_memcpy(&key, data + i + 1, sizeof(int));
    if (action % 7 == 4) {
      // Find
      bool res1 = tree1.find(key);
      bool res2 = tree2.find(key);
      if (res1 != res2)
        __builtin_trap();

    } else if (action % 7 == 5 || action % 7 == 6) {
      // Erase
      bool res1 = tree1.erase(key);
      bool res2 = tree2.erase(key);
      if (res1 != res2)
        __builtin_trap();
      if (tree1.find(key))
        __builtin_trap();
      if (tree2.find(key))
        __builtin_trap();
    } else {
      // FindOrInsert
      bool res1 = tree1.find_or_insert(key);
      bool res2 = tree2.find_or_insert(key);
      if (res1 != res2)
        __builtin_trap();
      if (!tree1.find(key))
        __builtin_trap();
      if (!tree2.find(key))
        __builtin_trap();
    }
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
