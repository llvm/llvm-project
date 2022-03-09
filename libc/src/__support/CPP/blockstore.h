//===-- A data structure which stores data in blocks  -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SUPPORT_CPP_BLOCKSTORE_H
#define LLVM_LIBC_SUPPORT_CPP_BLOCKSTORE_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

namespace __llvm_libc {
namespace cpp {

// The difference between BlockStore a traditional vector types is that,
// when more capacity is desired, a new block is added instead of allocating
// a larger sized array and copying over existing items to the new allocation.
// Also, the initial block does not need heap allocation. Hence, a BlockStore is
// suitable for global objects as it does not require explicit construction.
// Also, the destructor of this class does nothing, which eliminates the need
// for an atexit global object destruction. But, it also means that the global
// object should be explicitly cleaned up at the appropriate time.
//
// If REVERSE_ORDER is true, the iteration of elements will in the reverse
// order. Also, since REVERSE_ORDER is a constexpr, conditionals branching
// on its value will be optimized out in the code below.
template <typename T, size_t BLOCK_SIZE, bool REVERSE_ORDER = false>
class BlockStore {
protected:
  struct Block {
    alignas(T) uint8_t data[BLOCK_SIZE * sizeof(T)] = {0};
    Block *next = nullptr;
  };

  Block first;
  Block *current = &first;
  size_t fill_count = 0;

public:
  constexpr BlockStore() = default;
  ~BlockStore() = default;

  class iterator {
    Block *block;
    size_t index;

  public:
    constexpr iterator(Block *b, size_t i) : block(b), index(i) {}

    iterator &operator++() {
      if (REVERSE_ORDER) {
        if (index == 0)
          return *this;

        --index;
        if (index == 0 && block->next != nullptr) {
          index = BLOCK_SIZE;
          block = block->next;
        }
      } else {
        if (index == BLOCK_SIZE)
          return *this;

        ++index;
        if (index == BLOCK_SIZE && block->next != nullptr) {
          index = 0;
          block = block->next;
        }
      }

      return *this;
    }

    T &operator*() {
      size_t true_index = REVERSE_ORDER ? index - 1 : index;
      return *reinterpret_cast<T *>(block->data + sizeof(T) * true_index);
    }

    bool operator==(const iterator &rhs) const {
      return block == rhs.block && index == rhs.index;
    }

    bool operator!=(const iterator &rhs) const {
      return block != rhs.block || index != rhs.index;
    }
  };

  static void destroy(BlockStore<T, BLOCK_SIZE, REVERSE_ORDER> *block_store);

  T *new_obj() {
    if (fill_count == BLOCK_SIZE) {
      auto new_block = reinterpret_cast<Block *>(::malloc(sizeof(Block)));
      if (REVERSE_ORDER) {
        new_block->next = current;
      } else {
        new_block->next = nullptr;
        current->next = new_block;
      }
      current = new_block;
      fill_count = 0;
    }
    T *obj = reinterpret_cast<T *>(current->data + fill_count * sizeof(T));
    ++fill_count;
    return obj;
  }

  void push_back(const T &value) {
    T *ptr = new_obj();
    *ptr = value;
  }

  iterator begin() {
    if (REVERSE_ORDER)
      return iterator(current, fill_count);
    else
      return iterator(&first, 0);
  }

  iterator end() {
    if (REVERSE_ORDER)
      return iterator(&first, 0);
    else
      return iterator(current, fill_count);
  }
};

template <typename T, size_t BLOCK_SIZE, bool REVERSE_ORDER>
void BlockStore<T, BLOCK_SIZE, REVERSE_ORDER>::destroy(
    BlockStore<T, BLOCK_SIZE, REVERSE_ORDER> *block_store) {
  if (REVERSE_ORDER) {
    auto current = block_store->current;
    while (current->next != nullptr) {
      auto temp = current;
      current = current->next;
      free(temp);
    }
  } else {
    auto current = block_store->first.next;
    while (current != nullptr) {
      auto temp = current;
      current = current->next;
      free(temp);
    }
  }
  block_store->current = nullptr;
  block_store->fill_count = 0;
}

// A convenience type for reverse order block stores.
template <typename T, size_t BLOCK_SIZE>
using ReverseOrderBlockStore = BlockStore<T, BLOCK_SIZE, true>;

} // namespace cpp
} // namespace __llvm_libc

#endif // LLVM_LIBC_SUPPORT_CPP_BLOCKSTORE_H
