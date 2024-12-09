//===-- hashtable_fuzz.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Fuzzing test for llvm-libc hashtable implementations.
///
//===----------------------------------------------------------------------===//
#include "include/llvm-libc-types/ENTRY.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/HashTable/table.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// A fuzzing payload starts with
// - uint16_t: initial capacity for table A
// - uint64_t: seed for table A
// - uint16_t: initial capacity for table B
// - uint64_t: seed for table B
// Followed by a sequence of actions:
// - CrossCheck: only a single byte valued (4 mod 5)
// - Find: a single byte valued (3 mod 5) followed by a null-terminated string
// - Insert: a single byte valued (0,1,2 mod 5) followed by a null-terminated
// string
static constexpr size_t INITIAL_HEADER_SIZE =
    2 * (sizeof(uint16_t) + sizeof(uint64_t));
extern "C" size_t LLVMFuzzerMutate(uint8_t *data, size_t size, size_t max_size);
extern "C" size_t LLVMFuzzerCustomMutator(uint8_t *data, size_t size,
                                          size_t max_size, unsigned int seed) {
  size = LLVMFuzzerMutate(data, size, max_size);
  // not enough to read the initial capacities and seeds
  if (size < INITIAL_HEADER_SIZE)
    return 0;

  // skip the initial capacities and seeds
  size_t i = INITIAL_HEADER_SIZE;
  while (i < size) {
    // cross check
    if (static_cast<uint8_t>(data[i]) % 5 == 4) {
      // skip the cross check byte
      ++i;
      continue;
    }

    // find or insert
    // check if there is enough space for the action byte and the
    // null-terminator
    if (i + 2 >= max_size)
      return i;
    // skip the action byte
    ++i;
    // skip the null-terminated string
    while (i < max_size && data[i] != 0)
      ++i;
    // in the case the string is not null-terminated, null-terminate it
    if (i == max_size && data[i - 1] != 0) {
      data[i - 1] = 0;
      return max_size;
    }

    // move to the next action
    ++i;
  }
  // return the new size
  return i;
}

// a tagged union
struct Action {
  enum class Tag { Find, Insert, CrossCheck } tag;
  cpp::string_view key;
};

static struct {
  size_t remaining;
  const char *buffer;

  template <typename T> T next() {
    static_assert(cpp::is_integral<T>::value, "T must be an integral type");

    char data[sizeof(T)];

    for (size_t i = 0; i < sizeof(T); i++)
      data[i] = buffer[i];
    buffer += sizeof(T);
    remaining -= sizeof(T);
    return cpp::bit_cast<T>(data);
  }

  cpp::string_view next_string() {
    cpp::string_view result(buffer);
    buffer = result.end() + 1;
    remaining -= result.size() + 1;
    return result;
  }

  Action next_action() {
    uint8_t byte = next<uint8_t>();
    switch (byte % 5) {
    case 4:
      return {Action::Tag::CrossCheck, {}};
    case 3:
      return {Action::Tag::Find, next_string()};
    default:
      return {Action::Tag::Insert, next_string()};
    }
  }
} global_status;

class HashTable {
  internal::HashTable *table;

public:
  HashTable(uint64_t size, uint64_t seed)
      : table(internal::HashTable::allocate(size, seed)) {}
  HashTable(internal::HashTable *table) : table(table) {}
  ~HashTable() { internal::HashTable::deallocate(table); }
  HashTable(HashTable &&other) : table(other.table) { other.table = nullptr; }
  bool is_valid() const { return table != nullptr; }
  ENTRY *find(const char *key) { return table->find(key); }
  ENTRY *insert(const ENTRY &entry) {
    return internal::HashTable::insert(this->table, entry);
  }
  using iterator = internal::HashTable::iterator;
  iterator begin() const { return table->begin(); }
  iterator end() const { return table->end(); }
};

HashTable next_hashtable() {
  size_t size = global_status.next<uint16_t>();
  uint64_t seed = global_status.next<uint64_t>();
  return HashTable(size, seed);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  global_status.buffer = reinterpret_cast<const char *>(data);
  global_status.remaining = size;
  if (global_status.remaining < INITIAL_HEADER_SIZE)
    return 0;

  HashTable table_a = next_hashtable();
  HashTable table_b = next_hashtable();
  for (;;) {
    if (global_status.remaining == 0)
      break;
    Action action = global_status.next_action();
    switch (action.tag) {
    case Action::Tag::Find: {
      if (static_cast<bool>(table_a.find(action.key.data())) !=
          static_cast<bool>(table_b.find(action.key.data())))
        __builtin_trap();
      break;
    }
    case Action::Tag::Insert: {
      char *ptr = const_cast<char *>(action.key.data());
      ENTRY *a = table_a.insert(ENTRY{ptr, ptr});
      ENTRY *b = table_b.insert(ENTRY{ptr, ptr});
      if (a->data != b->data)
        __builtin_trap();
      break;
    }
    case Action::Tag::CrossCheck: {
      for (ENTRY a : table_a)
        if (const ENTRY *b = table_b.find(a.key); a.data != b->data)
          __builtin_trap();

      for (ENTRY b : table_b)
        if (const ENTRY *a = table_a.find(b.key); a->data != b.data)
          __builtin_trap();

      break;
    }
    }
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
