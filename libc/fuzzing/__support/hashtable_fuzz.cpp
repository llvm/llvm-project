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
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/string.h"
#include "src/__support/CPP/utility/forward.h"
#include "src/__support/HashTable/table.h"
#include <stdint.h>
namespace LIBC_NAMESPACE {

template <typename T> class UniquePtr {
  T *ptr;

public:
  UniquePtr(T *ptr) : ptr(ptr) {}
  ~UniquePtr() { delete ptr; }
  UniquePtr(UniquePtr &&other) : ptr(other.ptr) { other.ptr = nullptr; }
  UniquePtr &operator=(UniquePtr &&other) {
    delete ptr;
    ptr = other.ptr;
    other.ptr = nullptr;
    return *this;
  }
  T *operator->() { return ptr; }
  template <typename... U> static UniquePtr create(U &&...x) {
    AllocChecker ac;
    T *ptr = new (ac) T(cpp::forward<U>(x)...);
    if (!ac)
      return {nullptr};
    return UniquePtr(ptr);
  }
  operator bool() { return ptr != nullptr; }
  T *get() { return ptr; }
};

// a tagged union
struct Action {
  enum class Tag { Find, Insert, CrossCheck } tag;
  cpp::string key;
  UniquePtr<Action> next;
  Action(Tag tag, cpp::string key, UniquePtr<Action> next)
      : tag(tag), key(cpp::move(key)), next(cpp::move(next)) {}
};

static struct {
  UniquePtr<Action> actions = nullptr;
  size_t remaining;
  const char *buffer;

  template <typename T> cpp::optional<T> next() {
    static_assert(cpp::is_integral<T>::value, "T must be an integral type");
    union {
      T result;
      char data[sizeof(T)];
    };
    if (remaining < sizeof(result))
      return cpp::nullopt;
    for (size_t i = 0; i < sizeof(result); i++)
      data[i] = buffer[i];
    buffer += sizeof(result);
    remaining -= sizeof(result);
    return result;
  }

  cpp::optional<cpp::string> next_string() {
    if (cpp::optional<uint16_t> len = next<uint16_t>()) {
      uint64_t length;
      for (length = 0; length < *len && length < remaining; length++)
        if (buffer[length] == '\0')
          break;
      cpp::string result(buffer, length);
      result += '\0';
      buffer += length;
      remaining -= length;
      return result;
    }
    return cpp::nullopt;
  }
  Action *next_action() {
    if (cpp::optional<uint8_t> action = next<uint8_t>()) {
      switch (*action % 3) {
      case 0: {
        if (cpp::optional<cpp::string> key = next_string())
          actions = UniquePtr<Action>::create(
              Action::Tag::Find, cpp::move(*key), cpp::move(actions));
        else
          return nullptr;
        break;
      }
      case 1: {
        if (cpp::optional<cpp::string> key = next_string())
          actions = UniquePtr<Action>::create(
              Action::Tag::Insert, cpp::move(*key), cpp::move(actions));
        else
          return nullptr;
        break;
      }
      case 2: {
        actions = UniquePtr<Action>::create(Action::Tag::CrossCheck, "",
                                            cpp::move(actions));
        break;
      }
      }
      return actions.get();
    }
    return nullptr;
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
  if (cpp::optional<uint16_t> size = global_status.next<uint16_t>())
    if (cpp::optional<uint64_t> seed = global_status.next<uint64_t>())
      return HashTable(*size, *seed);

  return HashTable(0, 0);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  global_status.buffer = reinterpret_cast<const char *>(data);
  global_status.remaining = size;
  HashTable table_a = next_hashtable();
  HashTable table_b = next_hashtable();
  if (!table_a.is_valid() || !table_b.is_valid())
    return 0;

  for (;;) {
    Action *action = global_status.next_action();
    if (!action)
      return 0;
    switch (action->tag) {
    case Action::Tag::Find: {
      if (static_cast<bool>(table_a.find(action->key.c_str())) !=
          static_cast<bool>(table_b.find(action->key.c_str())))
        __builtin_trap();
      break;
    }
    case Action::Tag::Insert: {
      ENTRY *a = table_a.insert(ENTRY{action->key.data(), action->key.data()});
      ENTRY *b = table_b.insert(ENTRY{action->key.data(), action->key.data()});
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

} // namespace LIBC_NAMESPACE
