#include "src/__support/CPP/new.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/HashTable/table.h"
#include "src/string/memcpy.h"
#include <search.h>
#include <stdint.h>
namespace LIBC_NAMESPACE {

enum class Action { Find, Insert, CrossCheck };
static uint8_t *global_buffer = nullptr;
static size_t remaining = 0;

static cpp::optional<uint8_t> next_u8() {
  if (remaining == 0)
    return cpp::nullopt;
  uint8_t result = *global_buffer;
  global_buffer++;
  remaining--;
  return result;
}

static cpp::optional<uint64_t> next_uint64() {
  uint64_t result;
  if (remaining < sizeof(result))
    return cpp::nullopt;
  memcpy(&result, global_buffer, sizeof(result));
  global_buffer += sizeof(result);
  remaining -= sizeof(result);
  return result;
}

static cpp::optional<Action> next_action() {
  if (cpp::optional<uint8_t> action = next_u8()) {
    switch (*action % 3) {
    case 0:
      return Action::Find;
    case 1:
      return Action::Insert;
    case 2:
      return Action::CrossCheck;
    }
  }
  return cpp::nullopt;
}

static cpp::optional<char *> next_cstr() {
  char *result = reinterpret_cast<char *>(global_buffer);
  if (cpp::optional<uint64_t> len = next_uint64()) {
    uint64_t length;
    for (length = 0; length < *len % 128; length++) {
      if (length >= remaining)
        return cpp::nullopt;
      if (*global_buffer == '\0')
        break;
    }
    if (length >= remaining)
      return cpp::nullopt;
    global_buffer[length] = '\0';
    global_buffer += length + 1;
    remaining -= length + 1;
    return result;
  }
  return cpp::nullopt;
}

#define GET_VAL(op)                                                            \
  __extension__({                                                              \
    auto val = op();                                                           \
    if (!val)                                                                  \
      return 0;                                                                \
    *val;                                                                      \
  })

template <typename Fn> struct CleanUpHook {
  cpp::optional<Fn> fn;
  ~CleanUpHook() {
    if (fn)
      (*fn)();
  }
  CleanUpHook(Fn fn) : fn(cpp::move(fn)) {}
  CleanUpHook(const CleanUpHook &) = delete;
  CleanUpHook(CleanUpHook &&other) : fn(cpp::move(other.fn)) {
    other.fn = cpp::nullopt;
  }
};

#define register_cleanup(ID, ...)                                              \
  auto cleanup_hook##ID = __extension__({                                      \
    auto a = __VA_ARGS__;                                                      \
    CleanUpHook<decltype(a)>{a};                                               \
  });

static void trap_with_message(const char *msg) { __builtin_trap(); }

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  AllocChecker ac;
  global_buffer = static_cast<uint8_t *>(::operator new(size, ac));
  register_cleanup(0, [global_buffer = global_buffer, size] {
    ::operator delete(global_buffer, size);
  });
  if (!ac)
    return 0;
  memcpy(global_buffer, data, size);

  remaining = size;
  uint64_t size_a = GET_VAL(next_uint64) % 256;
  uint64_t size_b = GET_VAL(next_uint64) % 256;
  uint64_t rand_a = GET_VAL(next_uint64);
  uint64_t rand_b = GET_VAL(next_uint64);
  internal::HashTable *table_a = internal::HashTable::allocate(size_a, rand_a);
  register_cleanup(1, [&table_a] {
    if (table_a)
      internal::HashTable::deallocate(table_a);
  });
  internal::HashTable *table_b = internal::HashTable::allocate(size_b, rand_b);
  register_cleanup(2, [&table_b] {
    if (table_b)
      internal::HashTable::deallocate(table_b);
  });
  if (!table_a || !table_b)
    return 0;
  for (;;) {
    Action action = GET_VAL(next_action);
    switch (action) {
    case Action::Find: {
      const char *key = GET_VAL(next_cstr);
      if (static_cast<bool>(table_a->find(key)) !=
          static_cast<bool>(table_b->find(key)))
        trap_with_message(key);
      break;
    }
    case Action::Insert: {
      char *key = GET_VAL(next_cstr);
      ENTRY *a = internal::HashTable::insert(table_a, ENTRY{key, key});
      ENTRY *b = internal::HashTable::insert(table_b, ENTRY{key, key});
      if (a->data != b->data)
        __builtin_trap();
      break;
    }
    case Action::CrossCheck: {
      for (ENTRY a : *table_a) {
        if (const ENTRY *b = table_b->find(a.key)) {
          if (a.data != b->data)
            __builtin_trap();
        }
      }
      for (ENTRY b : *table_b) {
        if (const ENTRY *a = table_a->find(b.key)) {
          if (a->data != b.data)
            __builtin_trap();
        }
      }
      break;
    }
    }
  }
}

} // namespace LIBC_NAMESPACE
