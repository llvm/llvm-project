```{title} clang-tidy - misc-function-pointer-equality
```

# misc-function-pointer-equality

Detects equality comparisons between function pointers and standard containers
keyed by function pointers, which can behave unpredictably or unsoundly when
aggressive Identical Code Folding (`--icf=all`) is enabled.

For example, when linking with `--icf=all`, the linker merges distinct
functions that have identical machine code into a single copy in the final
binary. As a result, two distinct functions can share the same address at
runtime.

This warns on:

- Equality (`==`) and inequality (`!=`) comparisons between two function
  pointers may evaluate unexpectedly.
- Various associative and unordered standard library containers (e.g.
  `std::set`, `std::multiset`, or `std::map`) that are keyed by
  function pointers rely on address ordering or equality, which can cause
  distinct functions to collide as equivalent keys or produce non-deterministic
  ordering. This can be configured with `KeyedContainerTypes`.

This check exists to determine if `--icf=all` could be introducing miscompiles
in some common cases. **If you do not ever intend to enable `--icf=all`, you
can safely ignore this check.**

Some examples of what is linted:

```c++
#include <unordered_map>

void f();
void g();

bool compare() {
  // Warning: function pointer equality comparison is unsound with '--icf=all'
  // enabled.
  return f == g;
}

// Warning: container 'callbacks' with function pointer key is unsound
// with '--icf=all' enabled.
std::unordered_map<void (*)(), int> callbacks;
```

Comparisons against null pointer constants (`nullptr`, `NULL`, `0`) or `nullptr_t`
are not flagged.

```c++
bool is_null(void (*fp)()) {
  return fp == nullptr;
}
```

## Options

```{option} KeyedContainerTypes

Semicolon-separated list of container regex patterns to check for whether their
first template argument is a function pointer. For many associative and
associative-like containers, this is the key type. Default is
`::std::flat_map;::std::flat_set;::std::map;::std::multimap;::std::multiset;::std::set;::std::flat_multimap;::std::flat_multiset;::std::unordered_map;::std::unordered_multimap;::std::unordered_multiset;::std::unordered_set`
```


