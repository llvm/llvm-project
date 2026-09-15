```{title} clang-tidy - modernize-use-if-consteval
```

# modernize-use-if-consteval

Replaces direct `std::is_constant_evaluated()` checks in `if` statements with
C++23's `if consteval` syntax.

```cpp
if (std::is_constant_evaluated())
  return slow_but_constexpr_path();
else
  return fast_runtime_path();
```

is rewritten as:

```cpp
if consteval {
  return slow_but_constexpr_path();
} else {
  return fast_runtime_path();
}
```

The direct negated form is rewritten to `if !consteval`.
