```{title} clang-tidy - bugprone-unordered-equal-compare
```

# bugprone-unordered-equal-compare

Flags uses of `std::equal` to compare the ranges of two unordered containers
(`std::unordered_set`, `std::unordered_multiset`, `std::unordered_map` and
`std::unordered_multimap`).

The iteration order of an unordered container is unspecified, so two containers
holding the same elements may iterate them in a different order. Comparing their
ranges element by element with `std::equal` is therefore order-dependent and
can report equal containers as different. Use `operator==` instead, which
compares the containers as sets.

```cpp
std::unordered_set<int> a, b;

bool wrong = std::equal(a.begin(), a.end(), b.begin());  // warning
bool ok = (a == b);
```

## Options

```{option} Containers
A semicolon-separated list of the fully-qualified names of the unordered
container class templates to flag. This allows out-of-tree containers such as
the Boost ones to be added. Defaults to
`::std::unordered_set;::std::unordered_multiset;::std::unordered_map;::std::unordered_multimap`.
```
