# misc-use-bulk-insert

Detects range-based `for` loops that insert elements into associative containers one at a time and suggests replacing them with a bulk `insert()` call.

For example:

```cpp
for (int i : in) {
  out.insert(i);
}
```

becomes:

```cpp
out.insert(in.begin(), in.end());
```

The initial implementation covers standard associative containers, including `std::set`, `std::map`, and `std::unordered_*` variants.
