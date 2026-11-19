```{title} clang-tidy - performance-inefficient-container-assignment
```

# performance-inefficient-container-assignment

Finds assignments of a temporary container to a container of the same type
that could reuse the storage of the destination instead. The temporary
allocates its own buffer, and the move assignment then frees the one that the
destination already owns.

```cpp
v = std::vector<int>(n, 0);                // v.assign(n, 0);
v = std::vector<int>(first, last);         // v.assign(first, last);
v = {first, last};                         // v.assign(first, last);
v = std::vector<int>(std::from_range, r);  // v.assign_range(r);
s = std::string(p, n);                     // s.assign(p, n);
```

The check does not diagnose assignments where the replacement would not
compile or would behave differently: when the elements are not assignable,
when an argument refers into the destination, and when an allocator is
passed. `v = Container()` is not diagnosed either, because it releases the
storage.

The fix-it is omitted inside macros, when the value of the assignment is
used, and when evaluating the destination before the arguments could change
the result.

Unlike the assignment, the replacements leave the destination in a valid but
unspecified state if copying an element throws.

## Options

```{option} ContainerClasses

Semicolon-separated list of fully qualified names of container classes with
the interface of the standard sequence containers. Default is
`::std::vector;::std::deque;::std::list;::std::forward_list;::std::basic_string`.
```
