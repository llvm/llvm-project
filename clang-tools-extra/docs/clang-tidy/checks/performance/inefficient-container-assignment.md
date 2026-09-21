```{title} clang-tidy - performance-inefficient-container-assignment
```

# performance-inefficient-container-assignment

Finds assignments of a freshly constructed temporary container to a container
of the same type, such as `v = std::vector<int>(n, 0);`, and suggests the
`assign` member function or an equivalent in-place rewrite. The temporary
allocates its own buffer, and the move assignment then discards the buffer the
destination already owns. `assign` writes into the existing buffer and only
allocates when the new contents do not fit.

```cpp
std::vector<int> v;
std::string s;

v = std::vector<int>(n, 0);           // fix-it: v.assign(n, 0);
v = std::vector<int>(first, last);    // fix-it: v.assign(first, last);
v = std::vector<int>{1, 2, 3};        // fix-it: v = {1, 2, 3};
v = std::vector<int>(n);              // fix-it: v.clear(); v.resize(n);
v = std::vector<int>(other);          // fix-it: v = other;
s = std::string(n, ' ');              // fix-it: s.assign(n, ' ');
s = std::string(str, pos, count);     // fix-it: s.assign(str, pos, count);
```

A braced list that does not fit the element type, such as
`v = {first, last};`, list-initializes the same kind of temporary through a
multi-argument constructor and is diagnosed as well.

Every rewrite leaves the destination with exactly the elements the assignment
would have produced:

- The arguments of a constructor taking two or more arguments are passed to
  `assign` unchanged. Every such constructor of a standard sequence container
  has an `assign` overload with the same parameters. In C++23,
  `Container(std::from_range, r)` becomes `assign_range(r)`.
- An initializer list is assigned directly; `operator=(std::initializer_list)`
  is specified to behave like `assign`.
- A single element count, `Container(n)`, becomes `clear()` followed by
  `resize(n)`, which value-initializes the elements in place. This takes two
  statements, so it is only rewritten when the assignment is a statement of
  its own inside a block, the destination has no side effects, and the count
  neither has side effects nor mentions the destination, because the count
  is evaluated after `clear()`.
- A copy or move of another container of the same type, `Container(other)`,
  becomes `= other`.

The rewrites work in place. If copying an element throws partway through, the
destination is left in a valid but unspecified state, whereas the original
assignment would have left it unchanged. For trivially copyable elements, the
common case, there is no difference.

The following are not diagnosed:

- `v = Container();` and `v = Container{};` release the storage of `v`;
  `clear()` would keep the capacity, which is a different behavior.
- `v = std::vector<int>(v);`, the idiom for trimming the capacity of `v` to
  its size; there the temporary is the point.
- Constructors with an explicitly passed allocator; `assign` has no allocator
  parameter.
- A single argument of another type, such as `std::string(ptr)`; the temporary
  is the conversion itself, not a copy that `assign` could avoid.
- Initializations such as `std::vector<int> v = std::vector<int>(n, 0);`;
  guaranteed copy elision already makes them cheap.

The warning is emitted without a fix-it when an iterator, pointer or element
argument refers to the destination, for example
`v = std::vector<int>(v.begin() + 1, v.end());`, because `assign` must not be
given iterators into the container it replaces (an `erase` is the right
rewrite for that example; a count such as `v.size()` is computed before
`assign` runs and is fine); when the value of the assignment is used, because
`assign`, `clear` and `resize` do not yield the container; and inside macro
expansions. The check cannot see through pointers or iterators that are stored
in other variables, so the fix-it assumes such arguments do not refer into the
destination.

## Options

```{option} ContainerClasses

Semicolon-separated list of fully qualified names of container classes to
consider. The classes must follow the conventions of the standard sequence
containers: every constructor taking two or more arguments, other than a
trailing allocator, has an `assign` overload with the same parameters; a
single integral constructor argument is an element count for `resize`; and
`operator=` accepts an initializer list. `::llvm::SmallVector` is an example
of a compatible class. Default is
`::std::vector;::std::deque;::std::list;::std::forward_list;::std::basic_string`.
```
