```{title} clang-tidy - readability-redundant-zero-initializer
```

# readability-redundant-zero-initializer

Finds explicit zero initializers of arrays that can be replaced with empty
braces.

In C++ and since C23, an empty braced initializer zero-initializes every element
of an array, so an explicit `{0}` is redundant.

```cpp
char a[12] = {0};
int b[5] = {0};

// becomes

char a[12] = {};
int b[5] = {};
```

The check is only enabled in C++ and in C23 or later.

## Limitations

To keep the fix always safe, the check is intentionally conservative and only
handles single-element `{0}` initializers of arrays with an explicit bound.
It does not flag, among others:

- scalars (`int x = {0};`) and class or struct types (`S s = {0};`);
- arrays whose bound is deduced from the initializer (`char a[] = {0};`),
  where `{}` would change the deduced size;
- multi-dimensional arrays (`int m[2][3] = {0};`);
- initializers with more than one element (`int a[3] = {0, 0};`);
- zero written as something other than the integer literal `0`, for example
  `'\0'`, `0.0` or `nullptr`.
