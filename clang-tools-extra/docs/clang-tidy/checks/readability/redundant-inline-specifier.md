```{title} clang-tidy - readability-redundant-inline-specifier
```

# readability-redundant-inline-specifier

Detects redundant `inline` specifiers on function and variable declarations.

Examples:

```c++
constexpr inline void f() {}
```

In the example above the keyword `inline` is redundant since constexpr
functions are implicitly inlined

```c++
class MyClass {
    inline void myMethod() {}
};
```

In the example above the keyword `inline` is redundant since member functions
defined entirely inside a class/struct/union definition are implicitly inlined.

## Options

```{option} StrictMode
When `true`, the check will also flag functions and variables that
already have internal linkage as redundant. Default is `false`.
```
