```{title} clang-tidy - readability-const-return-type
```

# readability-const-return-type

Checks for functions with a `const`-qualified return type and recommends
removal of the `const` keyword. Such use of `const` is usually superfluous,
and can prevent valuable compiler optimizations. Does not (yet) fix trailing
return types.

Examples:

```c++
const int foo();
const Clazz foo();
Clazz *const foo();
```

Note that this applies strictly to top-level qualification, which excludes
pointers or references to const values. For example, these are fine:

```c++
const int* foo();
const int& foo();
const Clazz* foo();
```

## Options

```{option} IgnoreMacros
When `true`, the check will not give warnings inside macros.
Default is `true`.
```
