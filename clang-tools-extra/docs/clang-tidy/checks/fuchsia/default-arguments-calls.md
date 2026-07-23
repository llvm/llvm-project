```{title} clang-tidy - fuchsia-default-arguments-calls
```

# fuchsia-default-arguments-calls

Warns if a function or method is called with default arguments.

For example, given the declaration:

```cpp
int foo(int value = 5) { return value; }
```

A function call expression that uses a default argument will be diagnosed.
Calling it without defaults will not cause a warning:

```cpp
foo();  // warning
foo(0); // no warning
```

See the features disallowed in Fuchsia at <https://fuchsia.dev/fuchsia-src/development/languages/c-cpp/cxx?hl=en>
