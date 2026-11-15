```{title} clang-tidy - bugprone-rethrow-caught-exception
```

# bugprone-rethrow-caught-exception

Flags `throw` expressions that copy a caught exception variable instead of rethrowing the active exception with a bare `throw;`.

Throwing the catch variable by name (e.g. `throw e;`) copies the exception into a new object with the static catch type. This can slice a derived exception, adds a copy/move, and suggests the code intentionally throws a new exception when it meant to rethrow.

```c++
try {
  f();
} catch (const std::exception &e) {
  log(e.what());
  throw e; // warning: copies `e`
}
```

Use a bare `throw;` to rethrow the original exception object:

```c++
try {
  f();
} catch (const std::exception &e) {
  log(e.what());
  throw;
}
```

Only exceptions caught by reference are flagged. Exceptions caught by value are ignored because modifying and throwing the local copy is a different pattern. Throws inside nested handlers, lambda bodies, or blocks that merely name an outer handler's variable are ignored because a bare `throw;` there would not rethrow the same exception. Lambda capture initializers are not traversed by matchers, so they are never flagged either. An explicit construction such as `throw E(e)` is also ignored as it expresses a new object rather than a plain rethrow. Throws in unevaluated operands (`requires`, `noexcept`, `decltype`) are ignored because nothing is thrown there.
