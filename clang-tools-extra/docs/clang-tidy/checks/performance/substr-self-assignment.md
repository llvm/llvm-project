```{title} clang-tidy - performance-substr-self-assignment
```

# performance-substr-self-assignment

Finds cases where a string variable is assigned the result of calling
`substr()` on itself. This pattern materializes an unnecessary temporary
string (an allocation, a copy of the surviving characters, and a
deallocation) and discards the original capacity; the same effect can be
achieved in-place with `erase()`.

```cpp
std::string s = "hello world";

s = s.substr(5);                     // warning; fix-it: s.erase(0, 5)
s = s.substr(5, std::string::npos);  // warning; fix-it: s.erase(0, 5)
s = s.substr(0, 3);                  // warning; no fix-it (see below)
```

The fix-it for the prefix-stripping forms replaces the assignment with
`s.erase(0, pos)`. The two expressions differ only when `pos > s.size()`:
`substr` throws `std::out_of_range`, while `erase(0, pos)` clamps and
erases the whole string. Code that relies on that exception changes
behavior under the fix-it.

The truncation form `s = s.substr(0, count)` is diagnosed without a
fix-it: the tempting replacement `s.erase(count)` throws
`std::out_of_range` whenever `count > s.size()`, whereas `substr` clamps
`count` and leaves the string unchanged. A safe manual rewrite is
`s.resize(std::min(count, s.size()))`.

In C++23, `s = std::move(s).substr(pos, count)` is an exact,
allocation-free rewrite for every form: the rvalue `substr` overload
([P2438R2](https://wg21.link/p2438r2)) constructs the result by moving
from the string, reusing its buffer, and preserves the exception behavior
of the original code. A future version of this check may suggest it
automatically in C++23 mode. Note that the same spelling compiles before
C++23 but silently performs a full copy.

No diagnostic is emitted for the general form `s = s.substr(pos, count)`,
which has no single-call in-place equivalent, nor in unevaluated contexts
such as `decltype` or `sizeof`, where no temporary is ever materialized.
Inside macro expansions the warning is emitted without a fix-it. Only
self-assignments to plain variables are diagnosed; assignments through
class members or pointers are not.

## Options

```{option} StringLikeClasses

Semicolon-separated list of names of string-like classes. By default only
`::std::basic_string` is considered. Classes listed here must provide
`substr`, `erase`, and `npos` with `std::basic_string` semantics.
```
