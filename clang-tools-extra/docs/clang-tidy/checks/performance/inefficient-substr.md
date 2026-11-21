```{title} clang-tidy - performance-inefficient-substr
```

# performance-inefficient-substr

Finds assignments and appends of a `substr()` result to another string and
suggests the `assign()`/`append()` overloads taking a string, position,
and count, which avoid materializing a temporary string (an allocation, a
copy of the surviving characters, and a deallocation).

```cpp
std::string dst, s;
std::string src = "hello world";

dst = src.substr(6);     // fix-it: dst.assign(src, 6)
s += src.substr(0, 5);   // fix-it: s.append(src, 0, 5)
```

The rewrite is exact: `substr(pos, count)`, `assign(str, pos, count)`, and
`append(str, pos, count)` all throw `std::out_of_range` if and only if
`pos > str.size()` and clamp `count` the same way, so the fix-it preserves
behavior for every argument value, including `npos` counts.

`s += s.substr(pos)` is diagnosed but not rewritten: the replacement would
introduce a self-aliasing `append` call, which the check conservatively
avoids. `s = s.substr(pos)` is not diagnosed by this check at all: the
`assign` rewrite would self-alias too, and the strictly better rewrite for
that case is an in-place `erase` of the removed prefix.

Inside macro expansions the warning is emitted without a fix-it. Only plain
variables are matched on both sides; class members and pointers are not.
Initializations such as `std::string t = s.substr(1);` are not diagnosed:
guaranteed copy elision already makes them cheap.

## Options

```{option} StringLikeClasses

Semicolon-separated list of names of string-like classes. By default only
`::std::basic_string` is considered. Classes listed here must provide
`substr` and the `(string, position, count)` overloads of `assign` and
`append` with `std::basic_string` semantics.
```
