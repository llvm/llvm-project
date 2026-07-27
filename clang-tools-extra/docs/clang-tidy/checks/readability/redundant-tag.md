# readability-redundant-tag

Finds redundant uses of the `class`, `struct`, `union`, and `enum`
keywords in C++ declarations.

In C++, elaborated type specifiers are unnecessary when the type name is
already unambiguous.

For example:

```c++
struct S {};

void f() {
  struct S s;
}
```

becomes:

```c++
struct S {};

void f() {
  S s;
}
```

The check does not diagnose cases where removing the keyword would change name
lookup semantics. For example:

```c++
struct Hidden {} Hidden;

void f() {
  struct Hidden h;
}
```

Removing `struct` would cause `Hidden` to refer to the variable rather
than the type.

Similarly:

```c++
struct Foo {};

void Foo();

void f() {
  struct Foo x;
}
```

Removing `struct` would cause `Foo` to refer to the function instead of
the type.

The check also avoids issuing diagnostics when removing the elaborated type
specifier would change name lookup. For example:

```c++
namespace NS {
struct S {};
}

using NS::S;

namespace NS1 {
  int S;

  namespace NS2 {
    using T = struct S;
  }
}
```

Although `NS::S` is visible via the using declaration, the name `S` is
hidden by the variable declaration in `NS1`. Removing `struct` would make
`S` refer to the variable rather than the type, so no diagnostic is
produced.
