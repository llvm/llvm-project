```{title} clang-tidy - modernize-use-default-member-init
```

# modernize-use-default-member-init

This check converts constructors' member initializers into the new
default member initializers in C++11. Other member initializers that match the
default member initializer are removed. This can reduce repeated code or allow
use of '= default'.

```c++
struct A {
  A() : i(5), j(10.0) {}
  A(int i) : i(i), j(10.0) {}
  int i;
  double j;
};

// becomes

struct A {
  A() {}
  A(int i) : i(i) {}
  int i{5};
  double j{10.0};
};
```

```{note}
Only converts member initializers for built-in types, enums, and pointers.
The {doc}`readability-redundant-member-init <../readability/redundant-member-init>` check will remove redundant member
initializers for classes.
```

## Options

(modernize-use-default-member-init-use-assignment)=

````{option} UseAssignment
If this option is set to `true`, the check will initialize
members with an assignment. For example:

```c++
struct A {
  A() {}
  A(int i) : i(i) {}
  int i = 5;
  double j = 10.0;
};
```
Default is `false`.
````

(modernize-use-default-member-init-ignore-macros)=

```{option} IgnoreMacros
If this option is set to `true`, the check will not warn
about members declared inside macros.
Default is `true`.
```

(modernize-use-default-member-init-ignore-non-visible-references)=

```{option} IgnoreNonVisibleReferences
If this option is set to `true`, the check will not warn
about member initializers that refer to declarations that would not be
visible from the default member initializer created by the fix-it. If set to
`false`, the check will warn about these cases without emitting fix-its.
Default is `true`.
```
