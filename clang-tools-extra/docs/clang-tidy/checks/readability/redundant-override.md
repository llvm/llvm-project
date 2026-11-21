```{title} clang-tidy - readability-redundant-override
```

# readability-redundant-override

Finds virtual overrides that only forward their arguments to the inherited
implementation on the same object, without changing its interface.

For example:

```c++
struct Base {
  virtual void f() {}
};

struct Derived : Base {
  void f() override { Base::f(); }
};
```

becomes:

```c++
struct Base {
  virtual void f() {}
};

struct Derived : Base {};
```

Both inline definitions and out-of-line definitions are supported. When an
override is declared in a header and defined in a source file, the fix-it
removes both declarations. The check skips an override if any declaration
or definition cannot be safely removed, including shared declarations and
ranges containing preprocessor directives. It never offers partial fix-its.

Access is compared after accounting for the inheritance access specifier.
Only public methods inherited publicly are considered for removal.
Non-public methods and non-public inheritance are preserved. Even matching
non-public access can change access from code in the base class when an
override is removed.

The check is deliberately conservative. It requires one non-virtual direct
base class declaring the overridden method, a single forwarding call, and
unchanged function types and exception specifications. It does not diagnose
overrides involving:

- Pure virtual base methods, `final`, or additional attributes.
- Default arguments or overload sets that would affect name lookup.
- Class objects passed or returned by value, or changed argument order or
  expressions.
- Templates, multiple inheritance, or virtual inheritance.
- Macro-generated declarations or forwarding calls.
- References to the overriding method that can observe its member-pointer
  identity or type in the analyzed translation unit.

As with removing any member function, uses outside the analyzed translation
unit and binary compatibility require review. All affected translation
units should be rebuilt after applying the fix-it.
