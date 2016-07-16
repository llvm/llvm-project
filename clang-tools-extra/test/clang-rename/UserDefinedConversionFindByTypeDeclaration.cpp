// Currently unsupported test.
// RUN: cat %s > %t.cpp
// FIXME: while renaming class/struct clang-rename should be able to change
// this type name corresponding user-defined conversions, too.

class Foo {                         // CHECK: class Bar {
//    ^ offset must be here
public:
  Foo() {}                          // CHECK: Bar() {}
};

class Baz {
public:
  operator Foo() const {            // CHECK: operator Bar() const {
    Foo foo;                        // CHECK: Bar foo;
    return foo;
  }
};

int main() {
  Baz boo;
  Foo foo = static_cast<Foo>(boo);  // CHECK: Bar foo = static_cast<Bar>(boo);
  return 0;
}
