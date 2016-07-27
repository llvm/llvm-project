// RUN: clang-rename -offset=75 -new-name=Bar %s -- | FileCheck %s

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

// Use grep -FUbo 'Foo' <file> to get the correct offset of Foo when changing
// this file.
