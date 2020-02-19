class Foo { // CHECK: class Bar {
public:
  template <typename T>
  Foo(); // CHECK: Bar();

  template <typename T>
  Foo(Foo &); // CHECK: Bar(Bar &);
};

// RUN: clang-rename -offset=6 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s
