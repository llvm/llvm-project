// RUN: clang-reorder-fields -record-name Foo -fields-order y,x %s -- | FileCheck %s

struct Foo {
  int x;  // CHECK:      {{^  int y;}}
  int y;  // CHECK-NEXT: {{^  int x;}}
};

int main() {
  // The idiomatic zero initializer should remain the same.
  struct Foo foo0 = { 0 }; // CHECK: {{^ struct Foo foo0 = { 0 };}}
  struct Foo foo1 = { 1 }; // CHECK: {{^ struct Foo foo1 = { .x = 1 };}}

  return 0;
}
