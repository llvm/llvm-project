class Foo {         // CHECK: class Bar {
};

int main() {
  Foo *Pointer = 0; // CHECK: Bar *Pointer = 0;
  return 0;
}

// Test 1.
// RUN: clang-rename rename-all -old-name=Foo -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s
