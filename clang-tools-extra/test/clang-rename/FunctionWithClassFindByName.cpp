// RUN: clang-rename -old-name=Foo -new-name=Bar %s -- | FileCheck %s

void foo() {
}

class Foo {         // CHECK: class Bar
};

int main() {
  Foo *Pointer = 0; // CHECK: Bar *Pointer = 0;
  return 0;
}

