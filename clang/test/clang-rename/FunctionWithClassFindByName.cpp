void foo() {
}

class Foo {         // CHECK: class Bar
};

int main() {
  Foo *Pointer = 0; // CHECK: Bar *Pointer = 0;
  return 0;
}

// RUN: clang-rename -qualified-name=Foo -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s
