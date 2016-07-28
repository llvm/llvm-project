// RUN: clang-rename -offset=74 -new-name=Bar %s -- | FileCheck %s

class Foo {};       // CHECK: class Bar

int main() {
  Foo *Pointer = 0; // CHECK: Bar *Pointer = 0;
  return 0;
}

// Use grep -FUbo 'Foo' <file> to get the correct offset of Foo when changing
// this file.
