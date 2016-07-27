// RUN: clang-rename -offset=79 -new-name=llvm %s -- | FileCheck %s

namespace foo { // CHECK: namespace llvm {
  int x;
}

void boo() {
  foo::x = 42;  // CHECK: llvm::x = 42;
}

// Use grep -FUbo 'foo' <file> to get the correct offset of foo when changing
// this file.
