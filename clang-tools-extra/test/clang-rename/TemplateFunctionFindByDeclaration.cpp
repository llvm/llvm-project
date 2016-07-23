// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=241 -new-name=bar %t.cpp -i --
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s

// FIXME: clang-rename should be able to rename functions with templates.
// XFAIL: *

template <typename T>
T foo(T value) {    // CHECK: T boo(T value) {
  return value;
}

int main() {
  foo<bool>(false); // CHECK: bar<bool>(false);
  foo<int>(0);      // CHECK: bar<int>(0);
  return 0;
}

// Use grep -FUbo 'foo' <file> to get the correct offset of foo when changing
// this file.
