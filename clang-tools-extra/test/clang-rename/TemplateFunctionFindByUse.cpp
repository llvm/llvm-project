// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=290 -new-name=Bar %t.cpp -i --
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s

// FIXME: clang-rename should be able to rename functions with templates.
// XFAIL: *

template <typename T>
T foo(T value) {
  return value;
}

int main() {
  foo<bool>(false);
  foo<int>(0);
  return 0;
}
