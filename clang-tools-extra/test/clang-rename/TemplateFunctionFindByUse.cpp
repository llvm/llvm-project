// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=233 -new-name=bar %t.cpp -i --
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s

template <typename T>
T foo(T value) {    // CHECK: T bar(T value) {
  return value;
}

int main() {
  foo<bool>(false); // CHECK: bar<bool>(false);
  foo<int>(0);      // CHECK: bar<int>(0);
  return 0;
}
