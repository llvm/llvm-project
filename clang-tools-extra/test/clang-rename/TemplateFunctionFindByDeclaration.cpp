// RUN: clang-rename -offset=93 -new-name=bar %s -- | FileCheck %s

template <typename T>
T foo(T value) {    // CHECK: T bar(T value) {
  return value;
}

int main() {
  foo<bool>(false); // CHECK: bar<bool>(false);
  foo<int>(0);      // CHECK: bar<int>(0);
  return 0;
}

// Use grep -FUbo 'foo' <file> to get the correct offset of foo when changing
// this file.
