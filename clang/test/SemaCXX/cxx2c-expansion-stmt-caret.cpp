// RUN: not %clang_cc1 %s -std=c++2c -fsyntax-only 2>&1 | FileCheck %s

struct S {
  struct Iterator {};
  constexpr Iterator begin() const { return {}; }
  constexpr Iterator end() const { return {}; }
};

void f() {
  static constexpr S s;
  template for (auto x : s)
    ;
}

// CHECK:      cxx2c-expansion-stmt-caret.cpp:11:24: error: invalid operands to binary expression ('Iterator' and 'Iterator')
// CHECK-NEXT:       |   template for (auto x : s)
// CHECK-NEXT:       |                        ^
// CHECK-NEXT: <scratch space>:4:28: note: expanded from here
// CHECK-NEXT:       | __begin + decltype(__begin - __begin){__i}
// CHECK-NEXT:       |                            ^
