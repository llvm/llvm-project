// THIS IS A GENERATED TEST. DO NOT EDIT.
// To regenerate, see clang-doc/gen_test.py docstring.
//
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"

namespace A {
  
void f();

}  // namespace A

namespace A {

void f(){};

namespace B {

enum E { X };

E func(int i) { return X; }

}  // namespace B
}  // namespace A

// RUN: clang-doc --format=md --doxygen --public --extra-arg=-fmodules-ts -p %t %t/test.cpp -output=%t/docs


// RUN: cat %t/docs/./A.md | FileCheck %s --check-prefix CHECK-0
// CHECK-0: # namespace A
// CHECK-0: ## Functions
// CHECK-0: ### f
// CHECK-0: *void f()*
// CHECK-0: *Defined at line 17 of test*

// RUN: cat %t/docs/A/B.md | FileCheck %s --check-prefix CHECK-1
// CHECK-1: # namespace B
// CHECK-1: ## Functions
// CHECK-1: ### func
// CHECK-1: *enum A::B::E func(int i)*
// CHECK-1: *Defined at line 23 of test*
// CHECK-1: ## Enums
// CHECK-1: | enum E |
// CHECK-1: --
// CHECK-1: | X |
// CHECK-1: *Defined at line 21 of test*
