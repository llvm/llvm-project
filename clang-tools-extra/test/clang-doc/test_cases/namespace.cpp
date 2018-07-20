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
