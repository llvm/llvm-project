// Basic C++ test for update_cc_test_checks
// RUN: %clang_cc1 -no-opaque-pointers -triple=x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

class Foo {
  int x;

public:
  explicit Foo(int x);
  ~Foo();
  inline int function_defined_inline(int arg) const {
    return arg + x;
  }
  inline int function_defined_out_of_line(int arg) const;
};

Foo::Foo(int x) : x(x) {}
Foo::~Foo() {}
int Foo::function_defined_out_of_line(int arg) const { return x - arg; }

// Call the inline methods to ensure the LLVM IR is generated:
int main() {
  Foo f(1);
  f.function_defined_inline(2);
  f.function_defined_out_of_line(3);
}
