// RUN: %clang_cc1 -std=c++11 -triple %itanium_abi_triple -emit-llvm -o - %s | FileCheck %s

// PR9923
enum class Color { red, blue, green };

void f(Color);
void g() {
  f(Color::red);
}

// See that struct is handled equally.
enum struct Colour { grey };

void h(Colour);
void i() {
  h(Colour::grey);
}

enum struct PR17103 : int { a = -1, b = 1 };
bool cmp(PR17103 x, PR17103 y) { return x < y; }

// CHECK-LABEL: @_Z3cmp7PR17103S_(
// CHECK-NOT: }
// CHECK: icmp slt
