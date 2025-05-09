// Test BOLT is able to handle relative virtual function table, i.e., when
// code is compiled with `-fexperimental-relative-c++-abi-vtables`.

// REQUIRES: system-linux

// RUN: split-file %s %t
// RUN: %clang -fuse-ld=lld -o %t/main.so %t/tt.cpp %t/main.cpp -Wl,-q \
// RUN:     -fno-rtti -fexperimental-relative-c++-abi-vtables
// RUN: %t/main.so | FileCheck %s

// CHECK: derived_foo
// CHECK-NEXT: derived_bar
// CHECK-NEXT: derived_goo

// RUN: llvm-bolt %t/main.so -o %t/main.bolted.so --trap-old-code
// RUN: %t/main.bolted.so | FileCheck %s

;--- tt.h
#include <stdio.h>

class Base {
public:
  virtual void foo();
  virtual void bar();
  virtual void goo();
};

class Derived : public Base {
public:
  virtual void foo() override;
  virtual void bar() override;
  virtual void goo() override;
};

;--- tt.cpp
#include "tt.h"
void Derived::goo() { printf("derived_goo\n"); }

;--- main.cpp
#include "tt.h"
#pragma clang optimize off

void Base::foo() { printf("base_foo\n"); }
void Base::bar() { printf("base_bar\n"); }
void Base::goo() { printf("base_goo\n"); }

void Derived::foo() { printf("derived_foo\n"); }
void Derived::bar() { printf("derived_bar\n"); }

int main() {
  Derived D;
  Base *ptr = &D;
  ptr->foo();
  ptr->bar();
  ptr->goo();
  return 0;
}
