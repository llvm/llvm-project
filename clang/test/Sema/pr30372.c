// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple i386-pc-windows-msvc18.0.0 -disable-free -fms-volatile -fms-extensions -fms-compatibility -fms-compatibility-version=18 -std=c++11 -x c++

// Check that the parser catching an 'error' from forward declaration of "location" does not lexer out it's subsequent declaration.

void foo() {
  __asm {
    jl         location
 location:
    ret
  }
}
