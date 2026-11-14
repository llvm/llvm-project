// RUN: split-file %s %t
// RUN: %clang_cc1 -triple riscv64 -std=c++17 -emit-llvm -o /dev/null \
// RUN:   -fcf-protection=branch -mcf-branch-label-scheme=func-sig \
// RUN:   -verify %t/function-pointer.cpp
// RUN: %clang_cc1 -triple riscv64 -std=c++17 -emit-llvm -o /dev/null \
// RUN:   -fcf-protection=branch -mcf-branch-label-scheme=func-sig \
// RUN:   -verify %t/member-pointer.cpp

//--- function-pointer.cpp
template <typename T> struct Wrapper {};

// expected-error@+1 {{RISC-V landing pad label generation does not yet support template arguments containing exception spec}}
void f(Wrapper<void (*)() noexcept>) {}

//--- member-pointer.cpp
template <typename T> struct Wrapper {};
struct C {};

// expected-error@+1 {{RISC-V landing pad label generation does not yet support template arguments containing exception spec}}
void f(Wrapper<void (C::*)() noexcept>) {}