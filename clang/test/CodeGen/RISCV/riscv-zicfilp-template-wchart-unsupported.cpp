// RUN: split-file %s %t
// RUN: %clang_cc1 -triple riscv64 -std=c++17 -emit-llvm -o /dev/null \
// RUN:   -fcf-protection=branch -mcf-branch-label-scheme=func-sig \
// RUN:   -verify %t/direct.cpp
// RUN: %clang_cc1 -triple riscv64 -std=c++17 -emit-llvm -o /dev/null \
// RUN:   -fcf-protection=branch -mcf-branch-label-scheme=func-sig \
// RUN:   -verify %t/nested.cpp
// RUN: %clang_cc1 -triple riscv64 -std=c++17 -emit-llvm -o /dev/null \
// RUN:   -fcf-protection=branch -mcf-branch-label-scheme=func-sig \
// RUN:   -verify %t/function-pointer.cpp
// RUN: %clang_cc1 -triple riscv64 -std=c++17 -emit-llvm -o /dev/null \
// RUN:   -fcf-protection=branch -mcf-branch-label-scheme=func-sig \
// RUN:   -verify %t/member-pointer.cpp
// RUN: %clang_cc1 -triple riscv64 -std=c++17 -emit-llvm -o /dev/null \
// RUN:   -fcf-protection=branch -mcf-branch-label-scheme=func-sig \
// RUN:   -verify %t/integral.cpp
// RUN: %clang_cc1 -triple riscv64 -std=c++17 -emit-llvm -o /dev/null \
// RUN:   -fcf-protection=branch -mcf-branch-label-scheme=func-sig \
// RUN:   -I %t -verify %t/declaration.cpp

// wchar_t is supported in ordinary signature positions, where
// SignatureNormalizer rewrites it to the target's underlying type.
// It is currently unsupported within template arguments because template
// specializations are not structurally rewritten before mangling.

//--- direct.cpp
template <typename T> struct Wrapper {};

// expected-error@+1 {{RISC-V landing pad label generation does not yet support template arguments containing wchar_t}}
void f1(Wrapper<wchar_t>) {}

//--- nested.cpp
template <typename T> struct Wrapper {};

// expected-error@+1 {{RISC-V landing pad label generation does not yet support template arguments containing wchar_t}}
void f2(Wrapper<Wrapper<wchar_t>>) {}

//--- function-pointer.cpp
template <typename T> struct Wrapper {};

// expected-error@+1 {{RISC-V landing pad label generation does not yet support template arguments containing wchar_t}}
void f3(Wrapper<void (*)(wchar_t)>) {}

//--- member-pointer.cpp
template <typename T> struct Wrapper {};
struct C {};

// expected-error@+1 {{RISC-V landing pad label generation does not yet support template arguments containing wchar_t}}
void f4(Wrapper<void (C::*)(wchar_t)>) {}

//--- integral.cpp
template <wchar_t C> struct Character {};

// expected-error@+1 {{RISC-V landing pad label generation does not yet support template arguments containing wchar_t}}
void f5(Character<L'x'>) {}

//--- declaration.cpp
template <auto Value>
struct ValueWrapper {};

void target(wchar_t);

// Declaration-valued template arguments are conservatively unsupported.
// The referenced function's mangled name encodes its wchar_t parameter type.
// expected-error@+1 {{RISC-V landing pad label generation does not yet support this template argument form}}
void f(ValueWrapper<&target>) {}