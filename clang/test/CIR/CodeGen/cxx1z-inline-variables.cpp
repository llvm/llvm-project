// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck -check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// For compatibility with C++11 and C++14, an out-of-line declaration of a
// static constexpr local variable promotes the variable to weak_odr.
struct compat {
  static constexpr int a = 1;
  static constexpr int b = 2;
  static constexpr int c = 3;
  static inline constexpr int d = 4;
  static const int e = 5;
  static const int f = 6;
  static const int g = 7;
};
const int &compat_use_before_redecl = compat::b;
const int compat::a;
const int compat::b;
const int compat::c;
const int compat::d;
const int compat::e;
constexpr int compat::f;
constexpr inline int compat::g;
const int &compat_use_after_redecl1 = compat::c;
const int &compat_use_after_redecl2 = compat::d;
const int &compat_use_after_redecl3 = compat::g;

// CIR: cir.global  weak_odr comdat @_ZN6compat1bE = #cir.int<2> : !s32i {alignment = 4 : i64}
// CIR: cir.global  weak_odr comdat @_ZN6compat1aE = #cir.int<1> : !s32i {alignment = 4 : i64}
// CIR: cir.global  weak_odr comdat @_ZN6compat1cE = #cir.int<3> : !s32i {alignment = 4 : i64}
// CIR: cir.global  external @_ZN6compat1eE = #cir.int<5> : !s32i {alignment = 4 : i64}
// CIR: cir.global  weak_odr comdat @_ZN6compat1fE = #cir.int<6> : !s32i {alignment = 4 : i64}
// CIR: cir.global  linkonce_odr comdat @_ZN6compat1dE = #cir.int<4> : !s32i {alignment = 4 : i64}
// CIR: cir.global  linkonce_odr comdat @_ZN6compat1gE = #cir.int<7> : !s32i {alignment = 4 : i64}

// LLVM: $_ZN6compat1bE = comdat any
// LLVM: $_ZN6compat1aE = comdat any
// LLVM: $_ZN6compat1cE = comdat any
// LLVM: $_ZN6compat1fE = comdat any
// LLVM: $_ZN6compat1dE = comdat any
// LLVM: $_ZN6compat1gE = comdat any

// LLVM: @_ZN6compat1bE = weak_odr global i32 2, comdat, align 4
// LLVM: @_ZN6compat1aE = weak_odr global i32 1, comdat, align 4
// LLVM: @_ZN6compat1cE = weak_odr global i32 3, comdat, align 4
// LLVM: @_ZN6compat1eE = global i32 5, align 4
// LLVM: @_ZN6compat1fE = weak_odr global i32 6, comdat, align 4
// LLVM: @_ZN6compat1dE = linkonce_odr global i32 4, comdat, align 4
// LLVM: @_ZN6compat1gE = linkonce_odr global i32 7, comdat, align 4
