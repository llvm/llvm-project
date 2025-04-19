// RUN: %clang_cc1 -verify=NORMAL14,NORMAL_ALL -std=c++14 %s -fcolor-diagnostics -fcxx-exceptions -DCONSTEXPR=
// RUN: %clang_cc1 -verify=NORMAL17,NORMAL_ALL -std=c++17 %s -fcolor-diagnostics -fcxx-exceptions -DCONSTEXPR=
// RUN: %clang_cc1 -verify=NORMAL20,NORMAL_ALL -std=c++20 %s -fcolor-diagnostics -fcxx-exceptions -DCONSTEXPR=
// RUN: %clang_cc1 -verify=NORMAL23,NORMAL_ALL -std=c++23 %s -fcolor-diagnostics -fcxx-exceptions -DCONSTEXPR=
// RUN: %clang_cc1 -verify=NORMAL26,NORMAL_ALL -std=c++26 %s -fcolor-diagnostics -fcxx-exceptions -DCONSTEXPR=

// RUN: %clang_cc1 -verify=IMPLICIT14,IMPLICIT_ALL -std=c++14 %s -fcolor-diagnostics -fcxx-exceptions -DCONSTEXPR= -fimplicit-constexpr
// RUN: %clang_cc1 -verify=IMPLICIT17,IMPLICIT_ALL -std=c++17 %s -fcolor-diagnostics -fcxx-exceptions -DCONSTEXPR= -fimplicit-constexpr
// RUN: %clang_cc1 -verify=IMPLICIT20,IMPLICIT_ALL -std=c++20 %s -fcolor-diagnostics -fcxx-exceptions -DCONSTEXPR= -fimplicit-constexpr
// RUN: %clang_cc1 -verify=IMPLICIT23,IMPLICIT_ALL -std=c++23 %s -fcolor-diagnostics -fcxx-exceptions -DCONSTEXPR= -fimplicit-constexpr
// RUN: %clang_cc1 -verify=IMPLICIT26,IMPLICIT_ALL -std=c++26 %s -fcolor-diagnostics -fcxx-exceptions -DCONSTEXPR= -fimplicit-constexpr

// RUN: %clang_cc1 -verify=CONSTEXPR14,CONSTEXPR_BEFORE23,CONSTEXPR_BEFORE20,CONSTEXPR_ALL -std=c++14 %s -fcolor-diagnostics -fcxx-exceptions -DCONSTEXPR=constexpr
// RUN: %clang_cc1 -verify=CONSTEXPR17,CONSTEXPR_BEFORE23,CONSTEXPR_BEFORE20,CONSTEXPR_ALL -std=c++17 %s -fcolor-diagnostics -fcxx-exceptions -DCONSTEXPR=constexpr
// RUN: %clang_cc1 -verify=CONSTEXPR20,CONSTEXPR_BEFORE23,CONSTEXPR_ALL -std=c++20 %s -fcolor-diagnostics -fcxx-exceptions -DCONSTEXPR=constexpr
// RUN: %clang_cc1 -verify=CONSTEXPR23,CONSTEXPR_ALL -std=c++23 %s -fcolor-diagnostics -fcxx-exceptions -DCONSTEXPR=constexpr
// RUN: %clang_cc1 -verify=CONSTEXPR26,CONSTEXPR_ALL -std=c++26 %s -fcolor-diagnostics -fcxx-exceptions -DCONSTEXPR=constexpr

// Objective is to make sure features like allocation / throwing won't fail code by just adding implicit constexpr
// in an unevaluated code.

// NORMAL_ALL-no-diagnostics
// IMPLICIT_ALL-no-diagnostics
// CONSTEXPR23-no-diagnostics
// CONSTEXPR26-no-diagnostics

CONSTEXPR inline bool function_with_goto(int v) {
  if (v == 0) {
    return true;
  }
  
  goto label;
  // CONSTEXPR_BEFORE23-warning@-1 {{use of this statement in a constexpr function is a C++23 extension}}
  
  label:
  return false;
}

CONSTEXPR inline bool function_with_label(int v) {
  label:
  // CONSTEXPR_BEFORE23-warning@-1 {{use of this statement in a constexpr function is a C++23 extension}}
  if (v > 0) {
    return true;
  }
  v++;
  goto label;
}

CONSTEXPR inline bool function_with_try_catch(int v) {
  try {
    // CONSTEXPR_BEFORE20-warning@-1 {{use of this statement in a constexpr function is a C++20 extension}}
    return v;
  } catch (int) {
    return -v;
  }
}

CONSTEXPR inline bool function_with_inline_asm(int v) {
  if (v > 0) {
    asm("");
    // CONSTEXPR_BEFORE20-warning@-1 {{use of this statement in a constexpr function is a C++20 extension}}
  } 
  
  return v;
}

struct easy_type {
  // CONSTEXPR_BEFORE20-note@-1 {{declared here}}
  int * x;
};

CONSTEXPR inline bool function_with_no_initializer_variable(int v) {
  // CONSTEXPR_BEFORE20-error@-1 {{constexpr function never produces a constant expression}}
  easy_type easy;
  // CONSTEXPR_BEFORE20-note@-1 {{non-constexpr constructor 'easy_type' cannot be used in a constant expression}}
  return v;
}




