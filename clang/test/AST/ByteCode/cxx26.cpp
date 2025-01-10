// RUN: %clang_cc1 -std=c++26 -fsyntax-only -fcxx-exceptions -verify=ref,both %s
// RUN: %clang_cc1 -std=c++26 -fsyntax-only -fcxx-exceptions -verify=expected,both %s -fexperimental-new-constant-interpreter

// both-no-diagnostics

namespace VoidCast {
  constexpr void* p = nullptr;
  constexpr int* q = static_cast<int*>(p);
  static_assert(q == nullptr);
}
