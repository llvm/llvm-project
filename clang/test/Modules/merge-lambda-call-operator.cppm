// Test that a lambda closure type instantiated from a variable template in two
// modules is merged together with its members. The closure types are merged by
// identity, but the instantiated call operator is module-local to each
// instantiating module and used to be rejected by ASTContext::isSameEntity.
// That left the merged class with two distinct 'operator()' declarations,
// which was diagnosed as a false ODR violation.

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -I%t %t/m_a.cppm -emit-reduced-module-interface -o %t/m_a.pcm
// RUN: %clang_cc1 -std=c++20 -I%t %t/m_b.cppm -emit-reduced-module-interface -o %t/m_b.pcm
// RUN: %clang_cc1 -std=c++20 -I%t %t/main.cpp -fmodule-file=m_a=%t/m_a.pcm \
// RUN:     -fmodule-file=m_b=%t/m_b.pcm -fsyntax-only -verify
//
// Full module interfaces.
// RUN: %clang_cc1 -std=c++20 -I%t %t/m_a.cppm -emit-module-interface -o %t/m_a_full.pcm
// RUN: %clang_cc1 -std=c++20 -I%t %t/m_b.cppm -emit-module-interface -o %t/m_b_full.pcm
// RUN: %clang_cc1 -std=c++20 -I%t %t/main.cpp -fmodule-file=m_a=%t/m_a_full.pcm \
// RUN:     -fmodule-file=m_b=%t/m_b_full.pcm -fsyntax-only -verify

//--- template.h
#pragma once
template <typename T>
inline auto var = [](T x) { return x; };

//--- m_a.cppm
module;
#include "template.h"
export module m_a;
export auto Trigger1() {
  return var<int>;
}

//--- m_b.cppm
module;
#include "template.h"
export module m_b;
export auto Trigger2() {
  return var<int>;
}

//--- main.cpp
// expected-no-diagnostics
import m_a;
import m_b;

void use() {
  static_assert(__is_same(decltype(Trigger1()), decltype(Trigger2())));
}
