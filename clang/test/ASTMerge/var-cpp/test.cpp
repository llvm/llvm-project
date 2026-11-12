// RUN: %clang_cc1 -emit-pch -std=c++17 -o %t.1.ast %S/Inputs/var1.cpp
// RUN: %clang_cc1 -std=c++17 -ast-merge %t.1.ast -fsyntax-only %s 2>&1

static_assert(my_pi<double> == (double)3.1415926535897932385L);
static_assert(my_pi<char> == '3');

static_assert(Wrapper<int>::my_const<float> == 1.f);
static_assert(Wrapper<char>::my_const<const float *> == nullptr);
static_assert(Wrapper<float>::my_const<const char *> == a);
