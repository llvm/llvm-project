// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s

// expected-no-diagnostics


// TODO:
// R-U-N:
auto x = " //  %clang_cc1 -std=c++17 -fsyntax-only -verify -fexperimental-new-constant-interpreter %s";

// trunc
// truncf
// truncl

//   ASSERT_NOT_CONSTEXPR_CXX23(std::trunc(1.0f) == 1.0f);
//   ASSERT_NOT_CONSTEXPR_CXX23(std::trunc(1.0) == 1.0);
//   ASSERT_NOT_CONSTEXPR_CXX23(std::trunc(1.0L) == 1.0L);
//   ASSERT_NOT_CONSTEXPR_CXX23(std::truncf(1.0f) == 1.0f);
//   ASSERT_NOT_CONSTEXPR_CXX23(std::truncl(1.0L) == 1.0L);


//   std::trunc(1.0f) 
//   std::trunc(1.0)
//   std::trunc(1.0L)
//   std::truncf(1.0f)
//   std::truncl(1.0L)

constexpr float f = __builtin_trunc(+2.7f);
static_assert(f == 0.0); // FIXME: implement it later