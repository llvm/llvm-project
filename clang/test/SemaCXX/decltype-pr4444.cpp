// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11
// expected-no-diagnostics

template<typename T, T t>
struct TestStruct {
   typedef decltype(t+2) sum_type;
};
