// RUN: %clang_cc1 -Weverything -xc++ -std=c++11 -DCXX11 -verify %s
// RUN: %clang_cc1 -Weverything -xc++ -std=c++03 -DCXX03 -verify %s
// RUN: %clang_cc1 -Weverything -xobjective-c -DOBJC -verify %s
// RUN: %clang_cc1 -Weverything -std=c11 -xc -DC11 -verify %s
// RUN: %clang_cc1 -Weverything -std=c11 -xc -fms-extensions -DMS -verify %s

enum X : int {e};
#if defined(CXX11)
// expected-warning@-2{{enumeration types with a fixed underlying type are incompatible with C++98}}
#elif defined(CXX03)
// expected-warning@-4{{enumeration types with a fixed underlying type are a C++11 extension}}
#elif defined(OBJC)
// expected-no-diagnostics
#elif defined(C11)
// expected-warning@-8{{enumeration types with a fixed underlying type are a Clang extension}}
#elif defined(MS)
// expected-warning@-10{{enumeration types with a fixed underlying type are a Microsoft extension}}
#endif
