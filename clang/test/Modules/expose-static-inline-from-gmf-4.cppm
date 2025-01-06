// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang -std=c++20 %t/a.cppm --precompile -o %t/a.pcm \
// RUN:   -DTEST_INLINE
// RUN: %clang -std=c++20 %t/test.cc -fprebuilt-module-path=%t -fsyntax-only -Xclang -verify \
// RUN:   -DTEST_INLINE
//
// RUN: %clang -std=c++20 %t/a.cppm --precompile -o %t/a.pcm
// RUN: %clang -std=c++20 %t/test.cc -fprebuilt-module-path=%t -fsyntax-only -Xclang -verify

//--- a.h
#ifdef TEST_INLINE
#define INLINE inline
#else
#define INLINE
#endif
namespace ns {
template <typename G> static void func() {}
template <> INLINE void func<long>() {}
template <typename T = long> void a() { func<T>(); }
}

//--- a.cppm
module;
#include "a.h"
export module a;
export using ns::a;

//--- test.cc
import a;
auto m = (a(), 0);

#ifdef TEST_INLINE
// expected-no-diagnostics
#else
// expected-error@a.h:9 {{no matching function for call to 'func'}}
// expected-note@test.cc:2 {{in instantiation of function template specialization 'ns::a<long>' requested here}}
#endif
