// RUN: %clang_cc1 -fmodules -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fmodules -std=c++11 -include-pch %t %s -verify
//
// This test checks for a bug in the deserialization code that was only
// reachable with modules enabled, but actually building and using modules is
// not necessary in order to trigger it, so we just use PCH here to make the
// test simpler.

#ifndef HEADER_INCLUDED
#define HEADER_INCLUDED

struct X { template <typename T> X(T) {} };
struct Y { Y(X x = [] {}); };

#else

// This triggers us to load the specialization of X::X for Y's lambda. That
// lambda's context decl must not be loaded as a result of loading the lambda,
// as that would hit a deserialization cycle.
X x = [] {}; // expected-no-diagnostics

#endif
