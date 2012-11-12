// Test this without pch.
// RUN: %clang_cc1 -include %s -verify -std=c++11 %s

// Test with pch.
// RUN: %clang_cc1 -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -include-pch %t -verify -std=c++11 %s 

#ifndef HEADER
#define HEADER

template<int N> struct T {
    static_assert(N == 2, "N is not 2!");
};

#else

// expected-error@12 {{static_assert failed "N is not 2!"}}
T<1> t1; // expected-note {{in instantiation of template class 'T<1>' requested here}}
T<2> t2;

#endif
