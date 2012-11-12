// Test this without pch.
// RUN: %clang_cc1 %s -include %s -verify -fsyntax-only

// Test with pch.
// RUN: %clang_cc1 %s -emit-pch -o %t
// RUN: %clang_cc1 %s -include-pch %t -verify -fsyntax-only

#ifndef HEADER
#define HEADER

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wtautological-compare"
template <typename T>
struct TS1 {
    void m() {
      T a = 0;
      T b = a==a;
    }
};
#pragma clang diagnostic pop

#else


template <typename T>
struct TS2 {
    void m() {
      T a = 0;
      T b = a==a; // expected-warning {{self-comparison always evaluates to true}} expected-note@39 {{in instantiation of member function}}
    }
};

void f() {
    TS1<int> ts1;
    ts1.m();


    TS2<int> ts2;
    ts2.m();
}

#endif
