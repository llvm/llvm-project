// Test this without pch.
// RUN: %clang_cc1 %s -include %s -verify -fsyntax-only -Wuninitialized

// Test with pch.
// RUN: %clang_cc1 %s -emit-pch -o %t
// RUN: %clang_cc1 %s -include-pch %t -verify -fsyntax-only -Wuninitialized

// RUN: %clang_cc1 %s -emit-pch -fpch-instantiate-templates -o %t
// RUN: %clang_cc1 %s -include-pch %t -verify -fsyntax-only -Wuninitialized

#ifndef HEADER
#define HEADER

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
template <typename T>
struct TS1 {
    void m() {
      T a;
      T b = a;
    }
};
#pragma clang diagnostic pop

#else


template <typename T>
struct TS2 {
    void m() {
      T a;
      T b = a; // expected-warning {{variable 'a' is uninitialized}} \
                  expected-note@44 {{in instantiation of member function}} \
                  expected-note@31 {{initialize the variable 'a' to silence}}
    }
};

void f() {
    TS1<int> ts1;
    ts1.m();


    TS2<int> ts2;
    ts2.m();
}

#endif
