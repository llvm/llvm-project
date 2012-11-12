// Test this without pch.
// RUN: %clang_cc1 -include %S/cxx-using.h -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -x c++-header -emit-pch -o %t %S/cxx-using.h
// RUN: %clang_cc1 -include-pch %t -fsyntax-only -verify %s 

void m() {
    D s;   // expected-note {{candidate function}}
    s.f(); // expected-error {{no matching member}}
}



// expected-note {{candidate function}}
