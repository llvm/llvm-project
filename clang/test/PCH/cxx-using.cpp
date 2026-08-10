// Test this without pch.
// RUN: %clang_cc1 -include %S/cxx-using.h -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -x c++-header -emit-pch -o %t %S/cxx-using.h
// RUN: %clang_cc1 -include-pch %t -fsyntax-only -verify %s 

void m() {
    D s;
    s.f(); // expected-error {{no matching member}}
}

// expected-note@cxx-using.h:9  {{candidate function}}
// expected-note@cxx-using.h:15 {{candidate function}}
