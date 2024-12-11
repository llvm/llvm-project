
// Test without pch.
// RUN: %clang_cc1 -fbounds-safety -include %s -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -fbounds-safety -emit-pch -o %t %s
// RUN: %clang_cc1 -fbounds-safety -include-pch %t -fsyntax-only -verify %s

#ifndef HEADER
#define HEADER
int Test(int *singleArg) {
    int *local = singleArg;

    return *local;
}
#else

float Test(float); // expected-error{{conflicting types for 'Test'}}
                   // expected-note@-8{{previous definition is here}}

#endif
