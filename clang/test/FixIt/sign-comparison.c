// RUN: %clang_cc1 -triple=x86_64-apple-darwin -fsyntax-only -Wsign-compare -Wno-unused-comparison -Wno-empty-body -Wno-unused-value -verify %s
// RUN: %clang_cc1 -triple=armv7-apple-darwin -fsyntax-only -Wsign-compare -Wno-unused-comparison -Wno-empty-body -Wno-unused-value -verify %s
// RUN: cp %s %t
// RUN: %clang_cc1 -fsyntax-only -Wsign-compare -fixit -x c %t 2> /dev/null
// RUN: grep -v CHECK %t | FileCheck %s
// RUN: cp %s %t
// RUN: %clang_cc1 -fsyntax-only -Wsign-compare -fixit -x c++ %t 2> /dev/null
// RUN: grep -v CHECK %t | FileCheck --check-prefix=CHECKXX %s

unsigned Uf(void);
int Sf(void);

void test(signed S, unsigned U) {
    // expected-warning@+2{{comparison of integers of different signs implicitly casts left-side operand from 'int' to 'unsigned int'}}
    // expected-note@+1{{consider verifying that the signed value is non-negative}}
    S < U;
    // CHECK: S < 0 || (unsigned int)S < U;
    // CHECKXX: S < 0 || static_cast<unsigned int>(S) < U;

    // expected-warning@+2{{comparison of integers of different signs implicitly casts left-side operand}}
    // expected-note@+1{{consider verifying that the signed value is non-negative}}
    S <= U;
    // CHECK: S < 0 || (unsigned int)S <= U;
    // CHECKXX: S < 0 || static_cast<unsigned int>(S) <= U;

    // expected-warning@+2{{comparison of integers of different signs implicitly casts left-side operand}}
    // expected-note@+1{{consider verifying that the signed value is non-negative}}
    S == U;
    // CHECK: S >= 0 && (unsigned int)S == U;
    // CHECKXX: S >= 0 && static_cast<unsigned int>(S) == U;

    // expected-warning@+2{{comparison of integers of different signs implicitly casts left-side operand}}
    // expected-note@+1{{consider verifying that the signed value is non-negative}}
    S > U;
    // CHECK: S >= 0 && (unsigned int)S > U;
    // CHECKXX: S >= 0 && static_cast<unsigned int>(S) > U;

    // expected-warning@+2{{comparison of integers of different signs implicitly casts left-side operand}}
    // expected-note@+1{{consider verifying that the signed value is non-negative}}
    S >= U;
    // CHECK: S >= 0 && (unsigned int)S >= U;
    // CHECKXX: S >= 0 && static_cast<unsigned int>(S) >= U;

    // expected-warning@+2{{comparison of integers of different signs implicitly casts left-side operand}}
    // expected-note@+1{{consider verifying that the signed value is non-negative}}
    S != U;
    // CHECK: S >= 0 && (unsigned int)S != U;
    // CHECKXX: S >= 0 && static_cast<unsigned int>(S) != U;

    // expected-warning@+2{{comparison of integers of different signs implicitly casts right-side operand from 'int' to 'unsigned int'}}
    // expected-note@+1{{consider verifying that the signed value is non-negative}}
    U < S;
    // CHECK: S >= 0 && (unsigned int)U < S;
    // CHECKXX: S >= 0 && static_cast<unsigned int>(U) < S;

    // expected-warning@+2{{comparison of integers of different signs implicitly casts right-side operand}}
    // expected-note@+1{{consider verifying that the signed value is non-negative}}
    U <= S;
    // CHECK: S >= 0 && (unsigned int)U <= S;
    // CHECKXX: S >= 0 && static_cast<unsigned int>(U) <= S;

    // expected-warning@+2{{comparison of integers of different signs implicitly casts right-side operand}}
    // expected-note@+1{{consider verifying that the signed value is non-negative}}
    U == S;
    // CHECK: S >= 0 && (unsigned int)U == S;
    // CHECKXX: S >= 0 && static_cast<unsigned int>(U) == S;

    // expected-warning@+2{{comparison of integers of different signs implicitly casts right-side operand}}
    // expected-note@+1{{consider verifying that the signed value is non-negative}}
    U > S;
    // CHECK: S < 0 || (unsigned int)U > S;
    // CHECKXX: S < 0 || static_cast<unsigned int>(U) > S;

    // expected-warning@+2{{comparison of integers of different signs implicitly casts right-side operand}}
    // expected-note@+1{{consider verifying that the signed value is non-negative}}
    U >= S;
    // CHECK: S < 0 || (unsigned int)U >= S;
    // CHECKXX: S < 0 || static_cast<unsigned int>(U) >= S;

    // expected-warning@+2{{comparison of integers of different signs implicitly casts right-side operand}}
    // expected-note@+1{{consider verifying that the signed value is non-negative}}
    U != S;
    // CHECK: S >= 0 && (unsigned int)U != S;
    // CHECKXX: S >= 0 && static_cast<unsigned int>(U) != S;


    // expected-warning@+6{{comparison of integers of different signs implicitly casts left-side operand}}
    // expected-warning@+5{{comparison of integers of different signs implicitly casts left-side operand}}
    // expected-warning@+4{{comparison of integers of different signs implicitly casts left-side operand}}
    // expected-note@+3{{consider verifying that the signed value is non-negative}}
    // expected-note@+2{{consider verifying that the signed value is non-negative}}
    // expected-note@+1{{consider verifying that the signed value is non-negative}}
    for (S < U; S < U; S < U) ;
    // CHECK: for (S < 0 || (unsigned int)S < U; S < 0 || (unsigned int)S < U; S < 0 || (unsigned int)S < U) ;
    // CHECKXX: for (S < 0 || static_cast<unsigned int>(S) < U; S < 0 || static_cast<unsigned int>(S) < U; S < 0 || static_cast<unsigned int>(S) < U) ;

    // expected-warning@+2{{comparison of integers of different signs implicitly casts left-side operand}}
    // expected-note@+1{{consider verifying that the signed value is non-negative}}
    if (S < U) ;
    // CHECK: if (S < 0 || (unsigned int)S < U) ;
    // CHECKXX: if (S < 0 || static_cast<unsigned int>(S) < U) ;

    // expected-warning@+2{{comparison of integers of different signs implicitly casts left-side operand}}
    // expected-note@+1{{consider verifying that the signed value is non-negative}}
    1 && S < U;
    // CHECK: 1 && (S < 0 || (unsigned int)S < U);
    // CHECKXX: 1 && (S < 0 || static_cast<unsigned int>(S) < U);

    // expected-warning@+2{{comparison of integers of different signs implicitly casts left-side operand}}
    // expected-note@+1{{consider verifying that the signed value is non-negative}}
    S < U + U;
    // CHECK: S < 0 || (unsigned int)S < U + U;
    // CHECKXX: S < 0 || static_cast<unsigned int>(S) < U + U;

    // expected-warning@+2{{comparison of integers of different signs implicitly casts left-side operand}}
    // expected-note@+1{{consider verifying that the signed value is non-negative}}
    S + S < U;
    // CHECK: S + S < 0 || (unsigned int)(S + S) < U
    // CHECKXX: S + S < 0 || static_cast<unsigned int>(S + S) < U

    // expected-warning@+2{{comparison of integers of different signs implicitly casts left-side operand}}
    // expected-note@+1{{consider verifying that the signed value is non-negative}}
    Sf() < U;
    // CHECK: Sf() < U;
    // CHECKXX: Sf() < U;

    // expected-warning@+2{{comparison of integers of different signs implicitly casts left-side operand}}
    // expected-note@+1{{consider verifying that the signed value is non-negative}}
    S < Uf();
    // CHECK: S < Uf();
    // CHECKXX: S < Uf();

    // expected-warning@+2{{comparison of integers of different signs implicitly casts left-side operand}}
    // expected-note@+1{{consider verifying that the signed value is non-negative}}
    S++ < U;
    // CHECK: S++ < U;
    // CHECKXX: S++ < U;

    // expected-warning@+2{{comparison of integers of different signs implicitly casts left-side operand}}
    // expected-note@+1{{consider verifying that the signed value is non-negative}}
    S < U++;
    // CHECK: S < U++;
    // CHECKXX: S < U++;

#if !__LP64__
    // expected-warning@+3{{comparison of integers of different signs implicitly casts left-side operand}}
    // expected-note@+2{{consider verifying that the signed value is non-negative}}
#endif
    (long)S < U;
}

typedef unsigned uint32_t;
typedef int int32_t;

void test2(int32_t S, uint32_t U) {
    // expected-warning@+2{{comparison of integers of different signs}}
    // expected-note@+1{{consider verifying that the signed value is non-negative}}
    S < U;
    // CHECK: S < 0 || (uint32_t)S < U;
    // CHECKXX: S < 0 || static_cast<uint32_t>(S) < U;
}
