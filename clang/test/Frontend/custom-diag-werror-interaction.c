// RUN: %clang_cc1 -emit-llvm-only -fprofile-instrument=clang -fcoverage-mcdc -Werror -Wno-error=pgo-coverage -Wno-unused-value %s -verify -fmcdc-max-conditions=2

int foo(int x);

int main(void) {
    int a, b, c;
    a && foo( a && b && c ); // expected-warning{{unsupported MC/DC boolean expression; number of conditions (3) exceeds max (2). Expression will not be covered}}
    return 0;
}
