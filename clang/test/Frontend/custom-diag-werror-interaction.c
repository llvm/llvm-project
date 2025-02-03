// RUN: %clang_cc1 -emit-llvm-only -fprofile-instrument=clang -fcoverage-mcdc -Werror -Wno-unused-value %s -verify

int foo(int x);

int main(void) {
    int a, b, c;
    a && foo( b && c ); // expected-warning{{unsupported MC/DC boolean expression; contains an operation with a nested boolean expression. Expression will not be covered}}
    return 0;
}
