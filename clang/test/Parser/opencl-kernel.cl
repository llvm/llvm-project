// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only

__kernel void test() // expected-warning {{a function declaration without a prototype is deprecated in all versions of C}}
{
}

kernel void test1() // expected-warning {{a function declaration without a prototype is deprecated in all versions of C}}
{
}
