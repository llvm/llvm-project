// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only
// expected-no-diagnostics

__kernel void test()
{
}

kernel void test1()
{
}
