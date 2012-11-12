// RUN: %clang_cc1 %s -emit-llvm -o - -triple=x86_64-apple-darwin9

typedef unsigned long long uint64_t;
extern uint64_t numer;
extern uint64_t denom;

uint64_t
f(uint64_t val)
{
    __uint128_t tmp;

    tmp = val;
    tmp *= numer;
    tmp /= denom;

    return tmp;
}

