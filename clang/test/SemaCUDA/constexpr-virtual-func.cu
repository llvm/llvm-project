// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fsyntax-only \
// RUN:            -fcuda-is-device %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only %s

// expected-no-diagnostics

#include "Inputs/cuda.h"

class A
{
public:
    constexpr virtual int f() = 0;
};

class B : public A
{
public:
    int f() override
    {
        return 42;
    }
};

int test()
{
    B b;
    return b.f();
}
