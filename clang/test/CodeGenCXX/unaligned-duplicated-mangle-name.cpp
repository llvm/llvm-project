// RUN: %clang_cc1 -triple %itanium_abi_triple -fms-extensions -emit-llvm-only %s -verify

struct A
{
    int x;
    void foo() __unaligned;
    void foo();
};

void A::foo() __unaligned
{
    this->x++;
}

void A::foo() // expected-error {{definition with same mangled name as another definition}}
              // expected-note@-6 {{previous definition is here}}
{
    this->x++;
}

