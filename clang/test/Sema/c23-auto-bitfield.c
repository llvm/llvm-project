// RUN: %clang_cc1 -std=c23 -fsyntax-only -verify %s

void f()
{
    struct S { unsigned i : 1; };
    struct S s;
    auto si = s.i;
    static_assert(_Generic(si, unsigned int : 1, default : 0),
                  "the underlying type of the bit-field is 'unsigned int'");

    __auto_type si2 = s.i; // expected-error {{cannot pass bit-field as __auto_type initializer in C}}
}
