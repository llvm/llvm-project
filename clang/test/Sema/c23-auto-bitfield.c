// RUN: %clang_cc1 -std=c23 -fsyntax-only -verify %s
// expected-no-diagnostics

void f()
{
    struct S { unsigned i : 1; };
    struct S s;
    auto si = s.i;
    static_assert(_Generic(si, unsigned int : 1, default : 0), "the underlying type of the bit-field is 'unsigned int'");
}
