// RUN: %clang_cc1 -fenable-matrix -fsyntax-only -verify %s -std=c++11

using Matrix = int __attribute__((matrix_type(4, 3)));

template <__SIZE_TYPE__ a, __SIZE_TYPE__ b>
using TMatrix = int __attribute__((matrix_type(a, b)));

struct S {
    void* operator new(__SIZE_TYPE__, int);
    void* operator new(__SIZE_TYPE__, Matrix);
    void* operator new(__SIZE_TYPE__, TMatrix<2, 2>);
};

int main() {
    Matrix m;
    TMatrix<2, 2> tm;

    new (m) S {};
    new (tm) S {};

    new (m[1][1]) S {};
    new (tm[1][1]) S {};

    new (m[1]) S {}; // expected-error {{single subscript expressions are not allowed for matrix values}}
    new (tm[1]) S {}; // expected-error {{single subscript expressions are not allowed for matrix values}}
}
