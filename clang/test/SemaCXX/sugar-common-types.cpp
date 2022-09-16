// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++20 -fenable-matrix

enum class N {};

using B1 = int;
using X1 = B1;
using Y1 = B1;

using B2 = void;
using X2 = B2;
using Y2 = B2;

using A3 = char __attribute__((vector_size(4)));
using B3 = A3;
using X3 = B3;
using Y3 = B3;

using A4 = float;
using B4 = A4 __attribute__((matrix_type(4, 4)));
using X4 = B4;
using Y4 = B4;

using X5 = A4 __attribute__((matrix_type(3, 4)));
using Y5 = A4 __attribute__((matrix_type(4, 3)));

N t1 = 0 ? X1() : Y1(); // expected-error {{rvalue of type 'B1'}}
N t2 = 0 ? X2() : Y2(); // expected-error {{rvalue of type 'B2'}}

const X1 &xt3 = 0;
const Y1 &yt3 = 0;
N t3 = 0 ? xt3 : yt3; // expected-error {{lvalue of type 'const B1'}}

N t4 = X3() + Y3();   // expected-error {{rvalue of type 'B3'}}

N t5 = A3() ? X3() : Y3(); // expected-error {{rvalue of type 'B3'}}
N t6 = A3() ? X1() : Y1(); // expected-error {{vector condition type 'A3' (vector of 4 'char' values) and result type '__attribute__((__vector_size__(4 * sizeof(B1)))) B1' (vector of 4 'B1' values) do not have elements of the same size}}

N t7 = X4() + Y4(); // expected-error {{rvalue of type 'B4'}}
N t8 = X4() * Y4(); // expected-error {{rvalue of type 'B4'}}
N t9 = X5() * Y5(); // expected-error {{rvalue of type 'A4 __attribute__((matrix_type(3, 3)))'}}
