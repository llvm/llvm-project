// RUN: %clang_cc1 -triple=x86_64-apple-darwin -fsyntax-only -verify %s
//
// Ensure that when we use builtins in C++ code with templates that compute the
// valid immediate, the dead code with the invalid immediate doesn't error.
// expected-no-diagnostics

typedef short __v8hi __attribute__((__vector_size__(16)));

template <int Imm>
__v8hi test(__v8hi a) {
    if (Imm < 4)
      return __builtin_ia32_pshuflw(a, 0x55 * Imm);
    else
      return __builtin_ia32_pshuflw(a, 0x55 * (Imm - 4));
}

template __v8hi test<0>(__v8hi);
template __v8hi test<1>(__v8hi);
template __v8hi test<2>(__v8hi);
template __v8hi test<3>(__v8hi);
template __v8hi test<4>(__v8hi);
template __v8hi test<5>(__v8hi);
template __v8hi test<6>(__v8hi);
template __v8hi test<7>(__v8hi);
