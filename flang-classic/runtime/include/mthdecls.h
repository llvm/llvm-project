/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#if !defined(_WIN64) && !defined(HOST_MINGW)
#define FLOAT_COMPLEX_TYPE float complex
#define FLOAT_COMPLEX_CREATE(real, imag) (real + imag * I)
#define FLOAT_COMPLEX_MUL_CC(a, b) a * b
#define FLOAT_COMPLEX_ADD_CC(a, b) a + b
#define FLOAT_COMPLEX_EQ_CC(a, b) a == b
#define DOUBLE_COMPLEX_TYPE double complex
#define DOUBLE_COMPLEX_CREATE(real, imag) (real + imag * I)
#define DOUBLE_COMPLEX_MUL_CC(a, b) a * b
#define DOUBLE_COMPLEX_ADD_CC(a, b) a + b
#define DOUBLE_COMPLEX_EQ_CC(a, b) a == b
#else
#define FLOAT_COMPLEX_TYPE _Fcomplex
#define FLOAT_COMPLEX_CREATE(real, imag) _FCbuild(real, imag)
#define FLOAT_COMPLEX_MUL_CC(a, b) _FCmulcc(a, b)
#define FLOAT_COMPLEX_ADD_CC(a, b) _FCbuild(crealf(a) + crealf(b), cimagf(a) + cimagf(b))
#define FLOAT_COMPLEX_EQ_CC(a, b) (crealf(a) == crealf(b) && cimagf(a) == cimagf(b))
#define DOUBLE_COMPLEX_TYPE _Dcomplex
#define DOUBLE_COMPLEX_CREATE(real, imag) _Cbuild(real, imag)
#define DOUBLE_COMPLEX_MUL_CC(a, b) _Cmulcc(a, b)
#define DOUBLE_COMPLEX_ADD_CC(a, b) _Cbuild(creal(a) + creal(b), cimag(a) + cimag(b))
#define DOUBLE_COMPLEX_EQ_CC(a, b) (creal(a) == creal(b) && cimag(a) == cimag(b))
#endif
