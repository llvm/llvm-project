#pragma once

// Simplified MSVC STL <ymath.h>: C functions of the MSVC runtime that MSVC's
// <complex> calls.

#ifdef __cplusplus
extern "C" {
#endif

double _Cosh(double, double) noexcept;
double _Sinh(double, double) noexcept;
short _Exp(double *, double, short) noexcept;
float _FCosh(float, float) noexcept;
float _FSinh(float, float) noexcept;
short _FExp(float *, float, short) noexcept;

#ifdef __cplusplus
}
#endif
