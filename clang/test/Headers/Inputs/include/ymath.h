#pragma once

// Simplified MSVC STL <ymath.h>: C functions of the MSVC runtime that MSVC's
// <complex> calls.

#ifdef __cplusplus
extern "C" {
#endif

double _Cosh(double, double);
double _Sinh(double, double);
short _Exp(double *, double, short);
float _FCosh(float, float);
float _FSinh(float, float);
short _FExp(float *, float, short);

#ifdef __cplusplus
}
#endif
