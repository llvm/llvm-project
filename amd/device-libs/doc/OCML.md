# OCML User Guide

* [Introduction](#introduction)
  * [What Is OCML](#what-is-ocml)
* [Using OCML](#using-ocml)
  * [Standard Usage](#standard-usage)
  * [Controls](#controls)
* [Versioning](#versioning)
* [Tables](#tables)
* [Naming convention](#naming-convention)
* [Supported functions](#supported-functions)


## Introduction
### What Is OCML

OCML is an LLVM-IR bitcode library designed to relieve language compiler and
runtime implementers of the burden of implementing efficient and accurate
mathematical functions.  It is essentially a “libm” in intermediate
representation with a fixed, simple API that can be linked in to supply the
implementations of most standard low-level mathematical functions provided by the language.

## Using OCML
### Standard Usage

OCML is expected to be used in a standard LLVM compilation flow as follows:
  * Compile source modules to LLVM-IR bitcode (clang)
  * Link program bitcode, “wrapper” bitcode, OCML bitcode, and OCML control functions (llvm-link)
  * Generic optimizations (opt)
  * Code generation (llc)

Here, “wrapper” bitcode denotes a thin library responsible for mapping
mangled built-in function calls as produced by clang to the OCML API.  An example in C might look like

    inline float sqrt(float x) { return __ocml_sqrt_f32(x); }

The next section describes OCML controls and how to make them.

### Controls

OCML supports a number of controls that are provided by linking in specifically named inline
functions.  These functions are inlined at optimization time and result in specific paths
taken with no control flow overhead.  These functions all have the form (in C)

    __attribute__((always_inline, const)) int
    __oclc_control(void)
    { return 1; } // or 0 to disable

The currently supported control are
  * `finite_only_opt` - floating point Inf and NaN are never expected to be consumed or produced
  * `unsafe_math_opt` - lower accuracy results may be produced with higher performance
  * `daz_opt` - subnormal values consumed and produced may be flushed to zero
  * `correctly_rounded_sqrt32` - float square root must be correctly rounded
  * `ISA_version` - an integer representation of the ISA version of the target device

### Versioning

OCML ships as a single LLVM-IR bitcode file named

    ocml-{LLVM rev}-{OCLM rev}.bc

where `{LLVM rev}` is the version of LLVM used to create the file, of the
form X.Y, e.g. 3.8, and `{OCML rev}` is the OCML library version of the form X.Y, currently 0.9.

### Tables

Some OCML functions require access to tables of constants.  These tables are currently named
with the prefix `__ocmltbl_` and are placed in LLVM address space 2.

### Naming convention

OCML functions follow a simple naming convention:

    __ocml_{function}_{type suffix}

where {function} is generally the familiar libm name of the function, and {type suffix} indicates the type of the floating point arguments or results, and is one of
  * `f16` – 16 bit floating point (half precision)
  * `f32` – 32 bit floating point (single precision)
  * `f64` – 64 bit floating point (double precision)

For example, `__ocml_sqrt_f32` is the name of the OCML single precision square root function.

OCML does not currently support higher than double precision due to the lack of support on most devices. 

### Supported functions

The following table contains a list of {function} currently supported by OCML, a brief
description of each, and the maximum relative error in ULPs for each floating point
type.  A “c” in the last 3 columns indicates that the function is required to
be correctly rounded.

| **{function}** | **Description** | **f32 max err** | **f64 max err** | **f16 max err** |
| --- | --- | --- | --- | --- |
| acos | arc cosine | 4 | 4 | 2 |
| acosh | arc hyperbolic cosine | 4 | 4 | 2 |
| acospi | arc cosine / π | 5 | 5 | 2 |
| add_{rm} | add with specific rounding mode | c | c | c |
| asin | arc sine | 4 | 4 | 2 |
| asinh | arc hyperbolic sin | 4 | 4 | 2 |
| asinpi | arc sine / pi | 5 | 5 | 2 |
| atan2 | two argument arc tangent | 6 | 6 | 2 |
| atan2pi | two argument arc tangent / pi | 6 | 6 | 2 |
| atan | single argument arc tangent | 5 | 5 | 2 |
| atanh | arc hyperbolic tangent | 5 | 5 | 2 |
| atanpi | single argument arc tangent / pi | 5 | 5 | 2 |
| cbrt | cube root | 2 | 2 | 2 |
| ceil | round upwards to integer | c | c | c |
| copysign | copy sign of second argument to absolute value of first | 0 | 0 | 0 |
| cos | cosine | 4 | 4 | 2 |
| cosh | hyperbolic cosine | 4 | 4 | 2 |
| cospi | cosine of argument times pi | 4 | 4 | 2 |
| div_{rm} | correctly rounded division with specific rounding mode | c | c | c |
| erf | error function | 16 | 16 | 4 |
| erfc | complementary error function | 16 | 16 | 4 |
| erfcinv | inverse complementary error function | 7 | 8 | 3 |
| erfcx | scaled error function | 6 | 6 | 2 |
| erfinv | inverse error function | 3 | 8 | 2 |
| exp10 | 10x | 3 | 3 | 2 |
| exp2 | 2x | 3 | 3 | 2 |
| exp | ex | 3 | 3 | 2 |
| expm1 | ex -  1, accurate at 0 | 3 | 3 | 2 |
| fabs | absolute value | 0 | 0 | 0 |
| fdim | positive difference | c | c | c |
| floor | round downwards to integer | c | c | c |
| fma[_{rm}] | fused (i.e. singly rounded) multiply-add, with optional specific rounding | c | c | c |
| fmax | maximum, avoids NaN | 0 | 0 | 0 |
| fmin | minimum, avoids NaN | 0 | 0 | 0 |
| fmod | floating point remainder | 0 | 0 | 0 |
| fpclassify | classify floating point | - | - | - |
| fract | fractional part | c | c | c |
| frexp | extract significand and exponent | 0 | 0 | 0 |
| hypot | length, with overflow control | 4 | 4 | 2 |
| i0 | modified Bessel function of the first kind, order 0, I0 | 6 | 6 | 2 |
| i1 | modified Bessel function of the first kind, order 1, I1 | 6 | 6 | 2 |
| ilogb | extract exponent | 0 | 0 | 0 |
| isfinite | tests finiteness | - | - | - |
| isinf | test for Inf | - | - | - |
| isnan | test for NaN | - | - | - |
| isnormal | test for normal | - | - | - |
| j0 | Bessel function of the first kind, order 0, J0 | 6 (<12) | 6 (<12) | 2 (<12) |
| j1 | Bessel function of the first kind, order 1, J1 | 6 (<12) | 6 (<12) | 2 (<12) |
| ldexp | multiply by 2 raised to an integral power | c | c | c |
| len3 | three argument hypot | 2 | 2 | 2|
| len4 | four argument hypot | 2 | 2 | 2|
| lgamma | log Γ function | 6(>0) | 4(>0) | 3(>0) |
| lgamma_r | log Γ function with sign | 6(>0) | 4(>0) | 3(>0) |
| log10 | log base 10 | 3 | 3 | 2 |
| log1p | log base e accurate near 1 | 2 | 2 | 2 |
| log2 | log base 2 | 3 | 3 | 2 |
| log | log base e | 3 | 3 | 2 |
| logb | extract exponent | 0 | 0 | 0 |
| mad | multiply-add, implementation defined if fused | c | c | c |
| max | maximum without special NaN handling | 0 | 0 | 0 |
| maxmag | maximum magnitude | 0 | 0 | 0 |
| min | minimum without special NaN handling | 0 | 0 | 0 |
| minmag | minimum magnitude | 0 | 0 | 0 |
| modf | extract integer and fraction | 0 | 0 | 0 |
| mul_{rm} | multiply with specific rounding mode | c | c | c |
| nan | produce a NaN with a specific payload | 0 | 0 | 0 |
| ncdf | standard normal cumulateive distribution function | 16 | 16 | 4 |
| ncdfinv | inverse standard normal cumulative distribution function | 16 | 16 | 4 |
| nearbyint | round to nearest integer (see also rint) | 0 | 0 | 0 |
| nextafter | next closest value above or below | 0 | 0 | 0 |
| pow | general power | 16 | 16 | 4 |
| pown | power with integral exponent | 16 | 16 | 4 |
| powr | power with positive floating point exponent | 16 | 16 | 4 |
| rcbrt | reciprocal cube root | 2 | 2 | 2 |
| remainder | floating point remainder | 0 | 0 | 0 |
| remquo | floating point remainder and lowest integral quotient bits | 0 | 0 | 0 |
| rhypot | reciprocal hypot | 2 | 2 | 2 |
| rint | round to nearest integer | c | c | c |
| rlen3 | reciprocal len3 | 2 | 2 | 2 |
| rlen4 | reciprocal len4 | 2 | 2 | 2 |
| rootn | nth root | 16 | 16 | 4 |
| round | round to integer, always away from 0 | c | c | c |
| rsqrt | reciprocal square root | 2 | 2 | 1 |
| scalb | multiply by 2 raised to a power | c | c | c |
| scalbn | multiply by 2 raised to an integral power (see also ldexp) | c | c | c |
| signbit | nonzero if argument has sign bit set | - | - | - |
| sin | sine function | 4 | 4 | 2 |
| sincos | simultaneous sine and cosine evaluation | 4 | 4 | 2 |
| sincospi | sincos function of argument times pi | 4 | 4 | 2 |
| sinh | hyperbolic sin | 4 | 4 | 2 |
| sinpi | sine of argument times pi | 4 | 4 | 2 |
| sqrt | square root | 3/c | 3/c | c |
| sub_{rm} | subtract with specific rounding mode | c | c | c |
| tan | tangent | 5 | 5 | 2 |
| tanh | hyperbolic tangent | 5 | 5 | 2 |
| tanpi | tangent of argument times pi | 6 | 6 | 2 |
| tgamma | true Γ function | 16 | 16 | 4 |
| trunc | round to integer, towards zero | c | c | c |
| y0 | Bessel function of the second kind, order 0, Y0 | 2 (<12) | 6 (<12) | 6 (<12) |
| y1 | Bessel function of the second kind, order 1, Y1 | 2 (<12) | 6 (<12) | 6 (<12) |

For the functions supporting specific roundings, the rounding mode {rm} can be one of
  * `rte` – round towards nearest even
  * `rtp` – round towards positive infinity
  * `rtn` – round towards negative infinity
  * `rtz` – round towards zero
