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

OCML is an LLVM-IR bitcode library designed to relieve language compiler and runtime implementers of the burden of implementing efficient and accurate mathematical functions.  It is essentially a “libm” in intermediate representation with a fixed, simple API that can be linked in to supply the implementations of most standard low-level mathematical functions provided by the language.

## Using OCML
### Standard Usage

OCML is expected to be used in a standard LLVM compilation flow as follows:
  * Compile source modules to LLVM-IR bitcode (clang)
  * Link program bitcode, “wrapper” bitcode, OCML bitcode, other device library bitcode, and OCML control functions (llvm-link)
  * Generic optimizations (opt)
  * Code generation (llc)

Here, “wrapper” bitcode denotes a thin library responsible for mapping language specific mangled built-in function calls as produced by clang to the OCML API.  An example for handling "sqrt" might look like

    extern "C" __attribute__((const)) float __ocml_sqrt_f32(float);
    float sqrt(float x) { return __ocml_sqrt_f32(x); }

The next section describes OCML controls and how to use them.

### Controls

OCML (and a few other device libraries) requires a number of control variables definitions to be provided.  These definitions may be provided by linking in specific OCLC libraries which define one specifically named variable or via other runtime specific means.  These variables are known at optimization time and optimizations will result in specific paths taken with no control flow overhead.  These variables all have the form (in C)

`__constant const int __oclc_<name> = N;`


The currently supported control `<name>`s and values `N` are
  * `finite_only_opt` - floating point Inf and NaN are never expected to be consumed or produced.  `N` may be 1 (on/true/enabled), or 0 (off/false/disabled).
  * `unsafe_math_opt` - lower accuracy results may be produced with higher performance.  `N` may be 1 (on/true/enabled) or 0 (off/false/disabled).
  * `daz_opt` - subnormal values consumed and produced may be flushed to zero.  `N`may be 1 (on/true/enabled) or 0 (off/false/disabled).
  * `correctly_rounded_sqrt32` - float square root must be correctly rounded.  `N` may be 1 (on/true/enabled) or 0 (off/false/disabled).
  * `wavefrontsize64` - the wave front size is 64.  `N` may be 1 (on/true/enabled) or 0 (off/false/disabled).  Very few current devices support a value of 0.
  * `ISA_version` - an integer representation of the ISA version of the target device

The language runtime can link a specific set of OCLC control libraries to properly configure OCML and other device libraries which also use the controls.  If linking OCLC libraries is used to define the control variables, then the runtime must link in:

- Exactly one of `oclc_correctly_rounded_sqrt_on.amdgcn.bc` or `oclc_correctly_rounded_sqrt_off.amdgcn.bc` depending on the kernel's requirements
- Exactly one of `oclc_daz_opt_on.amdgcn.bc` or `oclc_daz_opt_off.amdgcn.bc` depending on the kernel's requirements
- Exactly one of `oclc_finite_only_on.amdgcn.bc` or `oclc_finite_only_off.amdgcn.bc` depending on the kernel's requirements
- Exactly one of `oclc_unsafe_math_on.amdgcn.bc` or `oclc_unsafe_math_off.amdgcn.bc` depending on the kernel's requirements
- Exactly one of `oclc_wavefrontsize64_on.amdgcn.bc` or `oclc_wavefrontsize64_off.amdgcn.bc` depending on the kernel's requirements
- Exactly one of `oclc_isa_version_XYZ.amdgcn.bc` where XYZ is the suffix of the `gfxXYZ` target name the kernel is being compiled for.

If these rules are not followed, link time or execution time errors may result.

### Versioning

OCML ships within the larger release as a single LLVM-IR bitcode file named

    ocml.amdgcn.bc

Bitcode linking errors are possible if the library is not in-sync with the compiler shipped with the same release.

### Tables

Some OCML functions require access to tables of constants.  These tables are currently named
with the prefix `__ocmltbl_` and are placed in LLVM address space 2.

### Naming convention

OCML functions follow a simple naming convention:

    __ocml_{function}_{type suffix}

where `{function}` is generally the familiar libm name of the function, and `{type suffix}` indicates the type of the floating point arguments or results, and is one of
  * `f16` – 16 bit floating point (half precision)
  * `f32` – 32 bit floating point (single precision)
  * `f64` – 64 bit floating point (double precision)

For example, `__ocml_sqrt_f32` is the name of the OCML single precision square root function.

OCML does not currently support higher precision than double precision due to the lack of hardware support for such precisions. 

### Supported functions

The following table contains a list of {function} currently supported by OCML, a brief description of each, and the maximum relative error in ULPs for each floating point type.  A “c” in the last 3 columns indicates that the function is required to be correctly rounded.

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
| ncdf | standard normal cumulative distribution function | 16 | 16 | 4 |
| ncdfinv | inverse standard normal cumulative distribution function | 16 | 16 | 4 |
| nearbyint | round to nearest integer (see also rint) | 0 | 0 | 0 |
| nextafter | next closest value above or below | 0 | 0 | 0 |
| pow | general power | 16 | 16 | 4 |
| pown | power with integral exponent | 16 | 16 | 4 |
| powr | power with positive floating point exponent | 16 | 16 | 4 |
| pred | predecessor | c | c | c |
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
| succ | successor | c | c | c |
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

Note that these functions are not currently available.

