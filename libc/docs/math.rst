===========================
Math Functions in LLVM-libc
===========================

.. role::  raw-html(raw)
    :format: html

.. |check| replace:: :raw-html:`&#x2705`

.. contents:: Table of Contents
  :depth: 4
  :local:

Summary
=======

* This document tracks the status of the implementation of math functions in
  LLVM libc.

Implementation Requirements / Goals
===================================

* The highest priority is to be as accurate as possible, according to the C and
  IEEE 754 standards.  By default, we will aim to be correctly rounded for `all rounding modes <https://en.cppreference.com/w/c/numeric/fenv/FE_round>`_.
  The current rounding mode of the floating point environment is used to perform
  computations and produce the final results.

  - To test for correctness, we compare the outputs with other correctly rounded
    multiple-precision math libraries such as the `GNU MPFR library <https://www.mpfr.org/>`_
    or the `CORE-MATH library <https://core-math.gitlabpages.inria.fr/>`_.

* Our next requirement is that the outputs are consistent across all platforms.
  Notice that the consistency requirement will be satisfied automatically if the
  implementation is correctly rounded.

* Our last requirement for the implementations is to have good and predicable
  performance:

  - The average performance should be comparable to other ``libc``
    implementations.
  - The worst case performance should be within 10X-20X of the average.
  - Platform-specific implementations or instructions could be added whenever it
    makes sense and provides significant performance boost.

* For other use cases that have strict requirements on the code size, memory
  footprint, or latency, such as embedded systems, we will aim to be as accurate
  as possible within the memory or latency budgets, and consistent across all
  platforms.


Source Locations
================

- The main source is located at: `libc/src/math <https://github.com/llvm/llvm-project/tree/main/libc/src/math>`_.
- The tests are located at: `libc/test/src/math <https://github.com/llvm/llvm-project/tree/main/libc/test/src/math>`_.
- The floating point utilities are located at: `libc/src/__support/FPUtil <https://github.com/llvm/llvm-project/tree/main/libc/src/__support/FPUtil>`_.

Add a new math function to LLVM libc
====================================

* To add a new math function, follow the steps at: `libc/src/math/docs/add_math_function.md <https://github.com/llvm/llvm-project/tree/main/libc/src/math/docs/add_math_function.md>`_.

Implementation Status
=====================

Basic Operations
----------------

============== ================ =============== ======================
<Func>         <Func_f> (float) <Func> (double) <Func_l> (long double)
============== ================ =============== ======================
ceil           |check|          |check|         |check|  
copysign       |check|          |check|         |check|  
fabs           |check|          |check|         |check|  
fdim           |check|          |check|         |check|  
floor          |check|          |check|         |check|  
fmax           |check|          |check|         |check|  
fmin           |check|          |check|         |check|  
fmod           |check|          |check|
fpclassify
frexp          |check|          |check|         |check|  
ilogb          |check|          |check|         |check|  
isfinite
isgreater
isgreaterequal
isinf
isless
islessequal
islessgreater
isnan
isnormal
isubordered
ldexp          |check|          |check|         |check|  
llrint         |check|          |check|         |check|  
llround        |check|          |check|         |check|  
logb           |check|          |check|         |check|  
lrint          |check|          |check|         |check|  
lround         |check|          |check|         |check|  
modf           |check|          |check|         |check|  
nan
nearbyint      |check|          |check|         |check|  
nextafter      |check|          |check|         |check|  
nexttoward
remainder      |check|          |check|         |check|  
remquo         |check|          |check|         |check|  
rint           |check|          |check|         |check|  
round          |check|          |check|         |check|  
scalbn
signbit
trunc          |check|          |check|         |check|  
============== ================ =============== ======================

Higher Math Functions
---------------------

============== ================ =============== ======================
<Func>         <Func_f> (float) <Func> (double) <Func_l> (long double)
============== ================ =============== ======================
acos
acosh
asin
asinh
atan
atan2
atanh
cbrt
cos            |check|          |check|
cosh
erf
erfc
exp            |check|
exp2           |check|
expm1          |check|
fma            |check|          |check|
hypot          |check|          |check|
lgamma
log            |check|
log10          |check|
log1p          |check|
log2           |check|
pow
sin            |check|          |check|
sincos         |check|          |check|
sinh
sqrt           |check|          |check|         |check|
tan
tanh
tgamma
============== ================ =============== ======================

Accuracy of Higher Math Functions
=================================

============== ================ =============== ======================
<Func>         <Func_f> (float) <Func> (double) <Func_l> (long double)
============== ================ =============== ======================
cos            0.776 ULPs       large
exp            |check|
exp2           |check|
expm1          |check|
fma            |check|          |check|
hypot          |check|          |check|
log            |check|
log10          |check|
log1p          |check|
log2           |check|
sin            |check|          large
sincos         0.776 ULPs       large
sqrt           |check|          |check|         |check|
============== ================ =============== ======================

Legends:

* |check|: correctly rounded for all 4 rounding modes.
* CR: correctly rounded for the default rounding mode (round-to-the-nearest,
  tie-to-even).
* x ULPs: largest errors recorded.

..
  TODO(lntue): Add a new page to discuss about the algorithms used in the
  implementations and include the link here.


Performance
===========

* Simple performance testings are located at: `libc/test/src/math/differential_testing <https://github.com/llvm/llvm-project/tree/main/libc/test/src/math/differential_testing>`_.

* We also use the *perf* tool from the `CORE-MATH <https://core-math.gitlabpages.inria.fr/>`_
  project: `link <https://gitlab.inria.fr/core-math/core-math/-/tree/master>`_.
  The performance results from the CORE-MATH's perf tool are reported in the
  table below, using the system library as reference (such as the `GNU C library <https://www.gnu.org/software/libc/>`_
  on Linux). Fmod performance results obtained with "differential_testing".

+--------------+-------------------------------+-------------------------------+-------------------------------------+---------------------------------------------------------------------+
| <Func>       | Reciprocal throughput (ns)    | Latency (ns)                  | Testing ranges                      | Testing configuration                                               |
|              +-----------+-------------------+-----------+-------------------+                                     +------------+-------------------------+--------------+---------------+
|              | LLVM libc | Reference (glibc) | LLVM libc | Reference (glibc) |                                     | CPU        | OS                      | Compiler     | Special flags |
+==============+===========+===================+===========+===================+=====================================+============+=========================+==============+===============+
| cosf         |        37 |                32 |        73 |                72 | :math:`[0, 2\pi]`                   | Ryzen 1700 | Ubuntu 20.04 LTS x86_64 | Clang 12.0.0 |               |
+--------------+-----------+-------------------+-----------+-------------------+-------------------------------------+------------+-------------------------+--------------+---------------+
| expf         |         9 |                 7 |        44 |                38 | :math:`[-10, 10]`                   | Ryzen 1700 | Ubuntu 20.04 LTS x86_64 | Clang 12.0.0 | FMA           |
+--------------+-----------+-------------------+-----------+-------------------+-------------------------------------+------------+-------------------------+--------------+---------------+
| exp2f        |        25 |                 8 |        81 |                37 | :math:`[-10, 10]`                   | Ryzen 1700 | Ubuntu 20.04 LTS x86_64 | Clang 12.0.0 | FMA           |
+--------------+-----------+-------------------+-----------+-------------------+-------------------------------------+------------+-------------------------+--------------+---------------+
| expm1f       |         9 |                44 |        42 |               121 | :math:`[-10, 10]`                   | Ryzen 1700 | Ubuntu 20.04 LTS x86_64 | Clang 12.0.0 | FMA           |
+--------------+-----------+-------------------+-----------+-------------------+-------------------------------------+------------+-------------------------+--------------+---------------+
| fmodf        |        73 |               263 |        -  |                 - | [MIN_NORMAL, MAX_NORMAL]            | i5 mobile  | Ubuntu 20.04 LTS x86_64 | Clang 12.0.0 |               |
|              +-----------+-------------------+-----------+-------------------+-------------------------------------+------------+-------------------------+--------------+---------------+
|              |         9 |                11 |        -  |                 - | [0, MAX_SUBNORMAL]                  | i5 mobile  | Ubuntu 20.04 LTS x86_64 | Clang 12.0.0 |               |
+--------------+-----------+-------------------+-----------+-------------------+-------------------------------------+------------+-------------------------+--------------+---------------+
| fmod         |       595 |              3297 |        -  |                 - | [MIN_NORMAL, MAX_NORMAL]            | i5 mobile  | Ubuntu 20.04 LTS x86_64 | Clang 12.0.0 |               |
|              +-----------+-------------------+-----------+-------------------+-------------------------------------+------------+-------------------------+--------------+---------------+
|              |        14 |                13 |        -  |                 - | [0, MAX_SUBNORMAL]                  | i5 mobile  | Ubuntu 20.04 LTS x86_64 | Clang 12.0.0 |               |
+--------------+-----------+-------------------+-----------+-------------------+-------------------------------------+------------+-------------------------+--------------+---------------+
| hypotf       |        25 |                15 |        64 |                49 | :math:`[-10, 10] \times [-10, 10]`  | Ryzen 1700 | Ubuntu 20.04 LTS x86_64 | Clang 12.0.0 |               |
+--------------+-----------+-------------------+-----------+-------------------+-------------------------------------+------------+-------------------------+--------------+---------------+
| logf         |        12 |                10 |        56 |                46 | :math:`[e^{-1}, e]`                 | Ryzen 1700 | Ubuntu 20.04 LTS x86_64 | Clang 12.0.0 | FMA           |
+--------------+-----------+-------------------+-----------+-------------------+-------------------------------------+------------+-------------------------+--------------+---------------+
| log10f       |        13 |                25 |        57 |                72 | :math:`[e^{-1}, e]`                 | Ryzen 1700 | Ubuntu 20.04 LTS x86_64 | Clang 12.0.0 | FMA           |
+--------------+-----------+-------------------+-----------+-------------------+-------------------------------------+------------+-------------------------+--------------+---------------+
| log1pf       |        16 |                33 |        61 |                97 | :math:`[e^{-0.5} - 1, e^{0.5} - 1]` | Ryzen 1700 | Ubuntu 20.04 LTS x86_64 | Clang 12.0.0 | FMA           |
+--------------+-----------+-------------------+-----------+-------------------+-------------------------------------+------------+-------------------------+--------------+---------------+
| log2f        |        13 |                10 |        57 |                46 | :math:`[e^{-1}, e]`                 | Ryzen 1700 | Ubuntu 20.04 LTS x86_64 | Clang 12.0.0 | FMA           |
+--------------+-----------+-------------------+-----------+-------------------+-------------------------------------+------------+-------------------------+--------------+---------------+
| sinf         |        14 |                26 |        65 |                59 | :math:`[-\pi, \pi]`                 | Ryzen 1700 | Ubuntu 20.04 LTS x86_64 | Clang 12.0.0 | FMA           |
+--------------+-----------+-------------------+-----------+-------------------+-------------------------------------+------------+-------------------------+--------------+---------------+

References
==========

* `CRLIBM <https://hal-ens-lyon.archives-ouvertes.fr/ensl-01529804/file/crlibm.pdf>`_.
* `RLIBM <https://people.cs.rutgers.edu/~sn349/rlibm/>`_.
* `Sollya <https://www.sollya.org/>`_.
* `The CORE-MATH Project <https://core-math.gitlabpages.inria.fr/>`_.
* `The GNU C Library (glibc) <https://www.gnu.org/software/libc/>`_.
* `The GNU MPFR Library <https://www.mpfr.org/>`_.
