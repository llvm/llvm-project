.. _math:

==============
Math Functions
==============

.. include:: check.rst

.. raw:: html

    <style> .green {color:green} </style>

.. role:: green

.. toctree::
  :hidden:

  math/log.rst


.. contents:: Table of Contents
  :depth: 4
  :local:

Source Locations
================

- The main source is located at: `libc/src/math <https://github.com/llvm/llvm-project/tree/main/libc/src/math>`_.
- The tests are located at: `libc/test/src/math <https://github.com/llvm/llvm-project/tree/main/libc/test/src/math>`_.
- The floating point utilities are located at: `libc/src/__support/FPUtil <https://github.com/llvm/llvm-project/tree/main/libc/src/__support/FPUtil>`_.

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
ceil           :green:`XA`      :green:`XA`     :green:`XA`
copysign       :green:`XA`      :green:`XA`     :green:`XA`
fabs           :green:`XA`      :green:`XA`     :green:`XA`
fdim           :green:`XA`      :green:`XA`     :green:`XA`
floor          :green:`XA`      :green:`XA`     :green:`XA`
fmax           :green:`XA`      :green:`XA`     :green:`XA`
fmin           :green:`XA`      :green:`XA`     :green:`XA`
fmod           :green:`XA`      :green:`XA`
fpclassify
frexp          :green:`XA`      :green:`XA`     :green:`XA`
ilogb          :green:`XA`      :green:`XA`     :green:`XA`
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
ldexp          :green:`XA`      :green:`XA`     :green:`XA`
llrint         :green:`XA`      :green:`XA`     :green:`XA`
llround        :green:`XA`      :green:`XA`     :green:`XA`
logb           :green:`XA`      :green:`XA`     :green:`XA`
lrint          :green:`XA`      :green:`XA`     :green:`XA`
lround         :green:`XA`      :green:`XA`     :green:`XA`
modf           :green:`XA`      :green:`XA`     :green:`XA`
nan
nearbyint      :green:`XA`      :green:`XA`     :green:`XA`
nextafter      :green:`XA`      :green:`XA`     :green:`XA`
nexttoward
remainder      :green:`XA`      :green:`XA`     :green:`XA`
remquo         :green:`XA`      :green:`XA`     :green:`XA`
rint           :green:`XA`      :green:`XA`     :green:`XA`
round          :green:`XA`      :green:`XA`     :green:`XA`
scalbn
signbit
trunc          :green:`XA`      :green:`XA`     :green:`XA`
============== ================ =============== ======================

Higher Math Functions
---------------------

============== ================ =============== ======================
<Func>         <Func_f> (float) <Func> (double) <Func_l> (long double)
============== ================ =============== ======================
acos           :green:`XA`
acosh          :green:`XA`
asin           :green:`XA`
asinh          :green:`XA`
atan           :green:`XA`
atan2
atanh          :green:`XA`
cbrt
cos            :green:`XA`      :green:`XA`
cosh           :green:`XA`
erf
erfc
exp            :green:`XA`
exp10          :green:`XA`
exp2           :green:`XA`
expm1          :green:`XA`
fma            :green:`XA`      :green:`XA`
hypot          :green:`XA`      :green:`XA`
lgamma
log            :green:`XA`
log10          :green:`XA`      :green:`XA`
log1p          :green:`XA`
log2           :green:`XA`
pow
sin            :green:`XA`      :green:`XA`
sincos         :green:`XA`      :green:`XA`
sinh           :green:`XA`
sqrt           :green:`XA`      :green:`XA`     :green:`XA`
tan            :green:`XA`
tanh           :green:`XA`
tgamma
============== ================ =============== ======================

Accuracy of Higher Math Functions
=================================

============== ================ =============== ======================
<Func>         <Func_f> (float) <Func> (double) <Func_l> (long double)
============== ================ =============== ======================
acos           :green:`XA`
acosh          :green:`XA`
asin           :green:`XA`
asinh          :green:`XA`
atan           :green:`XA`
atanh          :green:`XA`
cos            :green:`XA`      large
cosh           :green:`XA`
exp            :green:`XA`
exp10          :green:`XA`
exp2           :green:`XA`
expm1          :green:`XA`
fma            :green:`XA`      :green:`XA`
hypot          :green:`XA`      :green:`XA`
log            :green:`XA`
log10          :green:`XA`      :green:`XA`
log1p          :green:`XA`
log2           :green:`XA`
sin            :green:`XA`      large
sincos         :green:`XA`      large
sinh           :green:`XA`
sqrt           :green:`XA`      :green:`XA`     :green:`XA`
tan            :green:`XA`
tanh           :green:`XA`
============== ================ =============== ======================

Legends:

* `X` = x86_64, `A` = aarch64, `a` = arm32
* Green text (eg. :green:`XA`): correctly rounded for all 4 rounding modes.
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
| acosf        |        24 |                29 |        62 |                77 | :math:`[-1, 1]`                     | Ryzen 1700 | Ubuntu 22.04 LTS x86_64 | Clang 14.0.0 | FMA           |
+--------------+-----------+-------------------+-----------+-------------------+-------------------------------------+------------+-------------------------+--------------+---------------+
| acoshf       |        18 |                26 |        73 |                74 | :math:`[1, 21]`                     | Ryzen 1700 | Ubuntu 22.04 LTS x86_64 | Clang 14.0.0 | FMA           |
+--------------+-----------+-------------------+-----------+-------------------+-------------------------------------+------------+-------------------------+--------------+---------------+
| asinf        |        23 |                27 |        62 |                62 | :math:`[-1, 1]`                     | Ryzen 1700 | Ubuntu 22.04 LTS x86_64 | Clang 14.0.0 | FMA           |
+--------------+-----------+-------------------+-----------+-------------------+-------------------------------------+------------+-------------------------+--------------+---------------+
| asinhf       |        21 |                39 |        77 |                91 | :math:`[-10, 10]`                   | Ryzen 1700 | Ubuntu 22.04 LTS x86_64 | Clang 14.0.0 | FMA           |
+--------------+-----------+-------------------+-----------+-------------------+-------------------------------------+------------+-------------------------+--------------+---------------+
| atanf        |        27 |                29 |        79 |                68 | :math:`[-10, 10]`                   | Ryzen 1700 | Ubuntu 22.04 LTS x86_64 | Clang 14.0.0 | FMA           |
+--------------+-----------+-------------------+-----------+-------------------+-------------------------------------+------------+-------------------------+--------------+---------------+
| atanhf       |        18 |                66 |        68 |               133 | :math:`[-1, 1]`                     | Ryzen 1700 | Ubuntu 22.04 LTS x86_64 | Clang 14.0.0 | FMA           |
+--------------+-----------+-------------------+-----------+-------------------+-------------------------------------+------------+-------------------------+--------------+---------------+
| cosf         |        13 |                32 |        53 |                59 | :math:`[0, 2\pi]`                   | Ryzen 1700 | Ubuntu 20.04 LTS x86_64 | Clang 12.0.0 | FMA           |
+--------------+-----------+-------------------+-----------+-------------------+-------------------------------------+------------+-------------------------+--------------+---------------+
| coshf        |        14 |                20 |        50 |                48 | :math:`[-10, 10]`                   | Ryzen 1700 | Ubuntu 22.04 LTS x86_64 | Clang 14.0.0 | FMA           |
+--------------+-----------+-------------------+-----------+-------------------+-------------------------------------+------------+-------------------------+--------------+---------------+
| expf         |         9 |                 7 |        44 |                38 | :math:`[-10, 10]`                   | Ryzen 1700 | Ubuntu 20.04 LTS x86_64 | Clang 12.0.0 | FMA           |
+--------------+-----------+-------------------+-----------+-------------------+-------------------------------------+------------+-------------------------+--------------+---------------+
| exp10f       |        10 |                 8 |        40 |                38 | :math:`[-10, 10]`                   | Ryzen 1700 | Ubuntu 22.04 LTS x86_64 | Clang 14.0.0 | FMA           |
+--------------+-----------+-------------------+-----------+-------------------+-------------------------------------+------------+-------------------------+--------------+---------------+
| exp2f        |         9 |                 6 |        35 |                31 | :math:`[-10, 10]`                   | Ryzen 1700 | Ubuntu 22.04 LTS x86_64 | Clang 14.0.0 | FMA           |
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
| sinf         |        12 |                25 |        51 |                57 | :math:`[-\pi, \pi]`                 | Ryzen 1700 | Ubuntu 20.04 LTS x86_64 | Clang 12.0.0 | FMA           |
+--------------+-----------+-------------------+-----------+-------------------+-------------------------------------+------------+-------------------------+--------------+---------------+
| sincosf      |        19 |                30 |        57 |                68 | :math:`[-\pi, \pi]`                 | Ryzen 1700 | Ubuntu 20.04 LTS x86_64 | Clang 12.0.0 | FMA           |
+--------------+-----------+-------------------+-----------+-------------------+-------------------------------------+------------+-------------------------+--------------+---------------+
| sinhf        |        13 |                63 |        48 |               137 | :math:`[-10, 10]`                   | Ryzen 1700 | Ubuntu 22.04 LTS x86_64 | Clang 14.0.0 | FMA           |
+--------------+-----------+-------------------+-----------+-------------------+-------------------------------------+------------+-------------------------+--------------+---------------+
| tanf         |        16 |                50 |        61 |               107 | :math:`[-\pi, \pi]`                 | Ryzen 1700 | Ubuntu 22.04 LTS x86_64 | Clang 14.0.0 | FMA           |
+--------------+-----------+-------------------+-----------+-------------------+-------------------------------------+------------+-------------------------+--------------+---------------+
| tanhf        |        13 |                55 |        57 |               123 | :math:`[-10, 10]`                   | Ryzen 1700 | Ubuntu 22.04 LTS x86_64 | Clang 14.0.0 | FMA           |
+--------------+-----------+-------------------+-----------+-------------------+-------------------------------------+------------+-------------------------+--------------+---------------+

Algorithms + Implementation Details
===================================

* :doc:`math/log`

References
==========

* `CRLIBM <https://hal-ens-lyon.archives-ouvertes.fr/ensl-01529804/file/crlibm.pdf>`_.
* `RLIBM <https://people.cs.rutgers.edu/~sn349/rlibm/>`_.
* `Sollya <https://www.sollya.org/>`_.
* `The CORE-MATH Project <https://core-math.gitlabpages.inria.fr/>`_.
* `The GNU C Library (glibc) <https://www.gnu.org/software/libc/>`_.
* `The GNU MPFR Library <https://www.mpfr.org/>`_.
