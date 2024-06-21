.. include:: check.rst

======
fenv.h
======

Macros
======

.. list-table::
  :widths: auto
  :align: center
  :header-rows: 1

  * - Function
    - Implemented
    - C23 Standard Section
    - POSIX.1-2017 Standard Section
  * - FE_ALL_EXCEPT
    - |check|
    - 7.6.12
    -
  * - FE_DEC_DOWNWARD
    -
    - 7.6.14
    -
  * - FE_DEC_TONEAREST
    -
    - 7.6.14
    -
  * - FE_DEC_TONEARESTFROMZERO
    -
    - 7.6.14
    -
  * - FE_DEC_TOWARDZERO
    -
    - 7.6.14
    -
  * - FE_DEC_UPWARD
    -
    - 7.6.14
    -
  * - FE_DFL_ENV
    - |check|
    - 7.6.17
    -
  * - FE_DFL_MODE
    -
    - 7.6.11
    -
  * - FE_DIVBYZERO
    - |check|
    - 7.6.9
    -
  * - FE_DOWNARD
    -
    - 7.6.13
    -
  * - FE_INEXACT
    - |check|
    - 7.6.9
    -
  * - FE_INVALID
    - |check|
    - 7.6.9
    -
  * - FE_OVERFLOW
    - |check|
    - 7.6.9
    -
  * - FE_TONEAREST
    - |check|
    - 7.6.13
    -
  * - FE_TONEARESTFROMZERO
    -
    - 7.6.13
    -
  * - FE_TOWARDZERO
    - |check|
    - 7.6.13
    -
  * - FE_UNDERFLOW
    - |check|
    - 7.6.9
    -
  * - FE_UPWARD
    - |check|
    - 7.6.13
    -
  * - __STDC_VERSION_FENV_H__
    -
    - 7.6.5
    -

Functions
=========

.. list-table::
  :widths: auto
  :align: center
  :header-rows: 1

  * - Function
    - Implemented
    - C23 Standard Section
    - POSIX.1-2017 Standard Section
  * - fe_dec_getround
    -
    - 7.6.5.3
    -
  * - fe_dec_setround
    -
    - 7.6.5.6
    -
  * - feclearexcept
    - |check|
    - 7.6.4.1
    -
  * - fegetenv
    - |check|
    - 7.6.6.1
    -
  * - fegetexceptflag
    - |check|
    - 7.6.4.2
    -
  * - fegetmode
    -
    - 7.6.5.1
    -
  * - fegetround
    - |check|
    - 7.6.5.2
    -
  * - feholdexcept
    - |check|
    - 7.6.6.2
    -
  * - feraiseexcept
    - |check|
    - 7.6.4.3
    -
  * - fesetenv
    - |check|
    - 7.6.6.3
    -
  * - fesetexcept
    - |check|
    - 7.6.4.4
    -
  * - fesetexceptflag
    - |check|
    - 7.6.4.5
    -
  * - fesetmode
    -
    - 7.6.5.4
    -
  * - fesetround
    - |check|
    - 7.6.5.5
    -
  * - fetestexcept
    - |check|
    - 7.6.4.7
    -
  * - fetestexceptflag
    - |check|
    - 7.6.4.6
    -
  * - feupdateenv
    - |check|
    - 7.6.6.4
    -
