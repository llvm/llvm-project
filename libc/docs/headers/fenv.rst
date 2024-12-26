.. include:: ../check.rst

======
fenv.h
======

Macros
======

.. list-table::
  :widths: auto
  :align: center
  :header-rows: 1

  * - Macro
    - Implemented
    - C23 Standard Section
    - POSIX Docs
  * - FE_ALL_EXCEPT
    - |check|
    - 7.6.12
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/fenv.h.html>`__
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
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/fenv.h.html>`__
  * - FE_DFL_MODE
    -
    - 7.6.11
    -
  * - FE_DIVBYZERO
    - |check|
    - 7.6.9
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/fenv.h.html>`__
  * - FE_DOWNWARD
    - |check|
    - 7.6.13
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/fenv.h.html>`__
  * - FE_INEXACT
    - |check|
    - 7.6.9
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/fenv.h.html>`__
  * - FE_INVALID
    - |check|
    - 7.6.9
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/fenv.h.html>`__
  * - FE_OVERFLOW
    - |check|
    - 7.6.9
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/fenv.h.html>`__
  * - FE_TONEAREST
    - |check|
    - 7.6.13
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/fenv.h.html>`__
  * - FE_TONEARESTFROMZERO
    -
    - 7.6.13
    -
  * - FE_TOWARDZERO
    - |check|
    - 7.6.13
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/fenv.h.html>`__
  * - FE_UNDERFLOW
    - |check|
    - 7.6.9
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/fenv.h.html>`__
  * - FE_UPWARD
    - |check|
    - 7.6.13
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/fenv.h.html>`__
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
    - POSIX Docs
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
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/feclearexcept.html>`__
  * - fegetenv
    - |check|
    - 7.6.6.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/fegetenv.html>`__
  * - fegetexceptflag
    - |check|
    - 7.6.4.2
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/fegetexceptflag.html>`__
  * - fegetmode
    -
    - 7.6.5.1
    -
  * - fegetround
    - |check|
    - 7.6.5.2
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/fegetround.html>`__
  * - feholdexcept
    - |check|
    - 7.6.6.2
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/feholdexcept.html>`__
  * - feraiseexcept
    - |check|
    - 7.6.4.3
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/feraiseexcept.html>`__
  * - fesetenv
    - |check|
    - 7.6.6.3
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/fesetenv.html>`__
  * - fesetexcept
    - |check|
    - 7.6.4.4
    -
  * - fesetexceptflag
    - |check|
    - 7.6.4.5
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/fesetexceptflag.html>`__
  * - fesetmode
    -
    - 7.6.5.4
    -
  * - fesetround
    - |check|
    - 7.6.5.5
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/fesetround.html>`__
  * - fetestexcept
    - |check|
    - 7.6.4.7
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/fetestexcept.html>`__
  * - fetestexceptflag
    - |check|
    - 7.6.4.6
    -
  * - feupdateenv
    - |check|
    - 7.6.6.4
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/feupdateenv.html>`__
