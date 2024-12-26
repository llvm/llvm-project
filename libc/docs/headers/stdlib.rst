.. include:: ../check.rst

========
stdlib.h
========

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
  * - EXIT_FAILURE
    - |check|
    - 7.24
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/stdlib.h.html>`__
  * - EXIT_SUCCESS
    - |check|
    - 7.24
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/stdlib.h.html>`__
  * - MB_CUR_MAX
    - |check|
    - 7.24
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/stdlib.h.html>`__
  * - RAND_MAX
    - |check|
    - 7.24
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/stdlib.h.html>`__
  * - __STDC_VERSION_STDLIB_H__
    -
    - 7.24
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
  * - _Exit
    - |check|
    - 7.24.4.5
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/_Exit.html>`__
  * - abort
    - |check|
    - 7.24.4.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/abort.html>`__
  * - abs
    - |check|
    - 7.24.6.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/abs.html>`__
  * - aligned_alloc
    - |check|
    - 7.24.3.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/aligned_alloc.html>`__
  * - at_quick_exit
    - |check|
    - 7.24.4.3
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/at_quick_exit.html>`__
  * - atexit
    - |check|
    - 7.24.4.2
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/atexit.html>`__
  * - atof
    - |check|
    - 7.24.1.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/atof.html>`__
  * - atoi
    - |check|
    - 7.24.1.2
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/atoi.html>`__
  * - atol
    - |check|
    - 7.24.1.2
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/atol.html>`__
  * - atoll
    - |check|
    - 7.24.1.2
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/atoll.html>`__
  * - bsearch
    - |check|
    - 7.24.5.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/bsearch.html>`__
  * - calloc
    - |check|
    - 7.24.3.2
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/calloc.html>`__
  * - div
    - |check|
    - 7.24.6.2
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/div.html>`__
  * - exit
    - |check|
    - 7.24.4.4
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/exit.html>`__
  * - free
    - |check|
    - 7.24.3.3
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/free.html>`__
  * - free_aligned_sized
    -
    - 7.24.3.5
    -
  * - free_sized
    -
    - 7.24.3.4
    -
  * - getenv
    - |check|
    - 7.24.4.6
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/getenv.html>`__
  * - labs
    - |check|
    - 7.24.6.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/labs.html>`__
  * - ldiv
    - |check|
    - 7.24.6.2
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/ldiv.html>`__
  * - llabs
    - |check|
    - 7.24.6.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/llabs.html>`__
  * - lldiv
    - |check|
    - 7.24.6.2
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/lldiv.html>`__
  * - malloc
    - |check|
    - 7.24.3.6
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/malloc.html>`__
  * - mblen
    -
    - 7.24.7.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/mblen.html>`__
  * - mbstowcs
    -
    - 7.24.8.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/mbstowcs.html>`__
  * - mbtowc
    -
    - 7.24.7.2
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/mbtowc.html>`__
  * - memalignment
    -
    - 7.24.9.1
    -
  * - qsort
    - |check|
    - 7.24.5.2
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/qsort.html>`__
  * - quick_exit
    - |check|
    - 7.24.4.7
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/quick_exit.html>`__
  * - rand
    - |check|
    - 7.24.2.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/rand.html>`__
  * - realloc
    - |check|
    - 7.24.3.7
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/realloc.html>`__
  * - srand
    - |check|
    - 7.24.2.2
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/srand.html>`__
  * - strfromd
    - |check|
    - 7.24.1.3
    -
  * - strfromd128
    -
    - 7.24.1.4
    -
  * - strfromd32
    -
    - 7.24.1.4
    -
  * - strfromd64
    -
    - 7.24.1.4
    -
  * - strfromf
    - |check|
    - 7.24.1.3
    -
  * - strfroml
    - |check|
    - 7.24.1.3
    -
  * - strtod
    - |check|
    - 7.24.1.5
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/strtod.html>`__
  * - strtod128
    -
    - 7.24.1.6
    -
  * - strtod32
    -
    - 7.24.1.6
    -
  * - strtod64
    -
    - 7.24.1.6
    -
  * - strtof
    - |check|
    - 7.24.1.5
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/strtof.html>`__
  * - strtol
    - |check|
    - 7.24.1.7
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/strtol.html>`__
  * - strtold
    - |check|
    - 7.24.1.5
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/strtold.html>`__
  * - strtoll
    - |check|
    - 7.24.1.7
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/strtoll.html>`__
  * - strtoul
    - |check|
    - 7.24.1.7
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/strtoul.html>`__
  * - strtoull
    - |check|
    - 7.24.1.7
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/strtoull.html>`__
  * - system
    - |check|
    - 7.24.4.8
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/system.html>`__
  * - wcstombs
    -
    - 7.24.8.2
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/wcstombs.html>`__
  * - wctomb
    -
    - 7.24.7.3
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/wctomb.html>`__
