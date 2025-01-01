.. include:: ../check.rst

========
string.h
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
  * - __STDC_VERSION_STRING_H__
    -
    - 7.26.1
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
  * - memccpy
    - |check|
    - 7.26.2.2
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/memccpy.html>`__
  * - memchr
    - |check|
    - 7.26.5.2
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/memchr.html>`__
  * - memcmp
    - |check|
    - 7.26.4.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/memcmp.html>`__
  * - memcpy
    - |check|
    - 7.26.2.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/memcpy.html>`__
  * - memmove
    - |check|
    - 7.26.2.3
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/memmove.html>`__
  * - mempcpy
    - |check|
    - TODO: glibc extension
    -
  * - memset
    - |check|
    - 7.26.6.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/memset.html>`__
  * - memset_explicit
    - |check|
    - 7.26.6.2
    -
  * - stpcpy
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/stpcpy.html>`__
  * - stpncpy
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/stpncpy.html>`__
  * - strcat
    - |check|
    - 7.26.3.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/strcat.html>`__
  * - strchr
    - |check|
    - 7.26.5.3
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/strchr.html>`__
  * - strcmp
    - |check|
    - 7.26.4.2
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/strcmp.html>`__
  * - strcoll
    - |check|
    - 7.26.4.3
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/strcoll.html>`__
  * - strcoll_l
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/strcoll_l.html>`__
  * - strcpy
    - |check|
    - 7.26.2.4
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/strcpy.html>`__
  * - strcspn
    - |check|
    - 7.26.5.4
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/strcspn.html>`__
  * - strdup
    - |check|
    - 7.26.2.6
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/strdup.html>`__
  * - strerror
    - |check|
    - 7.26.6.3
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/strerror.html>`__
  * - strlen
    - |check|
    - 7.26.6.4
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/strlen.html>`__
  * - strncat
    - |check|
    - 7.26.3.2
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/strncat.html>`__
  * - strncmp
    - |check|
    - 7.26.4.4
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/strncmp.html>`__
  * - strncpy
    - |check|
    - 7.26.2.5
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/strncpy.html>`__
  * - strndup
    - |check|
    - 7.26.2.7
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/strndup.html>`__
  * - strpbrk
    - |check|
    - 7.26.5.5
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/strpbrk.html>`__
  * - strrchr
    - |check|
    - 7.26.5.6
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/strrchr.html>`__
  * - strspn
    - |check|
    - 7.26.5.7
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/strspn.html>`__
  * - strstr
    - |check|
    - 7.26.5.8
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/strstr.html>`__
  * - strtok
    - |check|
    - 7.26.5.9
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/strtok.html>`__
  * - strtok_r
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/strtok_r.html>`__
  * - strxfrm
    - |check|
    - 7.26.4.5
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/strxfrm.html>`__
  * - strxfrm_l
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/strxfrm_l.html>`__
