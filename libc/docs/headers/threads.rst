.. include:: ../check.rst

=========
threads.h
=========

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
  * - ONCE_FLAG_INIT
    -
    - 7.28.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/threads.h.html>`__
  * - TSS_DTOR_ITERATIONS
    -
    - 7.28.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/threads.h.html>`__
  * - __STDC_NO_THREADS__
    -
    - 7.28.1
    -
  * - thread_local
    -
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/threads.h.html>`__

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
  * - call_once
    - |check|
    - 7.28.2.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/call_once.html>`__
  * - cnd_broadcast
    - |check|
    - 7.28.3.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/cnd_broadcast.html>`__
  * - cnd_destroy
    - |check|
    - 7.28.3.2
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/cnd_destroy.html>`__
  * - cnd_init
    - |check|
    - 7.28.3.3
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/cnd_init.html>`__
  * - cnd_signal
    - |check|
    - 7.28.3.4
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/cnd_signal.html>`__
  * - cnd_timedwait
    -
    - 7.28.3.5
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/cnd_timedwait.html>`__
  * - cnd_wait
    - |check|
    - 7.28.3.6
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/cnd_wait.html>`__
  * - mtx_destroy
    - |check|
    - 7.28.4.2
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/mtx_destroy.html>`__
  * - mtx_init
    - |check|
    - 7.28.4.3
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/mtx_init.html>`__
  * - mtx_lock
    - |check|
    - 7.28.4.4
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/mtx_lock.html>`__
  * - mtx_timedlock
    -
    - 7.28.4.5
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/mtx_timedlock.html>`__
  * - mtx_trylock
    -
    - 7.28.4.6
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/mtx_trylock.html>`__
  * - mtx_unlock
    - |check|
    - 7.28.4.7
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/mtx_unlock.html>`__
  * - thrd_create
    - |check|
    - 7.28.5.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/thrd_create.html>`__
  * - thrd_current
    - |check|
    - 7.28.5.2
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/thrd_current.html>`__
  * - thrd_detach
    - |check|
    - 7.28.5.3
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/thrd_detach.html>`__
  * - thrd_equal
    - |check|
    - 7.28.5.4
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/thrd_equal.html>`__
  * - thrd_exit
    - |check|
    - 7.28.5.5
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/thrd_exit.html>`__
  * - thrd_join
    - |check|
    - 7.28.5.6
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/thrd_join.html>`__
  * - thrd_sleep
    -
    - 7.28.5.7
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/thrd_sleep.html>`__
  * - thrd_yield
    -
    - 7.28.5.8
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/thrd_yield.html>`__
  * - tss_create
    - |check|
    - 7.28.6.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/tss_create.html>`__
  * - tss_delete
    - |check|
    - 7.28.6.2
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/tss_delete.html>`__
  * - tss_get
    - |check|
    - 7.28.6.3
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/tss_get.html>`__
  * - tss_set
    - |check|
    - 7.28.6.4
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/tss_set.html>`__
