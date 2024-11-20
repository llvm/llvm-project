.. include:: check.rst

=========
threads.h
=========

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
  * - ONCE_FLAG_INIT
    -
    - 7.28.1.3
    -
  * - TSS_DTOR_ITERATIONS
    -
    - 7.28.1.3
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
  * - call_once
    - |check|
    - 7.28.2.1
    -
  * - cnd_broadcast
    - |check|
    - 7.28.3.1
    -
  * - cnd_destroy
    - |check|
    - 7.28.3.2
    -
  * - cnd_init
    - |check|
    - 7.28.3.3
    -
  * - cnd_signal
    - |check|
    - 7.28.3.4
    -
  * - cnd_timedwait
    -
    - 7.28.3.5
    -
  * - cnd_wait
    - |check|
    - 7.28.3.6
    -
  * - mtx_destroy
    - |check|
    - 7.28.4.1
    -
  * - mtx_init
    - |check|
    - 7.28.4.2
    -
  * - mtx_lock
    - |check|
    - 7.28.4.3
    -
  * - mtx_timedlock
    -
    - 7.28.4.4
    -
  * - mtx_trylock
    -
    - 7.28.4.5
    -
  * - mtx_unlock
    - |check|
    - 7.28.4.6
    -
  * - thrd_create
    - |check|
    - 7.28.5.1
    -
  * - thrd_current
    - |check|
    - 7.28.5.2
    -
  * - thrd_detach
    - |check|
    - 7.28.5.3
    -
  * - thrd_equal
    - |check|
    - 7.28.5.4
    -
  * - thrd_exit
    - |check|
    - 7.28.5.5
    -
  * - thrd_join
    - |check|
    - 7.28.5.6
    -
  * - thrd_sleep
    -
    - 7.28.5.7
    -
  * - thrd_yield
    -
    - 7.28.5.8
    -
  * - tss_create
    - |check|
    - 7.28.6.1
    -
  * - tss_delete
    - |check|
    - 7.28.6.2
    -
  * - tss_get
    - |check|
    - 7.28.6.3
    -
  * - tss_set
    - |check|
    - 7.28.6.4
    -
