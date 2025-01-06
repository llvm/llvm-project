.. include:: ../check.rst

========
signal.h
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
  * - SIGABRT
    - |check|
    - 7.14.3
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIGALRM
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIGBUS
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIGCHLD
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIGCONT
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIGFPE
    - |check|
    - 7.14.3
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIGHUP
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIGILL
    - |check|
    - 7.14.3
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIGINT
    - |check|
    - 7.14.3
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIGKILL
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIGPIPE
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIGPOLL
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIGPROF
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIGQUIT
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIGRTMAX
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIGRTMIN
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIGSEGV
    - |check|
    - 7.14.3
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIGSTOP
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIGSYS
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIGTERM
    - |check|
    - 7.14.3
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIGTRAP
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIGTSTP
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIGTTIN
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIGTTOU
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIGURG
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIGUSR1
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIGUSR2
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIGVTALRM
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIGXCPU
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIGXFSZ
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIG_DFL
    - |check|
    - 7.14.3
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIG_ERR
    - |check|
    - 7.14.3
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIG_HOLD
    -
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__
  * - SIG_IGN
    - |check|
    - 7.14.3
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/signal.h.html>`__

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
  * - kill
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/kill.html>`__
  * - raise
    - |check|
    - 7.14.2.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/raise.html>`__
  * - sigaction
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/sigaction.html>`__
  * - sigaddset
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/sigaddset.html>`__
  * - sigaltstack
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/sigaltstack.html>`__
  * - sigdelset
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/sigdelset.html>`__
  * - sigemptyset
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/sigemptyset.html>`__
  * - sigfillset
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/sigfillset.html>`__
  * - signal
    - |check|
    - 7.14.1.1
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/signal.html>`__
  * - sigprocmask
    - |check|
    -
    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/sigprocmask.html>`__
