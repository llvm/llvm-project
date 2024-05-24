.. include:: check.rst

========
signal.h
========

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
  * - SIGABRT
    - |check|
    - 7.14.3
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIGALRM
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIGBUS
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIGCHLD
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIGCONT
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIGFPE
    - |check|
    - 7.14.3
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIGHUP
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIGILL
    - |check|
    - 7.14.3
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIGINT
    - |check|
    - 7.14.3
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIGKILL
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIGPIPE
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIGPOLL
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIGPROF
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIGQUIT
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIGRTMAX
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIGRTMIN
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIGSEGV
    - |check|
    - 7.14.3
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIGSTOP
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIGSYS
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIGTERM
    - |check|
    - 7.14.3
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIGTRAP
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIGTSTP
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIGTTIN
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIGTTOU
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIGURG
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIGUSR1
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIGUSR2
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIGVTALRM
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIGXCPU
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIGXFSZ
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIG_DFL
    - |check|
    - 7.14.3
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIG_ERR
    - |check|
    - 7.14.3
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIG_HOLD
    -
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
  * - SIG_IGN
    - |check|
    - 7.14.3
    - https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html

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
  * - kill
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/functions/kill.html
  * - raise
    - |check|
    - 7.14.2.1
    - https://pubs.opengroup.org/onlinepubs/9699919799/functions/raise.html
  * - sigaction
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/functions/sigaction.html
  * - sigaddset
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/functions/sigaddset.html
  * - sigaltstack
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/functions/sigaltstack.html
  * - sigdelset
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/functions/sigdelset.html
  * - sigemptyset
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/functions/sigemptyset.html
  * - sigfillset
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/functions/sigfillset.html
  * - signal
    - |check|
    - 7.14.1.1
    - https://pubs.opengroup.org/onlinepubs/9699919799/functions/signal.html
  * - sigprocmask
    - |check|
    -
    - https://pubs.opengroup.org/onlinepubs/9699919799/functions/sigprocmask.html
