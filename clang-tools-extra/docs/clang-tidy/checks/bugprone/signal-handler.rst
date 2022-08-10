.. title:: clang-tidy - bugprone-signal-handler

bugprone-signal-handler
=======================

Finds specific constructs in signal handler functions that can cause undefined
behavior. The rules for what is allowed differ between C++ language versions.

Checked signal handler rules for C:

- Calls to non-asynchronous-safe functions are not allowed.

Checked signal handler rules for up to and including C++14:

- Calls to non-asynchronous-safe functions are not allowed.
- C++-specific code constructs are not allowed in signal handlers.
  In other words, only the common subset of C and C++ is allowed to be used.
- Calls to functions with non-C linkage are not allowed (including the signal
  handler itself).

The check is disabled on C++17 and later.

Asnychronous-safety is determined by comparing the function's name against a set
of known functions. In addition, the function must come from a system header
include and in a global namespace. The (possible) arguments passed to the
function are not checked. Any function that cannot be determined to be
asynchronous-safe is assumed to be non-asynchronous-safe by the check,
including user functions for which only the declaration is visible.
Calls to user-defined functions with visible definitions are checked
recursively.

This check implements the CERT C Coding Standard rule
`SIG30-C. Call only asynchronous-safe functions within signal handlers
<https://www.securecoding.cert.org/confluence/display/c/SIG30-C.+Call+only+asynchronous-safe+functions+within+signal+handlers>`_
and the rule
`MSC54-CPP. A signal handler must be a plain old function
<https://wiki.sei.cmu.edu/confluence/display/cplusplus/MSC54-CPP.+A+signal+handler+must+be+a+plain+old+function>`_.
It has the alias names ``cert-sig30-c`` and ``cert-msc54-cpp``.

Options
-------

.. option:: AsyncSafeFunctionSet

  Selects which set of functions is considered as asynchronous-safe
  (and therefore allowed in signal handlers). It can be set to the following values:
  
  ``minimal``
     Selects a minimal set that is defined in the CERT SIG30-C rule.
     and includes functions ``abort()``, ``_Exit()``, ``quick_exit()`` and
     ``signal()``.
  ``POSIX``
     Selects a larger set of functions that is listed in POSIX.1-2017 (see `this
     link
     <https://pubs.opengroup.org/onlinepubs/9699919799/functions/V2_chap02.html#tag_15_04_03>`_
     for more information). The following functions are included:
     ``_Exit``, ``_exit``, ``abort``, ``accept``, ``access``, ``aio_error``,
     ``aio_return``, ``aio_suspend``, ``alarm``, ``bind``, ``cfgetispeed``,
     ``cfgetospeed``, ``cfsetispeed``, ``cfsetospeed``, ``chdir``, ``chmod``,
     ``chown``, ``clock_gettime``, ``close``, ``connect``, ``creat``, ``dup``,
     ``dup2``, ``execl``, ``execle``, ``execv``, ``execve``, ``faccessat``,
     ``fchdir``, ``fchmod``, ``fchmodat``, ``fchown``, ``fchownat``, ``fcntl``,
     ``fdatasync``, ``fexecve``, ``ffs``, ``fork``, ``fstat``, ``fstatat``,
     ``fsync``, ``ftruncate``, ``futimens``, ``getegid``, ``geteuid``,
     ``getgid``, ``getgroups``, ``getpeername``, ``getpgrp``, ``getpid``,
     ``getppid``, ``getsockname``, ``getsockopt``, ``getuid``, ``htonl``,
     ``htons``, ``kill``, ``link``, ``linkat``, ``listen``, ``longjmp``,
     ``lseek``, ``lstat``, ``memccpy``, ``memchr``, ``memcmp``, ``memcpy``,
     ``memmove``, ``memset``, ``mkdir``, ``mkdirat``, ``mkfifo``, ``mkfifoat``,
     ``mknod``, ``mknodat``, ``ntohl``, ``ntohs``, ``open``, ``openat``,
     ``pause``, ``pipe``, ``poll``, ``posix_trace_event``, ``pselect``,
     ``pthread_kill``, ``pthread_self``, ``pthread_sigmask``, ``quick_exit``,
     ``raise``, ``read``, ``readlink``, ``readlinkat``, ``recv``, ``recvfrom``,
     ``recvmsg``, ``rename``, ``renameat``, ``rmdir``, ``select``, ``sem_post``,
     ``send``, ``sendmsg``, ``sendto``, ``setgid``, ``setpgid``, ``setsid``,
     ``setsockopt``, ``setuid``, ``shutdown``, ``sigaction``, ``sigaddset``,
     ``sigdelset``, ``sigemptyset``, ``sigfillset``, ``sigismember``,
     ``siglongjmp``, ``signal``, ``sigpause``, ``sigpending``, ``sigprocmask``,
     ``sigqueue``, ``sigset``, ``sigsuspend``, ``sleep``, ``sockatmark``,
     ``socket``, ``socketpair``, ``stat``, ``stpcpy``, ``stpncpy``,
     ``strcat``, ``strchr``, ``strcmp``, ``strcpy``, ``strcspn``, ``strlen``,
     ``strncat``, ``strncmp``, ``strncpy``, ``strnlen``, ``strpbrk``,
     ``strrchr``, ``strspn``, ``strstr``, ``strtok_r``, ``symlink``,
     ``symlinkat``, ``tcdrain``, ``tcflow``, ``tcflush``, ``tcgetattr``,
     ``tcgetpgrp``, ``tcsendbreak``, ``tcsetattr``, ``tcsetpgrp``,
     ``time``, ``timer_getoverrun``, ``timer_gettime``, ``timer_settime``,
     ``times``, ``umask``, ``uname``, ``unlink``, ``unlinkat``, ``utime``,
     ``utimensat``, ``utimes``, ``wait``, ``waitpid``, ``wcpcpy``,
     ``wcpncpy``, ``wcscat``, ``wcschr``, ``wcscmp``, ``wcscpy``, ``wcscspn``,
     ``wcslen``, ``wcsncat``, ``wcsncmp``, ``wcsncpy``, ``wcsnlen``, ``wcspbrk``,
     ``wcsrchr``, ``wcsspn``, ``wcsstr``, ``wcstok``, ``wmemchr``, ``wmemcmp``,
     ``wmemcpy``, ``wmemmove``, ``wmemset``, ``write``

     The function ``quick_exit`` is not included in the POSIX list but it
     is included here in the set of safe functions.

  The default value is ``POSIX``.
