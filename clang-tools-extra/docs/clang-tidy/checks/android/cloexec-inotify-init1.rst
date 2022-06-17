.. title:: clang-tidy - android-cloexec-inotify-init1

android-cloexec-inotify-init1
=============================

``inotify_init1()`` should include ``IN_CLOEXEC`` in its type argument to avoid the
file descriptor leakage. Without this flag, an opened sensitive file would
remain open across a fork+exec to a lower-privileged SELinux domain.

Examples:

.. code-block:: c++

  inotify_init1(IN_NONBLOCK);

  // becomes

  inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
