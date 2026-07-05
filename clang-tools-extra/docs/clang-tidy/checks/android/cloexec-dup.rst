.. title:: clang-tidy - android-cloexec-dup

android-cloexec-dup
===================

The usage of ``dup()`` is not recommended, it's better to use ``fcntl()``,
which can set the close-on-exec flag. Otherwise, an opened sensitive file would
remain open across a fork+exec to a lower-privileged SELinux domain.

Examples:

.. code-block:: c++

  int fd = dup(oldfd);

  // becomes

  int fd = fcntl(oldfd, F_DUPFD_CLOEXEC);
