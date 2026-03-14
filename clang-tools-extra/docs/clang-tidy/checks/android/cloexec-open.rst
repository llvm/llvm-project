.. title:: clang-tidy - android-cloexec-open

android-cloexec-open
====================

A common source of security bugs is code that opens a file without using the
``O_CLOEXEC`` flag. Without that flag, an opened sensitive file would remain
open across a fork+exec to a lower-privileged SELinux domain, leaking that
sensitive data. Open-like functions including ``open()``, ``openat()``, and
``open64()`` should include ``O_CLOEXEC`` in their flags argument.

Examples:

.. code-block:: c++

  open("filename", O_RDWR);
  open64("filename", O_RDWR);
  openat(0, "filename", O_RDWR);

  // becomes

  open("filename", O_RDWR | O_CLOEXEC);
  open64("filename", O_RDWR | O_CLOEXEC);
  openat(0, "filename", O_RDWR | O_CLOEXEC);
