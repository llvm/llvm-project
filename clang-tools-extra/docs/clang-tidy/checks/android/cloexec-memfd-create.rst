.. title:: clang-tidy - android-cloexec-memfd-create

android-cloexec-memfd-create
============================

``memfd_create()`` should include ``MFD_CLOEXEC`` in its type argument to avoid
the file descriptor leakage. Without this flag, an opened sensitive file would
remain open across a fork+exec to a lower-privileged SELinux domain.

Examples:

.. code-block:: c++

  memfd_create(name, MFD_ALLOW_SEALING);

  // becomes

  memfd_create(name, MFD_ALLOW_SEALING | MFD_CLOEXEC);
