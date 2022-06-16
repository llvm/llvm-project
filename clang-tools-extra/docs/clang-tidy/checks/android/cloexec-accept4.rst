.. title:: clang-tidy - android-cloexec-accept4

android-cloexec-accept4
=======================

``accept4()`` should include ``SOCK_CLOEXEC`` in its type argument to avoid the
file descriptor leakage. Without this flag, an opened sensitive file would
remain open across a fork+exec to a lower-privileged SELinux domain.

Examples:

.. code-block:: c++

  accept4(sockfd, addr, addrlen, SOCK_NONBLOCK);

  // becomes

  accept4(sockfd, addr, addrlen, SOCK_NONBLOCK | SOCK_CLOEXEC);
