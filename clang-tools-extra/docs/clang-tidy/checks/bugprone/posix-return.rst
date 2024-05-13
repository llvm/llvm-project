.. title:: clang-tidy - bugprone-posix-return

bugprone-posix-return
=====================

Checks if any calls to ``pthread_*`` or ``posix_*`` functions
(except ``posix_openpt``) expect negative return values. These functions return
either ``0`` on success or an ``errno`` on failure, which is positive only.

Example buggy usage looks like:

.. code-block:: c

  if (posix_fadvise(...) < 0) {

This will never happen as the return value is always non-negative. A simple fix could be:

.. code-block:: c

  if (posix_fadvise(...) > 0) {
