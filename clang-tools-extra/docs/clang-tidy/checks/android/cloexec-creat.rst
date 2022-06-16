.. title:: clang-tidy - android-cloexec-creat

android-cloexec-creat
=====================

The usage of ``creat()`` is not recommended, it's better to use ``open()``.

Examples:

.. code-block:: c++

  int fd = creat(path, mode);

  // becomes

  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, mode);
