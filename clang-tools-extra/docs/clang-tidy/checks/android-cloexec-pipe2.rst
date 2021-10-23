.. title:: clang-tidy - android-cloexec-pipe2

android-cloexec-pipe2
=====================

This check ensures that pipe2() is called with the O_CLOEXEC flag. The check also
adds the O_CLOEXEC flag that marks the file descriptor to be closed in child processes.
Without this flag a sensitive file descriptor can be leaked to a child process,
potentially into a lower-privileged SELinux domain.

Examples:

.. code-block:: c++

  pipe2(pipefd, O_NONBLOCK);

Suggested replacement:

.. code-block:: c++

  pipe2(pipefd, O_NONBLOCK | O_CLOEXEC);
