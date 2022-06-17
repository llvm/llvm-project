.. title:: clang-tidy - concurrency-mt-unsafe

concurrency-mt-unsafe
=====================

Checks for some thread-unsafe functions against a black list of
known-to-be-unsafe functions. Usually they access static variables without
synchronization (e.g. gmtime(3)) or utilize signals in a racy way.
The set of functions to check is specified with the `FunctionSet` option.

Note that using some thread-unsafe functions may be still valid in
concurrent programming if only a single thread is used (e.g. setenv(3)),
however, some functions may track a state in global variables which
would be clobbered by subsequent (non-parallel, but concurrent) calls to
a related function. E.g. the following code suffers from unprotected
accesses to a global state:

.. code-block:: c++

    // getnetent(3) maintains global state with DB connection, etc.
    // If a concurrent green thread calls getnetent(3), the global state is corrupted.
    netent = getnetent();
    yield();
    netent = getnetent();


Examples:

.. code-block:: c++

    tm = gmtime(timep); // uses a global buffer

    sleep(1); // implementation may use SIGALRM

.. option:: FunctionSet

  Specifies which functions in libc should be considered thread-safe,
  possible values are `posix`, `glibc`, or `any`.

  `posix` means POSIX defined thread-unsafe functions. POSIX.1-2001
  in "2.9.1 Thread-Safety" defines that all functions specified in the
  standard are thread-safe except a predefined list of thread-unsafe
  functions.

  Glibc defines some of them as thread-safe (e.g. dirname(3)), but adds
  non-POSIX thread-unsafe ones (e.g. getopt_long(3)). Glibc's list is
  compiled from GNU web documentation with a search for MT-Safe tag:
  https://www.gnu.org/software/libc/manual/html_node/POSIX-Safety-Concepts.html

  If you want to identify thread-unsafe API for at least one libc or
  unsure which libc will be used, use `any` (default).

