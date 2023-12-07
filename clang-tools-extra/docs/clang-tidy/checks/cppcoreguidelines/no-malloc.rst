.. title:: clang-tidy - cppcoreguidelines-no-malloc

cppcoreguidelines-no-malloc
===========================

This check handles C-Style memory management using ``malloc()``, ``realloc()``,
``calloc()`` and ``free()``. It warns about its use and tries to suggest the use
of an appropriate RAII object.
Furthermore, it can be configured to check against a user-specified list of functions
that are used for memory management (e.g. ``posix_memalign()``).

This check implements `R.10
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rr-mallocfree>`_
from the C++ Core Guidelines.

There is no attempt made to provide fix-it hints, since manual resource
management isn't easily transformed automatically into RAII.

.. code-block:: c++

  // Warns each of the following lines.
  // Containers like std::vector or std::string should be used.
  char* some_string = (char*) malloc(sizeof(char) * 20);
  char* some_string = (char*) realloc(sizeof(char) * 30);
  free(some_string);

  int* int_array = (int*) calloc(30, sizeof(int));

  // Rather use a smartpointer or stack variable.
  struct some_struct* s = (struct some_struct*) malloc(sizeof(struct some_struct));

Options
-------

.. option:: Allocations

   Semicolon-separated list of fully qualified names of memory allocation functions.
   Defaults to ``::malloc;::calloc``.

.. option:: Deallocations

   Semicolon-separated list of fully qualified names of memory allocation functions.
   Defaults to ``::free``.

.. option:: Reallocations

   Semicolon-separated list of fully qualified names of memory allocation functions.
   Defaults to ``::realloc``.
