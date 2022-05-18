.. title:: clang-tidy - bugprone-dynamic-static-initializers

bugprone-dynamic-static-initializers
====================================

Finds instances of static variables that are dynamically initialized
in header files.

This can pose problems in certain multithreaded contexts. For example,
when disabling compiler generated synchronization instructions for
static variables initialized at runtime (e.g. by ``-fno-threadsafe-statics``), even if a particular project
takes the necessary precautions to prevent race conditions during
initialization by providing their own synchronization, header files included from other projects may
not. Therefore, such a check is helpful for ensuring that disabling
compiler generated synchronization for static variable initialization will not cause
problems.

Consider the following code:

.. code-block:: c

  int foo() {
    static int k = bar();
    return k;
  }

When synchronization of static initialization is disabled, if two threads both call `foo` for the first time, there is the possibility that `k` will be double initialized, creating a race condition.
