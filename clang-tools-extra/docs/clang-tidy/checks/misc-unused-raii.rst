.. title:: clang-tidy - misc-unused-raii

misc-unused-raii
================


Finds temporaries that look like RAII objects.

The canonical example for this is a scoped lock.

.. code:: c++

  {
    scoped_lock(&global_mutex);
    critical_section();
  }

The destructor of the scoped_lock is called before the ``critical_section`` is
entered, leaving it unprotected.

We apply a number of heuristics to reduce the false positive count of this
check:

  * Ignore code expanded from macros. Testing frameworks make heavy use of this.
  * Ignore types with trivial destructors. They are very unlikely to be RAII
    objects and there's no difference when they are deleted.
  * Ignore objects at the end of a compound statement (doesn't change behavior).
  * Ignore objects returned from a call.
