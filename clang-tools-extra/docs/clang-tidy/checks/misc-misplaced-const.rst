.. title:: clang-tidy - misc-misplaced-const

misc-misplaced-const
====================

This check diagnoses when a ``const`` qualifier is applied to a ``typedef`` to a
pointer type rather than to the pointee, because such constructs are often
misleading to developers because the ``const`` applies to the pointer rather
than the pointee.

For instance, in the following code, the resulting type is ``int *`` ``const``
rather than ``const int *``:

.. code-block:: c++

  typedef int *int_ptr;
  void f(const int_ptr ptr);

The check does not diagnose when the underlying ``typedef`` type is a pointer to
a ``const`` type or a function pointer type. This is because the ``const``
qualifier is less likely to be mistaken because it would be redundant (or
disallowed) on the underlying pointee type.
