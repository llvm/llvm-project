.. title:: clang-tidy - modernize-use-equals-delete

modernize-use-equals-delete
===========================

This check marks unimplemented private special member functions with ``= delete``.
To avoid false-positives, this check only applies in a translation unit that has
all other member functions implemented.

.. code-block:: c++

  struct A {
  private:
    A(const A&);
    A& operator=(const A&);
  };

  // becomes

  struct A {
  private:
    A(const A&) = delete;
    A& operator=(const A&) = delete;
  };


.. option:: IgnoreMacros

   If this option is set to `true` (default is `true`), the check will not warn
   about functions declared inside macros.
