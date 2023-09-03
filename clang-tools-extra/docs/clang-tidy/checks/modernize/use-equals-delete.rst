.. title:: clang-tidy - modernize-use-equals-delete

modernize-use-equals-delete
===========================

Prior to C++11, the only way to "delete" a given function was to make it
``private`` and without definition, to generate a compiler error (calling
private function) or a linker error (undefined reference).

After C++11, the more idiomatic way to achieve this is by marking the functions
as ``= delete``, and keeping them in the ``public`` section.

This check warns only on unimplemented private **special member functions**.
To avoid false-positives, this check only applies in a translation unit that has
all other member functions implemented. The check will generate partial fixes
by adding ``= delete``, but the move the ``public`` section needs to be done
manually.

.. code-block:: c++

  // Example: bad
  class A {
   private:
    A(const A&);
    A& operator=(const A&);
  };

  // Example: good
  class A {
   public:
    A(const A&) = delete;
    A& operator=(const A&) = delete;
  };


.. option:: IgnoreMacros

   If this option is set to `true` (default is `true`), the check will not warn
   about functions declared inside macros.
