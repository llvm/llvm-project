.. title:: clang-tidy - modernize-use-equals-delete

modernize-use-equals-delete
===========================

Identifies unimplemented private special member functions, and recommends using
``= delete`` for them. Additionally, it recommends relocating any deleted
member function from the ``private`` to the ``public`` section.

Before the introduction of C++11, the primary method to effectively "erase" a
particular function involved declaring it as ``private`` without providing a
definition. This approach would result in either a compiler error (when
attempting to call a private function) or a linker error (due to an undefined
reference).

However, subsequent to the advent of C++11, a more conventional approach emerged
for achieving this purpose. It involves flagging functions as ``= delete`` and
keeping them in the ``public`` section of the class.

To prevent false positives, this check is only active within a translation
unit where all other member functions have been implemented. The check will
generate partial fixes by introducing ``= delete``, but the user is responsible
for manually relocating functions to the ``public`` section.

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
