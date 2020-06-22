.. title:: clang-tidy - modernize-replace-disallow-copy-and-assign-macro

modernize-replace-disallow-copy-and-assign-macro
================================================

Finds macro expansions of ``DISALLOW_COPY_AND_ASSIGN(Type)`` and replaces them
with a deleted copy constructor and a deleted assignment operator.

Before the ``delete`` keyword was introduced in C++11 it was common practice to
declare a copy constructor and an assignment operator as a private members. This
effectively makes them unusable to the public API of a class.

With the advent of the ``delete`` keyword in C++11 we can abandon the
``private`` access of the copy constructor and the assignment operator and
delete the methods entirely.

When running this check on a code like this:

.. code-block:: c++

  class Foo {
  private:
    DISALLOW_COPY_AND_ASSIGN(Foo);
  };

It will be transformed to this:

.. code-block:: c++

  class Foo {
  private:
    Foo(const Foo &) = delete;
    const Foo &operator=(const Foo &) = delete;
  };

Known Limitations
-----------------

* Notice that the migration example above leaves the ``private`` access
  specification untouched. You might want to run the check:doc:`modernize-use-equals-delete
  <modernize-use-equals-delete>` to get warnings for deleted functions in
  private sections.

Options
-------

.. option:: MacroName

   A string specifying the macro name whose expansion will be replaced.
   Default is `DISALLOW_COPY_AND_ASSIGN`.

See: https://en.cppreference.com/w/cpp/language/function#Deleted_functions
