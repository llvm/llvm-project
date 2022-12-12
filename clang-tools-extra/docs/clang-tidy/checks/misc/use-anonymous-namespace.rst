.. title:: clang-tidy - misc-use-anonymous-namespace

misc-use-anonymous-namespace
============================

Finds instances of ``static`` functions or variables declared at global scope
that could instead be moved into an anonymous namespace.

Anonymous namespaces are the "superior alternative" according to the C++
Standard. ``static`` was proposed for deprecation, but later un-deprecated to
keep C compatibility [1]. ``static`` is an overloaded term with different meanings in
different contexts, so it can create confusion.

The following uses of ``static`` will *not* be diagnosed:

* Functions or variables in header files, since anonymous namespaces in headers
  is considered an antipattern. Allowed header file extensions can be configured
  via the `HeaderFileExtensions` option (see below).
* ``const`` or ``constexpr`` variables, since they already have implicit internal
  linkage in C++.

Examples:

.. code-block:: c++

  // Bad
  static void foo();
  static int x;

  // Good
  namespace {
    void foo();
    int x;
  } // namespace

Options
-------

.. option:: HeaderFileExtensions

   A semicolon-separated list of filename extensions of header files (the filename
   extensions should not include "." prefix). Default is ";h;hh;hpp;hxx".
   For extension-less header files, using an empty string or leaving an
   empty string between ";" if there are other filename extensions.

[1] `Undeprecating static <https://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#1012>`_
