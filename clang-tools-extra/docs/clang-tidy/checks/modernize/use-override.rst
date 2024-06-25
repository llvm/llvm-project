.. title:: clang-tidy - modernize-use-override

modernize-use-override
======================

Adds ``override`` (introduced in C++11) to overridden virtual functions and
removes ``virtual`` from those functions as it is not required.

``virtual`` on non base class implementations was used to help indicate to the
user that a function was virtual. C++ compilers did not use the presence of
this to signify an overridden function.

In C++11 ``override`` and ``final`` keywords were introduced to allow
overridden functions to be marked appropriately. Their presence allows
compilers to verify that an overridden function correctly overrides a base
class implementation.

This can be useful as compilers can generate a compile time error when:

 - The base class implementation function signature changes.
 - The user has not created the override with the correct signature.

Options
-------

.. option:: IgnoreDestructors

   If set to `true`, this check will not diagnose destructors. Default is `false`.

.. option:: IgnoreTemplateInstantiations

   If set to `true`, instructs this check to ignore virtual function overrides
   that are part of template instantiations. Default is `false`.

.. option:: AllowOverrideAndFinal

   If set to `true`, this check will not diagnose ``override`` as redundant
   with ``final``. This is useful when code will be compiled by a compiler with
   warning/error checking flags requiring ``override`` explicitly on overridden
   members, such as ``gcc -Wsuggest-override``/``gcc -Werror=suggest-override``.
   Default is `false`.

.. option:: OverrideSpelling

   Specifies a macro to use instead of ``override``. This is useful when
   maintaining source code that also needs to compile with a pre-C++11
   compiler.

.. option:: FinalSpelling

   Specifies a macro to use instead of ``final``. This is useful when
   maintaining source code that also needs to compile with a pre-C++11
   compiler.

.. note::

   For more information on the use of ``override`` see https://en.cppreference.com/w/cpp/language/override
