.. title:: clang-tidy - misc-assert-side-effect

misc-assert-side-effect
=======================

Finds ``assert()`` with side effect.

The condition of ``assert()`` is evaluated only in debug builds so a
condition with side effect can cause different behavior in debug / release
builds.

Options
-------

.. option:: AssertMacros

   A comma-separated list of the names of assert macros to be checked.

.. option:: CheckFunctionCalls

   Whether to treat non-const member and non-member functions as they produce
   side effects. Disabled by default because it can increase the number of false
   positive warnings.
