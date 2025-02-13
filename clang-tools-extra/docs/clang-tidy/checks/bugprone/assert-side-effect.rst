.. title:: clang-tidy - bugprone-assert-side-effect

bugprone-assert-side-effect
===========================

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

.. option:: IgnoredFunctions

   A semicolon-separated list of the names of functions or methods to be
   considered as not having side-effects. Regular expressions are accepted,
   e.g. ``[Rr]ef(erence)?$`` matches every type with suffix ``Ref``, ``ref``,
   ``Reference`` and ``reference``. The default is empty. If a name in the list
   contains the sequence `::` it is matched against the qualified type name
   (i.e. ``namespace::Type``), otherwise it is matched against only
   the type name (i.e. ``Type``).
