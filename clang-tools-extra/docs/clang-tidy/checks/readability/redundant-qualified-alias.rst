.. title:: clang-tidy - readability-redundant-qualified-alias

readability-redundant-qualified-alias
=====================================

Finds redundant identity type aliases that re-expose a qualified name and can
be replaced with a ``using`` declaration.

.. code-block:: c++

  using seconds = std::chrono::seconds;

  // becomes

  using std::chrono::seconds;

The check is conservative and only warns when the alias name exactly matches
the unqualified name of a non-dependent, non-specialized named type written
with a qualifier. It skips alias templates, dependent forms, elaborated
keywords (``class``, ``struct``, ``enum``, ``typename``), and cases involving
macros.

Options
-------

.. option:: OnlyNamespaceScope

   When `true`, only consider aliases declared in a namespace or the
   translation unit. When `false`, also consider aliases declared inside
   classes, functions, and lambdas. Default is `false`.
