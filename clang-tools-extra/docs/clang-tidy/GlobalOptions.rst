==============
Global Options
==============

Some options apply to multiple checks. This page lists all the available
globally options.

.. option:: IncludeStyle

   A string specifying which include-style is used, `llvm` or `google`. Default
   is `llvm`.

.. option:: IgnoreMacros

   If set to `true`, the check will not give warnings inside macros. Default
   is `true`.

.. option:: StrictMode

  When `true`, some checkers will be more stringent. The default value depends
  on the checks.
