.. title:: clang-tidy - misc-move-constructor-init

misc-move-constructor-init
==========================

"cert-oop11-cpp" redirects here as an alias for this check.

The check flags user-defined move constructors that have a ctor-initializer
initializing a member or base class through a copy constructor instead of a
move constructor.

Options
-------

.. option:: IncludeStyle

   A string specifying which include-style is used, `llvm` or `google`. Default
   is `llvm`.
