.. title:: clang-tidy - misc-move-constructor-init

misc-move-constructor-init
==========================

"cert-oop11-cpp" redirects here as an alias for this check.

The check flags user-defined move constructors that have a ctor-initializer
initializing a member or base class through a copy constructor instead of a
move constructor.

It also flags constructor arguments that are passed by value, have a non-deleted
move-constructor and are assigned to a class field by copy construction.

Options
-------

.. option:: IncludeStyle

   A string specifying which include-style is used, `llvm` or `google`. Default
   is `llvm`.

.. option:: UseCERTSemantics

   When non-zero, the check conforms to the behavior expected by the CERT secure
   coding recommendation
   `OOP11-CPP <https://www.securecoding.cert.org/confluence/display/cplusplus/OOP11-CPP.+Do+not+copy-initialize+members+or+base+classes+from+a+move+constructor>`_.
   Default is `0` for misc-move-constructor-init and `1` for cert-oop11-cpp.
