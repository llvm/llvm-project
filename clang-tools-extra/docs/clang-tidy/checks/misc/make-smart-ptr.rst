.. title:: clang-tidy - misc-make-smart-ptr

misc-make-smart-ptr
===================

Finds constructions of custom smart pointer types from raw ``new`` expressions
and replaces them with a configurable factory function.

Unlike :doc:`modernize-make-shared <../modernize/make-shared>` and
:doc:`modernize-make-unique <../modernize/make-unique>`, this check has no
default smart pointer type or factory function. Both
:option:`MakeSmartPtrType` and :option:`MakeSmartPtrFunction` must be
configured for the check to produce
diagnostics.

This allows using the ``modernize-make-shared`` and ``modernize-make-unique``
checks with their default ``std::shared_ptr``/``std::unique_ptr`` types, while
also matching additional custom smart pointer types through this check.

.. code-block:: c++

  // Given MakeSmartPtrType = '::base::scoped_refptr'
  //       MakeSmartPtrFunction = 'base::MakeRefCounted'
  base::scoped_refptr<Foo> ptr(new Foo(1, 2));

  // becomes

  auto ptr = base::MakeRefCounted<Foo>(1, 2);

Options
-------

.. option:: MakeSmartPtrFunction

   A string specifying the name of the factory function. This option must be
   set for the check to work. Default is empty string.

.. option:: MakeSmartPtrFunctionHeader

   A string specifying the corresponding header of the factory function.
   Default is empty string.

.. option:: MakeSmartPtrType

   A string specifying the smart pointer type to match. This option must be
   set for the check to work. Default is empty string.

.. option:: IncludeStyle

   A string specifying which include-style is used, `llvm` or `google`. Default
   is `llvm`. This option can also be set globally via the
   ``IncludeStyle`` global option.

.. option:: IgnoreMacros

   If set to `true`, the check will not give warnings inside macros. Default
   is `true`.

.. option:: IgnoreDefaultInitialization

   If set to `false`, the check does not suggest edits that will transform
   default initialization into value initialization, as this can cause
   performance regressions. Default is `true`.
