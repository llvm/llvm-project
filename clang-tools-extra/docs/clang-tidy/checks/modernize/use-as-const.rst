.. title:: clang-tidy - modernize-use-as-const

modernize-use-as-const
======================

Replaces a ``static_cast`` that only adds ``const`` to an lvalue with a call to
``std::as_const`` (available since C++17), which states the intent more clearly
and cannot accidentally change the referenced type.

.. code-block:: c++

  void use(const std::string &);

  void f(std::string s) {
    use(static_cast<const std::string &>(s));
  }

becomes

.. code-block:: c++

  void use(const std::string &);

  void f(std::string s) {
    use(std::as_const(s));
  }

Casts of an already ``const`` operand, and casts that change the type rather than
only adding ``const``, are left untouched. A cast whose operand is an rvalue is
also left untouched, because ``std::as_const`` is deleted for rvalues.

Options
-------

.. option:: IgnoreMacros

   If set to `true`, the check will not give warnings inside macros. Default
   is `true`. A cast inside a macro is reported but not fixed, since the
   replacement would rewrite the macro body for every expansion.

.. option:: IncludeStyle

   A string specifying which include-style is used, `llvm` or `google`. Default
   is `llvm`.
