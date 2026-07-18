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
only adding ``const``, are left untouched.
