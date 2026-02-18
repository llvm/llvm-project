.. title:: clang-tidy - modernize-use-size-type

modernize-use-size-type
=======================

Finds local variables declared as signed integer types that are
initialized from an unsigned/``size_t`` source (e.g.
``container.size()``) and only used in contexts expecting unsigned
types, and suggests changing the type to ``size_t``.

Storing a ``size_t`` value in a signed ``int`` can cause implicit
narrowing conversions and sign-comparison warnings. Using ``size_t``
directly avoids these issues.

For example:

.. code-block:: c++

  std::vector<int> v;
  int n = v.size();
  v.resize(n);

  // transforms to:

  std::vector<int> v;
  size_t n = v.size();
  v.resize(n);

The check only triggers when all of the following are true:

- The variable is a local, non-static variable.
- The variable has a signed integer type (e.g. ``int``).
- The initializer is an unsigned integer expression.
- Every use of the variable is in an unsigned-compatible context (comparison,
  function argument expecting unsigned, array subscript, or implicit cast to
  unsigned).
- The variable is not ``constexpr``.
- The declaration is not in a macro.
