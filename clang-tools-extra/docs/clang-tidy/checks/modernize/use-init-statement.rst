.. title:: clang-tidy - modernize-use-init-statement

modernize-use-init-statement
============================

Finds variable declarations that immediately precede ``if`` or ``switch``
statements where the variable isn't used after the statement, and suggests
moving them into C++17 init statements.

This check helps to adopt the C++17 feature that allows variable initialization
within ``if`` and ``switch`` statements, narrowing variable scope and improving
code readability.

Examples
--------

.. code-block:: c++

  // Variable declaration before if - will be detected
  auto Pos = Str.find(Substr);
  if (Pos != std::string::npos) {
      // `Pos` is only used in condition, not in body
  }

  // Transforms to:
  if (auto Pos = Str.find(Substr); Pos != std::string::npos) {
  }

  // Variable declaration before switch - will be detected  
  auto Value = getValue();
  switch (Value.type) {
      case Type::A: break;
  }

  // Transforms to:
  switch (auto Value = getValue(); Value.type) {
      case Type::A: break;
  }

Options
-------

.. option:: IgnoreConditionVariableStatements

    When ``true``, the check will not suggest transformations for variable
    declarations that precede ``if`` or ``switch`` statements that already have
    a condition variable statement. When ``false`` (default), the check may
    suggest moving a variable declaration into an init statement even when the
    statement already has a condition variable statement.

    For example, with ``IgnoreConditionVariableStatements`` set to ``true``,
    the check will not suggest transforming:

    .. code-block:: c++

      auto x = getX();
      if (auto y = getY()) {
          // ...
      }

Limitations
-----------

* The check supports exclusively builtin types for both non-reference,
  non-pointer variable declarations and lifetime-extended references.

* The check may provide false-negative if you have template code.

Requirements
------------

Requires C++17 or later.
