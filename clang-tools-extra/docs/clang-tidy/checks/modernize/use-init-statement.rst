.. title:: clang-tidy - modernize-use-init-statement

modernize-use-init-statement
============================

Finds variable declarations that immediately precede ``if`` or ``switch`` statements where the variable isn't used after the statement, and suggests moving them into C++17 init statements.

This check helps to adopt the C++17 feature that allows variable initialization within ``if`` and ``switch`` statements, narrowing variable scope and improving code readability.

Examples
--------

.. code-block:: c++

  // Variable declaration before if - will be detected
  auto It = Map.find(Key);
  if (It != Map.end()) {
      // It is only used in condition, not in body
  }

  // Transforms to:
  if (auto It = Map.find(Key); It != Map.end()) {
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

None.

Requirements
------------

Requires C++17 or later.
