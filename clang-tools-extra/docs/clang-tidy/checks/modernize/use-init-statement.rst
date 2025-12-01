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

.. option:: StrictMode

    When ``true`` (default), the check will suggest transformations even in
    cases where moving the variable into the init statement doesn't reduce its
    scope. This includes situations where the variable is already in an inner
    scope (like inside a loop body or nested block).
    When ``false``, the check only suggests transformations when scope
    reduction would occur.

    In strict mode, the check prioritizes consistent use of C++17 init
    statements for readability, even when scope isn't affected.

Requirements
------------

Requires C++17 or later.
