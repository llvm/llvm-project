.. title:: clang-tidy - modernize-use-init-statement

modernize-use-init-statement
============================

Finds variable declarations immediately before ``if`` or ``switch``
statements where the variable is only used inside the conditional,
and suggests moving the declaration into the C++17 init-statement.

This reduces the scope of the variable to the minimum necessary,
improving readability and reducing the risk of accidental reuse.

For example:

.. code-block:: c++

  auto it = m.find(key);
  if (it != m.end()) {
    use(it);
  }

  // transforms to:

  if (auto it = m.find(key); it != m.end()) {
    use(it);
  }

Similarly for ``switch`` statements:

.. code-block:: c++

  Color c = getColor();
  switch (c) {
  case Red: break;
  case Green: break;
  }

  // transforms to:

  switch (Color c = getColor(); c) {
  case Red: break;
  case Green: break;
  }

The check only triggers when all of the following are true:

- The variable declaration immediately precedes the ``if``/``switch``.
- The declaration has a single variable with an initializer.
- The variable is referenced in the condition.
- The variable is not referenced after the ``if``/``switch``.
- The ``if``/``switch`` does not already have an init-statement.
- Neither the declaration nor the conditional is in a macro.
- The variable does not have static or extern storage.
