.. title:: clang-tidy - readability-redundant-nested-if

readability-redundant-nested-if
===============================

Finds nested ``if`` statements that can be merged by combining their
conditions with ``&&``.

Example:

.. code-block:: c++

  if (a) {
    if (b) {
      work();
    }
  }

becomes

.. code-block:: c++

  if ((a) && (b)) {
    work();
  }

The check can merge longer chains as well:

.. code-block:: c++

  if (a) {
    if (b) {
      if (c) {
        work();
      }
    }
  }

becomes

.. code-block:: c++

  if ((a) && (b) && (c)) {
    work();
  }

The check also supports outer declaration conditions in C++17 and later:

.. code-block:: c++

  if (bool X = ready()) {
    if (can_run()) {
      work();
    }
  }

becomes

.. code-block:: c++

  if (bool X = ready(); X && (can_run())) {
    work();
  }

Safety rules
------------

The check only transforms chains where:

- Neither outer nor nested ``if`` has an ``else`` branch.

- Nested merged ``if`` statements do not use condition variables.

- In C++17 and later, the outermost ``if`` may use a condition variable if it
  can be rewritten to an init-statement form, for example
  ``if (auto V = f())`` to ``if (auto V = f(); V && ...)``.

- When the outermost statement is already in ``if (init; cond)`` form, the
  check keeps ``init`` unchanged and merges only into ``cond``.

- By default, merged conditions avoid user-defined ``bool`` conversions to
  preserve built-in ``&&`` semantics. This can be changed with
  :option:`AllowUserDefinedBoolConversion`.

- Only the outermost ``if`` may have an init-statement.

- No merged ``if`` is ``if consteval``.

- All merged ``if`` statements are either all ``if constexpr`` or all regular
  ``if``.

- No merged ``if`` statement has statement attributes.

- All rewritten ranges are free of macro- and preprocessor-sensitive edits.

- Fix-its are suppressed when comments in removed nested headers cannot be
  preserved safely. Comments inside conditions are preserved, while other
  comments between the ``if`` statements disable fix-its.

For ``if constexpr``, an instantiation-dependent condition is merged only when
all preceding merged conditions are non-dependent constant ``true`` expressions.
This preserves discarded-branch behavior for nested template-dependent
``if constexpr`` statements.

Options
-------

.. option:: AllowUserDefinedBoolConversion

   When set to `true`, the check also merges chains whose conditions rely on
   user-defined conversion to ``bool``.

   The fix-it inserts ``static_cast<bool>(...)`` where needed so the merged
   condition still uses built-in ``&&`` semantics.

   Default is `false`.
