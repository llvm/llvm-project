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

.. option:: SafeDestructorTypes

    A comma-separated list of type name patterns (glob patterns) that are
    considered safe to move into init statements even if they have destructors.
    By default, the check never transforms variables with destructors when they
    are not the last statement in a compound statement, as moving them could
    change the destruction order. However, some types like  ``std::string`` are
    safe to move because their destructors are well-behaved and don't have side
    effects that depend on destruction order.

    The default value is ``-*,::std::*string,::std::*string_view,
    ::boost::*string,::boost::*string_view,::boost::*string_ref``.

    The glob pattern supports the ``*`` wildcard character to match zero or
    more characters. Patterns can be excluded by prefixing them with ``-``.

Limitations
-----------

* The check does not guarantee correct fixes for template code.

* The ``SafeDestructorTypes`` option does not include ``std::vector`` and
  other standard containers by default, because while the container itself
  has a safe destructor, it may contain elements with unsafe destructors.

  For example, ``std::vector<std::unique_lock<std::mutex>>`` has an unsafe
  destructor because the ``std::unique_lock`` elements perform mutex
  unlocking during destruction.

  For this reason, it is not recommended to manually add ``std::vector`` or
  other container types to the ``SafeDestructorTypes`` list, as the safety
  depends on the contained element types rather than the container itself.

Requirements
------------

Requires C++17 or later.
