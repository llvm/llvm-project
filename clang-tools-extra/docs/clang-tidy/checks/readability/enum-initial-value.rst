.. title:: clang-tidy - readability-enum-initial-value

readability-enum-initial-value
==============================

Enforces consistent style for enumerators' initialization, covering three
styles: none, first only, or all initialized explicitly.

When adding new enumerations, inconsistent initial value will cause potential
enumeration value conflicts.

In an enumeration, the following three cases are accepted. 
1. none of enumerators are explicit initialized.
2. the first enumerator is explicit initialized.
3. all of enumerators are explicit initialized.

.. code-block:: c++

  // valid, none of enumerators are initialized.
  enum A {
    e0,
    e1,
    e2,
  };

  // valid, the first enumerator is initialized.
  enum A {
    e0 = 0,
    e1,
    e2,
  };

  // valid, all of enumerators are initialized.
  enum A {
    e0 = 0,
    e1 = 1,
    e2 = 2,
  };

  // invalid, e1 is not explicit initialized.
  enum A {
    e0 = 0,
    e1,
    e2 = 2,
  };

Options
-------

.. option:: AllowExplicitZeroFirstInitialValue

  If set to `false`, the first enumerator must not be explicitly initialized.
  See examples below. Default is `true`.

  .. code-block:: c++

    enum A {
      e0 = 0, // not allowed if AllowExplicitZeroFirstInitialValue is false
      e1,
      e2,
    };


.. option:: AllowExplicitSequentialInitialValues

  If set to `false`, sequential initializations are not allowed.
  See examples below. Default is `true`.

  .. code-block:: c++

    enum A {
      e0 = 1, // not allowed if AllowExplicitSequentialInitialValues is false
      e1 = 2,
      e2 = 3,
    };
