.. title:: clang-tidy - readability-enum-initial-value

readability-enum-initial-value
==============================

Enforces consistent style for enumerators' initialization, covering three
styles: none, first only, or all initialized explicitly.

An inconsistent style and strictness to defining the initializing value of
enumerators may cause issues if the enumeration is extended with new
enumerators that obtain their integer representation implicitly.

The following three cases are accepted:

#. **No** enumerators are explicit initialized.
#. Exactly **the first** enumerator is explicit initialized.
#. **All** enumerators are explicit initialized.

.. code-block:: c++

  enum A {    // (1) Valid, none of enumerators are initialized.
    a0,
    a1,
    a2,
  };

  enum B {    // (2) Valid, the first enumerator is initialized.
    b0 = 0,
    b1,
    b2,
  };

  enum C {    // (3) Valid, all of enumerators are initialized.
    c0 = 0,
    c1 = 1,
    c2 = 2,
  };

  enum D {    // warning: initial values in enum 'D' are not consistent,
              //          consider explicit initialization of all, none or only
              //          the first enumerator
    d0 = 0,
    d1,       // note: uninitialized enumerator 'd1' defined here
    d2 = 2,
  };

  enum E {    // warning: initial values in enum 'E' are not consistent,
              //          consider explicit initialization of all, none or only
              //          the first enumerator
    e0 = 0,
    e1,       // note: uninitialized enumerator 'e1' defined here
    e2 = 2,
    e3,       // note: uninitialized enumerator 'e3' defined here
              // Dangerous, as the numeric values of e3 and e5 are both 3,
              // and this is not explicitly visible in the code!
    e4 = 2,
    e5,       // note: uninitialized enumerator 'e5' defined here
  };

This check corresponds to the CERT C Coding Standard recommendation `INT09-C. Ensure enumeration constants map to unique values
<https://wiki.sei.cmu.edu/confluence/display/c/INT09-C.+Ensure+enumeration+constants+map+to+unique+values>`_.

`cert-int09-c` redirects here as an alias of this check.

Options
-------

.. option:: AllowExplicitZeroFirstInitialValue

  If set to `false`, the first enumerator must not be explicitly initialized to
  a literal ``0``.
  Default is `true`.

  .. code-block:: c++

    enum F {
      f0 = 0, // Not allowed if AllowExplicitZeroFirstInitialValue is false.
      f1,
      f2,
    };


.. option:: AllowExplicitSequentialInitialValues

  If set to `false`, explicit initialization to sequential values are not
  allowed.
  Default is `true`.

  .. code-block:: c++

    enum G {
      g0 = 1, // Not allowed if AllowExplicitSequentialInitialValues is false.
      g1 = 2,
      g2 = 3,
