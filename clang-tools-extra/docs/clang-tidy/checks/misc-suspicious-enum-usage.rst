.. title:: clang-tidy - misc-suspicious-enum-usage

misc-suspicious-enum-usage
==========================

The checker detects various cases when an enum is probably misused (as a bitmask
).
  
1. When "ADD" or "bitwise OR" is used between two enum which come from different
   types and these types value ranges are not disjoint.

The following cases will be investigated only using :option:`StrictMode`. We 
regard the enum as a (suspicious)
bitmask if the three conditions below are true at the same time:

* at most half of the elements of the enum are non pow-of-2 numbers (because of
  short enumerations)
* there is another non pow-of-2 number than the enum constant representing all
  choices (the result "bitwise OR" operation of all enum elements)
* enum type variable/enumconstant is used as an argument of a `+` or "bitwise OR
  " operator

So whenever the non pow-of-2 element is used as a bitmask element we diagnose a
misuse and give a warning.

2. Investigating the right hand side of `+=` and `|=` operator.
3. Check only the enum value side of a `|` and `+` operator if one of them is not
   enum val.
4. Check both side of `|` or `+` operator where the enum values are from the
   same enum type.

Examples:

.. code-block:: c++

  enum { A, B, C };
  enum { D, E, F = 5 };
  enum { G = 10, H = 11, I = 12 };
  
  unsigned flag;
  flag =
      A |
      H; // OK, disjoint value intervalls in the enum types ->probably good use.
  flag = B | F; // Warning, have common values so they are probably misused.
  
  // Case 2:
  enum Bitmask {
    A = 0,
    B = 1,
    C = 2,
    D = 4,
    E = 8,
    F = 16,
    G = 31 // OK, real bitmask.
  };
  
  enum Almostbitmask {
    AA = 0,
    BB = 1,
    CC = 2,
    DD = 4,
    EE = 8,
    FF = 16,
    GG // Problem, forgot to initialize.
  };
  
  unsigned flag = 0;
  flag |= E; // OK.
  flag |=
      EE; // Warning at the decl, and note that it was used here as a bitmask.

Options
-------
.. option:: StrictMode

   Default value: 0.
   When non-null the suspicious bitmask usage will be investigated additionally
   to the different enum usage check.
