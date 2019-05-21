.. title:: clang-tidy - misc-throw-by-value-catch-by-reference

misc-throw-by-value-catch-by-reference
======================================

`cert-err09-cpp` redirects here as an alias for this check.
`cert-err61-cpp` redirects here as an alias for this check.

Finds violations of the rule "Throw by value, catch by reference" presented for
example in "C++ Coding Standards" by H. Sutter and A. Alexandrescu.

Exceptions:
  * Throwing string literals will not be flagged despite being a pointer. They
    are not susceptible to slicing and the usage of string literals is idomatic.
  * Catching character pointers (``char``, ``wchar_t``, unicode character types)
    will not be flagged to allow catching sting literals.
  * Moved named values will not be flagged as not throwing an anonymous
    temporary. In this case we can be sure that the user knows that the object
    can't be accessed outside catch blocks handling the error.
  * Throwing function parameters will not be flagged as not throwing an
    anonymous temporary. This allows helper functions for throwing.
  * Re-throwing caught exception variables will not be flragged as not throwing
    an anonymous temporary. Although this can usually be done by just writing
    ``throw;`` it happens often enough in real code.

Options
-------

.. option:: CheckThrowTemporaries

   Triggers detection of violations of the rule `Throw anonymous temporaries
   <https://www.securecoding.cert.org/confluence/display/cplusplus/ERR09-CPP.+Throw+anonymous+temporaries>`_.
   Default is `1`.

.. option:: WarnOnLargeObject

   Also warns for any large, trivial object caught by value. Catching a large
   object by value is not dangerous but affects the performance negatively. The
   maximum size of an object allowed to be caught without warning can be set
   using the `MaxSize` option.
   Default is `0`.

.. option:: MaxSize

   Determines the maximum size of an object allowed to be caught without
   warning. Only applicable if `WarnOnLargeObject` is set to `1`. If option is
   set by the user to `std::numeric_limits<uint64_t>::max()` then it reverts to
   the default value.
   Default is the size of `size_t`.
