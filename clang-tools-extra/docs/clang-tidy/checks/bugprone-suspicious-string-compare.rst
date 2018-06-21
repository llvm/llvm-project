.. title:: clang-tidy - bugprone-suspicious-string-compare

bugprone-suspicious-string-compare
==================================

Find suspicious usage of runtime string comparison functions.
This check is valid in C and C++.

Checks for calls with implicit comparator and proposed to explicitly add it.

.. code-block:: c++

    if (strcmp(...))       // Implicitly compare to zero
    if (!strcmp(...))      // Won't warn
    if (strcmp(...) != 0)  // Won't warn

Checks that compare function results (i,e, ``strcmp``) are compared to valid
constant. The resulting value is

.. code::

    <  0    when lower than,
    >  0    when greater than,
    == 0    when equals.

A common mistake is to compare the result to `1` or `-1`.

.. code-block:: c++

    if (strcmp(...) == -1)  // Incorrect usage of the returned value.

Additionally, the check warns if the results value is implicitly cast to a
*suspicious* non-integer type. It's happening when the returned value is used in
a wrong context.

.. code-block:: c++

    if (strcmp(...) < 0.)  // Incorrect usage of the returned value.

Options
-------

.. option:: WarnOnImplicitComparison

   When non-zero, the check will warn on implicit comparison. `1` by default.

.. option:: WarnOnLogicalNotComparison

   When non-zero, the check will warn on logical not comparison. `0` by default.

.. option:: StringCompareLikeFunctions

   A string specifying the comma-separated names of the extra string comparison
   functions. Default is an empty string.
   The check will detect the following string comparison functions:
   `__builtin_memcmp`, `__builtin_strcasecmp`, `__builtin_strcmp`,
   `__builtin_strncasecmp`, `__builtin_strncmp`, `_mbscmp`, `_mbscmp_l`,
   `_mbsicmp`, `_mbsicmp_l`, `_mbsnbcmp`, `_mbsnbcmp_l`, `_mbsnbicmp`,
   `_mbsnbicmp_l`, `_mbsncmp`, `_mbsncmp_l`, `_mbsnicmp`, `_mbsnicmp_l`,
   `_memicmp`, `_memicmp_l`, `_stricmp`, `_stricmp_l`, `_strnicmp`,
   `_strnicmp_l`, `_wcsicmp`, `_wcsicmp_l`, `_wcsnicmp`, `_wcsnicmp_l`,
   `lstrcmp`, `lstrcmpi`, `memcmp`, `memicmp`, `strcasecmp`, `strcmp`,
   `strcmpi`, `stricmp`, `strncasecmp`, `strncmp`, `strnicmp`, `wcscasecmp`,
   `wcscmp`, `wcsicmp`, `wcsncmp`, `wcsnicmp`, `wmemcmp`.
