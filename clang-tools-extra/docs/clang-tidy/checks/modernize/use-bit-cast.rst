.. title:: clang-tidy - modernize-use-bit-cast

modernize-use-bit-cast
======================

Finds ``memcpy``-based type punning that can be rewritten as ``std::bit_cast``
in C++20 and later.

.. code-block:: c++

  float src = 1.0f;
  unsigned int dst;
  std::memcpy(&dst, &src, sizeof(src));

This is rewritten to:

.. code-block:: c++

  float src = 1.0f;
  unsigned int dst;
  dst = std::bit_cast<unsigned int>(src);

The fix replaces only the ``memcpy`` call. It does not rewrite a preceding
declaration into ``auto dst = ...``.

It matches only object-to-object copies where:

* both object types are trivially copyable, and neither is a pointer,
  function, or ``volatile``-qualified type,
* the destination can be assigned from ``std::bit_cast``, so raw C array
  destinations are excluded,
* the source and destination are not the same type after removing aliases and
  cv-qualifiers,
* the size argument is ``sizeof`` of either copied type, and
* the ``memcpy`` result is not used.

It intentionally does not diagnose:

* macro expansions,
* dependent template cases,
* unevaluated contexts such as ``sizeof(memcpy(...))``,
* unrelated overloads such as a user-defined ``memcpy``.

If needed, the fix also inserts ``#include <bit>``.

Options
-------

.. option:: IncludeStyle

   A string specifying which include-style is used, `llvm` or `google`.
   Default is `llvm`.
