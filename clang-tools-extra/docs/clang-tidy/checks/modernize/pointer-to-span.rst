.. title:: clang-tidy - modernize-pointer-to-span

modernize-pointer-to-span
=========================

Finds function parameter pairs of (pointer, size) that could be replaced with
``std::span``.

Passing a pointer and a separate size is a common C-style pattern that
``std::span`` (C++20) was designed to replace. Using ``std::span`` bundles the
pointer and size together, preventing mismatches and improving readability.

.. code-block:: c++

  // Before
  void process(int *Data, int Size);

  // After
  void process(std::span<int> Data);

The check uses a name-based heuristic to identify size parameters. Names like
``size``, ``len``, ``length``, ``count``, ``n``, ``num``, or names ending in
``_size``, ``_len``, etc. are recognized.

The check will not flag a parameter pair if:

- the pointer is ``void*`` or a function pointer,
- the size parameter is not an integer type,
- the size parameter name does not suggest a size,
- the function is a virtual method, or
- the size parameter is unnamed.

.. note::

   This check requires C++20 or later and does not provide fix-its because
   changing the signature requires updating all call sites.
