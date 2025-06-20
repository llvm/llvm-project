.. title:: clang-tidy - modernize-make-direct

modernize-make-direct
=====================

Replaces ``std::make_*`` function calls with direct constructor calls using class template
argument deduction (CTAD).

================================== ====================================
  Before                             After
---------------------------------- ------------------------------------
``std::make_optional<int>(42)``    ``std::optional(42)``
``std::make_pair(1, "test")``      ``std::pair(1, "test")``
``std::make_tuple(1, 2.0, "hi")``  ``std::tuple(1, 2.0, "hi")``
================================== ====================================

.. note::

   This check does not transform ``std::make_unique`` or ``std::make_shared`` because:
   
   1. These smart pointer types cannot be constructed using CTAD from raw pointers.
   2. ``std::make_shared`` provides performance benefits (single allocation) and 
      exception safety that would be lost with direct construction.
   3. Direct use of ``new`` is discouraged in modern C++ code.
   
   Use the dedicated ``modernize-make-unique`` and ``modernize-make-shared`` checks
   for transforming these functions.

Options
-------

.. option:: CheckMakePair

   When `true`, transforms ``std::make_pair`` calls to direct constructor calls.
   Default is `true`.

.. option:: CheckMakeOptional

   When `true`, transforms ``std::make_optional`` calls to direct constructor calls.
   Default is `true`.

.. option:: CheckMakeTuple

   When `true`, transforms ``std::make_tuple`` calls to direct constructor calls.
   Default is `true`.

