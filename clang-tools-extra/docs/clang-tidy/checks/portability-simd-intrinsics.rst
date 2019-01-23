.. title:: clang-tidy - portability-simd-intrinsics

portability-simd-intrinsics
===========================

Finds SIMD intrinsics calls and suggests ``std::experimental::simd`` (`P0214`_)
alternatives.

If the option ``Suggest`` is set to non-zero, for

.. code-block:: c++

  _mm_add_epi32(a, b); // x86
  vec_add(a, b);       // Power

the check suggests an alternative: ``operator+`` on ``std::experimental::simd``
objects.

Otherwise, it just complains the intrinsics are non-portable (and there are
`P0214`_ alternatives).

Many architectures provide SIMD operations (e.g. x86 SSE/AVX, Power AltiVec/VSX,
ARM NEON). It is common that SIMD code implementing the same algorithm, is
written in multiple target-dispatching pieces to optimize for different
architectures or micro-architectures.

The C++ standard proposal `P0214`_ and its extensions cover many common SIMD
operations. By migrating from target-dependent intrinsics to `P0214`_
operations, the SIMD code can be simplified and pieces for different targets can
be unified.

Refer to `P0214`_ for introduction and motivation for the data-parallel standard
library.

Options
-------

.. option:: Suggest

   If this option is set to non-zero (default is `0`), the check will suggest
   `P0214`_ alternatives, otherwise it only points out the intrinsic function is
   non-portable.

.. option:: Std

   The namespace used to suggest `P0214`_ alternatives. If not specified, `std::`
   for `-std=c++2a` and `std::experimental::` for `-std=c++11`.

.. _P0214: https://wg21.link/p0214
