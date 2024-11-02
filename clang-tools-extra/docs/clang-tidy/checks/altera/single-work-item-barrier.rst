.. title:: clang-tidy - altera-single-work-item-barrier

altera-single-work-item-barrier
===============================

Finds OpenCL kernel functions that call a barrier function but do not call
an ID function (``get_local_id``, ``get_local_id``, ``get_group_id``, or
``get_local_linear_id``).

These kernels may be viable single work-item kernels, but will be forced to
execute as NDRange kernels if using a newer version of the Altera Offline
Compiler (>= v17.01).

If using an older version of the Altera Offline Compiler, these kernel
functions will be treated as single work-item kernels, which could be
inefficient or lead to errors if NDRange semantics were intended.

Based on the `Altera SDK for OpenCL: Best Practices Guide
<https://www.altera.com/en_US/pdfs/literature/hb/opencl-sdk/aocl_optimization_guide.pdf>`_.

Examples:

.. code-block:: c++

  // error: function calls barrier but does not call an ID function.
  void __kernel barrier_no_id(__global int * foo, int size) {
    for (int i = 0; i < 100; i++) {
      foo[i] += 5;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
  }

  // ok: function calls barrier and an ID function.
  void __kernel barrier_with_id(__global int * foo, int size) {
    for (int i = 0; i < 100; i++) {
      int tid = get_global_id(0);
      foo[tid] += 5;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
  }

  // ok with AOC Version 17.01: the reqd_work_group_size turns this into
  // an NDRange.
  __attribute__((reqd_work_group_size(2,2,2)))
  void __kernel barrier_with_id(__global int * foo, int size) {
    for (int i = 0; i < 100; i++) {
      foo[tid] += 5;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
  }

Options
-------

.. option:: AOCVersion

   Defines the version of the Altera Offline Compiler. Defaults to ``1600``
   (corresponding to version 16.00).
