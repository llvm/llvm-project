.. title:: clang-tidy - altera-unroll-loops

altera-unroll-loops
===================

Finds inner loops that have not been unrolled, as well as fully unrolled loops
with unknown loop bounds or a large number of iterations.

Unrolling inner loops could improve the performance of OpenCL kernels. However,
if they have unknown loop bounds or a large number of iterations, they cannot
be fully unrolled, and should be partially unrolled.

Notes:

- This check is unable to determine the number of iterations in a ``while`` or
  ``do..while`` loop; hence if such a loop is fully unrolled, a note is emitted
  advising the user to partially unroll instead.

- In ``for`` loops, our check only works with simple arithmetic increments (
  ``+``, ``-``, ``*``, ``/``). For all other increments, partial unrolling is
  advised.

- Depending on the exit condition, the calculations for determining if the
  number of iterations is large may be off by 1. This should not be an issue
  since the cut-off is generally arbitrary.

Based on the `Altera SDK for OpenCL: Best Practices Guide
<https://www.altera.com/en_US/pdfs/literature/hb/opencl-sdk/aocl_optimization_guide.pdf>`_.

.. code-block:: c++

   for (int i = 0; i < 10; i++) {  // ok: outer loops should not be unrolled
      int j = 0;
      do {  // warning: this inner do..while loop should be unrolled
         j++;
      } while (j < 15);

      int k = 0;
      #pragma unroll
      while (k < 20) {  // ok: this inner loop is already unrolled
         k++;
      }
   }

   int A[1000];
   #pragma unroll
   // warning: this loop is large and should be partially unrolled
   for (int a : A) {
      printf("%d", a);
   }

   #pragma unroll 5
   // ok: this loop is large, but is partially unrolled
   for (int a : A) {
      printf("%d", a);
   }

   #pragma unroll
   // warning: this loop is large and should be partially unrolled
   for (int i = 0; i < 1000; ++i) {
      printf("%d", i);
   }

   #pragma unroll 5
   // ok: this loop is large, but is partially unrolled
   for (int i = 0; i < 1000; ++i) {
      printf("%d", i);
   }

   #pragma unroll
   // warning: << operator not supported, recommend partial unrolling
   for (int i = 0; i < 1000; i<<1) {
      printf("%d", i);
   }

   std::vector<int> someVector (100, 0);
   int i = 0;
   #pragma unroll
   // note: loop may be large, recommend partial unrolling
   while (i < someVector.size()) {
      someVector[i]++;
   }

   #pragma unroll
   // note: loop may be large, recommend partial unrolling
   while (true) {
      printf("In loop");
   }

   #pragma unroll 5
   // ok: loop may be large, but is partially unrolled
   while (i < someVector.size()) {
      someVector[i]++;
   }

Options
-------

.. option:: MaxLoopIterations

   Defines the maximum number of loop iterations that a fully unrolled loop
   can have. By default, it is set to `100`.

   In practice, this refers to the integer value of the upper bound
   within the loop statement's condition expression.
