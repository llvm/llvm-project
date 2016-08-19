.. title:: clang-tidy - mpi-buffer-deref

mpi-buffer-deref
================

This check verifies if a buffer passed to an MPI (Message Passing Interface)
function is sufficiently dereferenced. Buffers should be passed as a single
pointer or array. As MPI function signatures specify ``void *`` for their buffer
types, insufficiently dereferenced buffers can be passed, like for example as
double pointers or multidimensional arrays, without a compiler warning emitted.

Examples:

.. code-block:: c++

   // A double pointer is passed to the MPI function.
   char *buf;
   MPI_Send(&buf, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);

   // A multidimensional array is passed to the MPI function.
   short buf[1][1];
   MPI_Send(buf, 1, MPI_SHORT, 0, 0, MPI_COMM_WORLD);

   // A pointer to an array is passed to the MPI function.
   short *buf[1];
   MPI_Send(buf, 1, MPI_SHORT, 0, 0, MPI_COMM_WORLD);
