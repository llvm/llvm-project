Fuzzing for LLVM-libc functions
===============================

Fuzz tests are used to ensure quality and security of LLVM-libc implementations.
All fuzz tests live under the directory named ``fuzzing``. Within this
directory, the fuzz test for a libc function lives in the same nested directory
as its implementation in the toplevel ``src`` directory. The build target
``libc-fuzzer`` builds all of the enabled fuzz tests (but does not run them).

Types of fuzz tests
===================

As of this writing, there are two different kinds of fuzz tests. One kind are
the traditional fuzz tests which test one function at a time and only that
particular function. The other kind of tests are what we call as the
differential fuzz tests. These tests compare the behavior of LLVM libc
implementations with the behavior of the corresponding functions from the system
libc.
