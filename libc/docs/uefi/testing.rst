.. _libc_uefi_testing:

==========================
Testing the UEFI C library
==========================

.. contents:: Table of Contents
  :depth: 4
  :local:

Testing infrastructure
======================

The LLVM C library supports different kinds of :ref:`tests <build_and_test>`
depending on the build configuration. The UEFI target is considered a full build
and therefore provides all of its own utilities to build and run the generated
tests. Currently UEFI supports two kinds of tests.

#. **Hermetic tests** - These are unit tests built with a test suite similar to
   Google's ``gtest`` infrastructure. These use the same infrastructure as unit
   tests except that the entire environment is self-hosted.

#. **Integration tests** - These are lightweight tests that simply call a
   ``main`` function and checks if it returns non-zero.

The UEFI target uses the same testing infrastructure as the other supported
``libc`` targets. We do this by treating UEFI as a standard hosted environment
capable of launching a ``main`` function. This only requires us to run the
tests in a UEFI environment.
