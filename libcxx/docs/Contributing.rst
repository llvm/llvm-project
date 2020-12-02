.. _ContributingToLibcxx:

======================
Contributing to libc++
======================

.. contents::
  :local:

Please read `this document <https://www.llvm.org/docs/Contributing.html>`__ on general rules to contribute to LLVM projects.

Tasks and processes
===================

This file contains notes about various tasks and processes specific to libc++.

Post-Release TODO
=================

After branching for an LLVM release:

1. Update ``_LIBCPP_VERSION`` in ``include/__config``
2. Update the ``include/__libcpp_version`` file
3. Update the version number in ``docs/conf.py``

Adding a new header TODO
========================

When adding a new header to libc++:

1. Add a test under ``test/libcxx`` that the new header defines ``_LIBCPP_VERSION``. See ``test/libcxx/algorithms/version.pass.cpp`` for an example.
2. Update the following test files to include the new header:

  * ``test/libcxx/double_include.sh.cpp``
  * ``test/libcxx/min_max_macros.compile.pass.cpp``
  * ``test/libcxx/no_assert_include.compile.pass.cpp``

3. Create a submodule in ``include/module.modulemap`` for the new header.
4. Update the ``include/CMakeLists.txt`` file to include the new header.

Exporting new symbols from the library
======================================

When exporting new symbols from libc++, one must update the ABI lists located in ``lib/abi``.
To test whether the lists are up-to-date, please run the target ``check-cxx-abilist``.
To regenerate the lists, use the target ``generate-cxx-abilist``.
