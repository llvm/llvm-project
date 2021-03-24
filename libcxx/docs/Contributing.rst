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

Looking for pre-existing reviews
================================

Before you start working on any feature, please take a look at the open reviews
to avoid duplicating someone else's work. You can do that by going to the website
where code reviews are held, `Differential <https://reviews.llvm.org/differential>`__,
and clicking on ``Libc++ Open Reviews`` in the sidebar to the left. If you see
that your feature is already being worked on, please consider chiming in instead
of duplicating work!

Post-Release TODO
=================

After branching for an LLVM release:

1. Update ``_LIBCPP_VERSION`` in ``include/__config``
2. Update the ``include/__libcpp_version`` file
3. Update the version number in ``docs/conf.py``

Modifying feature test macros
=============================

When adding or updating feature test macros, you should update the corresponding tests.
To do that, modify ``feature_test_macros`` table in the script ``utils/generate_feature_test_macro_components.py``, run it, and commit updated files.

Adding a new header TODO
========================

When adding a new header to libc++:

1. Add a test under ``test/libcxx`` that the new header defines ``_LIBCPP_VERSION``. See ``test/libcxx/algorithms/version.pass.cpp`` for an example.
2. Run ``python utils/generate_header_tests.py``; verify and commit the changes.
3. Modify ``python utils/generate_header_inclusion_tests.py``; run it; verify and commit the changes.
4. Create a submodule in ``include/module.modulemap`` for the new header.
5. Update the ``include/CMakeLists.txt`` file to include the new header.

Exporting new symbols from the library
======================================

When exporting new symbols from libc++, one must update the ABI lists located in ``lib/abi``.
To test whether the lists are up-to-date, please run the target ``check-cxx-abilist``.
To regenerate the lists, use the target ``generate-cxx-abilist``.
