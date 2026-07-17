========================
Libsycl Coding Standards
========================

.. contents::
   :local:

Introduction
============

The ``libsycl`` project follows the
`LLVM Coding Standards <https://llvm.org/docs/CodingStandards.html>`_ with
exceptions as described in this document.

Naming
------

Names of Macros, Types, Functions, Variables, and Enumerators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Entities specified by the SYCL specification are named as required by the SYCL
specification. Names of all other entities follow the guidance in the LLVM
Coding Standards.

Names of Files and Directories
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Directory Names** should be in snake case (e.g. ``test_e2e``) except in
  cases where LLVM project wide conventions are used. For example, LIT tests
  often use an ``Inputs`` directory to hold files that are used by tests but
  that should be excluded from test discovery.

* **File Names in snake case** should be used for all C++ implementation files.
  For example files in directories ``include``, ``src``, ``test``, ``utils``,
  and ``tools`` should be named in snake case.

* **File Names in camel case** should be used for most other files. For example
  files in directories ``cmake/modules`` and ``docs`` should be named in camel
  case.
