========================
Libsycl Coding Standards
========================

.. contents::
   :local:

Introduction
============

In general ``libsycl`` project adopts
`LLVM Coding Standards <https://llvm.org/docs/CodingStandards.html>`_.
Although ``libsycl`` code base has a reason to deviate from the Coding
Standards. API classes interface is defined by SYCL 2020 specification and
don't match the LLVM Coding Standards for Naming.
This document describes points of deviation from LLVM coding standards for the
``libsycl`` project.

Naming
------

Name of Types, Functions, Variables, and Enumerators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are 2 kinds of declarations that have different rules:

* **SYCL API types** should be declared as it is stated in SYCL 2020
  specification. Those declarations are in snake case (e.g.
  ``create_sub_devices``).
  This rule can be applicable to type traits helpers that are used by SYCL API
  methods (e.g. see platform::get_info) for style alignment within a
  declaration. Decision to use snake case or camel case in this case remains
  with a developer.

* **Other types (implementation details)** should follow LLVM Coding Standards
  and should be declared in camel case.

Name of Files and Directories
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Directory Names** within ``libsycl`` directory should be in the snake case
  (e.g. ``test_e2e``). In some case exceptions may apply: for example
  ``Inputs`` is usually used as default name for directories, containing helper
  classes for LIT tests and that should be excluded from test descovery.

* **File Names in snake case** should be used for all C++ implementation files.
  For example files in directories ``include``, ``src``, ``test``, ``utils``,
  ``tools`` should be named in snake case.

* **File Names in camel case** should be used for other files. For example
  files in directories ``cmake/modules``, ``docs`` should be named in camel
  case. Exception is extension files (in ``docs``) whose naming should be in
  snake case to align with Khronos extensions style.
