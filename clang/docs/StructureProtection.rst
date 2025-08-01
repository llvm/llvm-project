====================
Structure Protection
====================

.. contents::
   :local:


Introduction
============

Structure protection is an experimental mitigation against use-after-free
vulnerabilities. For more details, please see the original `RFC
<https://discourse.llvm.org/t/rfc-structure-protection-a-family-of-uaf-mitigation-techniques/85555>`_.
An independent set of documentation will be added here when the feature
is promoted to non-experimental.

Usage
=====

To use structure protection, build your program using one of the flags:

- ``-fexperimental-pointer-field-protection=untagged``: Enable pointer
  field protection with untagged pointers.

- ``-fexperimental-pointer-field-protection=tagged``: Enable pointer
  field protection with heap pointers assumed to be tagged by the allocator:

The entire C++ part of the program must be built with a consistent
``-fexperimental-pointer-field-protection`` flag, and the C++ standard
library must also be built with the same flag and statically linked into
the program.
