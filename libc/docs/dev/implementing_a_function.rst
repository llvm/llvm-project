.. _implementing_a_function:

===========================
Implementing a New Function
===========================

This guide provides a step-by-step walkthrough for adding a new function to LLVM-libc.

.. contents:: Table of Contents
   :depth: 2
   :local:

Overview
========

Adding a new function involves several steps, from updating the public specification to implementing and testing the code. Below is the standard checklist for contributors.

Step-by-Step Checklist
======================

1. Header Entry
---------------

Update the standard YAML file that describes the public header to ensure the function is included in the generated public header.

*   **File**: ``libc/include/<header>.yaml`` (or ``libc/include/sys/<header>.yaml`` for system headers)
*   Add the new function to the ``functions`` list.
*   Specify its name, return type, and arguments.
*   List the standards it complies with (e.g., ``stdc``, ``POSIX``).

2. Header Declaration
---------------------

Declare the function in the internal implementation header file. This file is used by other internal code.

*   **File**: ``libc/src/<header>/<func>.h``
*   Follow the structure defined in :ref:`implementation_standard`.
*   Ensure the declaration is inside the ``LIBC_NAMESPACE_DECL`` namespace.

3. Implementation
-----------------

Write the actual code for the function.

*   **File**: ``libc/src/<header>/<func>.cpp`` (or ``libc/src/<header>/<os>/<func>.cpp`` for platform-specific implementations)
*   Use the ``LLVM_LIBC_FUNCTION`` macro.
*   Refer to :ref:`code_style` for naming and layout conventions.

4. CMake Rule
-------------

Add a CMake target for the new function so it can be compiled.

*   **File**: ``libc/src/<header>/CMakeLists.txt``
*   Add an ``add_entrypoint_object`` rule for the new file.
*   List all internal dependencies correctly to ensure proper build order.

5. Platform Registration
------------------------

Register the new entrypoint for the target platforms to include it in the build.

*   **File**: ``libc/config/<os>/<arch>/entrypoints.txt``
*   Add the new function to the list of active entrypoints.

6. Testing
----------

Create tests to verify the implementation.

*   **File**: ``libc/test/src/<header>/<func>_test.cpp``
*   Add corresponding tests using the internal testing framework.
*   Update the ``CMakeLists.txt`` in the test directory (``libc/test/src/<header>/CMakeLists.txt``) to include the new test target.
