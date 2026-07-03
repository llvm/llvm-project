.. _entrypoints:

========================
Entrypoints in LLVM libc
========================

A public function or a global variable provided by LLVM-libc is called an
*entrypoint*. The notion of entrypoints is central to LLVM-libc's source layout,
build system, and configuration management. This document provides a technical
reference for how entrypoints are defined, implemented, and integrated into
the final library.

What is an Entrypoint?
----------------------

In a typical C library, all functions are part of a monolithic archive. In
LLVM-libc, each function (e.g., ``malloc``, ``printf``, ``isalpha``) is treated
as a discrete "entrypoint" unit. This allows for:

- **Granular build targets**: You can build just the objects you need.
- **Configuration-driven selection**: Different operating systems and
  architectures can pick specific implementations for the same function.
- **Support for multiple build modes**: Selectively replacing parts of a host's
  libc in :ref:`overlay_mode` or building a complete library in :ref:`full_host_build`.

The Lifecycle of an Entrypoint
------------------------------

1. **Implementation**: The function is implemented in a ``.cpp`` file using
   LLVM-libc's coding and implementation standards.
2. **Registration**: The entrypoint is defined as a CMake target using the
   ``add_entrypoint_object`` rule.
3. **Configuration**: The target name is added to an ``entrypoints.txt`` file
   to include it in a specific OS/Architecture configuration.

Implementation Standards
------------------------

Implementations live in the ``src/`` directory, organized by the public header
they belong to (e.g., ``src/ctype/isalpha.cpp`` for ``ctype.h``).

Header File Structure
^^^^^^^^^^^^^^^^^^^^^

Every entrypoint has an internal implementation header file (e.g.,
``src/ctype/isalpha.h``). This header declares the function within the
``LIBC_NAMESPACE_DECL`` namespace::

   namespace LIBC_NAMESPACE_DECL {
   int isalpha(int c);
   } // namespace LIBC_NAMESPACE_DECL

Source File Structure
^^^^^^^^^^^^^^^^^^^^^

The implementation file (e.g., ``src/ctype/isalpha.cpp``) defines the function
using the ``LLVM_LIBC_FUNCTION`` macro. This macro handles C-linkage and
aliasing::

   namespace LIBC_NAMESPACE_DECL {
   LLVM_LIBC_FUNCTION(int, isalpha, (int c)) {
     // ... implementation ...
   }
   } // namespace LIBC_NAMESPACE_DECL

For more details on implementation conventions, see the
:ref:`implementation_standard` page.

Registration: CMake Rules
-------------------------

Entrypoints are registered as CMake targets to make them available to the
build system. These rules are usually defined in the ``CMakeLists.txt`` file
within the function's source directory.

``add_entrypoint_object``
^^^^^^^^^^^^^^^^^^^^^^^^^

This rule generates a single object file containing the implementation of the
entrypoint.

.. code-block:: cmake

   add_entrypoint_object(
     isalpha
     SRCS isalpha.cpp
     HDRS isalpha.h
     DEPENDS
       .some_internal_dependency
   )

For redirecting entrypoints (e.g., when one function is a simple alias for
another), the ``REDIRECTED`` option can be specified to the rule.

``add_entrypoint_library``
^^^^^^^^^^^^^^^^^^^^^^^^^^

Standard library files like ``libc.a`` and ``libm.a`` are produced by
aggregating multiple entrypoint objects. The ``add_entrypoint_library`` target
takes a list of ``add_entrypoint_object`` targets and produces a static library.

Configuration: ``entrypoints.txt``
----------------------------------

The final selection of which entrypoints are included in a specific build is
determined by ``entrypoints.txt`` files located in the ``libc/config`` tree.

- **Location**: Typically found in ``libc/config/<os>/entrypoints.txt`` or
  ``libc/config/<os>/<arch>/entrypoints.txt``.
- **Role**: This file acts as the "source of truth" for what is supported on a
  given platform. A typical bring-up procedure involves progressively adding
  targets to this file as they are implemented and tested.

If you are implementing a new entrypoint, you must add its target name to the
relevant ``entrypoints.txt`` files for it to be included in the library build.
For more details on platform configuration, see the :ref:`porting` guide.
