.. _cmake_build_rules:

===========================
The libc CMake build system
===========================

At the cost of verbosity, we want to keep the build system of LLVM libc
as simple as possible. We also want to be highly modular with our build
targets. This makes picking and choosing desired pieces a straightforward
task.

Targets for entrypoints
-----------------------

Every entrypoint in LLVM-libc has its own build target, listed using the
``add_entrypoint_object`` rule. This rule generates a single object file
containing the implementation.

For more technical details on how to register entrypoints, see the
:ref:`entrypoints` documentation.

Targets for entrypoint libraries
--------------------------------

Standards like POSIX require that a libc provide certain library files like
``libc.a``, ``libm.a``, etc. The targets for such library files are listed in
the ``lib`` directory as ``add_entrypoint_library`` targets.

