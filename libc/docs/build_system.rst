LLVM libc build rules
=====================

At the cost of verbosity, we want to keep the build system of LLVM libc
as simple as possible. We also want to be highly modular with our build
targets. This makes picking and choosing desired pieces a straighforward
task.

Targets for entrypoints
-----------------------

Every entrypoint in LLVM-libc has its own build target. This target is listed
using the ``add_entrypoint_object`` rule. This rule generates a single object
file containing the implementation of the entrypoint.

Targets for redirecting entrypoints are also listed using the
``add_entrypoint_object`` rule. However, one will have to additionally specify
the ``REDIRECTED`` option with the rule.

Targets for entrypoint libraries
--------------------------------

Standards like POSIX require that a libc provide certain library files like
``libc.a``, ``libm.a``, etc. The targets for such library files are listed in
the ``lib`` directory as ``add_entrypoint_library`` targets. An
``add_entrypoint_library`` target  takes a list of ``add_entrypoint_object``
targets and produces a static library containing the object files corresponding
to the ``add_entrypoint_targets``.

Targets for redirectors
-----------------------

Similar to how every entrypoint in LLVM-libc has its own build target, every
redirector function also has its own build target. This target is listed using
the ``add_redirector_object`` rule. This rule generates a single object file
which can be packaged along with other redirector objects into shared library
of redirectors (see below).

Targets for library of redirectors
----------------------------------

Targets for shared libraries of redirectors are listed using the
``add_redirector_library`` rule.
