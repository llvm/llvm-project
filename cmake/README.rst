=======================
LLVM Common CMake Utils
=======================

What goes here
--------------

These are CMake modules to be shared between LLVM projects strictly at build
time. In other words, they must not be included from an installed CMake module,
such as the ``Add*.cmake`` ones. Modules that are reachable from installed
modules should instead go in ``${project}/cmake/modules`` of the most upstream
project that uses them.

The advantage of not putting these modules in an existing location like
``llvm/cmake/modules`` is two-fold:

- Since they are not installed, we don't have to worry about any out-of-tree
  downstream usage, and thus there is no need for stability.

- Since they are available as part of the source at build-time, we don't have
  to do the usual stand-alone vs combined-build dances, avoiding much
  complexity.

How to use
----------

For tools, please do:

.. code-block:: cmake

  if(NOT DEFINED LLVM_COMMON_CMAKE_UTILS)
    set(LLVM_COMMON_CMAKE_UTILS ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
  endif()

  # Add path for custom modules.
  list(INSERT CMAKE_MODULE_PATH 0
    # project-specific module dirs first
    "${LLVM_COMMON_CMAKE_UTILS}/Modules"
    )

Notes:

- The ``if(NOT DEFINED ...)`` guard is there because in combined builds, LLVM
  will set this variable.  This is useful for legacy builds where projects are
  found in ``llvm/tools`` instead.

- ``INSERT ... 0`` ensures these new entries are prepended to the front of the
  module path, so nothing might shadow them by mistake.

For runtime libs, we skip the ``if(NOT DEFINED`` part:

.. code-block:: cmake

  set(LLVM_COMMON_CMAKE_UTILS ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)

  ... # same as before

If ``llvm/tools`` legacy-style combined builds are deprecated, we should then
skip it everywhere, bringing the tools and runtimes boilerplate back in line.
