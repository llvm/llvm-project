====================
Structure Protection
====================

.. contents::
   :local:


Introduction
============

Structure protection is an *experimental* mitigation
against use-after-free vulnerabilities. For
more information, please see the original `RFC
<https://discourse.llvm.org/t/rfc-structure-protection-a-family-of-uaf-mitigation-techniques/85555>`_.
An independent set of documentation will be contributed when the feature
is promoted to stable.

Usage
=====

To use structure protection, build your program using one of the flags:

- ``-fexperimental-pointer-field-protection=untagged``: Enable pointer
  field protection with untagged pointers.

- ``-fexperimental-pointer-field-protection=tagged``: Enable pointer
  field protection with heap pointers assumed to be tagged by the allocator.

The entire C++ part of the program must be built with a consistent
``-fexperimental-pointer-field-protection`` flag, and the C++ standard
library must also be built with the same flag and statically linked into
the program.

To build libc++ with pointer field protection support, pass the following
CMake flags:

.. code-block:: console
 
    "-DRUNTIMES_${triple}_LIBCXXABI_ENABLE_SHARED=OFF" \
    "-DRUNTIMES_${triple}_LIBCXX_USE_COMPILER_RT=ON" \
    "-DRUNTIMES_${triple}_LIBCXX_PFP=untagged" \
    "-DRUNTIMES_${triple}_LIBCXX_ENABLE_SHARED=OFF" \
    "-DRUNTIMES_${triple}_LIBCXX_TEST_CONFIG=llvm-libc++-static.cfg.in" \
    "-DRUNTIMES_${triple}_LIBUNWIND_ENABLE_SHARED=OFF" \

where ``${triple}`` is your target triple, such as
``aarch64-unknown-linux``.

The resulting toolchain may then be used to build programs
with pointer field protection by passing ``-stdlib=libc++
-fexperimental-pointer-field-protection=untagged`` at compile time
and ``-Wl,-Bstatic -lc++ -lc++abi -Wl,-Bdynamic -lm -fuse-ld=lld
-static-libstdc++`` at link time.
