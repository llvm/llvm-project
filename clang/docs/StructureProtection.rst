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

To use structure protection, build your program using one or more of these flags:

- ``-fexperimental-pointer-field-protection``: Enable pointer
  field protection on all types that are not considered standard-layout
  according to the C++ rules for standard layout. Specifying this flag
  also defines the predefined macro ``__POINTER_FIELD_PROTECTION__``.

- ``-fexperimental-pointer-field-protection-tagged``: On architectures
  that support it (currently only AArch64), for types that are not considered
  trivially copyable, use the address of the object to compute the pointer
  encoding. Specifying this flag also defines the predefined macro
  ``__POINTER_FIELD_PROTECTION_TAGGED__``.

It is also possible to specify the attribute
``[[clang::pointer_field_protection]]`` on a struct type to opt the
struct's pointer fields into pointer field protection, even if the type is
standard layout or none of the command line flags are specified. Note that
this means that the type will not comply with pointer interconvertibility
and other standard layout rules.

Pointer field protection is inherited from bases and non-static data
members.

In order to avoid ABI breakage, the entire C++ part
of the program must be built with a consistent set of
``-fexperimental-pointer-field-protection*`` flags, and the C++ standard
library must also be built with the same flags and statically linked
into the program.

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
-fexperimental-pointer-field-protection`` at compile time
and ``-Wl,-Bstatic -lc++ -lc++abi -Wl,-Bdynamic -lm -fuse-ld=lld
-static-libstdc++`` at link time.
