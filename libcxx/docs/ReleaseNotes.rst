=========================================
Libc++ 14.0.0 (In-Progress) Release Notes
=========================================

.. contents::
   :local:
   :depth: 2

Written by the `Libc++ Team <https://libcxx.llvm.org>`_

.. warning::

   These are in-progress notes for the upcoming libc++ 14 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the libc++ C++ Standard Library,
part of the LLVM Compiler Infrastructure, release 14.0.0. Here we describe the
status of libc++ in some detail, including major improvements from the previous
release and new feature work. For the general LLVM release notes, see `the LLVM
documentation <https://llvm.org/docs/ReleaseNotes.html>`_. All LLVM releases may
be downloaded from the `LLVM releases web site <https://llvm.org/releases/>`_.

For more information about libc++, please see the `Libc++ Web Site
<https://libcxx.llvm.org>`_ or the `LLVM Web Site <https://llvm.org>`_.

Note that if you are reading this file from a Git checkout or the
main Libc++ web page, this document applies to the *next* release, not
the current one. To see the release notes for a specific release, please
see the `releases page <https://llvm.org/releases/>`_.

What's New in Libc++ 14.0.0?
============================

New Features
------------

- There's initial support for the C++20 header ``<format>``. The implementation
  is incomplete. Some functions are known to be inefficient; both in memory
  usage and performance. The implementation is considered experimental and isn't
  considered ABI stable.

- There's a new CMake option ``LIBCXX_ENABLE_UNICODE`` to disable Unicode
  support in the ``<format>`` header. This only affects the estimation of the
  output width of the format functions.

- Support for building libc++ on top of a C Standard Library that does not support ``wchar_t`` was
  added. This is useful for building libc++ in an embedded setting, and it adds itself to the various
  freestanding-friendly options provided by libc++.

- ``_LIBCPP_DEBUG`` equals to ``1`` enables the randomization of unspecified
  behavior of standard algorithms (e.g. equal elements in ``std::sort`` or
  randomization of both sides of partition for ``std::nth_element``)

- Floating-point support for ``std::to_chars`` support has been added.
  Thanks to Stephan T. Lavavej and Microsoft for providing their implementation
  to libc++.

API Changes
-----------

- The functions ``std::atomic<T*>::fetch_(add|sub)`` and
  ``std::atomic_fetch_(add|sub)`` no longer accept a function pointer. While
  this is technically an API break, the invalid syntax isn't supported by
  libstdc++ and MSVC STL.  See https://godbolt.org/z/49fvzz98d.

- The call of the functions ``std::atomic_(add|sub)(std::atomic<T*>*, ...)``
  with the explicit template argument ``T`` are now ill-formed. While this is
  technically an API break, the invalid syntax isn't supported by libstdc++ and
  MSVC STL. See https://godbolt.org/z/v9959re3v.

  Due to this change it's now possible to call these functions with the
  explicit template argument ``T*``. This allows using the same syntax on the
  major Standard library implementations.
  See https://godbolt.org/z/oEfzPhTTb.

  Calls to these functions where the template argument was deduced by the
  compiler are unaffected by this change.

- The functions ``std::allocator<T>::allocate`` and
  ``std::experimental::pmr::polymorphic_allocator<T>::allocate`` now throw
  an exception of type ``std::bad_array_new_length`` when the requested size
  exceeds the maximum supported size, as required by the C++ standard.
  Previously the type ``std::length_error`` was used.

- Removed the nonstandard methods ``std::chrono::file_clock::to_time_t`` and
  ``std::chrono::file_clock::from_time_t``; neither libstdc++ nor MSVC STL
  had such methods. Instead, in C++20, you can use ``std::chrono::file_clock::from_sys``
  and ``std::chrono::file_clock::to_sys``, which are specified in the Standard.
  If you are not using C++20, you should move to it.

- The declarations of functions ``declare_reachable``, ``undeclare_reachable``, ``declare_no_pointers``,
  ``undeclare_no_pointers``, and ``get_pointer_safety`` have been removed not only from C++2b but
  from all modes. Their symbols are still provided by the dynamic library for the benefit of
  existing compiled code. All of these functions have always behaved as no-ops.

ABI Changes
-----------

- The C++17 variable templates ``is_error_code_enum_v`` and
  ``is_error_condition_enum_v`` are now of type ``bool`` instead of ``size_t``.

- The C++03 emulation type for ``std::nullptr_t`` has been removed in favor of
  using ``decltype(nullptr)`` in all standard modes. This is an ABI break for
  anyone compiling in C++03 mode and who has ``std::nullptr_t`` as part of their
  ABI. However, previously, these users' ABI would be incompatible with any other
  binary or static archive compiled with C++11 or later. If you start seeing linker
  errors involving ``std::nullptr_t`` against previously compiled binaries, this may
  be the cause. You can define the ``_LIBCPP_ABI_USE_CXX03_NULLPTR_EMULATION`` macro
  to return to the previous behavior. That macro will be removed in LLVM 15. Please
  comment `here <https://reviews.llvm.org/D109459>`_ if you are broken by this change
  and need to define the macro.

Build System Changes
--------------------

- Building the libc++ shared or static library requires a C++ 20 capable compiler.
  Consider using a Bootstrapping build to build libc++ with a fresh Clang if you
  can't use the system compiler to build libc++ anymore.

- Historically, there has been numerous ways of building libc++ and libc++abi. This has
  culminated in over 5 different ways to build the runtimes, which made it impossible to
  maintain with a good level of support. Starting with this release, the runtimes support
  exactly two ways of being built, which should cater to all use-cases. Furthermore,
  these builds are as lightweight as possible and will work consistently even when targeting
  embedded platforms, which used not to be the case. Please see the documentation on building
  libc++ to see those two ways of building and migrate over to the appropriate build instructions
  as soon as possible.

  All other ways to build are deprecated and will not be supported in the next release.
  We understand that making these changes can be daunting. For that reason, here's a
  summary of how to migrate from the two most common ways to build:

  - If you were rooting your CMake invocation at ``<monorepo>/llvm`` and passing ``-DLLVM_ENABLE_PROJECTS=<...>``
    (which was the previously advertised way to build the runtimes), please simply root your CMake invocation at
    ``<monorepo>/runtimes`` and pass ``-DLLVM_ENABLE_RUNTIMES=<...>``.

  - If you were doing two CMake invocations, one rooted at ``<monorepo>/libcxx`` and one rooted at
    ``<monorepo>/libcxxabi`` (this used to be called a "Standalone build"), please move them to a
    single invocation like so:

    .. code-block:: bash

        $ cmake -S <monorepo>/libcxx -B libcxx-build <LIBCXX-OPTIONS>
        $ cmake -S <monorepo>/libcxxabi -B libcxxabi-build <LIBCXXABI-OPTIONS>

    should become

    .. code-block:: bash

        $ cmake -S <monorepo>/runtimes -B build -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi" <LIBCXX-OPTIONS> <LIBCXXABI-OPTIONS>

- Support for building the runtimes using the GCC 32 bit multilib flag (``-m32``) has been removed. Support
  for this had been flaky for a while, and we didn't know of anyone depending on this. Instead, please perform
  a normal cross-compilation of the runtimes using the appropriate target, such as passing the following to
  your bootstrapping build:

  .. code-block:: bash

      -DLLVM_RUNTIME_TARGETS=i386-unknown-linux

- Libc++, libc++abi and libunwind will not be built with ``-fPIC`` by default anymore.
  If you want to build those runtimes with position independent code, please specify
  ``-DCMAKE_POSITION_INDEPENDENT_CODE=ON`` explicitly when configuring the build, or
  ``-DRUNTIMES_<target-name>_CMAKE_POSITION_INDEPENDENT_CODE=ON`` if using the
  bootstrapping build.
