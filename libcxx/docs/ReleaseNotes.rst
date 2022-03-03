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

- There's support for the C++20 header ``<format>``. Some parts are still
  missing, most notably the compile-time format string validation. Some
  functions are known to be inefficient, both in memory usage and performance.
  The implementation isn't API- or ABI-stable and therefore considered
  experimental. (Some not-yet-implemented papers require an API-break.)
  As a result, it is disabled by default, however vendors can enable the
  header by using ``-DLIBCXX_ENABLE_INCOMPLETE_FEATURES=ON`` when
  configuring their build.

- More parts of ``<ranges>`` have been implemented. Since we still expect to make
  some API and ABI breaking changes, those are disabled by default. However,
  vendors that wish to enable ``<ranges>`` in their distribution may do so
  by defining ``-DLIBCXX_ENABLE_INCOMPLETE_FEATURES=ON`` when configuring
  their build.

- There's a new CMake option ``LIBCXX_ENABLE_UNICODE`` to disable Unicode
  support in the ``<format>`` header. This only affects the estimation of the
  output width of the format functions.

- Support for building libc++ on top of a C Standard Library that does not support ``wchar_t`` was
  added. This is useful for building libc++ in an embedded setting, and it adds itself to the various
  freestanding-friendly options provided by libc++.

- Defining ``_LIBCPP_DEBUG`` to ``1`` enables the randomization of unspecified
  behavior in standard algorithms (e.g. the ordering of equal elements in ``std::sort``, or
  the ordering of both sides of the partition in ``std::nth_element``).

- Floating-point support for ``std::to_chars`` support has been added.
  Thanks to Stephan T. Lavavej and Microsoft for providing their implementation
  to libc++.

- The C++20 ``<coroutine>`` implementation has been completed.

- More C++20 features have been implemented. :doc:`Status/Cxx20` has the full
  overview of libc++'s C++20 implementation status.

- More C++2b features have been implemented. :doc:`Status/Cxx2b` has the full
  overview of libc++'s C++2b implementation status.

- 16-bit ``wchar_t`` handling added for ``codecvt_utf8``, ``codecvt_utf16`` and
  ``codecvt_utf8_utf16``.

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

- ``std::filesystem::path::iterator``, which (in our implementation) stashes
  a ``path`` value inside itself similar to ``istream_iterator``, now sets its
  ``reference`` type to ``path`` and its ``iterator_category`` to ``input_iterator_tag``,
  so that it is a conforming input iterator in C++17 and a conforming
  ``std::bidirectional_iterator`` in C++20. Before this release, it had set its
  ``reference`` type to ``const path&`` and its ``iterator_category`` to
  ``bidirectional_iterator_tag``, making it a non-conforming bidirectional iterator.
  After this change, ``for`` loops of the form ``for (auto& c : path)`` must be rewritten
  as either ``for (auto&& c : path)`` or ``for (const auto& c : path)``.
  ``std::reverse_iterator<path::iterator>`` is no longer rejected.

- Removed the nonstandard default constructor from ``std::chrono::month_weekday``.
  You must now explicitly initialize with a ``chrono::month`` and
  ``chrono::weekday_indexed`` instead of "meh, whenever".

- C++20 requires that ``std::basic_string::reserve(n)`` never reduce the capacity
  of the string. (For that, use ``shrink_to_fit()``.) Prior to this release, libc++'s
  ``std::basic_string::reserve(n)`` could reduce capacity in C++17 and before, but
  not in C++20 and later. This caused ODR violations when mixing code compiled under
  different Standard modes. After this change, libc++'s ``std::basic_string::reserve(n)``
  never reduces capacity, even in C++17 and before.
  C++20 deprecates the zero-argument overload of ``std::basic_string::reserve()``,
  but specifically permits it to reduce capacity. To avoid breaking existing code
  assuming that ``std::basic_string::reserve()`` will shrink, libc++ maintains
  the behavior to shrink, even though that makes ``std::basic_string::reserve()`` not
  a synonym for ``std::basic_string::reserve(0)`` in any Standard mode anymore.

- The ``<experimental/coroutine>`` header is deprecated, as is any
  use of coroutines without C++20. Use C++20's ``<coroutine>`` header
  instead. The ``<experimental/coroutine>`` header will be removed
  in LLVM 16.

- ``_VSTD`` is now an alias for ``std`` instead of ``std::_LIBCPP_ABI_NAMESPACE``.
  This is technically not a functional change, except for folks that might have been
  using ``_VSTD`` in creative ways (which has never been officially supported).

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
  comment `on D109459 <https://reviews.llvm.org/D109459>`_ if you are broken by this change
  and need to define the macro.

- On Apple platforms, ``std::random_device`` is now implemented on top of ``arc4random()``
  instead of reading from ``/dev/urandom``. Any implementation-defined token used when
  constructing a ``std::random_device`` will now be ignored instead of interpreted as a
  file to read entropy from.

- ``std::lognormal_distribution::param_type`` used to store a data member of type
  ``std::normal_distribution``; now this member is stored in the ``lognormal_distribution``
  class itself, and the ``param_type`` stores only the mean and standard deviation,
  as required by the Standard. This changes ``sizeof(std::lognormal_distribution::param_type)``.
  You can define the ``_LIBCPP_ABI_OLD_LOGNORMAL_DISTRIBUTION`` macro to return to the
  previous behavior. That macro will be removed in LLVM 15. Please comment
  `on PR52906 <https://llvm.org/PR52906>`_ if you are broken by this change and need to
  define the macro.

Build System Changes
--------------------

- Building the libc++ shared or static library requires a C++ 20 capable compiler.
  Consider using a Bootstrapping build to build libc++ with a fresh Clang if you
  can't use the system compiler to build libc++ anymore.

- Historically, there have been numerous ways of building libc++, libc++abi, and libunwind.
  This has led to at least 5 different ways to build the runtimes, which was impossible to
  maintain with a good level of support. Starting with this release, libc++, libc++abi, and
  libunwind support exactly two ways of being built, which should cater to all use-cases.
  Furthermore, these builds are as lightweight as possible and will work consistently even
  when targeting embedded platforms, which used not to be the case. :doc:`BuildingLibcxx`
  describes those two ways of building. Please migrate over to the appropriate build
  instructions as soon as possible.

  All other ways to build are deprecated and will not be supported in the next release.
  We understand that making these changes can be daunting. For that reason, here's a
  summary of how to migrate from the two most common ways to build:

  - If you were rooting your CMake invocation at ``<monorepo>/llvm`` and passing ``-DLLVM_ENABLE_PROJECTS=<...>``
    (which was the previously advertised way to build the runtimes), please simply root your CMake invocation at
    ``<monorepo>/runtimes`` and pass ``-DLLVM_ENABLE_RUNTIMES=<...>``.

  - If you were doing multiple CMake invocations, e.g. one rooted at ``<monorepo>/libcxx`` and one rooted
    at ``<monorepo>/libcxxabi`` (this used to be called a "Standalone build"), please move them to a
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

- Libc++, libc++abi, and libunwind will not be built with ``-fPIC`` by default anymore.
  If you want to build those runtimes with position-independent code, please specify
  ``-DCMAKE_POSITION_INDEPENDENT_CODE=ON`` explicitly when configuring the build, or
  ``-DRUNTIMES_<target-name>_CMAKE_POSITION_INDEPENDENT_CODE=ON`` if using the
  bootstrapping build.

- The ``{LIBCXX,LIBCXXABI,LIBUNWIND}_TARGET_TRIPLE``, ``{LIBCXX,LIBCXXABI,LIBUNWIND}_SYSROOT`` and
  ``{LIBCXX,LIBCXXABI,LIBUNWIND}_GCC_TOOLCHAIN`` CMake variables are deprecated. Instead, please use
  the ``CMAKE_CXX_COMPILER_TARGET``, ``CMAKE_SYSROOT`` and ``CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN``
  variables provided by CMake.
