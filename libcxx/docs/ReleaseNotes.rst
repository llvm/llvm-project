==========================
Libc++ 9.0.0 Release Notes
==========================

.. contents::
   :local:
   :depth: 2

Written by the `Libc++ Team <https://libcxx.llvm.org>`_


Introduction
============

This document contains the release notes for the libc++ C++ Standard Library,
part of the LLVM Compiler Infrastructure, release 9.0.0. Here we describe the
status of libc++ in some detail, including major improvements from the previous
release and new feature work. For the general LLVM release notes, see `the LLVM
documentation <https://llvm.org/docs/ReleaseNotes.html>`_. All LLVM releases may
be downloaded from the `LLVM releases web site <https://llvm.org/releases/>`_.

For more information about libc++, please see the `Libc++ Web Site
<https://libcxx.llvm.org>`_ or the `LLVM Web Site <https://llvm.org>`_.


What's New in Libc++ 9.0.0?
===========================

Fixes
-----

* Minor fixes to ``std::chrono`` operators.
* libc++ now correctly handles Objective-C++ ARC qualifiers in ``std::is_pointer``.
* ``std::span`` general updates and fixes.
* Updates to the ``std::abs`` implementation.
* ``std::to_chars`` now adds leading zeros.
* Ensure ``std::tuple`` is trivially constructible.
* ``std::aligned_union`` now works in C++03.
* Output of nullptr to ``std::basic_ostream`` is formatted properly.

Features
--------

* Implemented P0608: sane variant converting constructor.
* Added ``ssize`` function.
* Added ``front`` and ``back`` methods in ``std::span``.
* ``std::is_unbounded_array`` and ``std::is_bounded_array`` added to type traits.
* ``std::atomic`` now includes many new features and specialization including improved Freestanding support.
* Added ``std::midpoint`` and ``std::lerp`` math functions.
* Added the function ``std::is_constant_evaluated``.
* Erase-like algorithms now return size type.
* Added ``contains`` method to container types.
* ``std::swap`` is now a constant expression.

Updates
-------

* libc++ dropped support for GCC 4.9; we now support GCC 5.1 and above.
* libc++ added explicit support for WebAssembly System Interface (WASI).
* Progress towards full support of rvalues and variadics in C++03 mode. ``std::move`` and ``std::forward`` now both work in C++03 mode.
