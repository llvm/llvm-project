==========================
Libc++ 8.0.0 Release Notes
==========================

.. contents::
   :local:
   :depth: 2

Written by the `Libc++ Team <https://libcxx.llvm.org>`_

Introduction
============

This document contains the release notes for the libc++ C++ Standard Library,
part of the LLVM Compiler Infrastructure, release 8.0.0. Here we describe the
status of libc++ in some detail, including major improvements from the previous
release and new feature work. For the general LLVM release notes, see `the LLVM
documentation <https://llvm.org/docs/ReleaseNotes.html>`_. All LLVM releases may
be downloaded from the `LLVM releases web site <https://releases.llvm.org/>`_.

For more information about libc++, please see the `Libc++ Web Site
<https://libcxx.llvm.org>`_ or the `LLVM Web Site <https://llvm.org>`_.

What's New in Libc++ 8.0.0?
===========================

API Changes
-----------
- Building libc++ for Mac OSX 10.6 is not supported anymore.
- Starting with LLVM 8.0.0, users that wish to link together translation units
  built with different versions of libc++'s headers into the same final linked
  image MUST define the _LIBCPP_HIDE_FROM_ABI_PER_TU macro to 1 when building
  those translation units. Not defining _LIBCPP_HIDE_FROM_ABI_PER_TU to 1 and
  linking translation units built with different versions of libc++'s headers
  together may lead to ODR violations and ABI issues. On the flipside, code
  size improvements should be expected for everyone not defining the macro.
- Starting with LLVM 8.0.0, std::dynarray has been removed from the library.
  std::dynarray was a feature proposed for C++14 that was pulled from the
  Standard at the last minute and was never standardized. Since there are no
  plans to standardize this facility it is being removed.
- Starting with LLVM 8.0.0, std::bad_array_length has been removed from the
  library. std::bad_array_length was a feature proposed for C++14 alongside
  std::dynarray, but it never actually made it into the C++ Standard. There
  are no plans to standardize this feature at this time. Formally speaking,
  this removal constitutes an ABI break because the symbols were shipped in
  the shared library. However, on macOS systems, the feature was not usable
  because it was hidden behind availability annotations. We do not expect
  any actual breakage to happen from this change.
