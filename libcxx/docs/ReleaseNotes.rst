=========================================
Libc++ 12.0.0 (In-Progress) Release Notes
=========================================

.. contents::
   :local:
   :depth: 2

Written by the `Libc++ Team <https://libcxx.llvm.org>`_

.. warning::

   These are in-progress notes for the upcoming libc++ 12 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the libc++ C++ Standard Library,
part of the LLVM Compiler Infrastructure, release 12.0.0. Here we describe the
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

What's New in Libc++ 12.0.0?
============================

New Features
------------

- ...

API Changes
-----------
- By default, libc++ will _not_ include the definition for new and delete,
  since those are provided in libc++abi. Vendors wishing to provide new and
  delete in libc++ can build the library with ``-DLIBCXX_ENABLE_NEW_DELETE_DEFINITIONS=ON``
  to get back the old behavior. This was done to avoid providing new and delete
  in both libc++ and libc++abi, which is technically an ODR violation. Also
  note that we couldn't decide to put the operators in libc++ only, because
  they are needed from libc++abi (which would create a circular dependency).
