=========================================
Libc++ 17.0.0 (In-Progress) Release Notes
=========================================

.. contents::
   :local:
   :depth: 2

Written by the `Libc++ Team <https://libcxx.llvm.org>`_

.. warning::

   These are in-progress notes for the upcoming libc++ 17 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the libc++ C++ Standard Library,
part of the LLVM Compiler Infrastructure, release 17.0.0. Here we describe the
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

What's New in Libc++ 17.0.0?
============================

Implemented Papers
------------------
- P2520R0 - ``move_iterator<T*>`` should be a random access iterator
- P1328R1 - ``constexpr type_info::operator==()``

Improvements and New Features
-----------------------------
- ``std::equal`` and ``std::ranges::equal`` are now forwarding to ``std::memcmp`` for integral types and pointers,
  which can lead up to 40x performance improvements.

- ``std::string_view`` now provides iterators that check for out-of-bounds accesses when the safe
  libc++ mode is enabled.

Deprecations and Removals
-------------------------

- The ``<experimental/coroutine>`` header has been removed in this release. The ``<coroutine>`` header
  has been shipping since LLVM 14, so the Coroutines TS implementation is being removed per our policy
  for removing TSes.

- Several incidental transitive includes have been removed from libc++. Those
  includes are removed based on the language version used. Incidental transitive
  inclusions of the following headers have been removed:

  - C++2b: ``bit``, ``cstring``, ``type_traits``

- The headers ``<experimental/algorithm>`` and ``<experimental/functional>`` have been removed, since all the contents
  have been implemented in namespace ``std`` for at least two releases.

- The formatter specialization ``template<size_t N> struct formatter<const charT[N], charT>``
  has been removed. Since libc++'s format library was marked experimental there
  is no backwards compatibility option. This specialization has been removed
  from the Standard since it was never used, the proper specialization to use
  instead is ``template<size_t N> struct formatter<charT[N], charT>``.

Upcoming Deprecations and Removals
----------------------------------

- The ``_LIBCPP_AVAILABILITY_CUSTOM_VERBOSE_ABORT_PROVIDED`` macro will not be honored anymore in LLVM 18.
  Please see the updated documentation about the safe libc++ mode and in particular the ``_LIBCPP_VERBOSE_ABORT``
  macro for details.

- The headers ``<experimental/deque>``, ``<experimental/forward_list>``, ``<experimental/list>``,
  ``<experimental/map>``, ``<experimental/memory_resource>``, ``<experimental/regex>``, ``<experimental/set>``,
  ``<experimental/string>``, ``<experimental/unordered_map>``, ``<experimental/unordered_set>``,
  and ``<experimental/vector>`` will be removed in LLVM 18, as all their contents will have been implemented in
  namespace ``std`` for at least two releases.

API Changes
-----------

ABI Affecting Changes
---------------------

Build System Changes
--------------------
