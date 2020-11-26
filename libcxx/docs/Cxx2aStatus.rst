.. _cxx2a-status:

================================
libc++ C++2a Status
================================

.. include:: Styles.rst

.. contents::
   :local:


Overview
================================

In July 2017, the C++ standard committee created a draft for the next version of the C++ standard, known here as "C++2a" (probably to be C++20).

This page shows the status of libc++; the status of clang's support of the language features is `here <https://clang.llvm.org/cxx_status.html#cxx2a>`__.

.. attention:: Features in unreleased drafts of the standard are subject to change.

The groups that have contributed papers:

-  CWG - Core Language Working group
-  LWG - Library working group
-  SG1 - Study group #1 (Concurrency working group)

.. note:: "Nothing to do" means that no library changes were needed to implement this change.

.. _paper-status-cxx2a:

Paper Status
====================================

.. csv-table::
   :file: Cxx2aStatusPaperStatus.csv
   :widths: auto

.. note::

   .. [#note-P0202] P0202: The missing bits in P0202 are in ``copy`` and ``copy_backwards`` (and the ones that call them: ``copy_n``, ``set_union``, ``set_difference``, and ``set_symmetric_difference``). This is because the first two algorithms have specializations that call ``memmove`` which is not constexpr. See `Bug 25165 <https://bugs.llvm.org/show_bug.cgi?id=25165>`__
   .. [#note-P0600] P0600: The missing bits in P0600 are in |sect|\ [mem.res.class], |sect|\ [mem.poly.allocator.class], and |sect|\ [container.node.overview].

   .. [#note-P0619] P0619: Only ``std::allocator`` part is implemented.


.. _issues-status-cxx2a:

Library Working Group Issues Status
====================================

.. csv-table::
   :file: Cxx2aStatusIssuesStatus.csv
   :widths: auto

Last Updated: 24-Nov-2020
