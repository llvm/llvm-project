.. _cxx2a-status:

================================
libc++ C++20 Status
================================

.. include:: Helpers/Styles.rst

.. contents::
   :local:


Overview
================================

In July 2017, the C++ standard committee created a draft for the next version of the C++ standard, initially known as "C++2a".
In September 2020, the C++ standard committee approved this draft, and sent it to ISO for approval as C++20.

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
   :header-rows: 1
   :widths: auto

.. note::

   .. [#note-P0600] P0600: The missing bits in P0600 are in |sect|\ [mem.res.class], |sect|\ [mem.poly.allocator.class], and |sect|\ [container.node.overview].
   .. [#note-P0966] P0966: It was previously erroneously marked as complete in version 8.0. See `bug 45368 <https://llvm.org/PR45368>`__.

   .. [#note-P0619] P0619: Only ``std::allocator`` part is implemented.


.. _issues-status-cxx2a:

Library Working Group Issues Status
====================================

.. csv-table::
   :file: Cxx2aStatusIssuesStatus.csv
   :header-rows: 1
   :widths: auto

Last Updated: 24-Nov-2020
