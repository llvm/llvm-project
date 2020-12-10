.. _cxx1z-status:

================================
libc++ C++17 Status
================================

.. include:: Helpers/Styles.rst

.. contents::
   :local:


Overview
================================

In November 2014, the C++ standard committee created a draft for the next version of the C++ standard, initially known as "C++1z".
In February 2017, the C++ standard committee approved this draft, and sent it to ISO for approval as C++17.

This page shows the status of libc++; the status of clang's support of the language features is `here <https://clang.llvm.org/cxx_status.html#cxx17>`__.

.. attention:: Features in unreleased drafts of the standard are subject to change.

The groups that have contributed papers:

-  CWG - Core Language Working group
-  LWG - Library working group
-  SG1 - Study group #1 (Concurrency working group)

.. note:: "Nothing to do" means that no library changes were needed to implement this change.

.. _paper-status-cxx1z:

Paper Status
====================================

.. csv-table::
   :file: Cxx1zStatusPaperStatus.csv
   :header-rows: 1
   :widths: auto

.. note::

   .. [#note-P0433] P0433: So far, only the ``<string>``, sequence containers, container adaptors and ``<regex>`` portions of P0433 have been implemented.
   .. [#note-P0607] P0607: The parts of P0607 that are not done are the ``<regex>`` bits.


.. _issues-status-cxx1z:

Library Working Group Issues Status
====================================

.. csv-table::
   :file: Cxx1zStatusIssuesStatus.csv
   :header-rows: 1
   :widths: auto

Last Updated: 17-Nov-2020
