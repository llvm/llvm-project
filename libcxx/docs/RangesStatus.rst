.. ranges-status:

================================
libc++ Ranges Status
================================

.. include:: Helpers/Styles.rst

.. contents::
   :local:


Overview
================================

This document contains the status of the C++20 Ranges library in libc++. It is used to
track both the status of the sub-projects of the ranges library and who is assigned to
these sub-projects. This is imperative to effective implementation so that work is not
duplicated and implementors are not blocked by each other.

If you are interested in contributing to the libc++ Ranges library, please send a message
to the #libcxx channel in the LLVM discord. Please *do not* start working on any of the
assigned items below.


Sub-Projects in the One Ranges Proposal
=======================================

.. csv-table::
   :file: OneRangesProposalStatus.csv
   :header-rows: 1
   :widths: auto


Misc. Items and TODOs
====================================

(Note: files with required updates will contain the TODO at the beginning of the list item
so they can be easily found via global search.)

    * TODO(XX_SPACESHIP_CONCEPTS): when spaceship support is added to various STL types, we need to update some concept tests.

Paper and Issue Status
====================================

(Note: stolen from MSVC `here <https://github.com/microsoft/STL/issues/39>`_.)

.. csv-table::
   :file: RangesIssuePaperStatus.csv
   :header-rows: 1
   :widths: auto