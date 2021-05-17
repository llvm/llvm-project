.. ranges-status:

================================
libc++ Format Status
================================

.. include:: Helpers/Styles.rst

.. contents::
   :local:


Overview
========

This document contains the status of the C++20 Format library in libc++. It is used to
track both the status of the sub-projects of the Format library and who is assigned to
these sub-projects. This is imperative to effective implementation so that work is not
duplicated and implementors are not blocked by each other.


If you are interested in contributing to the libc++ Format library, please send
a message to the #libcxx channel in the LLVM discord. Please *do not* start
working on any of the assigned items below.


Sub-Projects in the Format library
==================================

.. csv-table::
   :file: FormatProposalStatus.csv
   :header-rows: 1
   :widths: auto


Misc. Items and TODOs
=====================

(Please mark all Format-related TODO comments with the string ``TODO FMT``, so we
can find them easily.)

    * C++23 may break the ABI with `P2216 <https://wg21.link/P2216>`_.
      This ABI break may be backported to C++20. Therefore the library will not
      be available on platforms where the ABI break is an issue.


Paper and Issue Status
======================

.. csv-table::
   :file: FormatIssuePaperStatus.csv
   :header-rows: 1
   :widths: auto
