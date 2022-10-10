.. _cxx20-status:

================================
libc++ C++20 Status
================================

.. include:: ../Helpers/Styles.rst

.. contents::
   :local:


Overview
================================

In July 2017, the C++ standard committee created a draft for the next version of the C++ standard, initially known as "C++2a".
In September 2020, the C++ standard committee approved this draft, and sent it to ISO for approval as C++20.

This page shows the status of libc++; the status of clang's support of the language features is `here <https://clang.llvm.org/cxx_status.html#cxx20>`__.

.. attention:: Features in unreleased drafts of the standard are subject to change.

The groups that have contributed papers:

-  CWG - Core Language Working group
-  LWG - Library working group
-  SG1 - Study group #1 (Concurrency working group)

.. note:: "Nothing to do" means that no library changes were needed to implement this change.

.. _paper-status-cxx20:

Paper Status
====================================

.. csv-table::
   :file: Cxx20Papers.csv
   :header-rows: 1
   :widths: auto

.. note::

   .. [#note-P0591] P0591: The changes in [mem.poly.allocator.mem] are missing.
   .. [#note-P0600] P0600: The missing bits in P0600 are in |sect|\ [mem.res.class] and |sect|\ [mem.poly.allocator.class].
   .. [#note-P0645] P0645: The paper is implemented but still marked as an incomplete feature
      (the feature-test macro is not set and the libary is only available when built with ``-fexperimental-library``).
      Not yet implemented LWG-issues will cause API and ABI breakage.
   .. [#note-P0966] P0966: It was previously erroneously marked as complete in version 8.0. See `bug 45368 <https://llvm.org/PR45368>`__.
   .. [#note-P0619] P0619: Only sections D.8, D.9, D.10 and D.13 are implemented. Sections D.4, D.7, D.11, D.12, and D.14 remain undone.
   .. [#note-P0883.1] P0883: shared_ptr and floating-point changes weren't applied as they themselves aren't implemented yet.
   .. [#note-P0883.2] P0883: ``ATOMIC_FLAG_INIT`` was marked deprecated in version 14.0, but was undeprecated with the implementation of LWG3659 in version 15.0.
   .. [#note-P2231] P2231: Optional is complete. The changes to variant haven't been implemented yet.
   .. [#note-P0482] P0482: Declarations of ``std::pmr::u8string`` and ``std::hash<std::pmr::u8string>`` are blocked pending implementation
      of polymorphic allocator support from P0220.


.. _issues-status-cxx20:

Library Working Group Issues Status
====================================

.. csv-table::
   :file: Cxx20Issues.csv
   :header-rows: 1
   :widths: auto
