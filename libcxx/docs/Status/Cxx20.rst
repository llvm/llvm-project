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
   .. [#note-P0645] P0645: The implementation was complete since Clang 14,
      except the feature-test macro was not set until Clang 19.
   .. [#note-P0966] P0966: It was previously erroneously marked as complete in version 8.0. See `bug 45368 <https://llvm.org/PR45368>`__.
   .. [#note-P0619] P0619: Only sections D.8, D.9, D.10 and D.13 are implemented. Sections D.4, D.7, D.11, and D.12 remain undone.
   .. [#note-P0883.1] P0883: shared_ptr and floating-point changes weren't applied as they themselves aren't implemented yet.
   .. [#note-P0883.2] P0883: ``ATOMIC_FLAG_INIT`` was marked deprecated in version 14.0, but was undeprecated with the implementation of LWG3659 in version 15.0.
   .. [#note-P0660] P0660: The paper is implemented but the features are experimental and can be enabled via ``-fexperimental-library``.
   .. [#note-P1614] P1614: ``std::strong_order(long double, long double)`` is partly implemented.
   .. [#note-P0542] P0542: That paper was pulled out of the draft at the 2019-07 meeting in Cologne.
   .. [#note-P0788] P0788: That paper was pulled out of the draft at the 2019-07 meeting in Cologne.
   .. [#note-P0920] P0920: That paper was reverted by `P1661 <https://wg21.link/P1661>`__.
   .. [#note-P1424] P1424: That paper was superseded by `P1902 <https://wg21.link/P1902>`__.
   .. [#note-LWG2070] LWG2070: That LWG issue was resolved by `P0674R1 <https://wg21.link/P0674R1>`__.
   .. [#note-LWG2499] LWG2499: That LWG issue was resolved by `P0487R1 <https://wg21.link/P0487R1>`__.
   .. [#note-LWG2797] LWG2797: That LWG issue was resolved by `P1285R0 <https://wg21.link/P1285R0>`__.
   .. [#note-LWG3022] LWG3022: That LWG issue was resolved by `P1285R0 <https://wg21.link/P1285R0>`__.
   .. [#note-LWG3134] LWG3134: That LWG issue was resolved by `P1210R0 <https://wg21.link/P1210R0>`__.
   .. [#note-P0355] P0355: The implementation status is:

      * ``Calendars`` mostly done in Clang 7
      * ``Input parsers`` not done
      * ``Stream output`` Obsolete due to `P1361R2 <https://wg21.link/P1361R2>`_ "Integration of chrono with text formatting"
      * ``Time zone and leap seconds`` In Progress
      * ``TAI clock`` not done
      * ``GPS clock`` not done
      * ``UTC clock`` not done

.. _issues-status-cxx20:

Library Working Group Issues Status
====================================

.. csv-table::
   :file: Cxx20Issues.csv
   :header-rows: 1
   :widths: auto
