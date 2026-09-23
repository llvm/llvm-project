.. _NewStandardProcedure:

==========================
New standard procedure
==========================

Roughly every three years, WG21 finishes a version of the C++ standard and starts
working on the next one. This page describes the procedure that libc++ developers
must follow to introduce support for a new version of the standard in the library.
Since introducing a new Standard does not happen often, this document may not be
fully exhaustive and is meant as a starting point. Keep it up-to-date when drift
is noticed.

* Create status pages for tracking conformance of C++ZZ (``CxxZZIssues.csv``, ``CxxZZPapers.csv`` and related).
* Create the associated views in the `libc++ Conformance project <https://github.com/orgs/llvm/projects/31>`__.
* CI updates

  * Add a new job testing C++ZZ
  * Move jobs that specify the previous standard over to C++ZZ (except jobs which intend to test older
    standard specifically)

* Teach the test suite about C++ZZ (for example ``--param std=c++zz`` in the ``Lit`` configuration)
* Add files to track the transitive includes for C++ZZ
* Add a new version for ``_LIBCPP_STD_VER`` and ``TEST_STD_VER`` for the test suite

  * Note that we don't add various versioned macros until we need them (e.g. ``_LIBCPP_CONSTEXPR_SINCE_CXXZZ``)

* Feature-test macros:

  * Update the FTM generation script to account for C++ZZ
  * Add any missing FTMs for the new standard version in the FTM generation script
  * Regenerate the FTM files
  * Update the tests for the FTM generation script itself
