.. _ReleaseProcedure:

=================
Release procedure
=================

The LLVM project creates a new release twice a year following a fixed
`schedule <https://llvm.org/docs/HowToReleaseLLVM.html#annual-release-schedule>`__.
This page describes the libc++ procedure for that release.

Prepare the release
===================

It should be finished before the Release managers start branching the new
release:

* Make sure ``libcxx/docs/ReleaseNotes/<VERSION>.rst`` is up to date. Typically
  this file is updated when contributing patches. Still there might be some
  information added regarding the general improvements of larger projects.

* Make sure the deprecated features on this page are up to date. Typically a
  new deprecated feature should be added to the release notes and this page.
  However this should be verified so removals won't get forgotten.

* Make sure the latest Unicode version is used. The C++ Standard
  `refers to the Unicode Standard <https://wg21.link/intro.refs#1.10>`__

  ``The Unicode Consortium. The Unicode Standard. Available from: https://www.unicode.org/versions/latest/``

  Typically the Unicode Consortium has one release per year. The libc++
  format library uses the Unicode Standard. Libc++ should be updated to the
  latest Unicode version. Updating means using the latest data files and, if
  needed, adapting the code to changes in the Unicode Standard.

* Make sure all libc++ supported compilers in the CI are updated to their
  latest release.

After the branch is created
===========================

After branching for an LLVM release:

1. Update ``_LIBCPP_VERSION`` in ``libcxx/include/__config``
2. Update the version number in ``libcxx/docs/conf.py``
3. Update ``_LIBCPPABI_VERSION`` in ``libcxxabi/include/cxxabi.h``
4. Update ``_LIBUNWIND_VERSION`` in ``libunwind/include/__libunwind_config.h``
5. Create a release notes file for the next release from the previous ones and point to it from
   ``libcxx/docs/ReleaseNotes.rst``. Remove entries that do not apply anymore, but keep in mind
   that some entries (such as upcoming deprecations) may still apply, may need rewording and may
   also require follow up PRs to implement.
6. Update the set of runners targeted by the CI on the release branch to ``llvm-premerge-libcxx-release-runners``, and
   make sure that runner set is using the appropriate image. This ensures that the release branch CI keeps working even
   if the main branch starts using newer images.
7. Update the pre-commit CI to use the new ToT version of Clang available from Compiler Explorer. In order
   to make sure patches can be backported to the release branch, we don't remove the oldest compiler yet.

Post release
============

Once the release is done and cherry-picks are not expected, we remove support for the ToT - 3 version Clang.
We also perform associated cleanups:

- Search for ``LLVM RELEASE`` and address their comments
- Search for test that have ``UNSUPPORTED`` or ``XFAIL`` for the no longer supported version
- Search for ``TODO(LLVM-<ToT>)`` and address their comments
