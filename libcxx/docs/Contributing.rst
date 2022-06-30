.. _ContributingToLibcxx:

======================
Contributing to libc++
======================

This file contains notes about various tasks and processes specific to contributing
to libc++. If this is your first time contributing, please also read `this document
<https://www.llvm.org/docs/Contributing.html>`__ on general rules for contributing to LLVM.

For libc++, please make sure you follow `these instructions <https://www.llvm.org/docs/Phabricator.html#requesting-a-review-via-the-command-line>`_
for submitting a code review from the command-line using ``arc``, since we have some
automation (e.g. CI) that depends on the review being submitted that way.

If you plan on contributing to libc++, it can be useful to join the ``#libcxx`` channel
on `LLVM's Discord server <https://discord.gg/jzUbyP26tQ>`__.

Looking for pre-existing reviews
================================

Before you start working on any feature, please take a look at the open reviews
to avoid duplicating someone else's work. You can do that by going to the website
where code reviews are held, `Differential <https://reviews.llvm.org/differential>`__,
and clicking on ``Libc++ Open Reviews`` in the sidebar to the left. If you see
that your feature is already being worked on, please consider chiming in instead
of duplicating work!

Pre-commit check list
=====================

Before committing or creating a review, please go through this check-list to make
sure you don't forget anything:

- Do you have tests for every public class and/or function you're adding or modifying?
- Did you update the synopsis of the relevant headers?
- Did you update the relevant files to track implementation status (in ``docs/Status/``)?
- Did you mark all functions and type declarations with the :ref:`proper visibility macro <visibility-macros>`?
- If you added a header:

  - Did you add it to ``include/module.modulemap.in``?
  - Did you add it to ``include/CMakeLists.txt``?
  - If it's a public header, did you add a test under ``test/libcxx`` that the new header defines ``_LIBCPP_VERSION``? See ``test/libcxx/algorithms/version.pass.cpp`` for an example. NOTE: This should be automated.
  - If it's a public header, did you update ``utils/generate_header_inclusion_tests.py``?

- Did you add the relevant feature test macro(s) for your feature? Did you update the ``generate_feature_test_macro_components.py`` script with it?
- Did you run the ``libcxx-generate-files`` target and verify its output?

The review process
==================

After uploading your patch, you should see that the "libc++" review group is automatically
added as a reviewer for your patch. Once the group is marked as having approved your patch,
you can commit it. However, if you get an approval very quickly for a significant patch,
please try to wait a couple of business days before committing to give the opportunity for
other reviewers to chime in. If you need someone else to commit the patch for you, please
mention it and provide your ``Name <email@domain>`` for us to attribute the commit properly.

Note that the rule for accepting as the "libc++" review group is to wait for two members
of the group to have approved the patch, excluding the patch author. This is not a hard
rule -- for very simple patches, use your judgement. The `"libc++" review group <https://reviews.llvm.org/project/members/64/>`__
consists of frequent libc++ contributors with a good understanding of the project's
guidelines -- if you would like to be added to it, please reach out on Discord.

Post-release check list
=======================

After branching for an LLVM release:

1. Update ``_LIBCPP_VERSION`` in ``libcxx/include/__config``
2. Update the version number in ``libcxx/docs/conf.py``
3. Update ``_LIBCPPABI_VERSION`` in ``libcxxabi/include/cxxabi.h``
4. Update ``_LIBUNWIND_VERSION`` in ``libunwind/include/__libunwind_config.h``

Exporting new symbols from the library
======================================

When exporting new symbols from libc++, you must update the ABI lists located in ``lib/abi``.
To test whether the lists are up-to-date, please run the target ``check-cxx-abilist``.
To regenerate the lists, use the target ``generate-cxx-abilist``.
The ABI lists must be updated for all supported platforms; currently Linux and
Apple.  If you don't have access to one of these platforms, you can download an
updated list from the failed build at
`Buildkite <https://buildkite.com/llvm-project/libcxx-ci>`__.
Look for the failed build and select the ``artifacts`` tab. There, download the
abilist for the platform, e.g.:

* C++20 for the Linux platform.
* MacOS C++20 for the Apple platform.
