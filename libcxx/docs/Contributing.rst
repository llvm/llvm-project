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

  - Did you add it to ``include/module.modulemap``?
  - Did you add it to ``include/CMakeLists.txt``?
  - If it's a public header, did you add a test under ``test/libcxx`` that the new header defines ``_LIBCPP_VERSION``? See ``test/libcxx/algorithms/version.pass.cpp`` for an example. NOTE: This should be automated.
  - If it's a public header, did you update ``utils/generate_header_inclusion_tests.py``?

- Did you add the relevant feature test macro(s) for your feature? Did you update the ``generate_feature_test_macro_components.py`` script with it?
- Did you run the ``libcxx-generate-files`` target and verify its output?

Post-release check list
=======================

After branching for an LLVM release:

1. Update ``_LIBCPP_VERSION`` in ``include/__config``
2. Update the ``include/__libcpp_version`` file
3. Update the version number in ``docs/conf.py``

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

Working on large features
=========================

Libc++ makes no guarantees about the implementation status or the ABI stability
of features that have not yet been ratified in the C++ Standard. After the C++
Standard is ratified libc++ promises a conforming and ABI-stable
implementation. When working on a large new feature in the ratified version of
the C++ Standard that can't be finished before the next release branch is
created, we can't honor this promise. Another reason for not being able to
promise ABI stability happens when the C++ Standard committee retroactively
accepts ABI breaking papers as defect reports against the ratified C++
Standard.

When working on these features it should be possible for libc++ vendors to
disable these incomplete features, so they can promise ABI stability to their
customers. This is done by the CMake option
``LIBCXX_ENABLE_INCOMPLETE_FEATURES``. When start working on a large feature
the following steps are required to guard the new library with the CMake
option.

* ``libcxx/CMakeLists.txt``: Add

  .. code-block:: cmake

    config_define_if_not(LIBCXX_ENABLE_INCOMPLETE_FEATURES _LIBCPP_HAS_NO_INCOMPLETE_FOO)

* ``libcxx/include/__config_site.in``: Add

  .. code-block:: c++

    #cmakedefine _LIBCPP_HAS_NO_INCOMPLETE_FOO

* ``libcxx/include/foo``: The contents of the file should be guarded in an
  ``ifdef`` and always include ``<version>``

  .. code-block:: c++

    #ifndef _LIBCPP_FOO
    #define _LIBCPP_FOO

    // Make sure all feature-test macros are available.
    #include <version>
    // Enable the contents of the header only when libc++ was built with LIBCXX_ENABLE_INCOMPLETE_FEATURES.
    #if !defined(_LIBCPP_HAS_NO_INCOMPLETE_FOO)

    ...

    #endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_FO0)
    #endif // _LIBCPP_FOO

* ``libcxx/src/CMakeLists.txt``: When the library has a file ``foo.cpp`` it
  should only be added when ``LIBCXX_ENABLE_INCOMPLETE_FEATURES`` is enabled

  .. code-block:: cmake

    if(LIBCXX_ENABLE_INCOMPLETE_FEATURES)
      list(APPEND LIBCXX_SOURCES
        foo.cpp
      )
    endif()

* ``libcxx/utils/generate_feature_test_macro_components.py``: Add to
  ``lit_markup``

  .. code-block:: python

    "foo": ["UNSUPPORTED: libcpp-has-no-incomplete-foo"],

* ``libcxx/utils/generate_header_inclusion_tests.py``: Add to ``lit_markup``

  .. code-block:: python

    "foo": ["UNSUPPORTED: libcpp-has-no-incomplete-foo"],

* ``libcxx/utils/generate_header_tests.py``: Add to ``header_markup``

  .. code-block:: python

    "foo": ["ifndef _LIBCPP_HAS_NO_INCOMPLETE_FOO"],

* ``libcxx/utils/libcxx/test/features.py``: Add to ``macros``

  .. code-block:: python

    '_LIBCPP_HAS_NO_INCOMPLETE_FOO': 'libcpp-has-no-incomplete-foo',

* All tests that include ``<foo>`` should contain

  .. code-block:: c++

    // UNSUPPORTED: libcpp-has-no-incomplete-foo

Once the library is complete these changes and guards should be removed.
