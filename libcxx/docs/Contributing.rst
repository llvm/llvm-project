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

* C++<version>.
* MacOS X86_64 and MacOS arm64 for the Apple platform.


Pre-commit CI
=============

Introduction
------------

Unlike most parts of the LLVM project, libc++ uses a pre-commit CI [#]_. This
CI is hosted on `Buildkite <https://buildkite.com/llvm-project/libcxx-ci>`__ and
the build results are visible in the review on Phabricator. Please make sure
the CI is green before committing a patch.

The CI tests libc++ for all :ref:`supported platforms <SupportedPlatforms>`.
The build is started for every diff uploaded to Phabricator. A complete CI run
takes approximately one hour. To reduce the load:

* The build is cancelled when a new diff for the same revision is uploaded.
* The build is done in several stages and cancelled when a stage fails.

Typically, the libc++ jobs use a Ubuntu Docker image. This image contains
recent `nightly builds <https://apt.llvm.org>`__ of all supported versions of
Clang and the current version of the ``main`` branch. These versions of Clang
are used to build libc++ and execute its tests.

Unless specified otherwise, the configurations:

* use a nightly build of the ``main`` branch of Clang,
* execute the tests using the language C++<latest>. This is the version
  "developed" by the C++ committee.

.. note:: Updating the Clang nightly builds in the Docker image is a manual
   process and is done at an irregular interval on purpose. When you need to
   have the latest nightly build to test recent Clang changes, ask in the
   ``#libcxx`` channel on `LLVM's Discord server
   <https://discord.gg/jzUbyP26tQ>`__.

.. [#] There's `LLVM Dev Meeting talk <https://www.youtube.com/watch?v=B7gB6van7Bw>`__
   explaining the benefits of libc++'s pre-commit CI.

Builds
------

Below is a short description of the most interesting CI builds [#]_:

* ``Format`` runs ``clang-format`` and uploads its output as an artifact. At the
  moment this build is a soft error and doesn't fail the build.
* ``Generated output`` runs the ``libcxx-generate-files`` build target and
  tests for non-ASCII characters in libcxx. Some files are excluded since they
  use Unicode, mainly tests. The output of these commands are uploaded as
  artifact.
* ``Documentation`` builds the documentation. (This is done early in the build
  process since it is cheap to run.)
* ``C++<version>`` these build steps test the various C++ versions, making sure all
  C++ language versions work with the changes made.
* ``Clang <version>`` these build steps test whether the changes work with all
  supported Clang versions.
* ``Booststrapping build`` builds Clang using the revision of the patch and
  uses that Clang version to build and test libc++. This validates the current
  Clang and lib++ are compatible.

  When a crash occurs in this build, the crash reproducer is available as an
  artifact.

* ``Modular build`` tests libc++ using Clang modules [#]_.
* ``GCC <version>`` tests libc++ with the latest stable GCC version. Only C++11
  and the latest C++ version are tested.
* ``Santitizers`` tests libc++ using the Clang sanitizers.
* ``Parts disabled`` tests libc++ with certain libc++ features disabled.
* ``Windows`` tests libc++ using MinGW and clang-cl.
* ``Apple`` tests libc++ on MacOS.
* ``ARM`` tests libc++ on various Linux ARM platforms.
* ``AIX`` tests libc++ on AIX.

.. [#] Not all all steps are listed: steps are added and removed when the need
   arises.
.. [#] Clang modules are not the same as C++20's modules.

Infrastructure
--------------

All files of the CI infrastructure are in the directory ``libcxx/utils/ci``.
Note that quite a bit of this infrastructure is heavily Linux focused. This is
the platform used by most of libc++'s Buildkite runners and developers.

Dockerfile
~~~~~~~~~~

Contains the Docker image for the Ubuntu CI. Because the same Docker image is
used for the ``main`` and ``release`` branch, it should contain no hard-coded
versions.  It contains the used versions of Clang, various clang-tools,
GCC, and CMake.

.. note:: This image is pulled from Docker hub and not rebuild when changing
   the Dockerfile.

run-buildbot-container
~~~~~~~~~~~~~~~~~~~~~~

Helper script that pulls and runs the Docker image. This image mounts the LLVM
monorepo at ``/llvm``. This can be used to test with compilers not available on
your system.

run-buildbot
~~~~~~~~~~~~

Contains the buld script executed on Buildkite. This script can be executed
locally or inside ``run-buildbot-container``. The script must be called with
the target to test. For example, ``run-buildbot generic-cxx20`` will build
libc++ and test it using C++20.

.. warning:: This script will overwrite the directory ``<llvm-root>/build/XX``
  where ``XX`` is the target of ``run-buildbot``.

This script contains as little version information as possible. This makes it
easy to use the script with a different compiler. This allows testing a
combination not in the libc++ CI. It can be used to add a new (temporary)
job to the CI. For example, testing the C++17 build with Clang-14 can be done
like:

.. code-block:: bash

  CC=clang-14 CXX=clang++-14 run-buildbot generic-cxx17

buildkite-pipeline.yml
~~~~~~~~~~~~~~~~~~~~~~

Contains the jobs executed in the CI. This file contains the version
information of the jobs being executed. Since this script differs between the
``main`` and ``release`` branch, both branches can use different compiler
versions.
