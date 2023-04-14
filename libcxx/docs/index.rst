.. _index:

=============================
"libc++" C++ Standard Library
=============================

Overview
========

libc++ is a new implementation of the C++ standard library, targeting C++11 and
above.

* Features and Goals

  * Correctness as defined by the C++11 standard.
  * Fast execution.
  * Minimal memory use.
  * Fast compile times.
  * ABI compatibility with gcc's libstdc++ for some low-level features
    such as exception objects, rtti and memory allocation.
  * Extensive unit tests.

* Design and Implementation:

  * Extensive unit tests
  * Internal linker model can be dumped/read to textual format
  * Additional linking features can be plugged in as "passes"
  * OS specific and CPU specific code factored out


Getting Started with libc++
===========================

.. toctree::
   :maxdepth: 1

   ReleaseNotes
   UsingLibcxx
   BuildingLibcxx
   TestingLibcxx
   Contributing
   Status/Cxx14
   Status/Cxx17
   Status/Cxx20
   Status/Cxx2b
   Status/Format
   Status/Parallelism
   Status/Ranges
   Status/Spaceship
   Status/Zip


.. toctree::
    :hidden:

    AddingNewCIJobs
    FeatureTestMacroTable


Current Status
==============

After its initial introduction, many people have asked "why start a new
library instead of contributing to an existing library?" (like Apache's
libstdcxx, GNU's libstdc++, STLport, etc).  There are many contributing
reasons, but some of the major ones are:

* From years of experience (including having implemented the standard
  library before), we've learned many things about implementing
  the standard containers which require ABI breakage and fundamental changes
  to how they are implemented.  For example, it is generally accepted that
  building std::string using the "short string optimization" instead of
  using Copy On Write (COW) is a superior approach for multicore
  machines (particularly in C++11, which has rvalue references).  Breaking
  ABI compatibility with old versions of the library was
  determined to be critical to achieving the performance goals of
  libc++.

* Mainline libstdc++ has switched to GPL3, a license which the developers
  of libc++ cannot use.  libstdc++ 4.2 (the last GPL2 version) could be
  independently extended to support C++11, but this would be a fork of the
  codebase (which is often seen as worse for a project than starting a new
  independent one).  Another problem with libstdc++ is that it is tightly
  integrated with G++ development, tending to be tied fairly closely to the
  matching version of G++.

* STLport and the Apache libstdcxx library are two other popular
  candidates, but both lack C++11 support.  Our experience (and the
  experience of libstdc++ developers) is that adding support for C++11 (in
  particular rvalue references and move-only types) requires changes to
  almost every class and function, essentially amounting to a rewrite.
  Faced with a rewrite, we decided to start from scratch and evaluate every
  design decision from first principles based on experience.
  Further, both projects are apparently abandoned: STLport 5.2.1 was
  released in Oct'08, and STDCXX 4.2.1 in May'08.

.. _SupportedPlatforms:

Platform and Compiler Support
=============================

Libc++ aims to support common compilers that implement the C++11 Standard. In order to strike a
good balance between stability for users and maintenance cost, testing coverage and development
velocity, libc++ drops support for older compilers as newer ones are released.

============ =============== ========================== =====================
Compiler     Versions        Restrictions               Support policy
============ =============== ========================== =====================
Clang        15, 16, 17-git                             latest two stable releases per `LLVM's release page <https://releases.llvm.org>`_ and the development version
AppleClang   14                                         latest stable release per `Xcode's release page <https://developer.apple.com/documentation/xcode-release-notes>`_
Open XL      17.1 (AIX)                                 latest stable release per `Open XL's documentation page <https://www.ibm.com/docs/en/openxl-c-and-cpp-aix>`_
GCC          12              In C++11 or later only     latest stable release per `GCC's release page <https://gcc.gnu.org/releases.html>`_
============ =============== ========================== =====================

Libc++ also supports common platforms and architectures:

=============== ========================= ============================
Target platform Target architecture       Notes
=============== ========================= ============================
macOS 10.9+     i386, x86_64, arm64       Building the shared library itself requires targetting macOS 10.13+
FreeBSD 12+     i386, x86_64, arm
Linux           i386, x86_64, arm, arm64  Only glibc-2.24 and later and no other libc is officially supported
Windows         i386, x86_64              Both MSVC and MinGW style environments
AIX 7.2TL5+     powerpc, powerpc64
=============== ========================= ============================

Generally speaking, libc++ should work on any platform that provides a fairly complete
C Standard Library. It is also possible to turn off parts of the library for use on
systems that provide incomplete support.

However, libc++ aims to provide a high-quality implementation of the C++ Standard
Library, especially when it comes to correctness. As such, we aim to have test coverage
for all the platforms and compilers that we claim to support. If a platform or compiler
is not listed here, it is not officially supported. It may happen to work, and
in practice the library is known to work on some platforms not listed here, but
we don't make any guarantees. If you would like your compiler and/or platform
to be formally supported and listed here, please work with the libc++ team to set
up testing for your configuration.


C++ Dialect Support
===================

* C++11 - Complete
* :ref:`C++14 - Complete <cxx14-status>`
* :ref:`C++17 - In Progress <cxx17-status>`
* :ref:`C++20 - In Progress <cxx20-status>`
* :ref:`C++2b - In Progress <cxx2b-status>`
* :ref:`C++ Feature Test Macro Status <feature-status>`


Notes and Known Issues
======================

This list contains known issues with libc++

* Building libc++ with ``-fno-rtti`` is not supported. However
  linking against it with ``-fno-rtti`` is supported.


A full list of currently open libc++ bugs can be `found here`__.

.. __:  https://github.com/llvm/llvm-project/labels/libc%2B%2B


Design Documents
================

.. toctree::
   :maxdepth: 1

   DesignDocs/ABIVersioning
   DesignDocs/AtomicDesign
   DesignDocs/CapturingConfigInfo
   DesignDocs/DebugMode
   DesignDocs/ExperimentalFeatures
   DesignDocs/ExtendedCXX03Support
   DesignDocs/FeatureTestMacros
   DesignDocs/FileTimeType
   DesignDocs/HeaderRemovalPolicy
   DesignDocs/NoexceptPolicy
   DesignDocs/ThreadingSupportAPI
   DesignDocs/UniquePtrTrivialAbi
   DesignDocs/UnspecifiedBehaviorRandomization
   DesignDocs/VisibilityMacros


Build Bots and Test Coverage
============================

* `Buildkite CI pipeline <https://buildkite.com/llvm-project/libcxx-ci>`_
* `LLVM Buildbot Builders <https://lab.llvm.org/buildbot>`_
* :ref:`Adding New CI Jobs <AddingNewCIJobs>`


Getting Involved
================

First please review our `Developer's Policy <https://llvm.org/docs/DeveloperPolicy.html>`__
and `Getting started with LLVM <https://llvm.org/docs/GettingStarted.html>`__.

**Bug Reports**

If you think you've found a bug in libc++, please report it using
the `LLVM bug tracker`_. If you're not sure, you
can ask for support on the `libcxx forum`_ or on IRC.

**Patches**

If you want to contribute a patch to libc++, the best place for that is
`Phabricator <https://llvm.org/docs/Phabricator.html>`_. Please add `libcxx-commits` as a subscriber.
Also make sure you are subscribed to the `libcxx-commits mailing list`_.

**Discussion and Questions**

Send discussions and questions to the `libcxx forum`_.


Quick Links
===========
* `LLVM Homepage <https://llvm.org/>`_
* `libc++abi Homepage <http://libcxxabi.llvm.org/>`_
* `LLVM Bug Tracker <https://github.com/llvm/llvm-project/labels/libc++/>`_
* `libcxx-commits Mailing List <http://lists.llvm.org/mailman/listinfo/libcxx-commits>`_
* `libcxx Forum <https://discourse.llvm.org/c/runtimes/libcxx/>`_
* `Browse libc++ Sources <https://github.com/llvm/llvm-project/tree/main/libcxx/>`_
