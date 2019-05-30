.. _index:

=======================
libunwind LLVM Unwinder
=======================

Overview
========

libunwind is an implementation of the interface defined by the HP libunwind
project. It was contributed by Apple as a way to enable clang++ to port to
platforms that do not have a system unwinder. It is intended to be a small and
fast implementation of the ABI, leaving off some features of HP's libunwind
that never materialized (e.g. remote unwinding).

The unwinder has two levels of API. The high level APIs are the `_Unwind_*`
functions which implement functionality required by `__cxa_*` exception
functions. The low level APIs are the `unw_*` functions which are an interface
defined by the old HP libunwind project.

Getting Started with libunwind
------------------------------

.. toctree::
   :maxdepth: 2

   BuildingLibunwind

Current Status
--------------

libunwind is a production-quality unwinder, with platform support for DWARF
unwind info, SjLj, and ARM EHABI.

The low level libunwind API was designed to work either in-process (aka local)
or to operate on another process (aka remote), but only the local path has been
implemented. Remote unwinding remains as future work.

Platform and Compiler Support
-----------------------------

libunwind is known to work on the following platforms:

============ ======================== ============ ========================
OS           Arch                     Compilers    Unwind Info
============ ======================== ============ ========================
Any          i386, x86_64, ARM        Clang        SjLj
Bare Metal   ARM                      Clang, GCC   EHABI
FreeBSD      i386, x86_64, ARM64      Clang        DWARF CFI
iOS          ARM                      Clang        SjLj
Linux        ARM                      Clang, GCC   EHABI
Linux        i386, x86_64, ARM64      Clang, GCC   DWARF CFI
macOS        i386, x86_64             Clang, GCC   DWARF CFI
NetBSD       x86_64                   Clang, GCC   DWARF CFI
Windows      i386, x86_64, ARM, ARM64 Clang        DWARF CFI
============ ======================== ============ ========================

The following minimum compiler versions are strongly recommended.

* Clang 3.5 and above
* GCC 4.7 and above.

Anything older *may* work.

Notes and Known Issues
----------------------

* TODO


Getting Involved
================

First please review our `Developer's Policy <http://llvm.org/docs/DeveloperPolicy.html>`__
and `Getting started with LLVM <http://llvm.org/docs/GettingStarted.html>`__.

**Bug Reports**

If you think you've found a bug in libunwind, please report it using
the `LLVM Bugzilla`_. If you're not sure, you
can post a message to the `cfe-dev mailing list`_ or on IRC.
Please include "libunwind" in your subject.

**Patches**

If you want to contribute a patch to libunwind, the best place for that is
`Phabricator <http://llvm.org/docs/Phabricator.html>`_. Please include [libunwind] in the subject and
add `cfe-commits` as a subscriber. Also make sure you are subscribed to the
`cfe-commits mailing list <http://lists.llvm.org/mailman/listinfo/cfe-commits>`_.

**Discussion and Questions**

Send discussions and questions to the
`cfe-dev mailing list <http://lists.llvm.org/mailman/listinfo/cfe-dev>`_.
Please include [libunwind] in the subject.


Quick Links
===========
* `LLVM Homepage <http://llvm.org/>`_
* `LLVM Bugzilla <https://bugs.llvm.org/>`_
* `cfe-commits Mailing List`_
* `cfe-dev Mailing List`_
* `Browse libunwind Sources <https://github.com/llvm/llvm-project/blob/master/libunwind/>`_
