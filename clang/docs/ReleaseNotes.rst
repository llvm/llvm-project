=========================
Clang 4.0.0 Release Notes
=========================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <http://llvm.org/>`_

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C/OpenCL
frontend, part of the LLVM Compiler Infrastructure, release 4.0.0. Here we
describe the status of Clang in some detail, including major
improvements from the previous release and new feature work. For the
general LLVM release notes, see `the LLVM
documentation <http://llvm.org/docs/ReleaseNotes.html>`_. All LLVM
releases may be downloaded from the `LLVM releases web
site <http://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about
the latest release, please check out the main please see the `Clang Web
Site <http://clang.llvm.org>`_ or the `LLVM Web
Site <http://llvm.org>`_.

What's New in Clang 4.0.0?
==========================

Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

Major New Features
------------------

- The `diagnose_if <AttributeReference.html#diagnose-if>`_ attribute has been
  added to clang. This attribute allows
  clang to emit a warning or error if a function call meets one or more
  user-specified conditions.

- Enhanced devirtualization with
  `-fstrict-vtable-pointers <UsersManual.html#cmdoption-fstrict-vtable-pointers>`_.
  Clang devirtualizes across different basic blocks, like loops:

  .. code-block:: c++

       struct A {
           virtual void foo();
       };
       void indirect(A &a, int n) {
           for (int i = 0 ; i < n; i++)
               a.foo();
       }
       void test(int n) {
           A a;
           indirect(a, n);
       }


Improvements to ThinLTO (-flto=thin)
------------------------------------
- Integration with profile data (PGO). When available, profile data enables
  more accurate function importing decisions, as well as cross-module indirect
  call promotion.
- Significant build-time and binary-size improvements when compiling with debug
  info (``-g``).

New Compiler Flags
------------------

- The option ``-Og`` has been added to optimize the debugging experience.
  For now, this option is exactly the same as ``-O1``. However, in the future,
  some other optimizations might be enabled or disabled.

- The option ``-MJ`` has been added to simplify adding JSON compilation
  database output into existing build systems.


OpenCL C Language Changes in Clang
----------------------------------

**The following bugs in the OpenCL header have been fixed:**

* Added missing ``overloadable`` and ``convergent`` attributes.
* Removed some erroneous extra ``native_*`` functions.

**The following bugs in the generation of metadata have been fixed:**

* Corrected the SPIR version depending on the OpenCL version.
* Source level address spaces are taken from the SPIR specification.
* Image types now contain no access qualifier.

**The following bugs in the AMD target have been fixed:**

* Corrected the bitwidth of ``size_t`` and NULL pointer value with respect to
  address spaces.
* Added ``cl_khr_subgroups``, ``cl_amd_media_ops`` and ``cl_amd_media_ops2``
  extensions.
* Added ``cl-denorms-are-zero`` support.
* Changed address spaces for image objects to be ``constant``.
* Added little-endian.

**The following bugs in OpenCL 2.0 have been fixed:**

* Fixed pipe builtin function return type, added extra argument to generated
  IR intrinsics to propagate size and alignment information of the pipe packed
  type.
* Improved pipe type to accommodate access qualifiers.
* Added correct address space to the ObjC block generation and ``enqueue_kernel``
  prototype.
* Improved handling of integer parameters of ``enqueue_kernel`` prototype. We
  now allow ``size_t`` instead of ``int`` for specifying block parameter sizes.
* Allow using NULL (aka ``CLK_NULL_QUEUE``) with ``queue_t``.


**Improved the following diagnostics:**

* Disallow address spaces other than ``global`` for kernel pointer parameters.
* Correct the use of half type argument and pointer assignment with
  dereferencing.
* Disallow variadic arguments in functions and blocks.
* Allow partial initializer for array and struct.

**Some changes to OpenCL extensions have been made:**

* Added ``cl_khr_mipmap_image``.
* Added ``-cl-ext`` flag to allow overwriting supported extensions otherwise
  set by the target compiled for (Example: ``-cl-ext=-all,+cl_khr_fp16``).
* New types and functions can now be flexibly added to extensions using the
  following pragmas instead of modifying the Clang source code:

  .. code-block:: c

       #pragma OPENCL EXTENSION the_new_extension_name : begin
       // declare types and functions associated with the extension here
       #pragma OPENCL EXTENSION the_new_extension_name : end


**Miscellaneous changes:**

* Fix ``__builtin_astype`` to cast between different address space objects.
* Allow using ``opencl_unroll_hint`` with earlier OpenCL versions than 2.0.
* Improved handling of floating point literal to default to single precision if
  fp64 extension is not enabled.
* Refactor ``sampler_t`` implementation to simplify initializer representation
  which is now handled as a compiler builtin function with an integer value
  passed into it.
* Change fake address space map to use the SPIR convention.
* Added `the OpenCL manual <UsersManual.html#opencl-features>`_ to Clang
  documentation.


Static Analyzer
---------------

With the option ``--show-description``, scan-build's list of defects will also
show the description of the defects.

The analyzer now provides better support of code that uses gtest.

Several new checks were added:

- The analyzer warns when virtual calls are made from constructors or
  destructors. This check is off by default but can be enabled by passing the
  following command to scan-build: ``-enable-checker optin.cplusplus.VirtualCall``.
- The analyzer checks for synthesized copy properties of mutable types in
  Objective C, such as ``NSMutableArray``. Calling the setter for these properties
  will store an immutable copy of the value.
- The analyzer checks for calls to ``dispatch_once()`` that use an Objective-C
  instance variable as the predicate. Using an instance variable as a predicate
  may result in the passed-in block being executed multiple times or not at all.
  These calls should be rewritten either to use a lock or to store the predicate
  in a global or static variable.
- The analyzer checks for unintended comparisons of ``NSNumber``, ``CFNumberRef``, and
  other Cocoa number objects to scalar values.


Additional Information
======================

A wide variety of additional information is available on the `Clang web
page <http://clang.llvm.org/>`_. The web page contains versions of the
API documentation which are up-to-date with the Subversion version of
the source code. You can access versions of these documents specific to
this release by going into the "``clang/docs/``" directory in the Clang
tree.

If you have any questions or comments about Clang, please feel free to
contact us via the `mailing
list <http://lists.llvm.org/mailman/listinfo/cfe-dev>`_.
