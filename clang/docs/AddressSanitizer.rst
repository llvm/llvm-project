================
AddressSanitizer
================

.. contents::
   :local:

Introduction
============

AddressSanitizer is a fast memory error detector. It consists of a compiler
instrumentation module and a run-time library. The tool can detect the
following types of bugs:

* Out-of-bounds accesses to heap, stack and globals
* Use-after-free
* Use-after-return (to some extent)
* Double-free, invalid free
* Memory leaks (experimental)

Typical slowdown introduced by AddressSanitizer is **2x**.

How to build
============

Follow the `clang build instructions <../get_started.html>`_. CMake build is
supported.

Usage
=====

Simply compile and link your program with ``-fsanitize=address`` flag.  The
AddressSanitizer run-time library should be linked to the final executable, so
make sure to use ``clang`` (not ``ld``) for the final link step.  When linking
shared libraries, the AddressSanitizer run-time is not linked, so
``-Wl,-z,defs`` may cause link errors (don't use it with AddressSanitizer).  To
get a reasonable performance add ``-O1`` or higher.  To get nicer stack traces
in error messages add ``-fno-omit-frame-pointer``.  To get perfect stack traces
you may need to disable inlining (just use ``-O1``) and tail call elimination
(``-fno-optimize-sibling-calls``).

.. code-block:: console

    % cat example_UseAfterFree.cc
    int main(int argc, char **argv) {
      int *array = new int[100];
      delete [] array;
      return array[argc];  // BOOM
    }

    # Compile and link
    % clang -O1 -g -fsanitize=address -fno-omit-frame-pointer example_UseAfterFree.cc

or:

.. code-block:: console

    # Compile
    % clang -O1 -g -fsanitize=address -fno-omit-frame-pointer -c example_UseAfterFree.cc
    # Link
    % clang -g -fsanitize=address example_UseAfterFree.o

If a bug is detected, the program will print an error message to stderr and
exit with a non-zero exit code. Currently, AddressSanitizer does not symbolize
its output, so you may need to use a separate script to symbolize the result
offline (this will be fixed in future).

.. code-block:: console

    % ./a.out 2> log
    % projects/compiler-rt/lib/asan/scripts/asan_symbolize.py / < log | c++filt
    ==9442== ERROR: AddressSanitizer heap-use-after-free on address 0x7f7ddab8c084 at pc 0x403c8c bp 0x7fff87fb82d0 sp 0x7fff87fb82c8
    READ of size 4 at 0x7f7ddab8c084 thread T0
        #0 0x403c8c in main example_UseAfterFree.cc:4
        #1 0x7f7ddabcac4d in __libc_start_main ??:0
    0x7f7ddab8c084 is located 4 bytes inside of 400-byte region [0x7f7ddab8c080,0x7f7ddab8c210)
    freed by thread T0 here:
        #0 0x404704 in operator delete[](void*) ??:0
        #1 0x403c53 in main example_UseAfterFree.cc:4
        #2 0x7f7ddabcac4d in __libc_start_main ??:0
    previously allocated by thread T0 here:
        #0 0x404544 in operator new[](unsigned long) ??:0
        #1 0x403c43 in main example_UseAfterFree.cc:2
        #2 0x7f7ddabcac4d in __libc_start_main ??:0
    ==9442== ABORTING

AddressSanitizer exits on the first detected error. This is by design.
One reason: it makes the generated code smaller and faster (both by
~5%). Another reason: this makes fixing bugs unavoidable. With Valgrind,
it is often the case that users treat Valgrind warnings as false
positives (which they are not) and don't fix them.

``__has_feature(address_sanitizer)``
------------------------------------

In some cases one may need to execute different code depending on whether
AddressSanitizer is enabled.
:ref:`\_\_has\_feature <langext-__has_feature-__has_extension>` can be used for
this purpose.

.. code-block:: c

    #if defined(__has_feature)
    #  if __has_feature(address_sanitizer)
    // code that builds only under AddressSanitizer
    #  endif
    #endif

``__attribute__((no_sanitize_address))``
-----------------------------------------------

Some code should not be instrumented by AddressSanitizer. One may use the
function attribute
:ref:`no_sanitize_address <langext-address_sanitizer>`
(or a deprecated synonym `no_address_safety_analysis`)
to disable instrumentation of a particular function. This attribute may not be
supported by other compilers, so we suggest to use it together with
``__has_feature(address_sanitizer)``.

Initialization order checking
-----------------------------

AddressSanitizer can optionally detect dynamic initialization order problems,
when initialization of globals defined in one translation unit uses
globals defined in another translation unit. To enable this check at runtime,
you should set environment variable
``ASAN_OPTIONS=check_initialization_order=1``.

Blacklist
---------

AddressSanitizer supports ``src`` and ``fun`` entity types in
:doc:`SanitizerSpecialCaseList`, that can be used to suppress error reports
in the specified source files or functions. Additionally, AddressSanitizer
introduces ``global`` and ``type`` entity types that can be used to
suppress error reports for out-of-bound access to globals with certain
names and types (you may only specify class or struct types).

You may use an ``init`` category to suppress reports about initialization-order
problems happening in certain source files or with certain global variables.

.. code-block:: bash

    # Suppress error reports for code in a file or in a function:
    src:bad_file.cpp
    # Ignore all functions with names containing MyFooBar:
    fun:*MyFooBar*
    # Disable out-of-bound checks for global:
    global:bad_array
    # Disable out-of-bound checks for global instances of a given class ...
    type:class.Namespace::BadClassName
    # ... or a given struct. Use wildcard to deal with anonymous namespace.
    type:struct.Namespace2::*::BadStructName
    # Disable initialization-order checks for globals:
    global:bad_init_global=init
    type:*BadInitClassSubstring*=init
    src:bad/init/files/*=init

Memory leak detection
---------------------

For the experimental memory leak detector in AddressSanitizer, see
:doc:`LeakSanitizer`.

Supported Platforms
===================

AddressSanitizer is supported on

* Linux i386/x86\_64 (tested on Ubuntu 10.04 and 12.04);
* MacOS 10.6, 10.7 and 10.8 (i386/x86\_64).

Support for Linux ARM (and Android ARM) is in progress (it may work, but
is not guaranteed too).

Limitations
===========

* AddressSanitizer uses more real memory than a native run. Exact overhead
  depends on the allocations sizes. The smaller the allocations you make the
  bigger the overhead is.
* AddressSanitizer uses more stack memory. We have seen up to 3x increase.
* On 64-bit platforms AddressSanitizer maps (but not reserves) 16+ Terabytes of
  virtual address space. This means that tools like ``ulimit`` may not work as
  usually expected.
* Static linking is not supported.

Current Status
==============

AddressSanitizer is fully functional on supported platforms starting from LLVM
3.1. The test suite is integrated into CMake build and can be run with ``make
check-asan`` command.

More Information
================

`http://code.google.com/p/address-sanitizer <http://code.google.com/p/address-sanitizer/>`_

