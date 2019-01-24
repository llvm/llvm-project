==========================
UndefinedBehaviorSanitizer
==========================

.. contents::
   :local:

Introduction
============

UndefinedBehaviorSanitizer (UBSan) is a fast undefined behavior detector.
UBSan modifies the program at compile-time to catch various kinds of undefined
behavior during program execution, for example:

* Using misaligned or null pointer
* Signed integer overflow
* Conversion to, from, or between floating-point types which would
  overflow the destination

See the full list of available :ref:`checks <ubsan-checks>` below.

UBSan has an optional run-time library which provides better error reporting.
The checks have small runtime cost and no impact on address space layout or ABI.

How to build
============

Build LLVM/Clang with `CMake <https://llvm.org/docs/CMake.html>`_.

Usage
=====

Use ``clang++`` to compile and link your program with ``-fsanitize=undefined``
flag. Make sure to use ``clang++`` (not ``ld``) as a linker, so that your
executable is linked with proper UBSan runtime libraries. You can use ``clang``
instead of ``clang++`` if you're compiling/linking C code.

.. code-block:: console

  % cat test.cc
  int main(int argc, char **argv) {
    int k = 0x7fffffff;
    k += argc;
    return 0;
  }
  % clang++ -fsanitize=undefined test.cc
  % ./a.out
  test.cc:3:5: runtime error: signed integer overflow: 2147483647 + 1 cannot be represented in type 'int'

You can enable only a subset of :ref:`checks <ubsan-checks>` offered by UBSan,
and define the desired behavior for each kind of check:

* ``-fsanitize=...``: print a verbose error report and continue execution (default);
* ``-fno-sanitize-recover=...``: print a verbose error report and exit the program;
* ``-fsanitize-trap=...``: execute a trap instruction (doesn't require UBSan run-time support).

For example if you compile/link your program as:

.. code-block:: console

  % clang++ -fsanitize=signed-integer-overflow,null,alignment -fno-sanitize-recover=null -fsanitize-trap=alignment

the program will continue execution after signed integer overflows, exit after
the first invalid use of a null pointer, and trap after the first use of misaligned
pointer.

.. _ubsan-checks:

Available checks
================

Available checks are:

  -  ``-fsanitize=alignment``: Use of a misaligned pointer or creation
     of a misaligned reference. Also sanitizes assume_aligned-like attributes.
  -  ``-fsanitize=bool``: Load of a ``bool`` value which is neither
     ``true`` nor ``false``.
  -  ``-fsanitize=builtin``: Passing invalid values to compiler builtins.
  -  ``-fsanitize=bounds``: Out of bounds array indexing, in cases
     where the array bound can be statically determined.
  -  ``-fsanitize=enum``: Load of a value of an enumerated type which
     is not in the range of representable values for that enumerated
     type.
  -  ``-fsanitize=float-cast-overflow``: Conversion to, from, or
     between floating-point types which would overflow the
     destination.
  -  ``-fsanitize=float-divide-by-zero``: Floating point division by
     zero.
  -  ``-fsanitize=function``: Indirect call of a function through a
     function pointer of the wrong type (Darwin/Linux, C++ and x86/x86_64
     only).
  -  ``-fsanitize=implicit-unsigned-integer-truncation``,
     ``-fsanitize=implicit-signed-integer-truncation``: Implicit conversion from
     integer of larger bit width to smaller bit width, if that results in data
     loss. That is, if the demoted value, after casting back to the original
     width, is not equal to the original value before the downcast.
     The ``-fsanitize=implicit-unsigned-integer-truncation`` handles conversions
     between two ``unsigned`` types, while
     ``-fsanitize=implicit-signed-integer-truncation`` handles the rest of the
     conversions - when either one, or both of the types are signed.
     Issues caught by these sanitizers are not undefined behavior,
     but are often unintentional.
  -  ``-fsanitize=implicit-integer-sign-change``: Implicit conversion between
     integer types, if that changes the sign of the value. That is, if the the
     original value was negative and the new value is positive (or zero),
     or the original value was positive, and the new value is negative.
     Issues caught by this sanitizer are not undefined behavior,
     but are often unintentional.
  -  ``-fsanitize=integer-divide-by-zero``: Integer division by zero.
  -  ``-fsanitize=nonnull-attribute``: Passing null pointer as a function
     parameter which is declared to never be null.
  -  ``-fsanitize=null``: Use of a null pointer or creation of a null
     reference.
  -  ``-fsanitize=nullability-arg``: Passing null as a function parameter
     which is annotated with ``_Nonnull``.
  -  ``-fsanitize=nullability-assign``: Assigning null to an lvalue which
     is annotated with ``_Nonnull``.
  -  ``-fsanitize=nullability-return``: Returning null from a function with
     a return type annotated with ``_Nonnull``.
  -  ``-fsanitize=object-size``: An attempt to potentially use bytes which
     the optimizer can determine are not part of the object being accessed.
     This will also detect some types of undefined behavior that may not
     directly access memory, but are provably incorrect given the size of
     the objects involved, such as invalid downcasts and calling methods on
     invalid pointers. These checks are made in terms of
     ``__builtin_object_size``, and consequently may be able to detect more
     problems at higher optimization levels.
  -  ``-fsanitize=pointer-overflow``: Performing pointer arithmetic which
     overflows.
  -  ``-fsanitize=return``: In C++, reaching the end of a
     value-returning function without returning a value.
  -  ``-fsanitize=returns-nonnull-attribute``: Returning null pointer
     from a function which is declared to never return null.
  -  ``-fsanitize=shift``: Shift operators where the amount shifted is
     greater or equal to the promoted bit-width of the left hand side
     or less than zero, or where the left hand side is negative. For a
     signed left shift, also checks for signed overflow in C, and for
     unsigned overflow in C++. You can use ``-fsanitize=shift-base`` or
     ``-fsanitize=shift-exponent`` to check only left-hand side or
     right-hand side of shift operation, respectively.
  -  ``-fsanitize=signed-integer-overflow``: Signed integer overflow, where the
     result of a signed integer computation cannot be represented in its type.
     This includes all the checks covered by ``-ftrapv``, as well as checks for
     signed division overflow (``INT_MIN/-1``), but not checks for
     lossy implicit conversions performed before the computation
     (see ``-fsanitize=implicit-conversion``). Both of these two issues are
     handled by ``-fsanitize=implicit-conversion`` group of checks.
  -  ``-fsanitize=unreachable``: If control flow reaches an unreachable
     program point.
  -  ``-fsanitize=unsigned-integer-overflow``: Unsigned integer overflow, where
     the result of an unsigned integer computation cannot be represented in its
     type. Unlike signed integer overflow, this is not undefined behavior, but
     it is often unintentional. This sanitizer does not check for lossy implicit
     conversions performed before such a computation
     (see ``-fsanitize=implicit-conversion``).
  -  ``-fsanitize=vla-bound``: A variable-length array whose bound
     does not evaluate to a positive value.
  -  ``-fsanitize=vptr``: Use of an object whose vptr indicates that it is of
     the wrong dynamic type, or that its lifetime has not begun or has ended.
     Incompatible with ``-fno-rtti``. Link must be performed by ``clang++``, not
     ``clang``, to make sure C++-specific parts of the runtime library and C++
     standard libraries are present.

You can also use the following check groups:
  -  ``-fsanitize=undefined``: All of the checks listed above other than
     ``unsigned-integer-overflow``, ``implicit-conversion`` and the
     ``nullability-*`` group of checks.
  -  ``-fsanitize=undefined-trap``: Deprecated alias of
     ``-fsanitize=undefined``.
  -  ``-fsanitize=implicit-integer-truncation``: Catches lossy integral
     conversions. Enables ``implicit-signed-integer-truncation`` and
     ``implicit-unsigned-integer-truncation``.
  -  ``-fsanitize=implicit-integer-arithmetic-value-change``: Catches implicit
     conversions that change the arithmetic value of the integer. Enables
     ``implicit-signed-integer-truncation`` and ``implicit-integer-sign-change``.
  -  ``-fsanitize=implicit-conversion``: Checks for suspicious
     behaviour of implicit conversions. Enables
     ``implicit-unsigned-integer-truncation``,
     ``implicit-signed-integer-truncation`` and
     ``implicit-integer-sign-change``.
  -  ``-fsanitize=integer``: Checks for undefined or suspicious integer
     behavior (e.g. unsigned integer overflow).
     Enables ``signed-integer-overflow``, ``unsigned-integer-overflow``,
     ``shift``, ``integer-divide-by-zero``,
     ``implicit-unsigned-integer-truncation``,
     ``implicit-signed-integer-truncation`` and
     ``implicit-integer-sign-change``.
  -  ``-fsanitize=nullability``: Enables ``nullability-arg``,
     ``nullability-assign``, and ``nullability-return``. While violating
     nullability does not have undefined behavior, it is often unintentional,
     so UBSan offers to catch it.

Volatile
--------

The ``null``, ``alignment``, ``object-size``, and ``vptr`` checks do not apply
to pointers to types with the ``volatile`` qualifier.

Minimal Runtime
===============

There is a minimal UBSan runtime available suitable for use in production
environments. This runtime has a small attack surface. It only provides very
basic issue logging and deduplication, and does not support ``-fsanitize=vptr``
checking.

To use the minimal runtime, add ``-fsanitize-minimal-runtime`` to the clang
command line options. For example, if you're used to compiling with
``-fsanitize=undefined``, you could enable the minimal runtime with
``-fsanitize=undefined -fsanitize-minimal-runtime``.

Stack traces and report symbolization
=====================================
If you want UBSan to print symbolized stack trace for each error report, you
will need to:

#. Compile with ``-g`` and ``-fno-omit-frame-pointer`` to get proper debug
   information in your binary.
#. Run your program with environment variable
   ``UBSAN_OPTIONS=print_stacktrace=1``.
#. Make sure ``llvm-symbolizer`` binary is in ``PATH``.

Silencing Unsigned Integer Overflow
===================================
To silence reports from unsigned integer overflow, you can set
``UBSAN_OPTIONS=silence_unsigned_overflow=1``.  This feature, combined with
``-fsanitize-recover=unsigned-integer-overflow``, is particularly useful for
providing fuzzing signal without blowing up logs.

Issue Suppression
=================

UndefinedBehaviorSanitizer is not expected to produce false positives.
If you see one, look again; most likely it is a true positive!

Disabling Instrumentation with ``__attribute__((no_sanitize("undefined")))``
----------------------------------------------------------------------------

You disable UBSan checks for particular functions with
``__attribute__((no_sanitize("undefined")))``. You can use all values of
``-fsanitize=`` flag in this attribute, e.g. if your function deliberately
contains possible signed integer overflow, you can use
``__attribute__((no_sanitize("signed-integer-overflow")))``.

This attribute may not be
supported by other compilers, so consider using it together with
``#if defined(__clang__)``.

Suppressing Errors in Recompiled Code (Blacklist)
-------------------------------------------------

UndefinedBehaviorSanitizer supports ``src`` and ``fun`` entity types in
:doc:`SanitizerSpecialCaseList`, that can be used to suppress error reports
in the specified source files or functions.

Runtime suppressions
--------------------

Sometimes you can suppress UBSan error reports for specific files, functions,
or libraries without recompiling the code. You need to pass a path to
suppression file in a ``UBSAN_OPTIONS`` environment variable.

.. code-block:: bash

    UBSAN_OPTIONS=suppressions=MyUBSan.supp

You need to specify a :ref:`check <ubsan-checks>` you are suppressing and the
bug location. For example:

.. code-block:: bash

  signed-integer-overflow:file-with-known-overflow.cpp
  alignment:function_doing_unaligned_access
  vptr:shared_object_with_vptr_failures.so

There are several limitations:

* Sometimes your binary must have enough debug info and/or symbol table, so
  that the runtime could figure out source file or function name to match
  against the suppression.
* It is only possible to suppress recoverable checks. For the example above,
  you can additionally pass
  ``-fsanitize-recover=signed-integer-overflow,alignment,vptr``, although
  most of UBSan checks are recoverable by default.
* Check groups (like ``undefined``) can't be used in suppressions file, only
  fine-grained checks are supported.

Supported Platforms
===================

UndefinedBehaviorSanitizer is supported on the following operating systems:

* Android
* Linux
* NetBSD
* FreeBSD
* OpenBSD
* OS X 10.6 onwards
* Windows

The runtime library is relatively portable and platform independent. If the OS
you need is not listed above, UndefinedBehaviorSanitizer may already work for
it, or could be made to work with a minor porting effort.

Current Status
==============

UndefinedBehaviorSanitizer is available on selected platforms starting from LLVM
3.3. The test suite is integrated into the CMake build and can be run with
``check-ubsan`` command.

Additional Configuration
========================

UndefinedBehaviorSanitizer adds static check data for each check unless it is
in trap mode. This check data includes the full file name. The option
``-fsanitize-undefined-strip-path-components=N`` can be used to trim this
information. If ``N`` is positive, file information emitted by
UndefinedBehaviorSanitizer will drop the first ``N`` components from the file
path. If ``N`` is negative, the last ``N`` components will be kept.

Example
-------

For a file called ``/code/library/file.cpp``, here is what would be emitted:

* Default (No flag, or ``-fsanitize-undefined-strip-path-components=0``): ``/code/library/file.cpp``
* ``-fsanitize-undefined-strip-path-components=1``: ``code/library/file.cpp``
* ``-fsanitize-undefined-strip-path-components=2``: ``library/file.cpp``
* ``-fsanitize-undefined-strip-path-components=-1``: ``file.cpp``
* ``-fsanitize-undefined-strip-path-components=-2``: ``library/file.cpp``

More Information
================

* From LLVM project blog:
  `What Every C Programmer Should Know About Undefined Behavior
  <http://blog.llvm.org/2011/05/what-every-c-programmer-should-know.html>`_
* From John Regehr's *Embedded in Academia* blog:
  `A Guide to Undefined Behavior in C and C++
  <https://blog.regehr.org/archives/213>`_
