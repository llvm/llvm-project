======================
Control Flow Integrity
======================

.. toctree::
   :hidden:

   ControlFlowIntegrityDesign

.. contents::
   :local:

Introduction
============

Clang includes an implementation of a number of control flow integrity (CFI)
schemes, which are designed to abort the program upon detecting certain forms
of undefined behavior that can potentially allow attackers to subvert the
program's control flow. These schemes have been optimized for performance,
allowing developers to enable them in release builds.

To enable Clang's available CFI schemes, use the flag ``-fsanitize=cfi``.
You can also enable a subset of available :ref:`schemes <cfi-schemes>`.
As currently implemented, all schemes rely on link-time optimization (LTO);
so it is required to specify ``-flto``, and the linker used must support LTO,
for example via the `gold plugin`_.

To allow the checks to be implemented efficiently, the program must
be structured such that certain object files are compiled with CFI
enabled, and are statically linked into the program. This may preclude
the use of shared libraries in some cases.

The compiler will only produce CFI checks for a class if it can infer hidden
LTO visibility for that class. LTO visibility is a property of a class that
is inferred from flags and attributes. For more details, see the documentation
for :doc:`LTO visibility <LTOVisibility>`.

The ``-fsanitize=cfi-{vcall,nvcall,derived-cast,unrelated-cast}`` flags
require that a ``-fvisibility=`` flag also be specified. This is because the
default visibility setting is ``-fvisibility=default``, which would disable
CFI checks for classes without visibility attributes. Most users will want
to specify ``-fvisibility=hidden``, which enables CFI checks for such classes.

Experimental support for :ref:`cross-DSO control flow integrity
<cfi-cross-dso>` exists that does not require classes to have hidden LTO
visibility. This cross-DSO support has unstable ABI at this time.

.. _gold plugin: http://llvm.org/docs/GoldPlugin.html

.. _cfi-schemes:

Available schemes
=================

Available schemes are:

  -  ``-fsanitize=cfi-cast-strict``: Enables :ref:`strict cast checks
     <cfi-strictness>`.
  -  ``-fsanitize=cfi-derived-cast``: Base-to-derived cast to the wrong
     dynamic type.
  -  ``-fsanitize=cfi-unrelated-cast``: Cast from ``void*`` or another
     unrelated type to the wrong dynamic type.
  -  ``-fsanitize=cfi-nvcall``: Non-virtual call via an object whose vptr is of
     the wrong dynamic type.
  -  ``-fsanitize=cfi-vcall``: Virtual call via an object whose vptr is of the
     wrong dynamic type.
  -  ``-fsanitize=cfi-icall``: Indirect call of a function with wrong dynamic
     type.

You can use ``-fsanitize=cfi`` to enable all the schemes and use
``-fno-sanitize`` flag to narrow down the set of schemes as desired.
For example, you can build your program with
``-fsanitize=cfi -fno-sanitize=cfi-nvcall,cfi-icall``
to use all schemes except for non-virtual member function call and indirect call
checking.

Remember that you have to provide ``-flto`` if at least one CFI scheme is
enabled.

Trapping and Diagnostics
========================

By default, CFI will abort the program immediately upon detecting a control
flow integrity violation. You can use the :ref:`-fno-sanitize-trap=
<controlling-code-generation>` flag to cause CFI to print a diagnostic
similar to the one below before the program aborts.

.. code-block:: console

    bad-cast.cpp:109:7: runtime error: control flow integrity check for type 'B' failed during base-to-derived cast (vtable address 0x000000425a50)
    0x000000425a50: note: vtable is of type 'A'
     00 00 00 00  f0 f1 41 00 00 00 00 00  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  20 5a 42 00
                  ^ 

If diagnostics are enabled, you can also configure CFI to continue program
execution instead of aborting by using the :ref:`-fsanitize-recover=
<controlling-code-generation>` flag.

Forward-Edge CFI for Virtual Calls
==================================

This scheme checks that virtual calls take place using a vptr of the correct
dynamic type; that is, the dynamic type of the called object must be a
derived class of the static type of the object used to make the call.
This CFI scheme can be enabled on its own using ``-fsanitize=cfi-vcall``.

For this scheme to work, all translation units containing the definition
of a virtual member function (whether inline or not), other than members
of :ref:`blacklisted <cfi-blacklist>` types or types with public :doc:`LTO
visibility <LTOVisibility>`, must be compiled with ``-flto`` or ``-flto=thin``
enabled and be statically linked into the program.

Performance
-----------

A performance overhead of less than 1% has been measured by running the
Dromaeo benchmark suite against an instrumented version of the Chromium
web browser. Another good performance benchmark for this mechanism is the
virtual-call-heavy SPEC 2006 xalancbmk.

Note that this scheme has not yet been optimized for binary size; an increase
of up to 15% has been observed for Chromium.

Bad Cast Checking
=================

This scheme checks that pointer casts are made to an object of the correct
dynamic type; that is, the dynamic type of the object must be a derived class
of the pointee type of the cast. The checks are currently only introduced
where the class being casted to is a polymorphic class.

Bad casts are not in themselves control flow integrity violations, but they
can also create security vulnerabilities, and the implementation uses many
of the same mechanisms.

There are two types of bad cast that may be forbidden: bad casts
from a base class to a derived class (which can be checked with
``-fsanitize=cfi-derived-cast``), and bad casts from a pointer of
type ``void*`` or another unrelated type (which can be checked with
``-fsanitize=cfi-unrelated-cast``).

The difference between these two types of casts is that the first is defined
by the C++ standard to produce an undefined value, while the second is not
in itself undefined behavior (it is well defined to cast the pointer back
to its original type) unless the object is uninitialized and the cast is a
``static_cast`` (see C++14 [basic.life]p5).

If a program as a matter of policy forbids the second type of cast, that
restriction can normally be enforced. However it may in some cases be necessary
for a function to perform a forbidden cast to conform with an external API
(e.g. the ``allocate`` member function of a standard library allocator). Such
functions may be :ref:`blacklisted <cfi-blacklist>`.

For this scheme to work, all translation units containing the definition
of a virtual member function (whether inline or not), other than members
of :ref:`blacklisted <cfi-blacklist>` types or types with public :doc:`LTO
visibility <LTOVisibility>`, must be compiled with ``-flto`` or ``-flto=thin``
enabled and be statically linked into the program.

Non-Virtual Member Function Call Checking
=========================================

This scheme checks that non-virtual calls take place using an object of
the correct dynamic type; that is, the dynamic type of the called object
must be a derived class of the static type of the object used to make the
call. The checks are currently only introduced where the object is of a
polymorphic class type.  This CFI scheme can be enabled on its own using
``-fsanitize=cfi-nvcall``.

For this scheme to work, all translation units containing the definition
of a virtual member function (whether inline or not), other than members
of :ref:`blacklisted <cfi-blacklist>` types or types with public :doc:`LTO
visibility <LTOVisibility>`, must be compiled with ``-flto`` or ``-flto=thin``
enabled and be statically linked into the program.

.. _cfi-strictness:

Strictness
----------

If a class has a single non-virtual base and does not introduce or override
virtual member functions or fields other than an implicitly defined virtual
destructor, it will have the same layout and virtual function semantics as
its base. By default, casts to such classes are checked as if they were made
to the least derived such class.

Casting an instance of a base class to such a derived class is technically
undefined behavior, but it is a relatively common hack for introducing
member functions on class instances with specific properties that works under
most compilers and should not have security implications, so we allow it by
default. It can be disabled with ``-fsanitize=cfi-cast-strict``.

Indirect Function Call Checking
===============================

This scheme checks that function calls take place using a function of the
correct dynamic type; that is, the dynamic type of the function must match
the static type used at the call. This CFI scheme can be enabled on its own
using ``-fsanitize=cfi-icall``.

For this scheme to work, each indirect function call in the program, other
than calls in :ref:`blacklisted <cfi-blacklist>` functions, must call a
function which was either compiled with ``-fsanitize=cfi-icall`` enabled,
or whose address was taken by a function in a translation unit compiled with
``-fsanitize=cfi-icall``.

If a function in a translation unit compiled with ``-fsanitize=cfi-icall``
takes the address of a function not compiled with ``-fsanitize=cfi-icall``,
that address may differ from the address taken by a function in a translation
unit not compiled with ``-fsanitize=cfi-icall``. This is technically a
violation of the C and C++ standards, but it should not affect most programs.

Each translation unit compiled with ``-fsanitize=cfi-icall`` must be
statically linked into the program or shared library, and calls across
shared library boundaries are handled as if the callee was not compiled with
``-fsanitize=cfi-icall``.

This scheme is currently only supported on the x86 and x86_64 architectures.

``-fsanitize-cfi-icall-generalize-pointers``
--------------------------------------------

Mismatched pointer types are a common cause of cfi-icall check failures.
Translation units compiled with the ``-fsanitize-cfi-icall-generalize-pointers``
flag relax pointer type checking for call sites in that translation unit,
applied across all functions compiled with ``-fsanitize=cfi-icall``.

Specifically, pointers in return and argument types are treated as equivalent as
long as the qualifiers for the type they point to match. For example, ``char*``,
``char**``, and ``int*`` are considered equivalent types. However, ``char*`` and
``const char*`` are considered separate types.

``-fsanitize-cfi-icall-generalize-pointers`` is not compatible with
``-fsanitize-cfi-cross-dso``.


``-fsanitize=cfi-icall`` and ``-fsanitize=function``
----------------------------------------------------

This tool is similar to ``-fsanitize=function`` in that both tools check
the types of function calls. However, the two tools occupy different points
on the design space; ``-fsanitize=function`` is a developer tool designed
to find bugs in local development builds, whereas ``-fsanitize=cfi-icall``
is a security hardening mechanism designed to be deployed in release builds.

``-fsanitize=function`` has a higher space and time overhead due to a more
complex type check at indirect call sites, as well as a need for run-time
type information (RTTI), which may make it unsuitable for deployment. Because
of the need for RTTI, ``-fsanitize=function`` can only be used with C++
programs, whereas ``-fsanitize=cfi-icall`` can protect both C and C++ programs.

On the other hand, ``-fsanitize=function`` conforms more closely with the C++
standard and user expectations around interaction with shared libraries;
the identity of function pointers is maintained, and calls across shared
library boundaries are no different from calls within a single program or
shared library.

.. _cfi-blacklist:

Blacklist
=========

A :doc:`SanitizerSpecialCaseList` can be used to relax CFI checks for certain
source files, functions and types using the ``src``, ``fun`` and ``type``
entity types. Specific CFI modes can be be specified using ``[section]``
headers.

.. code-block:: bash

    # Suppress all CFI checking for code in a file.
    src:bad_file.cpp
    src:bad_header.h
    # Ignore all functions with names containing MyFooBar.
    fun:*MyFooBar*
    # Ignore all types in the standard library.
    type:std::*
    # Disable only unrelated cast checks for this function
    [cfi-unrelated-cast]
    fun:*UnrelatedCast*
    # Disable CFI call checks for this function without affecting cast checks
    [cfi-vcall|cfi-nvcall|cfi-icall]
    fun:*BadCall*


.. _cfi-cross-dso:

Shared library support
======================

Use **-f[no-]sanitize-cfi-cross-dso** to enable the cross-DSO control
flow integrity mode, which allows all CFI schemes listed above to
apply across DSO boundaries. As in the regular CFI, each DSO must be
built with ``-flto``.

Normally, CFI checks will only be performed for classes that have hidden LTO
visibility. With this flag enabled, the compiler will emit cross-DSO CFI
checks for all classes, except for those which appear in the CFI blacklist
or which use a ``no_sanitize`` attribute.

Design
======

Please refer to the :doc:`design document<ControlFlowIntegrityDesign>`.

Publications
============

`Control-Flow Integrity: Principles, Implementations, and Applications <http://research.microsoft.com/pubs/64250/ccs05.pdf>`_.
Martin Abadi, Mihai Budiu, Úlfar Erlingsson, Jay Ligatti.

`Enforcing Forward-Edge Control-Flow Integrity in GCC & LLVM <http://www.pcc.me.uk/~peter/acad/usenix14.pdf>`_.
Caroline Tice, Tom Roeder, Peter Collingbourne, Stephen Checkoway,
Úlfar Erlingsson, Luis Lozano, Geoff Pike.
