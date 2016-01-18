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
As currently implemented, all of Clang's CFI schemes (``cfi-vcall``,
``cfi-derived-cast``, ``cfi-unrelated-cast``, ``cfi-nvcall``, ``cfi-icall``)
rely on link-time optimization (LTO); so it is required to specify
``-flto``, and the linker used must support LTO, for example via the `gold
plugin`_. To allow the checks to be implemented efficiently, the program must
be structured such that certain object files are compiled with CFI enabled,
and are statically linked into the program. This may preclude the use of
shared libraries in some cases.

Clang currently implements forward-edge CFI for member function calls and
bad cast checking. More schemes are under development.

.. _gold plugin: http://llvm.org/docs/GoldPlugin.html

Forward-Edge CFI for Virtual Calls
----------------------------------

This scheme checks that virtual calls take place using a vptr of the correct
dynamic type; that is, the dynamic type of the called object must be a
derived class of the static type of the object used to make the call.
This CFI scheme can be enabled on its own using ``-fsanitize=cfi-vcall``.

For this scheme to work, all translation units containing the definition
of a virtual member function (whether inline or not), other than members
of :ref:`blacklisted <cfi-blacklist>` types, must be compiled with
``-fsanitize=cfi-vcall`` enabled and be statically linked into the program.

Performance
~~~~~~~~~~~

A performance overhead of less than 1% has been measured by running the
Dromaeo benchmark suite against an instrumented version of the Chromium
web browser. Another good performance benchmark for this mechanism is the
virtual-call-heavy SPEC 2006 xalancbmk.

Note that this scheme has not yet been optimized for binary size; an increase
of up to 15% has been observed for Chromium.

Bad Cast Checking
-----------------

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
to its original type).

If a program as a matter of policy forbids the second type of cast, that
restriction can normally be enforced. However it may in some cases be necessary
for a function to perform a forbidden cast to conform with an external API
(e.g. the ``allocate`` member function of a standard library allocator). Such
functions may be :ref:`blacklisted <cfi-blacklist>`.

For this scheme to work, all translation units containing the definition
of a virtual member function (whether inline or not), other than members
of :ref:`blacklisted <cfi-blacklist>` types, must be compiled with
``-fsanitize=cfi-derived-cast`` or ``-fsanitize=cfi-unrelated-cast`` enabled
and be statically linked into the program.

Non-Virtual Member Function Call Checking
-----------------------------------------

This scheme checks that non-virtual calls take place using an object of
the correct dynamic type; that is, the dynamic type of the called object
must be a derived class of the static type of the object used to make the
call. The checks are currently only introduced where the object is of a
polymorphic class type.  This CFI scheme can be enabled on its own using
``-fsanitize=cfi-nvcall``.

For this scheme to work, all translation units containing the definition
of a virtual member function (whether inline or not), other than members
of :ref:`blacklisted <cfi-blacklist>` types, must be compiled with
``-fsanitize=cfi-nvcall`` enabled and be statically linked into the program.

.. _cfi-strictness:

Strictness
~~~~~~~~~~

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
-------------------------------

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

``-fsanitize=cfi-icall`` and ``-fsanitize=function``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
---------

A :doc:`SanitizerSpecialCaseList` can be used to relax CFI checks for certain
source files, functions and types using the ``src``, ``fun`` and ``type``
entity types.

In addition, if a type has a ``uuid`` attribute and the blacklist contains
the type entry ``attr:uuid``, CFI checks are suppressed for that type. This
allows all COM types to be easily blacklisted, which is useful as COM types
are typically defined outside of the linked program.

.. code-block:: bash

    # Suppress checking for code in a file.
    src:bad_file.cpp
    src:bad_header.h
    # Ignore all functions with names containing MyFooBar.
    fun:*MyFooBar*
    # Ignore all types in the standard library.
    type:std::*
    # Ignore all types with a uuid attribute.
    type:attr:uuid

Design
------

Please refer to the :doc:`design document<ControlFlowIntegrityDesign>`.

Publications
------------

`Control-Flow Integrity: Principles, Implementations, and Applications <http://research.microsoft.com/pubs/64250/ccs05.pdf>`_.
Martin Abadi, Mihai Budiu, Úlfar Erlingsson, Jay Ligatti.

`Enforcing Forward-Edge Control-Flow Integrity in GCC & LLVM <http://www.pcc.me.uk/~peter/acad/usenix14.pdf>`_.
Caroline Tice, Tom Roeder, Peter Collingbourne, Stephen Checkoway,
Úlfar Erlingsson, Luis Lozano, Geoff Pike.
