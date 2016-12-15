.. raw:: html

  <style type="text/css">
    .none { background-color: #FFCCCC }
    .partial { background-color: #FFFF99 }
    .good { background-color: #CCFF99 }
  </style>

.. role:: none
.. role:: partial
.. role:: good

==================
MSVC compatibility
==================

When Clang compiles C++ code for Windows, it attempts to be compatible with
MSVC.  There are multiple dimensions to compatibility.

First, Clang attempts to be ABI-compatible, meaning that Clang-compiled code
should be able to link against MSVC-compiled code successfully.  However, C++
ABIs are particularly large and complicated, and Clang's support for MSVC's C++
ABI is a work in progress.  If you don't require MSVC ABI compatibility or don't
want to use Microsoft's C and C++ runtimes, the mingw32 toolchain might be a
better fit for your project.

Second, Clang implements many MSVC language extensions, such as
``__declspec(dllexport)`` and a handful of pragmas.  These are typically
controlled by ``-fms-extensions``.

Third, MSVC accepts some C++ code that Clang will typically diagnose as
invalid.  When these constructs are present in widely included system headers,
Clang attempts to recover and continue compiling the user's program.  Most
parsing and semantic compatibility tweaks are controlled by
``-fms-compatibility`` and ``-fdelayed-template-parsing``, and they are a work
in progress.

Finally, there is :ref:`clang-cl`, a driver program for clang that attempts to
be compatible with MSVC's cl.exe.

ABI features
============

The status of major ABI-impacting C++ features:

* Record layout: :good:`Complete`.  We've tested this with a fuzzer and have
  fixed all known bugs.

* Class inheritance: :good:`Mostly complete`.  This covers all of the standard
  OO features you would expect: virtual method inheritance, multiple
  inheritance, and virtual inheritance.  Every so often we uncover a bug where
  our tables are incompatible, but this is pretty well in hand.  This feature
  has also been fuzz tested.

* Name mangling: :good:`Ongoing`.  Every new C++ feature generally needs its own
  mangling.  For example, member pointer template arguments have an interesting
  and distinct mangling.  Fortunately, incorrect manglings usually do not result
  in runtime errors.  Non-inline functions with incorrect manglings usually
  result in link errors, which are relatively easy to diagnose.  Incorrect
  manglings for inline functions and templates result in multiple copies in the
  final image.  The C++ standard requires that those addresses be equal, but few
  programs rely on this.

* Member pointers: :good:`Mostly complete`.  Standard C++ member pointers are
  fully implemented and should be ABI compatible.  Both `#pragma
  pointers_to_members`_ and the `/vm`_ flags are supported. However, MSVC
  supports an extension to allow creating a `pointer to a member of a virtual
  base class`_.  Clang does not yet support this.

.. _#pragma pointers_to_members:
  http://msdn.microsoft.com/en-us/library/83cch5a6.aspx
.. _/vm: http://msdn.microsoft.com/en-us/library/yad46a6z.aspx
.. _pointer to a member of a virtual base class: http://llvm.org/PR15713

* Debug info: :good:`Mostly complete`.  Clang emits relatively complete CodeView
  debug information if ``/Z7`` or ``/Zi`` is passed. Microsoft's link.exe will
  transform the CodeView debug information into a PDB that works in Windows
  debuggers and other tools that consume PDB files like ETW. Work to teach lld
  about CodeView and PDBs is ongoing.

* RTTI: :good:`Complete`.  Generation of RTTI data structures has been
  finished, along with support for the ``/GR`` flag.

* C++ Exceptions: :good:`Mostly complete`.  Support for
  C++ exceptions (``try`` / ``catch`` / ``throw``) have been implemented for
  x86 and x64.  Our implementation has been well tested but we still get the
  odd bug report now and again.
  C++ exception specifications are ignored, but this is `consistent with Visual
  C++`_.

.. _consistent with Visual C++:
  https://msdn.microsoft.com/en-us/library/wfa0edys.aspx

* Asynchronous Exceptions (SEH): :partial:`Partial`.
  Structured exceptions (``__try`` / ``__except`` / ``__finally``) mostly
  work on x86 and x64.
  LLVM does not model asynchronous exceptions, so it is currently impossible to
  catch an asynchronous exception generated in the same frame as the catching
  ``__try``.

* Thread-safe initialization of local statics: :good:`Complete`.  MSVC 2015
  added support for thread-safe initialization of such variables by taking an
  ABI break.
  We are ABI compatible with both the MSVC 2013 and 2015 ABI for static local
  variables.

* Lambdas: :good:`Mostly complete`.  Clang is compatible with Microsoft's
  implementation of lambdas except for providing overloads for conversion to
  function pointer for different calling conventions.  However, Microsoft's
  extension is non-conforming.

Template instantiation and name lookup
======================================

MSVC allows many invalid constructs in class templates that Clang has
historically rejected.  In order to parse widely distributed headers for
libraries such as the Active Template Library (ATL) and Windows Runtime Library
(WRL), some template rules have been relaxed or extended in Clang on Windows.

The first major semantic difference is that MSVC appears to defer all parsing
an analysis of inline method bodies in class templates until instantiation
time.  By default on Windows, Clang attempts to follow suit.  This behavior is
controlled by the ``-fdelayed-template-parsing`` flag.  While Clang delays
parsing of method bodies, it still parses the bodies *before* template argument
substitution, which is not what MSVC does.  The following compatibility tweaks
are necessary to parse the template in those cases.

MSVC allows some name lookup into dependent base classes.  Even on other
platforms, this has been a `frequently asked question`_ for Clang users.  A
dependent base class is a base class that depends on the value of a template
parameter.  Clang cannot see any of the names inside dependent bases while it
is parsing your template, so the user is sometimes required to use the
``typename`` keyword to assist the parser.  On Windows, Clang attempts to
follow the normal lookup rules, but if lookup fails, it will assume that the
user intended to find the name in a dependent base.  While parsing the
following program, Clang will recover as if the user had written the
commented-out code:

.. _frequently asked question:
  http://clang.llvm.org/compatibility.html#dep_lookup

.. code-block:: c++

  template <typename T>
  struct Foo : T {
    void f() {
      /*typename*/ T::UnknownType x =  /*this->*/unknownMember;
    }
  };

After recovery, Clang warns the user that this code is non-standard and issues
a hint suggesting how to fix the problem.

As of this writing, Clang is able to compile a simple ATL hello world
application.  There are still issues parsing WRL headers for modern Windows 8
apps, but they should be addressed soon.
