===========================================
Clang |release| |ReleaseNotesTitle|
===========================================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <https://llvm.org/>`_

.. only:: PreRelease

  .. warning::
     These are in-progress notes for the upcoming Clang |version| release.
     Release notes for previous releases can be found on
     `the Releases Page <https://llvm.org/releases/>`_.

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release |release|. Here we
describe the status of Clang in some detail, including major
improvements from the previous release and new feature work. For the
general LLVM release notes, see `the LLVM
documentation <https://llvm.org/docs/ReleaseNotes.html>`_. For the libc++ release notes,
see `this page <https://libcxx.llvm.org/ReleaseNotes.html>`_. All LLVM releases
may be downloaded from the `LLVM releases web site <https://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about the
latest release, please see the `Clang Web Site <https://clang.llvm.org>`_ or the
`LLVM Web Site <https://llvm.org>`_.

Potentially Breaking Changes
============================
These changes are ones which we think may surprise users when upgrading to
Clang |release| because of the opportunity they pose for disruption to existing
code bases.

- The ``le32`` and ``le64`` targets have been removed.

C/C++ Language Potentially Breaking Changes
-------------------------------------------

C++ Specific Potentially Breaking Changes
-----------------------------------------

- The type trait builtin ``__is_nullptr`` has been removed, since it has very
  few users and can be written as ``__is_same(__remove_cv(T), decltype(nullptr))``,
  which GCC supports as well.

- Clang will now correctly diagnose as ill-formed a constant expression where an
  enum without a fixed underlying type is set to a value outside the range of
  the enumeration's values.

  .. code-block:: c++

    enum E { Zero, One, Two, Three, Four };
    constexpr E Val1 = (E)3;  // Ok
    constexpr E Val2 = (E)7;  // Ok
    constexpr E Val3 = (E)8;  // Now ill-formed, out of the range [0, 7]
    constexpr E Val4 = (E)-1; // Now ill-formed, out of the range [0, 7]

  Since Clang 16, it has been possible to suppress the diagnostic via
  `-Wno-enum-constexpr-conversion`, to allow for a transition period for users.
  Now, in Clang 20, **it is no longer possible to suppress the diagnostic**.

- Extraneous template headers are now ill-formed by default.
  This error can be disable with ``-Wno-error=extraneous-template-head``.

  .. code-block:: c++

    template <> // error: extraneous template head
    template <typename T>
    void f();

ABI Changes in This Version
---------------------------

AST Dumping Potentially Breaking Changes
----------------------------------------

Clang Frontend Potentially Breaking Changes
-------------------------------------------

Clang Python Bindings Potentially Breaking Changes
--------------------------------------------------
- Parts of the interface returning string results will now return
  the empty string ``""`` when no result is available, instead of ``None``.
- Calling a property on the ``CompletionChunk`` or ``CompletionString`` class
  statically now leads to an error, instead of returning a ``CachedProperty`` object
  that is used internally. Properties are only available on instances.
- For a single-line ``SourceRange`` and a ``SourceLocation`` in the same line,
  but after the end of the ``SourceRange``, ``SourceRange.__contains__``
  used to incorrectly return ``True``. (#GH22617), (#GH52827)

What's New in Clang |release|?
==============================
Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

C++ Language Changes
--------------------
- Allow single element access of GCC vector/ext_vector_type object to be
  constant expression. Supports the `V.xyzw` syntax and other tidbits
  as seen in OpenCL. Selecting multiple elements is left as a future work.

C++17 Feature Support
^^^^^^^^^^^^^^^^^^^^^

C++14 Feature Support
^^^^^^^^^^^^^^^^^^^^^

C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^

C++23 Feature Support
^^^^^^^^^^^^^^^^^^^^^
- Removed the restriction to literal types in constexpr functions in C++23 mode.

C++2c Feature Support
^^^^^^^^^^^^^^^^^^^^^

- Add ``__builtin_is_implicit_lifetime`` intrinsic, which supports
  `P2647R1 A trait for implicit lifetime types <https://wg21.link/p2674r1>`_

- Add ``__builtin_is_virtual_base_of`` intrinsic, which supports
  `P2985R0 A type trait for detecting virtual base classes <https://wg21.link/p2985r0>`_

- Implemented `P2893R3 Variadic Friends <https://wg21.link/P2893>`_

Resolutions to C++ Defect Reports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Allow calling initializer list constructors from initializer lists with
  a single element of the same type instead of always copying.
  (`CWG2137: List-initialization from object of same type <https://cplusplus.github.io/CWG/issues/2137.html>`)

- Speculative resolution for CWG2311 implemented so that the implementation of CWG2137 doesn't remove
  previous cases where guaranteed copy elision was done. Given a prvalue ``e`` of class type
  ``T``, ``T{e}`` will try to resolve an initializer list constructor and will use it if successful.
  Otherwise, if there is no initializer list constructor, the copy will be elided as if it was ``T(e)``.
  (`CWG2311: Missed case for guaranteed copy elision <https://cplusplus.github.io/CWG/issues/2311.html>`)

- Casts from a bit-field to an integral type is now not considered narrowing if the
  width of the bit-field means that all potential values are in the range
  of the target type, even if the type of the bit-field is larger.
  (`CWG2627: Bit-fields and narrowing conversions <https://cplusplus.github.io/CWG/issues/2627.html>`_)

C Language Changes
------------------

C2y Feature Support
^^^^^^^^^^^^^^^^^^^

C23 Feature Support
^^^^^^^^^^^^^^^^^^^

Non-comprehensive list of changes in this release
-------------------------------------------------

- The floating point comparison builtins (``__builtin_isgreater``,
  ``__builtin_isgreaterequal``, ``__builtin_isless``, etc.) and
  ``__builtin_signbit`` can now be used in constant expressions.

New Compiler Flags
------------------

Deprecated Compiler Flags
-------------------------

Modified Compiler Flags
-----------------------

- The compiler flag `-fbracket-depth` default value is increased from 256 to 2048.

- The ``-ffp-model`` option has been updated to enable a more limited set of
  optimizations when the ``fast`` argument is used and to accept a new argument,
  ``aggressive``. The behavior of ``-ffp-model=aggressive`` is equivalent
  to the previous behavior of ``-ffp-model=fast``. The updated
  ``-ffp-model=fast`` behavior no longer assumes finite math only and uses
  the ``promoted`` algorithm for complex division when possible rather than the
  less basic (limited range) algorithm.

Removed Compiler Flags
-------------------------

- The compiler flag `-Wenum-constexpr-conversion` (and the `Wno-`, `Wno-error-`
  derivatives) is now removed, since it's no longer possible to suppress the
  diagnostic (see above). Users can expect an `unknown warning` diagnostic if
  it's still in use.

Attribute Changes in Clang
--------------------------

- Clang now disallows more than one ``__attribute__((ownership_returns(class, idx)))`` with
  different class names attached to one function.

- Introduced a new format attribute ``__attribute__((format(syslog, 1, 2)))`` from OpenBSD.

- The ``hybrid_patchable`` attribute is now supported on ARM64EC targets. It can be used to specify
  that a function requires an additional x86-64 thunk, which may be patched at runtime.

- ``[[clang::lifetimebound]]`` is now explicitly disallowed on explicit object member functions
  where they were previously silently ignored.

Improvements to Clang's diagnostics
-----------------------------------

- Some template related diagnostics have been improved.

  .. code-block:: c++

     void foo() { template <typename> int i; } // error: templates can only be declared in namespace or class scope

     struct S {
      template <typename> int i; // error: non-static data member 'i' cannot be declared as a template
     };

- Clang now has improved diagnostics for functions with explicit 'this' parameters. Fixes #GH97878

- Clang now diagnoses dangling references to fields of temporary objects. Fixes #GH81589.

- Clang now diagnoses undefined behavior in constant expressions more consistently. This includes invalid shifts, and signed overflow in arithmetic.

- -Wdangling-assignment-gsl is enabled by default.
- Clang now always preserves the template arguments as written used
  to specialize template type aliases.

- Clang now diagnoses the use of ``main`` in an ``extern`` context as invalid according to [basic.start.main] p3. Fixes #GH101512.

- Clang now diagnoses when the result of a [[nodiscard]] function is discarded after being cast in C. Fixes #GH104391.

Improvements to Clang's time-trace
----------------------------------

Improvements to Coverage Mapping
--------------------------------

Bug Fixes in This Version
-------------------------

- Fixed the definition of ``ATOMIC_FLAG_INIT`` in ``<stdatomic.h>`` so it can
  be used in C++.
- Fixed a failed assertion when checking required literal types in C context. (#GH101304).
- Fixed a crash when trying to transform a dependent address space type. Fixes #GH101685.
- Fixed a crash when diagnosing format strings and encountering an empty
  delimited escape sequence (e.g., ``"\o{}"``). #GH102218

Bug Fixes to Compiler Builtins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to Attribute Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to C++ Support
^^^^^^^^^^^^^^^^^^^^^^^^

- Fixed a crash when an expression with a dependent ``__typeof__`` type is used as the operand of a unary operator. (#GH97646)
- Fixed incorrect pack expansion of init-capture references in requires expresssions.
- Fixed a failed assertion when checking invalid delete operator declaration. (#GH96191)
- Fix a crash when checking destructor reference with an invalid initializer. (#GH97230)
- Clang now correctly parses potentially declarative nested-name-specifiers in pointer-to-member declarators.
- Fix a crash when checking the initialzier of an object that was initialized
  with a string literal. (#GH82167)
- Fix a crash when matching template template parameters with templates which have
  parameters of different class type. (#GH101394)
- Clang now correctly recognizes the correct context for parameter
  substitutions in concepts, so it doesn't incorrectly complain of missing
  module imports in those situations. (#GH60336)
- Fix init-capture packs having a size of one before being instantiated. (#GH63677)
- Clang now preserves the unexpanded flag in a lambda transform used for pack expansion. (#GH56852), (#GH85667),
  (#GH99877).
- Fixed a bug when diagnosing ambiguous explicit specializations of constrained member functions.
- Fixed an assertion failure when selecting a function from an overload set that includes a
  specialization of a conversion function template.
- Correctly diagnose attempts to use a concept name in its own definition;
  A concept name is introduced to its scope sooner to match the C++ standard. (#GH55875)
- Properly reject defaulted relational operators with invalid types for explicit object parameters,
  e.g., ``bool operator==(this int, const Foo&)`` (#GH100329), and rvalue reference parameters.
- Properly reject defaulted copy/move assignment operators that have a non-reference explicit object parameter.
- Clang now properly handles the order of attributes in `extern` blocks. (#GH101990).
- Fixed an assertion failure by preventing null explicit object arguments from being deduced. (#GH102025).
- Correctly check constraints of explicit instantiations of member functions. (#GH46029)

Bug Fixes to AST Handling
^^^^^^^^^^^^^^^^^^^^^^^^^

- Fixed a crash that occurred when dividing by zero in complex integer division. (#GH55390).

Miscellaneous Bug Fixes
^^^^^^^^^^^^^^^^^^^^^^^

Miscellaneous Clang Crashes Fixed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Fixed a crash in C due to incorrect lookup that members in nested anonymous struct/union
  can be found as ordinary identifiers in struct/union definition. (#GH31295)

- Fixed a crash caused by long chains of ``sizeof`` and other similar operators
  that can be followed by a non-parenthesized expression. (#GH45061)

OpenACC Specific Changes
------------------------

Target Specific Changes
-----------------------

AMDGPU Support
^^^^^^^^^^^^^^

X86 Support
^^^^^^^^^^^

- The MMX vector intrinsic functions from ``*mmintrin.h`` which
  operate on `__m64` vectors, such as ``_mm_add_pi8``, have been
  reimplemented to use the SSE2 instruction-set and XMM registers
  unconditionally. These intrinsics are therefore *no longer
  supported* if MMX is enabled without SSE2 -- either from targeting
  CPUs from the Pentium-MMX through the Pentium 3, or explicitly via
  passing arguments such as ``-mmmx -mno-sse2``. MMX assembly code
  remains supported without requiring SSE2, including inside
  inline-assembly.

- The compiler builtins such as ``__builtin_ia32_paddb`` which
  formerly implemented the above MMX intrinsic functions have been
  removed. Any uses of these removed functions should migrate to the
  functions defined by the ``*mmintrin.h`` headers. A mapping can be
  found in the file ``clang/www/builtins.py``.

- Support ISA of ``AVX10.2``.
  * Supported MINMAX intrinsics of ``*_(mask(z)))_minmax(ne)_p[s|d|h|bh]`` and
  ``*_(mask(z)))_minmax_s[s|d|h]``.

Arm and AArch64 Support
^^^^^^^^^^^^^^^^^^^^^^^

Android Support
^^^^^^^^^^^^^^^

Windows Support
^^^^^^^^^^^^^^^

- Clang no longer allows references inside a union when emulating MSVC 1900+ even if `fms-extensions` is enabled.
  Starting with VS2015, MSVC 1900, this Microsoft extension is no longer allowed and always results in an error.
  Clang now follows the MSVC behavior in this scenario.
  When `-fms-compatibility-version=18.00` or prior is set on the command line this Microsoft extension is still
  allowed as VS2013 and prior allow it.

LoongArch Support
^^^^^^^^^^^^^^^^^

RISC-V Support
^^^^^^^^^^^^^^

CUDA/HIP Language Changes
^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA Support
^^^^^^^^^^^^

AIX Support
^^^^^^^^^^^

NetBSD Support
^^^^^^^^^^^^^^

WebAssembly Support
^^^^^^^^^^^^^^^^^^^

AVR Support
^^^^^^^^^^^

DWARF Support in Clang
----------------------

Floating Point Support in Clang
-------------------------------

Fixed Point Support in Clang
----------------------------

AST Matchers
------------

- Fixed an issue with the `hasName` and `hasAnyName` matcher when matching
  inline namespaces with an enclosing namespace of the same name.

- Fixed an ordering issue with the `hasOperands` matcher occuring when setting a
  binding in the first matcher and using it in the second matcher.

clang-format
------------

- Adds ``BreakBinaryOperations`` option.

libclang
--------
- Add ``clang_isBeforeInTranslationUnit``. Given two source locations, it determines
  whether the first one comes strictly before the second in the source code.

Static Analyzer
---------------

New features
^^^^^^^^^^^^

- MallocChecker now checks for ``ownership_returns(class, idx)`` and ``ownership_takes(class, idx)``
  attributes with class names different from "malloc". Clang static analyzer now reports an error
  if class of allocation and deallocation function mismatches.
  `Documentation <https://clang.llvm.org/docs/analyzer/checkers.html#unix-mismatcheddeallocator-c-c>`__.

Crash and bug fixes
^^^^^^^^^^^^^^^^^^^

Improvements
^^^^^^^^^^^^

- Improved the handling of the ``ownership_returns`` attribute. Now, Clang reports an
  error if the attribute is attached to a function that returns a non-pointer value.
  Fixes (#GH99501)

Moved checkers
^^^^^^^^^^^^^^

- The checker ``alpha.security.MallocOverflow`` was deleted because it was
  badly implemented and its agressive logic produced too many false positives.
  To detect too large arguments passed to malloc, consider using the checker
  ``alpha.taint.TaintedAlloc``.

.. _release-notes-sanitizers:

Sanitizers
----------

Python Binding Changes
----------------------
- Fixed an issue that led to crashes when calling ``Type.get_exception_specification_kind``.

OpenMP Support
--------------
- Added support for 'omp assume' directive.

Improvements
^^^^^^^^^^^^
- Improve the handling of mapping array-section for struct containing nested structs with user defined mappers

- `num_teams` and `thead_limit` now accept multiple expressions when it is used
  along in ``target teams ompx_bare`` construct. This allows the target region
  to be launched with multi-dim grid on GPUs.

Additional Information
======================

A wide variety of additional information is available on the `Clang web
page <https://clang.llvm.org/>`_. The web page contains versions of the
API documentation which are up-to-date with the Git version of
the source code. You can access versions of these documents specific to
this release by going into the "``clang/docs/``" directory in the Clang
tree.

If you have any questions or comments about Clang, please feel free to
contact us on the `Discourse forums (Clang Frontend category)
<https://discourse.llvm.org/c/clang/6>`_.
