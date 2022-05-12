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
     `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release |release|. Here we
describe the status of Clang in some detail, including major
improvements from the previous release and new feature work. For the
general LLVM release notes, see `the LLVM
documentation <https://llvm.org/docs/ReleaseNotes.html>`_. All LLVM
releases may be downloaded from the `LLVM releases web
site <https://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about the
latest release, please see the `Clang Web Site <https://clang.llvm.org>`_ or the
`LLVM Web Site <https://llvm.org>`_.

Note that if you are reading this file from a Git checkout or the
main Clang web page, this document applies to the *next* release, not
the current one. To see the release notes for a specific release, please
see the `releases page <https://llvm.org/releases/>`_.

What's New in Clang |release|?
==============================

Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

Major New Features
------------------

- Clang now supports the ``-fzero-call-used-regs`` feature for x86. The purpose
  of this feature is to limit Return-Oriented Programming (ROP) exploits and
  information leakage. It works by zeroing out a selected class of registers
  before function return --- e.g., all GPRs that are used within the function.
  There is an analogous ``zero_call_used_regs`` attribute to allow for finer
  control of this feature.

- Clang now supports randomizing structure layout in C. This feature is a
  compile-time hardening technique, making it more difficult for an attacker to
  retrieve data from structures. Specify randomization with the
  ``randomize_layout`` attribute. The corresponding ``no_randomize_layout``
  attribute can be used to turn the feature off.

  A seed value is required to enable randomization, and is deterministic based
  on a seed value. Use the ``-frandomize-layout-seed=`` or
  ``-frandomize-layout-seed-file=`` flags.

  .. note::

      Randomizing structure layout is a C-only feature.

Bug Fixes
---------
- ``CXXNewExpr::getArraySize()`` previously returned a ``llvm::Optional``
  wrapping a ``nullptr`` when the ``CXXNewExpr`` did not have an array
  size expression. This was fixed and ``::getArraySize()`` will now always
  either return ``None`` or a ``llvm::Optional`` wrapping a valid ``Expr*``.
  This fixes `Issue 53742 <https://github.com/llvm/llvm-project/issues/53742>`_.
- We now ignore full expressions when traversing cast subexpressions. This
  fixes `Issue 53044 <https://github.com/llvm/llvm-project/issues/53044>`_.
- Allow ``-Wno-gnu`` to silence GNU extension diagnostics for pointer
  arithmetic diagnostics. Fixes `Issue 54444
  <https://github.com/llvm/llvm-project/issues/54444>`_.
- Placeholder constraints, as in ``Concept auto x = f();``, were not checked
  when modifiers like ``auto&`` or ``auto**`` were added. These constraints are
  now checked.
  This fixes  `Issue 53911 <https://github.com/llvm/llvm-project/issues/53911>`_
  and  `Issue 54443 <https://github.com/llvm/llvm-project/issues/54443>`_.
- Previously invalid member variables with template parameters would crash clang.
  Now fixed by setting identifiers for them.
  This fixes `Issue 28475 (PR28101) <https://github.com/llvm/llvm-project/issues/28475>`_.
- Now allow the ``restrict`` and ``_Atomic`` qualifiers to be used in
  conjunction with ``__auto_type`` to match the behavior in GCC. This fixes
  `Issue 53652 <https://github.com/llvm/llvm-project/issues/53652>`_.
- No longer crash when specifying a variably-modified parameter type in a
  function with the ``naked`` attribute. This fixes
  `Issue 50541 <https://github.com/llvm/llvm-project/issues/50541>`_.
- Allow multiple ``#pragma weak`` directives to name the same undeclared (if an
  alias, target) identifier instead of only processing one such ``#pragma weak``
  per identifier.
  Fixes `Issue 28985 <https://github.com/llvm/llvm-project/issues/28985>`_.
- Assignment expressions in C11 and later mode now properly strip the _Atomic
  qualifier when determining the type of the assignment expression. Fixes
  `Issue 48742 <https://github.com/llvm/llvm-project/issues/48742>`_.
- Improved the diagnostic when accessing a member of an atomic structure or
  union object in C; was previously an unhelpful error, but now issues a
  ``-Watomic-access`` warning which defaults to an error. Fixes
  `Issue 54563 <https://github.com/llvm/llvm-project/issues/54563>`_.
- Unevaluated lambdas in dependant contexts no longer result in clang crashing.
  This fixes Issues `50376 <https://github.com/llvm/llvm-project/issues/50376>`_,
  `51414 <https://github.com/llvm/llvm-project/issues/51414>`_,
  `51416 <https://github.com/llvm/llvm-project/issues/51416>`_,
  and `51641 <https://github.com/llvm/llvm-project/issues/51641>`_.
- The builtin function __builtin_dump_struct would crash clang when the target 
  struct contains a bitfield. It now correctly handles bitfields.
  This fixes Issue `Issue 54462 <https://github.com/llvm/llvm-project/issues/54462>`_.
- Statement expressions are now disabled in default arguments in general.
  This fixes Issue `Issue 53488 <https://github.com/llvm/llvm-project/issues/53488>`_.
- According to `CWG 1394 <https://wg21.link/cwg1394>`_ and 
  `C++20 [dcl.fct.def.general]p2 <https://timsong-cpp.github.io/cppwp/n4868/dcl.fct.def#general-2.sentence-3>`_,
  Clang should not diagnose incomplete types in function definitions if the function body is "= delete;".
  This fixes Issue `Issue 52802 <https://github.com/llvm/llvm-project/issues/52802>`_.
- Unknown type attributes with a ``[[]]`` spelling are no longer diagnosed twice.
  This fixes Issue `Issue 54817 <https://github.com/llvm/llvm-project/issues/54817>`_.
- Clang should no longer incorrectly diagnose a variable declaration inside of
  a lambda expression that shares the name of a variable in a containing
  if/while/for/switch init statement as a redeclaration.
  This fixes `Issue 54913 <https://github.com/llvm/llvm-project/issues/54913>`_.
- Overload resolution for constrained function templates could use the partial
  order of constraints to select an overload, even if the parameter types of
  the functions were different. It now diagnoses this case correctly as an
  ambiguous call and an error. Fixes
  `Issue 53640 <https://github.com/llvm/llvm-project/issues/53640>`_.
- No longer crash when trying to determine whether the controlling expression
  argument to a generic selection expression has side effects in the case where
  the expression is result dependent. This fixes
  `Issue 50227 <https://github.com/llvm/llvm-project/issues/50227>`_.
- Fixed an assertion when constant evaluating an initializer for a GCC/Clang
  floating-point vector type when the width of the initialization is exactly
  the same as the elements of the vector being initialized.
  Fixes `Issue 50216 <https://github.com/llvm/llvm-project/issues/50216>`_.
- Fixed a crash when the ``__bf16`` type is used such that its size or
  alignment is calculated on a target which does not support that type. This
  fixes `Issue 50171 <https://github.com/llvm/llvm-project/issues/50171>`_.
- Fixed a false positive diagnostic about an unevaluated expression having no
  side effects when the expression is of VLA type and is an operand of the
  ``sizeof`` operator. Fixes `Issue 48010 <https://github.com/llvm/llvm-project/issues/48010>`_.
- Fixed a false positive diagnostic about scoped enumerations being a C++11
  extension in C mode. A scoped enumeration's enumerators cannot be named in C
  because there is no way to fully qualify the enumerator name, so this
  "extension" was unintentional and useless. This fixes
  `Issue 42372 <https://github.com/llvm/llvm-project/issues/42372>`_.

Improvements to Clang's diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- ``-Wliteral-range`` will warn on floating-point equality comparisons with
  constants that are not representable in a casted value. For example,
  ``(float) f == 0.1`` is always false.
- ``-Winline-namespace-reopened-noninline`` now takes into account that the
  ``inline`` keyword must appear on the original but not necessarily all
  extension definitions of an inline namespace and therefore points its note
  at the original definition. This fixes `Issue 50794 (PR51452)
  <https://github.com/llvm/llvm-project/issues/50794>`_.
- ``-Wunused-but-set-variable`` now also warns if the variable is only used
  by unary operators.
- ``-Wunused-variable`` no longer warn for references extending the lifetime
  of temporaries with side effects. This fixes `Issue 54489
  <https://github.com/llvm/llvm-project/issues/54489>`_.
- Modified the behavior of ``-Wstrict-prototypes`` and added a new, related
  diagnostic ``-Wdeprecated-non-prototype``. The strict prototypes warning will
  now only diagnose deprecated declarations and definitions of functions
  without a prototype where the behavior in C2x will remain correct. This
  diagnostic remains off by default but is now enabled via ``-pedantic`` due to
  it being a deprecation warning. ``-Wstrict-prototypes`` has no effect in C2x
  or when ``-fno-knr-functions`` is enabled. ``-Wdeprecated-non-prototype``
  will diagnose cases where the deprecated declarations or definitions of a
  function without a prototype will change behavior in C2x. Additionally, it
  will diagnose calls which pass arguments to a function without a prototype.
  This warning is enabled only when the ``-Wdeprecated-non-prototype`` option
  is enabled at the function declaration site, which allows a developer to
  disable the diagnostic for all callers at the point of declaration. This
  diagnostic is grouped under the ``-Wstrict-prototypes`` warning group, but is
  enabled by default. ``-Wdeprecated-non-prototype`` has no effect in C2x or
  when ``-fno-knr-functions`` is enabled.
- Clang now appropriately issues an error in C when a definition of a function
  without a prototype and with no arguments is an invalid redeclaration of a
  function with a prototype. e.g., ``void f(int); void f() {}`` is now properly
  diagnosed.
- The ``-Wimplicit-function-declaration`` warning diagnostic now defaults to
  an error in C99 and later. Prior to C2x, it may be downgraded to a warning
  with ``-Wno-error=implicit-function-declaration``, or disabled entirely with
  ``-Wno-implicit-function-declaration``. As of C2x, support for implicit
  function declarations has been removed, and the warning options will have no
  effect.
- The ``-Wimplicit-int`` warning diagnostic now defaults to an error in C99 and
  later. Prior to C2x, it may be downgraded to a warning with
  ``-Wno-error=implicit-int``, or disabled entirely with ``-Wno-implicit-int``.
  As of C2x, support for implicit int has been removed, and the warning options
  will have no effect. Specifying ``-Wimplicit-int`` in C89 mode will now issue
  warnings instead of being a noop.
- No longer issue a "declaration specifiers missing, defaulting to int"
  diagnostic in C89 mode because it is not an extension in C89, it was valid
  code. The diagnostic has been removed entirely as it did not have a
  diagnostic group to disable it, but it can be covered wholly by
  ``-Wimplicit-int``.
- ``-Wmisexpect`` warns when the branch weights collected during profiling
  conflict with those added by ``llvm.expect``.
- ``-Wthread-safety-analysis`` now considers overloaded compound assignment and
  increment/decrement operators as writing to their first argument, thus
  requiring an exclusive lock if the argument is guarded.
- ``-Wenum-conversion`` now warns on converting a signed enum of one type to an
  unsigned enum of a different type (or vice versa) rather than
  ``-Wsign-conversion``.
- Added the ``-Wunreachable-code-generic-assoc`` diagnostic flag (grouped under
  the ``-Wunreachable-code`` flag) which is enabled by default and warns the
  user about ``_Generic`` selection associations which are unreachable because
  the type specified is an array type or a qualified type.

Non-comprehensive list of changes in this release
-------------------------------------------------

- Improve __builtin_dump_struct:

  - Support bitfields in struct and union.
  - Improve the dump format, dump both bitwidth(if its a bitfield) and field
    value.
  - Remove anonymous tag locations and flatten anonymous struct members.
  - Beautify dump format, add indent for struct members.
  - Support passing additional arguments to the formatting function, allowing
    use with ``fprintf`` and similar formatting functions.
  - Support use within constant evaluation in C++, if a ``constexpr``
    formatting function is provided.
  - Support formatting of base classes in C++.
  - Support calling a formatting function template in C++, which can provide
    custom formatting for non-aggregate types.

- Previously disabled sanitizer options now enabled by default:
  - ASAN_OPTIONS=detect_stack_use_after_return=1 (only on Linux).
  - MSAN_OPTIONS=poison_in_dtor=1.

New Compiler Flags
------------------
- Added the ``-fno-knr-functions`` flag to allow users to opt into the C2x
  behavior where a function with an empty parameter list is treated as though
  the parameter list were ``void``. There is no ``-fknr-functions`` or
  ``-fno-no-knr-functions`` flag; this feature cannot be disabled in language
  modes where it is required, such as C++ or C2x.

Deprecated Compiler Flags
-------------------------

Modified Compiler Flags
-----------------------

Removed Compiler Flags
-------------------------
- Removed the ``-fno-concept-satisfaction-caching`` flag. The flag was added
  at the time when the draft of C++20 standard did not permit caching of
  atomic constraints. The final standard permits such caching, see
  `WG21 P2104R0 <http://wg21.link/p2104r0>`_.

New Pragmas in Clang
--------------------

- ...

Attribute Changes in Clang
--------------------------

- Added support for parameter pack expansion in ``clang::annotate``.

- The ``overloadable`` attribute can now be written in all of the syntactic
  locations a declaration attribute may appear.
  This fixes `Issue 53805 <https://github.com/llvm/llvm-project/issues/53805>`_.

- Improved namespace attributes handling:

  - Handle GNU attributes before a namespace identifier and subsequent
    attributes of different kinds.
  - Emit error on GNU attributes for a nested namespace definition.

- Statement attributes ``[[clang::noinline]]`` and  ``[[clang::always_inline]]``
  can be used to control inlining decisions at callsites.

- ``#pragma clang attribute push`` now supports multiple attributes within a single directive.

- The ``__declspec(naked)`` attribute can no longer be written on a member
  function in Microsoft compatibility mode, matching the behavior of cl.exe.

Windows Support
---------------

- Add support for MSVC-compatible ``/JMC``/``/JMC-`` flag in clang-cl (supports
  X86/X64/ARM/ARM64). ``/JMC`` could only be used when ``/Zi`` or ``/Z7`` is
  turned on. With this addition, clang-cl can be used in Visual Studio for the
  JustMyCode feature. Note, you may need to manually add ``/JMC`` as additional
  compile options in the Visual Studio since it currently assumes clang-cl does not support ``/JMC``.

C Language Changes in Clang
---------------------------

C2x Feature Support
-------------------

- Implemented `WG14 N2674 The noreturn attribute <http://www.open-std.org/jtc1/sc22/wg14/www/docs/n2764.pdf>`_.
- Implemented `WG14 N2935 Make false and true first-class language features <http://www.open-std.org/jtc1/sc22/wg14/www/docs/n2935.pdf>`_.
- Implemented `WG14 N2763 Adding a fundamental type for N-bit integers <http://www.open-std.org/jtc1/sc22/wg14/www/docs/n2763.pdf>`_.
- Implemented `WG14 N2775 Literal suffixes for bit-precise integers <http://www.open-std.org/jtc1/sc22/wg14/www/docs/n2775.pdf>`_.
- Implemented the ``*_WIDTH`` macros to complete support for
  `WG14 N2412 Two's complement sign representation for C2x <https://www9.open-std.org/jtc1/sc22/wg14/www/docs/n2412.pdf>`_.
- Implemented `WG14 N2418 Adding the u8 character prefix <http://www.open-std.org/jtc1/sc22/wg14/www/docs/n2418.pdf>`_.
- Removed support for implicit function declarations. This was a C89 feature
  that was removed in C99, but cannot be supported in C2x because it requires
  support for functions without prototypes, which no longer exist in C2x.
- Implemented `WG14 N2841 No function declarators without prototypes <https://www9.open-std.org/jtc1/sc22/wg14/www/docs/n2841.htm>`_
  and `WG14 N2432 Remove support for function definitions with identifier lists <https://www9.open-std.org/jtc1/sc22/wg14/www/docs/n2432.pdf>`_.

C++ Language Changes in Clang
-----------------------------

- Improved ``-O0`` code generation for calls to ``std::move``, ``std::forward``,
  ``std::move_if_noexcept``, ``std::addressof``, and ``std::as_const``. These
  are now treated as compiler builtins and implemented directly, rather than
  instantiating the definition from the standard library.
- Fixed mangling of nested dependent names such as ``T::a::b``, where ``T`` is a
  template parameter, to conform to the Itanium C++ ABI and be compatible with
  GCC. This breaks binary compatibility with code compiled with earlier versions
  of clang; use the ``-fclang-abi-compat=14`` option to get the old mangling.

C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^
- Diagnose consteval and constexpr issues that happen at namespace scope. This
  partially addresses `Issue 51593 <https://github.com/llvm/llvm-project/issues/51593>`_.
- No longer attempt to evaluate a consteval UDL function call at runtime when
  it is called through a template instantiation. This fixes
  `Issue 54578 <https://github.com/llvm/llvm-project/issues/54578>`_.

- Implemented ``__builtin_source_location()``, which enables library support
  for ``std::source_location``.

- The mangling scheme for C++20 modules has incompatibly changed. The
  initial mangling was discovered not to be reversible, and the weak
  ownership design decision did not give the backwards compatibility
  that was hoped for. C++20 since added ``extern "C++"`` semantics
  that can be used for such compatibility. The demangler now demangles
  symbols with named module attachment.

C++2b Feature Support
^^^^^^^^^^^^^^^^^^^^^

- Implemented `P2128R6: Multidimensional subscript operator <https://wg21.link/P2128R6>`_.
- Implemented `P0849R8: auto(x): decay-copy in the language <https://wg21.link/P0849R8>`_.
- Implemented `P2242R3: Non-literal variables (and labels and gotos) in constexpr functions	<https://wg21.link/P2242R3>`_.

CUDA Language Changes in Clang
------------------------------

Objective-C Language Changes in Clang
-------------------------------------

OpenCL C Language Changes in Clang
----------------------------------

...

ABI Changes in Clang
--------------------

OpenMP Support in Clang
-----------------------

...

CUDA Support in Clang
---------------------

- ...

X86 Support in Clang
--------------------

DWARF Support in Clang
----------------------

Arm and AArch64 Support in Clang
--------------------------------

Floating Point Support in Clang
-------------------------------

Internal API Changes
--------------------

- Added a new attribute flag ``AcceptsExprPack`` that when set allows
  expression pack expansions in the parsed arguments of the corresponding
  attribute. Additionally it introduces delaying of attribute arguments, adding
  common handling for creating attributes that cannot be fully initialized
  prior to template instantiation.

Build System Changes
--------------------

* CMake ``-DCLANG_DEFAULT_PIE_ON_LINUX=ON`` is now the default. This is used by
  linux-gnu systems to decide whether ``-fPIE -pie`` is the default (instead of
  ``-fno-pic -no-pie``). This matches GCC installations on many Linux distros.
  Note: linux-android and linux-musl always default to ``-fPIE -pie``, ignoring
  this variable. ``-DCLANG_DEFAULT_PIE_ON_LINUX`` may be removed in the future.

AST Matchers
------------

- Expanded ``isInline`` narrowing matcher to support c++17 inline variables.

clang-format
------------

- **Important change**: Renamed ``IndentRequires`` to ``IndentRequiresClause``
  and changed the default for all styles from ``false`` to ``true``.

- Reworked and improved handling of concepts and requires. Added the
  ``RequiresClausePosition`` option as part of that.

- Changed ``BreakBeforeConceptDeclarations`` from ``Boolean`` to an enum.

- Option ``InsertBraces`` has been added to insert optional braces after control
  statements.

libclang
--------

- ...

Static Analyzer
---------------

- Added a new checker ``alpha.unix.cstring.UninitializedRead`` this will check for uninitialized reads
  from common memory copy/manipulation functions such as ``memcpy``, ``mempcpy``, ``memmove``, ``memcmp``, `
  `strcmp``, ``strncmp``, ``strcpy``, ``strlen``, ``strsep`` and many more. Although 
  this checker currently is in list of alpha checkers due to a false positive.

.. _release-notes-ubsan:

Undefined Behavior Sanitizer (UBSan)
------------------------------------

Core Analysis Improvements
==========================

- ...

New Issues Found
================

- ...

Python Binding Changes
----------------------

The following methods have been added:

-  ...

Significant Known Problems
==========================

Additional Information
======================

A wide variety of additional information is available on the `Clang web
page <https://clang.llvm.org/>`_. The web page contains versions of the
API documentation which are up-to-date with the Git version of
the source code. You can access versions of these documents specific to
this release by going into the "``clang/docs/``" directory in the Clang
tree.

If you have any questions or comments about Clang, please feel free to
contact us via the `mailing
list <https://lists.llvm.org/mailman/listinfo/cfe-dev>`_.
