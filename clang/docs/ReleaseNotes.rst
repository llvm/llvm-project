=========================
Clang 5.0.0 Release Notes
=========================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <http://llvm.org/>`_

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release 5.0.0. Here we
describe the status of Clang in some detail, including major
improvements from the previous release and new feature work. For the
general LLVM release notes, see `the LLVM
documentation <http://llvm.org/docs/ReleaseNotes.html>`_. All LLVM
releases may be downloaded from the `LLVM releases web
site <http://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about the
latest release, please see the `Clang Web Site <http://clang.llvm.org>`_ or the
`LLVM Web Site <http://llvm.org>`_.

What's New in Clang 5.0.0?
==========================

Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

Major New Features
------------------

C++ coroutines
^^^^^^^^^^^^^^
`C++ coroutines TS
<http://open-std.org/jtc1/sc22/wg21/docs/papers/2017/n4680.pdf>`_
implementation has landed. Use ``-fcoroutines-ts -stdlib=libc++`` to enable
coroutine support. Here is `an example
<https://wandbox.org/permlink/Dth1IO5q8Oe31ew2>`_ to get you started.


Improvements to Clang's diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  ``-Wcast-qual`` was implemented for C++. C-style casts are now properly
   diagnosed.

-  ``-Wunused-lambda-capture`` warns when a variable explicitly captured
   by a lambda is not used in the body of the lambda.

-  ``-Wstrict-prototypes`` is a new warning that warns about non-prototype
   function and block declarations and types in C and Objective-C.

-  ``-Wunguarded-availability`` is a new warning that warns about uses of new
   APIs that were introduced in a system whose version is newer than the
   deployment target version. A new Objective-C expression ``@available`` has
   been introduced to perform system version checking at runtime. This warning
   is off by default to prevent unexpected warnings in existing projects.
   However, its less strict sibling ``-Wunguarded-availability-new`` is on by
   default. It warns about unguarded uses of APIs only when they were introduced
   in or after macOS 10.13, iOS 11, tvOS 11 or watchOS 4.

-  The ``-Wdocumentation`` warning now allows the use of ``\param`` and
   ``\returns`` documentation directives in the documentation comments for
   declarations with a function or a block pointer type.

-  The compiler no longer warns about unreachable ``__builtin_unreachable``
   statements.

New Compiler Flags
------------------

- ``--autocomplete`` was implemented to obtain a list of flags and its arguments.
  This is used for shell autocompletion.

Deprecated Compiler Flags
-------------------------

The following options are deprecated and ignored. They will be removed in
future versions of Clang.

- ``-fslp-vectorize-aggressive`` used to enable the BB vectorizing pass. They have been superseeded
  by the normal SLP vectorizer.
- ``-fno-slp-vectorize-aggressive`` used to be the default behavior of clang.

New Pragmas in Clang
-----------------------

- Clang now supports the ``clang attribute`` pragma that allows users to apply
  an attribute to multiple declarations.

- ``pragma pack`` directives that are included in a precompiled header are now
  applied correctly to the declarations in the compilation unit that includes
  that precompiled header.

Attribute Changes in Clang
--------------------------

-  The ``overloadable`` attribute now allows at most one function with a given
   name to lack the ``overloadable`` attribute. This unmarked function will not
   have its name mangled.
-  The ``ms_abi`` attribute and the ``__builtin_ms_va_list`` types and builtins
   are now supported on AArch64.

C Language Changes in Clang
---------------------------

- Added near complete support for implicit scalar to vector conversion, a GNU
  C/C++ language extension. With this extension, the following code is
  considered valid:

.. code-block:: c

    typedef unsigned v4i32 __attribute__((vector_size(16)));

    v4i32 foo(v4i32 a) {
      // Here 5 is implicitly casted to an unsigned value and replicated into a
      // vector with as many elements as 'a'.
      return a + 5;
    }

The implicit conversion of a scalar value to a vector value--in the context of
a vector expression--occurs when:

- The type of the vector is that of a ``__attribute__((vector_size(size)))``
  vector, not an OpenCL ``__attribute__((ext_vector_type(size)))`` vector type.

- The scalar value can be casted to that of the vector element's type without
  the loss of precision based on the type of the scalar and the type of the
  vector's elements.

- For compile time constant values, the above rule is weakened to consider the
  value of the scalar constant rather than the constant's type. However,
  for compatibility with GCC, floating point constants with precise integral
  representations are not implicitly converted to integer values.

Currently the basic integer and floating point types with the following
operators are supported: ``+``, ``/``, ``-``, ``*``, ``%``, ``>``, ``<``,
``>=``, ``<=``, ``==``, ``!=``, ``&``, ``|``, ``^`` and the corresponding
assignment operators where applicable.


C++ Language Changes in Clang
-----------------------------

- We expect this to be the last Clang release that defaults to ``-std=gnu++98``
  when using the GCC-compatible ``clang++`` driver. From Clang 6 onwards we
  expect to use ``-std=gnu++14`` or a later standard by default, to match the
  behavior of recent GCC releases. Users are encouraged to change their build
  files to explicitly specify their desired C++ standard.

- Support for the C++17 standard has been completed. This mode can be enabled
  using ``-std=c++17`` (the old flag ``-std=c++1z`` is still supported for
  compatibility).

- When targeting a platform that uses the Itanium C++ ABI, Clang implements a
  `recent change to the ABI`__ that passes objects of class type indirectly if they
  have a non-trivial move constructor. Previous versions of Clang only
  considered the copy constructor, resulting in an ABI change in rare cases,
  but GCC has already implemented this change for several releases.
  This affects all targets other than Windows and PS4. You can opt out of this
  ABI change with ``-fclang-abi-compat=4.0``.

- As mentioned in `C Language Changes in Clang`_, Clang's support for
  implicit scalar to vector conversions also applies to C++. Additionally
  the following operators are also supported: ``&&`` and ``||``.

.. __: https://github.com/itanium-cxx-abi/cxx-abi/commit/7099637aba11fed6bdad7ee65bf4fd3f97fbf076

Objective-C Language Changes in Clang
-------------------------------------

- Clang now guarantees that a ``readwrite`` property is synthesized when an
  ambiguous property (i.e. a property that's declared in multiple protocols)
  is synthesized. The ``-Wprotocol-property-synthesis-ambiguity`` warning that
  warns about incompatible property types is now promoted to an error when
  there's an ambiguity between ``readwrite`` and ``readonly`` properties.

- Clang now prohibits synthesis of ambiguous properties with incompatible
  explicit property attributes. The following property attributes are
  checked for differences: ``copy``, ``retain``/``strong``, ``atomic``,
  ``getter`` and ``setter``.

OpenCL C Language Changes in Clang
----------------------------------

Various bug fixes and improvements:

-  Extended OpenCL-related Clang tests.

-  Improved diagnostics across several areas: scoped address space
   qualified variables, function pointers, atomics, type rank for overloading,
   block captures, ``reserve_id_t``.

-  Several address space related fixes for constant address space function scope variables,
   IR generation, mangling of ``generic`` and alloca (post-fix from general Clang
   refactoring of address spaces).

-  Several improvements in extensions: fixed OpenCL version for ``cl_khr_mipmap_image``,
   added missing ``cl_khr_3d_image_writes``.

-  Improvements in ``enqueue_kernel``, especially the implementation of ``ndrange_t`` and blocks.

-  OpenCL type related fixes: global samplers, the ``pipe_t`` size, internal type redefinition,
   and type compatibility checking in ternary and other operations.

-  The OpenCL header has been extended with missing extension guards, and direct mapping of ``as_type``
   to ``__builtin_astype``.

-  Fixed ``kernel_arg_type_qual`` and OpenCL/SPIR version in metadata.

-  Added proper use of the kernel calling convention to various targets.

The following new functionalities have been added:

-  Added documentation on OpenCL to Clang user manual.

-  Extended Clang builtins with required ``cl_khr_subgroups`` support.

-  Add ``intel_reqd_sub_group_size`` attribute support.

-  Added OpenCL types to ``CIndex``.


clang-format
------------

* Option **BreakBeforeInheritanceComma** added to break before ``:`` and ``,``  in case of
  multiple inheritance in a class declaration. Enabled by default in the Mozilla coding style.

  +---------------------+----------------------------------------+
  | true                | false                                  |
  +=====================+========================================+
  | .. code-block:: c++ | .. code-block:: c++                    |
  |                     |                                        |
  |   class MyClass     |   class MyClass : public X, public Y { |
  |       : public X    |   };                                   |
  |       , public Y {  |                                        |
  |   };                |                                        |
  +---------------------+----------------------------------------+

* Align block comment decorations.

  +----------------------+---------------------+
  | Before               | After               |
  +======================+=====================+
  |  .. code-block:: c++ | .. code-block:: c++ |
  |                      |                     |
  |    /* line 1         |   /* line 1         |
  |      * line 2        |    * line 2         |
  |     */               |    */               |
  +----------------------+---------------------+

* The :doc:`ClangFormatStyleOptions` documentation provides detailed examples for most options.

* Namespace end comments are now added or updated automatically.

  +---------------------+---------------------+
  | Before              | After               |
  +=====================+=====================+
  | .. code-block:: c++ | .. code-block:: c++ |
  |                     |                     |
  |   namespace A {     |   namespace A {     |
  |   int i;            |   int i;            |
  |   int j;            |   int j;            |
  |   }                 |   } // namespace A  |
  +---------------------+---------------------+

* Comment reflow support added. Overly long comment lines will now be reflown with the rest of
  the paragraph instead of just broken. Option **ReflowComments** added and enabled by default.

libclang
--------

- Libclang now provides code-completion results for more C++ constructs
  and keywords. The following keywords/identifiers are now included in the
  code-completion results: ``static_assert``, ``alignas``, ``constexpr``,
  ``final``, ``noexcept``, ``override`` and ``thread_local``.

- Libclang now provides code-completion results for members from dependent
  classes. For example:

  .. code-block:: c++

    template<typename T>
    void appendValue(std::vector<T> &dest, const T &value) {
        dest. // Relevant completion results are now shown after '.'
    }

  Note that code-completion results are still not provided when the member
  expression includes a dependent base expression. For example:

  .. code-block:: c++

    template<typename T>
    void appendValue(std::vector<std::vector<T>> &dest, const T &value) {
        dest.at(0). // Libclang fails to provide completion results after '.'
    }

Static Analyzer
---------------

- The static analyzer now supports using the
  `z3 theorem prover <https://github.com/z3prover/z3>`_ from Microsoft Research
  as an external constraint solver. This allows reasoning over more complex
  queries, but performance is ~15x slower than the default range-based
  constraint solver. To enable the z3 solver backend, clang must be built with
  the ``CLANG_ANALYZER_BUILD_Z3=ON`` option, and the
  ``-Xanalyzer -analyzer-constraints=z3`` arguments passed at runtime.

Undefined Behavior Sanitizer (UBSan)
------------------------------------

- The Undefined Behavior Sanitizer has a new check for pointer overflow. This
  check is on by default. The flag to control this functionality is
  ``-fsanitize=pointer-overflow``.

  Pointer overflow is an indicator of undefined behavior: when a pointer
  indexing expression wraps around the address space, or produces other
  unexpected results, its result may not point to a valid object.

- UBSan has several new checks which detect violations of nullability
  annotations. These checks are off by default. The flag to control this group
  of checks is ``-fsanitize=nullability``. The checks can be individially enabled
  by ``-fsanitize=nullability-arg`` (which checks calls),
  ``-fsanitize=nullability-assign`` (which checks assignments), and
  ``-fsanitize=nullability-return`` (which checks return statements).

- UBSan can now detect invalid loads from bitfields and from ObjC BOOLs.

- UBSan can now avoid emitting unnecessary type checks in C++ class methods and
  in several other cases where the result is known at compile-time. UBSan can
  also avoid emitting unnecessary overflow checks in arithmetic expressions
  with promoted integer operands.


Python Binding Changes
----------------------

Python bindings now support both Python 2 and Python 3.

The following methods have been added:

- ``is_scoped_enum`` has been added to ``Cursor``.

- ``exception_specification_kind`` has been added to ``Cursor``.

- ``get_address_space`` has been added to ``Type``.

- ``get_typedef_name`` has been added to ``Type``.

- ``get_exception_specification_kind`` has been added to ``Type``.


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
