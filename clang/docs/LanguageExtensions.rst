=========================
Clang Language Extensions
=========================

.. contents::
   :local:
   :depth: 1

.. toctree::
   :hidden:

   ObjectiveCLiterals
   BlockLanguageSpec
   Block-ABI-Apple
   AutomaticReferenceCounting

Introduction
============

This document describes the language extensions provided by Clang.  In addition
to the language extensions listed here, Clang aims to support a broad range of
GCC extensions.  Please see the `GCC manual
<https://gcc.gnu.org/onlinedocs/gcc/C-Extensions.html>`_ for more information on
these extensions.

.. _langext-feature_check:

Feature Checking Macros
=======================

Language extensions can be very useful, but only if you know you can depend on
them.  In order to allow fine-grain features checks, we support three builtin
function-like macros.  This allows you to directly test for a feature in your
code without having to resort to something like autoconf or fragile "compiler
version checks".

``__has_builtin``
-----------------

This function-like macro takes a single identifier argument that is the name of
a builtin function.  It evaluates to 1 if the builtin is supported or 0 if not.
It can be used like this:

.. code-block:: c++

  #ifndef __has_builtin         // Optional of course.
    #define __has_builtin(x) 0  // Compatibility with non-clang compilers.
  #endif

  ...
  #if __has_builtin(__builtin_trap)
    __builtin_trap();
  #else
    abort();
  #endif
  ...

.. _langext-__has_feature-__has_extension:

``__has_feature`` and ``__has_extension``
-----------------------------------------

These function-like macros take a single identifier argument that is the name
of a feature.  ``__has_feature`` evaluates to 1 if the feature is both
supported by Clang and standardized in the current language standard or 0 if
not (but see :ref:`below <langext-has-feature-back-compat>`), while
``__has_extension`` evaluates to 1 if the feature is supported by Clang in the
current language (either as a language extension or a standard language
feature) or 0 if not.  They can be used like this:

.. code-block:: c++

  #ifndef __has_feature         // Optional of course.
    #define __has_feature(x) 0  // Compatibility with non-clang compilers.
  #endif
  #ifndef __has_extension
    #define __has_extension __has_feature // Compatibility with pre-3.0 compilers.
  #endif

  ...
  #if __has_feature(cxx_rvalue_references)
  // This code will only be compiled with the -std=c++11 and -std=gnu++11
  // options, because rvalue references are only standardized in C++11.
  #endif

  #if __has_extension(cxx_rvalue_references)
  // This code will be compiled with the -std=c++11, -std=gnu++11, -std=c++98
  // and -std=gnu++98 options, because rvalue references are supported as a
  // language extension in C++98.
  #endif

.. _langext-has-feature-back-compat:

For backward compatibility, ``__has_feature`` can also be used to test
for support for non-standardized features, i.e. features not prefixed ``c_``,
``cxx_`` or ``objc_``.

Another use of ``__has_feature`` is to check for compiler features not related
to the language standard, such as e.g. :doc:`AddressSanitizer
<AddressSanitizer>`.

If the ``-pedantic-errors`` option is given, ``__has_extension`` is equivalent
to ``__has_feature``.

The feature tag is described along with the language feature below.

The feature name or extension name can also be specified with a preceding and
following ``__`` (double underscore) to avoid interference from a macro with
the same name.  For instance, ``__cxx_rvalue_references__`` can be used instead
of ``cxx_rvalue_references``.

``__has_cpp_attribute``
-----------------------

This function-like macro is available in C++2a by default, and is provided as an
extension in earlier language standards. It takes a single argument that is the
name of a double-square-bracket-style attribute. The argument can either be a
single identifier or a scoped identifier. If the attribute is supported, a
nonzero value is returned. If the attribute is a standards-based attribute, this
macro returns a nonzero value based on the year and month in which the attribute
was voted into the working draft. See `WG21 SD-6
<https://isocpp.org/std/standing-documents/sd-6-sg10-feature-test-recommendations>`_
for the list of values returned for standards-based attributes. If the attribute
is not supported by the current compliation target, this macro evaluates to 0.
It can be used like this:

.. code-block:: c++

  #ifndef __has_cpp_attribute         // For backwards compatibility
    #define __has_cpp_attribute(x) 0
  #endif

  ...
  #if __has_cpp_attribute(clang::fallthrough)
  #define FALLTHROUGH [[clang::fallthrough]]
  #else
  #define FALLTHROUGH
  #endif
  ...

The attribute scope tokens ``clang`` and ``_Clang`` are interchangeable, as are
the attribute scope tokens ``gnu`` and ``__gnu__``. Attribute tokens in either
of these namespaces can be specified with a preceding and following ``__``
(double underscore) to avoid interference from a macro with the same name. For
instance, ``gnu::__const__`` can be used instead of ``gnu::const``.

``__has_c_attribute``
---------------------

This function-like macro takes a single argument that is the name of an
attribute exposed with the double square-bracket syntax in C mode. The argument
can either be a single identifier or a scoped identifier. If the attribute is
supported, a nonzero value is returned. If the attribute is not supported by the
current compilation target, this macro evaluates to 0. It can be used like this:

.. code-block:: c

  #ifndef __has_c_attribute         // Optional of course.
    #define __has_c_attribute(x) 0  // Compatibility with non-clang compilers.
  #endif

  ...
  #if __has_c_attribute(fallthrough)
    #define FALLTHROUGH [[fallthrough]]
  #else
    #define FALLTHROUGH
  #endif
  ...

The attribute scope tokens ``clang`` and ``_Clang`` are interchangeable, as are
the attribute scope tokens ``gnu`` and ``__gnu__``. Attribute tokens in either
of these namespaces can be specified with a preceding and following ``__``
(double underscore) to avoid interference from a macro with the same name. For
instance, ``gnu::__const__`` can be used instead of ``gnu::const``.

``__has_attribute``
-------------------

This function-like macro takes a single identifier argument that is the name of
a GNU-style attribute.  It evaluates to 1 if the attribute is supported by the
current compilation target, or 0 if not.  It can be used like this:

.. code-block:: c++

  #ifndef __has_attribute         // Optional of course.
    #define __has_attribute(x) 0  // Compatibility with non-clang compilers.
  #endif

  ...
  #if __has_attribute(always_inline)
  #define ALWAYS_INLINE __attribute__((always_inline))
  #else
  #define ALWAYS_INLINE
  #endif
  ...

The attribute name can also be specified with a preceding and following ``__``
(double underscore) to avoid interference from a macro with the same name.  For
instance, ``__always_inline__`` can be used instead of ``always_inline``.


``__has_declspec_attribute``
----------------------------

This function-like macro takes a single identifier argument that is the name of
an attribute implemented as a Microsoft-style ``__declspec`` attribute.  It
evaluates to 1 if the attribute is supported by the current compilation target,
or 0 if not.  It can be used like this:

.. code-block:: c++

  #ifndef __has_declspec_attribute         // Optional of course.
    #define __has_declspec_attribute(x) 0  // Compatibility with non-clang compilers.
  #endif

  ...
  #if __has_declspec_attribute(dllexport)
  #define DLLEXPORT __declspec(dllexport)
  #else
  #define DLLEXPORT
  #endif
  ...

The attribute name can also be specified with a preceding and following ``__``
(double underscore) to avoid interference from a macro with the same name.  For
instance, ``__dllexport__`` can be used instead of ``dllexport``.

``__is_identifier``
-------------------

This function-like macro takes a single identifier argument that might be either
a reserved word or a regular identifier. It evaluates to 1 if the argument is just
a regular identifier and not a reserved word, in the sense that it can then be
used as the name of a user-defined function or variable. Otherwise it evaluates
to 0.  It can be used like this:

.. code-block:: c++

  ...
  #ifdef __is_identifier          // Compatibility with non-clang compilers.
    #if __is_identifier(__wchar_t)
      typedef wchar_t __wchar_t;
    #endif
  #endif

  __wchar_t WideCharacter;
  ...

Include File Checking Macros
============================

Not all developments systems have the same include files.  The
:ref:`langext-__has_include` and :ref:`langext-__has_include_next` macros allow
you to check for the existence of an include file before doing a possibly
failing ``#include`` directive.  Include file checking macros must be used
as expressions in ``#if`` or ``#elif`` preprocessing directives.

.. _langext-__has_include:

``__has_include``
-----------------

This function-like macro takes a single file name string argument that is the
name of an include file.  It evaluates to 1 if the file can be found using the
include paths, or 0 otherwise:

.. code-block:: c++

  // Note the two possible file name string formats.
  #if __has_include("myinclude.h") && __has_include(<stdint.h>)
  # include "myinclude.h"
  #endif

To test for this feature, use ``#if defined(__has_include)``:

.. code-block:: c++

  // To avoid problem with non-clang compilers not having this macro.
  #if defined(__has_include)
  #if __has_include("myinclude.h")
  # include "myinclude.h"
  #endif
  #endif

.. _langext-__has_include_next:

``__has_include_next``
----------------------

This function-like macro takes a single file name string argument that is the
name of an include file.  It is like ``__has_include`` except that it looks for
the second instance of the given file found in the include paths.  It evaluates
to 1 if the second instance of the file can be found using the include paths,
or 0 otherwise:

.. code-block:: c++

  // Note the two possible file name string formats.
  #if __has_include_next("myinclude.h") && __has_include_next(<stdint.h>)
  # include_next "myinclude.h"
  #endif

  // To avoid problem with non-clang compilers not having this macro.
  #if defined(__has_include_next)
  #if __has_include_next("myinclude.h")
  # include_next "myinclude.h"
  #endif
  #endif

Note that ``__has_include_next``, like the GNU extension ``#include_next``
directive, is intended for use in headers only, and will issue a warning if
used in the top-level compilation file.  A warning will also be issued if an
absolute path is used in the file argument.

``__has_warning``
-----------------

This function-like macro takes a string literal that represents a command line
option for a warning and returns true if that is a valid warning option.

.. code-block:: c++

  #if __has_warning("-Wformat")
  ...
  #endif

Builtin Macros
==============

``__BASE_FILE__``
  Defined to a string that contains the name of the main input file passed to
  Clang.

``__COUNTER__``
  Defined to an integer value that starts at zero and is incremented each time
  the ``__COUNTER__`` macro is expanded.

``__INCLUDE_LEVEL__``
  Defined to an integral value that is the include depth of the file currently
  being translated.  For the main file, this value is zero.

``__TIMESTAMP__``
  Defined to the date and time of the last modification of the current source
  file.

``__clang__``
  Defined when compiling with Clang

``__clang_major__``
  Defined to the major marketing version number of Clang (e.g., the 2 in
  2.0.1).  Note that marketing version numbers should not be used to check for
  language features, as different vendors use different numbering schemes.
  Instead, use the :ref:`langext-feature_check`.

``__clang_minor__``
  Defined to the minor version number of Clang (e.g., the 0 in 2.0.1).  Note
  that marketing version numbers should not be used to check for language
  features, as different vendors use different numbering schemes.  Instead, use
  the :ref:`langext-feature_check`.

``__clang_patchlevel__``
  Defined to the marketing patch level of Clang (e.g., the 1 in 2.0.1).

``__clang_version__``
  Defined to a string that captures the Clang marketing version, including the
  Subversion tag or revision number, e.g., "``1.5 (trunk 102332)``".

.. _langext-vectors:

Vectors and Extended Vectors
============================

Supports the GCC, OpenCL, AltiVec and NEON vector extensions.

OpenCL vector types are created using ``ext_vector_type`` attribute.  It
support for ``V.xyzw`` syntax and other tidbits as seen in OpenCL.  An example
is:

.. code-block:: c++

  typedef float float4 __attribute__((ext_vector_type(4)));
  typedef float float2 __attribute__((ext_vector_type(2)));

  float4 foo(float2 a, float2 b) {
    float4 c;
    c.xz = a;
    c.yw = b;
    return c;
  }

Query for this feature with ``__has_extension(attribute_ext_vector_type)``.

Giving ``-maltivec`` option to clang enables support for AltiVec vector syntax
and functions.  For example:

.. code-block:: c++

  vector float foo(vector int a) {
    vector int b;
    b = vec_add(a, a) + a;
    return (vector float)b;
  }

NEON vector types are created using ``neon_vector_type`` and
``neon_polyvector_type`` attributes.  For example:

.. code-block:: c++

  typedef __attribute__((neon_vector_type(8))) int8_t int8x8_t;
  typedef __attribute__((neon_polyvector_type(16))) poly8_t poly8x16_t;

  int8x8_t foo(int8x8_t a) {
    int8x8_t v;
    v = a;
    return v;
  }

Vector Literals
---------------

Vector literals can be used to create vectors from a set of scalars, or
vectors.  Either parentheses or braces form can be used.  In the parentheses
form the number of literal values specified must be one, i.e. referring to a
scalar value, or must match the size of the vector type being created.  If a
single scalar literal value is specified, the scalar literal value will be
replicated to all the components of the vector type.  In the brackets form any
number of literals can be specified.  For example:

.. code-block:: c++

  typedef int v4si __attribute__((__vector_size__(16)));
  typedef float float4 __attribute__((ext_vector_type(4)));
  typedef float float2 __attribute__((ext_vector_type(2)));

  v4si vsi = (v4si){1, 2, 3, 4};
  float4 vf = (float4)(1.0f, 2.0f, 3.0f, 4.0f);
  vector int vi1 = (vector int)(1);    // vi1 will be (1, 1, 1, 1).
  vector int vi2 = (vector int){1};    // vi2 will be (1, 0, 0, 0).
  vector int vi3 = (vector int)(1, 2); // error
  vector int vi4 = (vector int){1, 2}; // vi4 will be (1, 2, 0, 0).
  vector int vi5 = (vector int)(1, 2, 3, 4);
  float4 vf = (float4)((float2)(1.0f, 2.0f), (float2)(3.0f, 4.0f));

Vector Operations
-----------------

The table below shows the support for each operation by vector extension.  A
dash indicates that an operation is not accepted according to a corresponding
specification.

============================== ======= ======= ======= =======
         Operator              OpenCL  AltiVec   GCC    NEON
============================== ======= ======= ======= =======
[]                               yes     yes     yes     --
unary operators +, --            yes     yes     yes     --
++, -- --                        yes     yes     yes     --
+,--,*,/,%                       yes     yes     yes     --
bitwise operators &,|,^,~        yes     yes     yes     --
>>,<<                            yes     yes     yes     --
!, &&, ||                        yes     --      --      --
==, !=, >, <, >=, <=             yes     yes     --      --
=                                yes     yes     yes     yes
:?                               yes     --      --      --
sizeof                           yes     yes     yes     yes
C-style cast                     yes     yes     yes     no
reinterpret_cast                 yes     no      yes     no
static_cast                      yes     no      yes     no
const_cast                       no      no      no      no
============================== ======= ======= ======= =======

See also :ref:`langext-__builtin_shufflevector`, :ref:`langext-__builtin_convertvector`.

Half-Precision Floating Point
=============================

Clang supports two half-precision (16-bit) floating point types: ``__fp16`` and
``_Float16``.  These types are supported in all language modes.

``__fp16`` is supported on every target, as it is purely a storage format; see below.
``_Float16`` is currently only supported on the following targets, with further
targets pending ABI standardization:
- 32-bit ARM
- 64-bit ARM (AArch64)
- SPIR
``_Float16`` will be supported on more targets as they define ABIs for it.

``__fp16`` is a storage and interchange format only.  This means that values of
``__fp16`` are immediately promoted to (at least) ``float`` when used in arithmetic
operations, so that e.g. the result of adding two ``__fp16`` values has type ``float``.
The behavior of ``__fp16`` is specified by the ARM C Language Extensions (`ACLE <http://infocenter.arm.com/help/topic/com.arm.doc.ihi0053d/IHI0053D_acle_2_1.pdf>`_).
Clang uses the ``binary16`` format from IEEE 754-2008 for ``__fp16``, not the ARM
alternative format.

``_Float16`` is an extended floating-point type.  This means that, just like arithmetic on
``float`` or ``double``, arithmetic on ``_Float16`` operands is formally performed in the
``_Float16`` type, so that e.g. the result of adding two ``_Float16`` values has type
``_Float16``.  The behavior of ``_Float16`` is specified by ISO/IEC TS 18661-3:2015
("Floating-point extensions for C").  As with ``__fp16``, Clang uses the ``binary16``
format from IEEE 754-2008 for ``_Float16``.

``_Float16`` arithmetic will be performed using native half-precision support
when available on the target (e.g. on ARMv8.2a); otherwise it will be performed
at a higher precision (currently always ``float``) and then truncated down to
``_Float16``.  Note that C and C++ allow intermediate floating-point operands
of an expression to be computed with greater precision than is expressible in
their type, so Clang may avoid intermediate truncations in certain cases; this may
lead to results that are inconsistent with native arithmetic.

It is recommended that portable code use ``_Float16`` instead of ``__fp16``,
as it has been defined by the C standards committee and has behavior that is
more familiar to most programmers.

Because ``__fp16`` operands are always immediately promoted to ``float``, the
common real type of ``__fp16`` and ``_Float16`` for the purposes of the usual
arithmetic conversions is ``float``.

A literal can be given ``_Float16`` type using the suffix ``f16``; for example:
```
3.14f16
```

Because default argument promotion only applies to the standard floating-point
types, ``_Float16`` values are not promoted to ``double`` when passed as variadic
or untyped arguments.  As a consequence, some caution must be taken when using
certain library facilities with ``_Float16``; for example, there is no ``printf`` format
specifier for ``_Float16``, and (unlike ``float``) it will not be implicitly promoted to
``double`` when passed to ``printf``, so the programmer must explicitly cast it to
``double`` before using it with an ``%f`` or similar specifier.

Messages on ``deprecated`` and ``unavailable`` Attributes
=========================================================

An optional string message can be added to the ``deprecated`` and
``unavailable`` attributes.  For example:

.. code-block:: c++

  void explode(void) __attribute__((deprecated("extremely unsafe, use 'combust' instead!!!")));

If the deprecated or unavailable declaration is used, the message will be
incorporated into the appropriate diagnostic:

.. code-block:: none

  harmless.c:4:3: warning: 'explode' is deprecated: extremely unsafe, use 'combust' instead!!!
        [-Wdeprecated-declarations]
    explode();
    ^

Query for this feature with
``__has_extension(attribute_deprecated_with_message)`` and
``__has_extension(attribute_unavailable_with_message)``.

Attributes on Enumerators
=========================

Clang allows attributes to be written on individual enumerators.  This allows
enumerators to be deprecated, made unavailable, etc.  The attribute must appear
after the enumerator name and before any initializer, like so:

.. code-block:: c++

  enum OperationMode {
    OM_Invalid,
    OM_Normal,
    OM_Terrified __attribute__((deprecated)),
    OM_AbortOnError __attribute__((deprecated)) = 4
  };

Attributes on the ``enum`` declaration do not apply to individual enumerators.

Query for this feature with ``__has_extension(enumerator_attributes)``.

'User-Specified' System Frameworks
==================================

Clang provides a mechanism by which frameworks can be built in such a way that
they will always be treated as being "system frameworks", even if they are not
present in a system framework directory.  This can be useful to system
framework developers who want to be able to test building other applications
with development builds of their framework, including the manner in which the
compiler changes warning behavior for system headers.

Framework developers can opt-in to this mechanism by creating a
"``.system_framework``" file at the top-level of their framework.  That is, the
framework should have contents like:

.. code-block:: none

  .../TestFramework.framework
  .../TestFramework.framework/.system_framework
  .../TestFramework.framework/Headers
  .../TestFramework.framework/Headers/TestFramework.h
  ...

Clang will treat the presence of this file as an indicator that the framework
should be treated as a system framework, regardless of how it was found in the
framework search path.  For consistency, we recommend that such files never be
included in installed versions of the framework.

Checks for Standard Language Features
=====================================

The ``__has_feature`` macro can be used to query if certain standard language
features are enabled.  The ``__has_extension`` macro can be used to query if
language features are available as an extension when compiling for a standard
which does not provide them.  The features which can be tested are listed here.

Since Clang 3.4, the C++ SD-6 feature test macros are also supported.
These are macros with names of the form ``__cpp_<feature_name>``, and are
intended to be a portable way to query the supported features of the compiler.
See `the C++ status page <https://clang.llvm.org/cxx_status.html#ts>`_ for
information on the version of SD-6 supported by each Clang release, and the
macros provided by that revision of the recommendations.

C++98
-----

The features listed below are part of the C++98 standard.  These features are
enabled by default when compiling C++ code.

C++ exceptions
^^^^^^^^^^^^^^

Use ``__has_feature(cxx_exceptions)`` to determine if C++ exceptions have been
enabled.  For example, compiling code with ``-fno-exceptions`` disables C++
exceptions.

C++ RTTI
^^^^^^^^

Use ``__has_feature(cxx_rtti)`` to determine if C++ RTTI has been enabled.  For
example, compiling code with ``-fno-rtti`` disables the use of RTTI.

C++11
-----

The features listed below are part of the C++11 standard.  As a result, all
these features are enabled with the ``-std=c++11`` or ``-std=gnu++11`` option
when compiling C++ code.

C++11 SFINAE includes access control
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_access_control_sfinae)`` or
``__has_extension(cxx_access_control_sfinae)`` to determine whether
access-control errors (e.g., calling a private constructor) are considered to
be template argument deduction errors (aka SFINAE errors), per `C++ DR1170
<http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#1170>`_.

C++11 alias templates
^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_alias_templates)`` or
``__has_extension(cxx_alias_templates)`` to determine if support for C++11's
alias declarations and alias templates is enabled.

C++11 alignment specifiers
^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_alignas)`` or ``__has_extension(cxx_alignas)`` to
determine if support for alignment specifiers using ``alignas`` is enabled.

Use ``__has_feature(cxx_alignof)`` or ``__has_extension(cxx_alignof)`` to
determine if support for the ``alignof`` keyword is enabled.

C++11 attributes
^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_attributes)`` or ``__has_extension(cxx_attributes)`` to
determine if support for attribute parsing with C++11's square bracket notation
is enabled.

C++11 generalized constant expressions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_constexpr)`` to determine if support for generalized
constant expressions (e.g., ``constexpr``) is enabled.

C++11 ``decltype()``
^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_decltype)`` or ``__has_extension(cxx_decltype)`` to
determine if support for the ``decltype()`` specifier is enabled.  C++11's
``decltype`` does not require type-completeness of a function call expression.
Use ``__has_feature(cxx_decltype_incomplete_return_types)`` or
``__has_extension(cxx_decltype_incomplete_return_types)`` to determine if
support for this feature is enabled.

C++11 default template arguments in function templates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_default_function_template_args)`` or
``__has_extension(cxx_default_function_template_args)`` to determine if support
for default template arguments in function templates is enabled.

C++11 ``default``\ ed functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_defaulted_functions)`` or
``__has_extension(cxx_defaulted_functions)`` to determine if support for
defaulted function definitions (with ``= default``) is enabled.

C++11 delegating constructors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_delegating_constructors)`` to determine if support for
delegating constructors is enabled.

C++11 ``deleted`` functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_deleted_functions)`` or
``__has_extension(cxx_deleted_functions)`` to determine if support for deleted
function definitions (with ``= delete``) is enabled.

C++11 explicit conversion functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_explicit_conversions)`` to determine if support for
``explicit`` conversion functions is enabled.

C++11 generalized initializers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_generalized_initializers)`` to determine if support for
generalized initializers (using braced lists and ``std::initializer_list``) is
enabled.

C++11 implicit move constructors/assignment operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_implicit_moves)`` to determine if Clang will implicitly
generate move constructors and move assignment operators where needed.

C++11 inheriting constructors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_inheriting_constructors)`` to determine if support for
inheriting constructors is enabled.

C++11 inline namespaces
^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_inline_namespaces)`` or
``__has_extension(cxx_inline_namespaces)`` to determine if support for inline
namespaces is enabled.

C++11 lambdas
^^^^^^^^^^^^^

Use ``__has_feature(cxx_lambdas)`` or ``__has_extension(cxx_lambdas)`` to
determine if support for lambdas is enabled.

C++11 local and unnamed types as template arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_local_type_template_args)`` or
``__has_extension(cxx_local_type_template_args)`` to determine if support for
local and unnamed types as template arguments is enabled.

C++11 noexcept
^^^^^^^^^^^^^^

Use ``__has_feature(cxx_noexcept)`` or ``__has_extension(cxx_noexcept)`` to
determine if support for noexcept exception specifications is enabled.

C++11 in-class non-static data member initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_nonstatic_member_init)`` to determine whether in-class
initialization of non-static data members is enabled.

C++11 ``nullptr``
^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_nullptr)`` or ``__has_extension(cxx_nullptr)`` to
determine if support for ``nullptr`` is enabled.

C++11 ``override control``
^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_override_control)`` or
``__has_extension(cxx_override_control)`` to determine if support for the
override control keywords is enabled.

C++11 reference-qualified functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_reference_qualified_functions)`` or
``__has_extension(cxx_reference_qualified_functions)`` to determine if support
for reference-qualified functions (e.g., member functions with ``&`` or ``&&``
applied to ``*this``) is enabled.

C++11 range-based ``for`` loop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_range_for)`` or ``__has_extension(cxx_range_for)`` to
determine if support for the range-based for loop is enabled.

C++11 raw string literals
^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_raw_string_literals)`` to determine if support for raw
string literals (e.g., ``R"x(foo\bar)x"``) is enabled.

C++11 rvalue references
^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_rvalue_references)`` or
``__has_extension(cxx_rvalue_references)`` to determine if support for rvalue
references is enabled.

C++11 ``static_assert()``
^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_static_assert)`` or
``__has_extension(cxx_static_assert)`` to determine if support for compile-time
assertions using ``static_assert`` is enabled.

C++11 ``thread_local``
^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_thread_local)`` to determine if support for
``thread_local`` variables is enabled.

C++11 type inference
^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_auto_type)`` or ``__has_extension(cxx_auto_type)`` to
determine C++11 type inference is supported using the ``auto`` specifier.  If
this is disabled, ``auto`` will instead be a storage class specifier, as in C
or C++98.

C++11 strongly typed enumerations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_strong_enums)`` or
``__has_extension(cxx_strong_enums)`` to determine if support for strongly
typed, scoped enumerations is enabled.

C++11 trailing return type
^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_trailing_return)`` or
``__has_extension(cxx_trailing_return)`` to determine if support for the
alternate function declaration syntax with trailing return type is enabled.

C++11 Unicode string literals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_unicode_literals)`` to determine if support for Unicode
string literals is enabled.

C++11 unrestricted unions
^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_unrestricted_unions)`` to determine if support for
unrestricted unions is enabled.

C++11 user-defined literals
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_user_literals)`` to determine if support for
user-defined literals is enabled.

C++11 variadic templates
^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_variadic_templates)`` or
``__has_extension(cxx_variadic_templates)`` to determine if support for
variadic templates is enabled.

C++14
-----

The features listed below are part of the C++14 standard.  As a result, all
these features are enabled with the ``-std=C++14`` or ``-std=gnu++14`` option
when compiling C++ code.

C++14 binary literals
^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_binary_literals)`` or
``__has_extension(cxx_binary_literals)`` to determine whether
binary literals (for instance, ``0b10010``) are recognized. Clang supports this
feature as an extension in all language modes.

C++14 contextual conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_contextual_conversions)`` or
``__has_extension(cxx_contextual_conversions)`` to determine if the C++14 rules
are used when performing an implicit conversion for an array bound in a
*new-expression*, the operand of a *delete-expression*, an integral constant
expression, or a condition in a ``switch`` statement.

C++14 decltype(auto)
^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_decltype_auto)`` or
``__has_extension(cxx_decltype_auto)`` to determine if support
for the ``decltype(auto)`` placeholder type is enabled.

C++14 default initializers for aggregates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_aggregate_nsdmi)`` or
``__has_extension(cxx_aggregate_nsdmi)`` to determine if support
for default initializers in aggregate members is enabled.

C++14 digit separators
^^^^^^^^^^^^^^^^^^^^^^

Use ``__cpp_digit_separators`` to determine if support for digit separators
using single quotes (for instance, ``10'000``) is enabled. At this time, there
is no corresponding ``__has_feature`` name

C++14 generalized lambda capture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_init_captures)`` or
``__has_extension(cxx_init_captures)`` to determine if support for
lambda captures with explicit initializers is enabled
(for instance, ``[n(0)] { return ++n; }``).

C++14 generic lambdas
^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_generic_lambdas)`` or
``__has_extension(cxx_generic_lambdas)`` to determine if support for generic
(polymorphic) lambdas is enabled
(for instance, ``[] (auto x) { return x + 1; }``).

C++14 relaxed constexpr
^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_relaxed_constexpr)`` or
``__has_extension(cxx_relaxed_constexpr)`` to determine if variable
declarations, local variable modification, and control flow constructs
are permitted in ``constexpr`` functions.

C++14 return type deduction
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_return_type_deduction)`` or
``__has_extension(cxx_return_type_deduction)`` to determine if support
for return type deduction for functions (using ``auto`` as a return type)
is enabled.

C++14 runtime-sized arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_runtime_array)`` or
``__has_extension(cxx_runtime_array)`` to determine if support
for arrays of runtime bound (a restricted form of variable-length arrays)
is enabled.
Clang's implementation of this feature is incomplete.

C++14 variable templates
^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(cxx_variable_templates)`` or
``__has_extension(cxx_variable_templates)`` to determine if support for
templated variable declarations is enabled.

C11
---

The features listed below are part of the C11 standard.  As a result, all these
features are enabled with the ``-std=c11`` or ``-std=gnu11`` option when
compiling C code.  Additionally, because these features are all
backward-compatible, they are available as extensions in all language modes.

C11 alignment specifiers
^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(c_alignas)`` or ``__has_extension(c_alignas)`` to determine
if support for alignment specifiers using ``_Alignas`` is enabled.

Use ``__has_feature(c_alignof)`` or ``__has_extension(c_alignof)`` to determine
if support for the ``_Alignof`` keyword is enabled.

C11 atomic operations
^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(c_atomic)`` or ``__has_extension(c_atomic)`` to determine
if support for atomic types using ``_Atomic`` is enabled.  Clang also provides
:ref:`a set of builtins <langext-__c11_atomic>` which can be used to implement
the ``<stdatomic.h>`` operations on ``_Atomic`` types. Use
``__has_include(<stdatomic.h>)`` to determine if C11's ``<stdatomic.h>`` header
is available.

Clang will use the system's ``<stdatomic.h>`` header when one is available, and
will otherwise use its own. When using its own, implementations of the atomic
operations are provided as macros. In the cases where C11 also requires a real
function, this header provides only the declaration of that function (along
with a shadowing macro implementation), and you must link to a library which
provides a definition of the function if you use it instead of the macro.

C11 generic selections
^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(c_generic_selections)`` or
``__has_extension(c_generic_selections)`` to determine if support for generic
selections is enabled.

As an extension, the C11 generic selection expression is available in all
languages supported by Clang.  The syntax is the same as that given in the C11
standard.

In C, type compatibility is decided according to the rules given in the
appropriate standard, but in C++, which lacks the type compatibility rules used
in C, types are considered compatible only if they are equivalent.

C11 ``_Static_assert()``
^^^^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(c_static_assert)`` or ``__has_extension(c_static_assert)``
to determine if support for compile-time assertions using ``_Static_assert`` is
enabled.

C11 ``_Thread_local``
^^^^^^^^^^^^^^^^^^^^^

Use ``__has_feature(c_thread_local)`` or ``__has_extension(c_thread_local)``
to determine if support for ``_Thread_local`` variables is enabled.

Modules
-------

Use ``__has_feature(modules)`` to determine if Modules have been enabled.
For example, compiling code with ``-fmodules`` enables the use of Modules.

More information could be found `here <https://clang.llvm.org/docs/Modules.html>`_.

Checks for Type Trait Primitives
================================

Type trait primitives are special builtin constant expressions that can be used
by the standard C++ library to facilitate or simplify the implementation of
user-facing type traits in the <type_traits> header.

They are not intended to be used directly by user code because they are
implementation-defined and subject to change -- as such they're tied closely to
the supported set of system headers, currently:

* LLVM's own libc++
* GNU libstdc++
* The Microsoft standard C++ library

Clang supports the `GNU C++ type traits
<https://gcc.gnu.org/onlinedocs/gcc/Type-Traits.html>`_ and a subset of the
`Microsoft Visual C++ Type traits
<https://msdn.microsoft.com/en-us/library/ms177194(v=VS.100).aspx>`_.

Feature detection is supported only for some of the primitives at present. User
code should not use these checks because they bear no direct relation to the
actual set of type traits supported by the C++ standard library.

For type trait ``__X``, ``__has_extension(X)`` indicates the presence of the
type trait primitive in the compiler. A simplistic usage example as might be
seen in standard C++ headers follows:

.. code-block:: c++

  #if __has_extension(is_convertible_to)
  template<typename From, typename To>
  struct is_convertible_to {
    static const bool value = __is_convertible_to(From, To);
  };
  #else
  // Emulate type trait for compatibility with other compilers.
  #endif

The following type trait primitives are supported by Clang:

* ``__has_nothrow_assign`` (GNU, Microsoft)
* ``__has_nothrow_copy`` (GNU, Microsoft)
* ``__has_nothrow_constructor`` (GNU, Microsoft)
* ``__has_trivial_assign`` (GNU, Microsoft)
* ``__has_trivial_copy`` (GNU, Microsoft)
* ``__has_trivial_constructor`` (GNU, Microsoft)
* ``__has_trivial_destructor`` (GNU, Microsoft)
* ``__has_virtual_destructor`` (GNU, Microsoft)
* ``__is_abstract`` (GNU, Microsoft)
* ``__is_aggregate`` (GNU, Microsoft)
* ``__is_base_of`` (GNU, Microsoft)
* ``__is_class`` (GNU, Microsoft)
* ``__is_convertible_to`` (Microsoft)
* ``__is_empty`` (GNU, Microsoft)
* ``__is_enum`` (GNU, Microsoft)
* ``__is_interface_class`` (Microsoft)
* ``__is_pod`` (GNU, Microsoft)
* ``__is_polymorphic`` (GNU, Microsoft)
* ``__is_union`` (GNU, Microsoft)
* ``__is_literal(type)``: Determines whether the given type is a literal type
* ``__is_final``: Determines whether the given type is declared with a
  ``final`` class-virt-specifier.
* ``__underlying_type(type)``: Retrieves the underlying type for a given
  ``enum`` type.  This trait is required to implement the C++11 standard
  library.
* ``__is_trivially_assignable(totype, fromtype)``: Determines whether a value
  of type ``totype`` can be assigned to from a value of type ``fromtype`` such
  that no non-trivial functions are called as part of that assignment.  This
  trait is required to implement the C++11 standard library.
* ``__is_trivially_constructible(type, argtypes...)``: Determines whether a
  value of type ``type`` can be direct-initialized with arguments of types
  ``argtypes...`` such that no non-trivial functions are called as part of
  that initialization.  This trait is required to implement the C++11 standard
  library.
* ``__is_destructible`` (MSVC 2013)
* ``__is_nothrow_destructible`` (MSVC 2013)
* ``__is_nothrow_assignable`` (MSVC 2013, clang)
* ``__is_constructible`` (MSVC 2013, clang)
* ``__is_nothrow_constructible`` (MSVC 2013, clang)
* ``__is_assignable`` (MSVC 2015, clang)
* ``__reference_binds_to_temporary(T, U)`` (Clang):  Determines whether a
  reference of type ``T`` bound to an expression of type ``U`` would bind to a
  materialized temporary object. If ``T`` is not a reference type the result
  is false. Note this trait will also return false when the initialization of
  ``T`` from ``U`` is ill-formed.

Blocks
======

The syntax and high level language feature description is in
:doc:`BlockLanguageSpec<BlockLanguageSpec>`. Implementation and ABI details for
the clang implementation are in :doc:`Block-ABI-Apple<Block-ABI-Apple>`.

Query for this feature with ``__has_extension(blocks)``.

Objective-C Features
====================

Related result types
--------------------

According to Cocoa conventions, Objective-C methods with certain names
("``init``", "``alloc``", etc.) always return objects that are an instance of
the receiving class's type.  Such methods are said to have a "related result
type", meaning that a message send to one of these methods will have the same
static type as an instance of the receiver class.  For example, given the
following classes:

.. code-block:: objc

  @interface NSObject
  + (id)alloc;
  - (id)init;
  @end

  @interface NSArray : NSObject
  @end

and this common initialization pattern

.. code-block:: objc

  NSArray *array = [[NSArray alloc] init];

the type of the expression ``[NSArray alloc]`` is ``NSArray*`` because
``alloc`` implicitly has a related result type.  Similarly, the type of the
expression ``[[NSArray alloc] init]`` is ``NSArray*``, since ``init`` has a
related result type and its receiver is known to have the type ``NSArray *``.
If neither ``alloc`` nor ``init`` had a related result type, the expressions
would have had type ``id``, as declared in the method signature.

A method with a related result type can be declared by using the type
``instancetype`` as its result type.  ``instancetype`` is a contextual keyword
that is only permitted in the result type of an Objective-C method, e.g.

.. code-block:: objc

  @interface A
  + (instancetype)constructAnA;
  @end

The related result type can also be inferred for some methods.  To determine
whether a method has an inferred related result type, the first word in the
camel-case selector (e.g., "``init``" in "``initWithObjects``") is considered,
and the method will have a related result type if its return type is compatible
with the type of its class and if:

* the first word is "``alloc``" or "``new``", and the method is a class method,
  or

* the first word is "``autorelease``", "``init``", "``retain``", or "``self``",
  and the method is an instance method.

If a method with a related result type is overridden by a subclass method, the
subclass method must also return a type that is compatible with the subclass
type.  For example:

.. code-block:: objc

  @interface NSString : NSObject
  - (NSUnrelated *)init; // incorrect usage: NSUnrelated is not NSString or a superclass of NSString
  @end

Related result types only affect the type of a message send or property access
via the given method.  In all other respects, a method with a related result
type is treated the same way as method that returns ``id``.

Use ``__has_feature(objc_instancetype)`` to determine whether the
``instancetype`` contextual keyword is available.

Automatic reference counting
----------------------------

Clang provides support for :doc:`automated reference counting
<AutomaticReferenceCounting>` in Objective-C, which eliminates the need
for manual ``retain``/``release``/``autorelease`` message sends.  There are three
feature macros associated with automatic reference counting:
``__has_feature(objc_arc)`` indicates the availability of automated reference
counting in general, while ``__has_feature(objc_arc_weak)`` indicates that
automated reference counting also includes support for ``__weak`` pointers to
Objective-C objects. ``__has_feature(objc_arc_fields)`` indicates that C structs
are allowed to have fields that are pointers to Objective-C objects managed by
automatic reference counting.

.. _objc-weak:

Weak references
---------------

Clang supports ARC-style weak and unsafe references in Objective-C even
outside of ARC mode.  Weak references must be explicitly enabled with
the ``-fobjc-weak`` option; use ``__has_feature((objc_arc_weak))``
to test whether they are enabled.  Unsafe references are enabled
unconditionally.  ARC-style weak and unsafe references cannot be used
when Objective-C garbage collection is enabled.

Except as noted below, the language rules for the ``__weak`` and
``__unsafe_unretained`` qualifiers (and the ``weak`` and
``unsafe_unretained`` property attributes) are just as laid out
in the :doc:`ARC specification <AutomaticReferenceCounting>`.
In particular, note that some classes do not support forming weak
references to their instances, and note that special care must be
taken when storing weak references in memory where initialization
and deinitialization are outside the responsibility of the compiler
(such as in ``malloc``-ed memory).

Loading from a ``__weak`` variable always implicitly retains the
loaded value.  In non-ARC modes, this retain is normally balanced
by an implicit autorelease.  This autorelease can be suppressed
by performing the load in the receiver position of a ``-retain``
message send (e.g. ``[weakReference retain]``); note that this performs
only a single retain (the retain done when primitively loading from
the weak reference).

For the most part, ``__unsafe_unretained`` in non-ARC modes is just the
default behavior of variables and therefore is not needed.  However,
it does have an effect on the semantics of block captures: normally,
copying a block which captures an Objective-C object or block pointer
causes the captured pointer to be retained or copied, respectively,
but that behavior is suppressed when the captured variable is qualified
with ``__unsafe_unretained``.

Note that the ``__weak`` qualifier formerly meant the GC qualifier in
all non-ARC modes and was silently ignored outside of GC modes.  It now
means the ARC-style qualifier in all non-GC modes and is no longer
allowed if not enabled by either ``-fobjc-arc`` or ``-fobjc-weak``.
It is expected that ``-fobjc-weak`` will eventually be enabled by default
in all non-GC Objective-C modes.

.. _objc-fixed-enum:

Enumerations with a fixed underlying type
-----------------------------------------

Clang provides support for C++11 enumerations with a fixed underlying type
within Objective-C.  For example, one can write an enumeration type as:

.. code-block:: c++

  typedef enum : unsigned char { Red, Green, Blue } Color;

This specifies that the underlying type, which is used to store the enumeration
value, is ``unsigned char``.

Use ``__has_feature(objc_fixed_enum)`` to determine whether support for fixed
underlying types is available in Objective-C.

Interoperability with C++11 lambdas
-----------------------------------

Clang provides interoperability between C++11 lambdas and blocks-based APIs, by
permitting a lambda to be implicitly converted to a block pointer with the
corresponding signature.  For example, consider an API such as ``NSArray``'s
array-sorting method:

.. code-block:: objc

  - (NSArray *)sortedArrayUsingComparator:(NSComparator)cmptr;

``NSComparator`` is simply a typedef for the block pointer ``NSComparisonResult
(^)(id, id)``, and parameters of this type are generally provided with block
literals as arguments.  However, one can also use a C++11 lambda so long as it
provides the same signature (in this case, accepting two parameters of type
``id`` and returning an ``NSComparisonResult``):

.. code-block:: objc

  NSArray *array = @[@"string 1", @"string 21", @"string 12", @"String 11",
                     @"String 02"];
  const NSStringCompareOptions comparisonOptions
    = NSCaseInsensitiveSearch | NSNumericSearch |
      NSWidthInsensitiveSearch | NSForcedOrderingSearch;
  NSLocale *currentLocale = [NSLocale currentLocale];
  NSArray *sorted
    = [array sortedArrayUsingComparator:[=](id s1, id s2) -> NSComparisonResult {
               NSRange string1Range = NSMakeRange(0, [s1 length]);
               return [s1 compare:s2 options:comparisonOptions
               range:string1Range locale:currentLocale];
       }];
  NSLog(@"sorted: %@", sorted);

This code relies on an implicit conversion from the type of the lambda
expression (an unnamed, local class type called the *closure type*) to the
corresponding block pointer type.  The conversion itself is expressed by a
conversion operator in that closure type that produces a block pointer with the
same signature as the lambda itself, e.g.,

.. code-block:: objc

  operator NSComparisonResult (^)(id, id)() const;

This conversion function returns a new block that simply forwards the two
parameters to the lambda object (which it captures by copy), then returns the
result.  The returned block is first copied (with ``Block_copy``) and then
autoreleased.  As an optimization, if a lambda expression is immediately
converted to a block pointer (as in the first example, above), then the block
is not copied and autoreleased: rather, it is given the same lifetime as a
block literal written at that point in the program, which avoids the overhead
of copying a block to the heap in the common case.

The conversion from a lambda to a block pointer is only available in
Objective-C++, and not in C++ with blocks, due to its use of Objective-C memory
management (autorelease).

Object Literals and Subscripting
--------------------------------

Clang provides support for :doc:`Object Literals and Subscripting
<ObjectiveCLiterals>` in Objective-C, which simplifies common Objective-C
programming patterns, makes programs more concise, and improves the safety of
container creation.  There are several feature macros associated with object
literals and subscripting: ``__has_feature(objc_array_literals)`` tests the
availability of array literals; ``__has_feature(objc_dictionary_literals)``
tests the availability of dictionary literals;
``__has_feature(objc_subscripting)`` tests the availability of object
subscripting.

Objective-C Autosynthesis of Properties
---------------------------------------

Clang provides support for autosynthesis of declared properties.  Using this
feature, clang provides default synthesis of those properties not declared
@dynamic and not having user provided backing getter and setter methods.
``__has_feature(objc_default_synthesize_properties)`` checks for availability
of this feature in version of clang being used.

.. _langext-objc-retain-release:

Objective-C retaining behavior attributes
-----------------------------------------

In Objective-C, functions and methods are generally assumed to follow the
`Cocoa Memory Management 
<https://developer.apple.com/library/mac/#documentation/Cocoa/Conceptual/MemoryMgmt/Articles/mmRules.html>`_
conventions for ownership of object arguments and
return values. However, there are exceptions, and so Clang provides attributes
to allow these exceptions to be documented. This are used by ARC and the
`static analyzer <https://clang-analyzer.llvm.org>`_ Some exceptions may be
better described using the ``objc_method_family`` attribute instead.

**Usage**: The ``ns_returns_retained``, ``ns_returns_not_retained``,
``ns_returns_autoreleased``, ``cf_returns_retained``, and
``cf_returns_not_retained`` attributes can be placed on methods and functions
that return Objective-C or CoreFoundation objects. They are commonly placed at
the end of a function prototype or method declaration:

.. code-block:: objc

  id foo() __attribute__((ns_returns_retained));

  - (NSString *)bar:(int)x __attribute__((ns_returns_retained));

The ``*_returns_retained`` attributes specify that the returned object has a +1
retain count.  The ``*_returns_not_retained`` attributes specify that the return
object has a +0 retain count, even if the normal convention for its selector
would be +1.  ``ns_returns_autoreleased`` specifies that the returned object is
+0, but is guaranteed to live at least as long as the next flush of an
autorelease pool.

**Usage**: The ``ns_consumed`` and ``cf_consumed`` attributes can be placed on
an parameter declaration; they specify that the argument is expected to have a
+1 retain count, which will be balanced in some way by the function or method.
The ``ns_consumes_self`` attribute can only be placed on an Objective-C
method; it specifies that the method expects its ``self`` parameter to have a
+1 retain count, which it will balance in some way.

.. code-block:: objc

  void foo(__attribute__((ns_consumed)) NSString *string);

  - (void) bar __attribute__((ns_consumes_self));
  - (void) baz:(id) __attribute__((ns_consumed)) x;

Further examples of these attributes are available in the static analyzer's `list of annotations for analysis
<https://clang-analyzer.llvm.org/annotations.html#cocoa_mem>`_.

Query for these features with ``__has_attribute(ns_consumed)``,
``__has_attribute(ns_returns_retained)``, etc.

Objective-C @available
----------------------

It is possible to use the newest SDK but still build a program that can run on
older versions of macOS and iOS by passing ``-mmacosx-version-min=`` /
``-miphoneos-version-min=``.

Before LLVM 5.0, when calling a function that exists only in the OS that's
newer than the target OS (as determined by the minimum deployment version),
programmers had to carefully check if the function exists at runtime, using
null checks for weakly-linked C functions, ``+class`` for Objective-C classes,
and ``-respondsToSelector:`` or ``+instancesRespondToSelector:`` for
Objective-C methods.  If such a check was missed, the program would compile
fine, run fine on newer systems, but crash on older systems.

As of LLVM 5.0, ``-Wunguarded-availability`` uses the `availability attributes
<https://clang.llvm.org/docs/AttributeReference.html#availability>`_ together
with the new ``@available()`` keyword to assist with this issue.
When a method that's introduced in the OS newer than the target OS is called, a
-Wunguarded-availability warning is emitted if that call is not guarded:

.. code-block:: objc

  void my_fun(NSSomeClass* var) {
    // If fancyNewMethod was added in e.g. macOS 10.12, but the code is
    // built with -mmacosx-version-min=10.11, then this unconditional call
    // will emit a -Wunguarded-availability warning:
    [var fancyNewMethod];
  }

To fix the warning and to avoid the crash on macOS 10.11, wrap it in
``if(@available())``:

.. code-block:: objc

  void my_fun(NSSomeClass* var) {
    if (@available(macOS 10.12, *)) {
      [var fancyNewMethod];
    } else {
      // Put fallback behavior for old macOS versions (and for non-mac
      // platforms) here.
    }
  }

The ``*`` is required and means that platforms not explicitly listed will take
the true branch, and the compiler will emit ``-Wunguarded-availability``
warnings for unlisted platforms based on those platform's deployment target.
More than one platform can be listed in ``@available()``:

.. code-block:: objc

  void my_fun(NSSomeClass* var) {
    if (@available(macOS 10.12, iOS 10, *)) {
      [var fancyNewMethod];
    }
  }

If the caller of ``my_fun()`` already checks that ``my_fun()`` is only called
on 10.12, then add an `availability attribute
<https://clang.llvm.org/docs/AttributeReference.html#availability>`_ to it,
which will also suppress the warning and require that calls to my_fun() are
checked:

.. code-block:: objc

  API_AVAILABLE(macos(10.12)) void my_fun(NSSomeClass* var) {
    [var fancyNewMethod];  // Now ok.
  }

``@available()`` is only available in Objective-C code.  To use the feature
in C and C++ code, use the ``__builtin_available()`` spelling instead.

If existing code uses null checks or ``-respondsToSelector:``, it should
be changed to use ``@available()`` (or ``__builtin_available``) instead.

``-Wunguarded-availability`` is disabled by default, but
``-Wunguarded-availability-new``, which only emits this warning for APIs
that have been introduced in macOS >= 10.13, iOS >= 11, watchOS >= 4 and
tvOS >= 11, is enabled by default.

.. _langext-overloading:

Objective-C++ ABI: protocol-qualifier mangling of parameters
------------------------------------------------------------

Starting with LLVM 3.4, Clang produces a new mangling for parameters whose
type is a qualified-``id`` (e.g., ``id<Foo>``).  This mangling allows such
parameters to be differentiated from those with the regular unqualified ``id``
type.

This was a non-backward compatible mangling change to the ABI.  This change
allows proper overloading, and also prevents mangling conflicts with template
parameters of protocol-qualified type.

Query the presence of this new mangling with
``__has_feature(objc_protocol_qualifier_mangling)``.

Initializer lists for complex numbers in C
==========================================

clang supports an extension which allows the following in C:

.. code-block:: c++

  #include <math.h>
  #include <complex.h>
  complex float x = { 1.0f, INFINITY }; // Init to (1, Inf)

This construct is useful because there is no way to separately initialize the
real and imaginary parts of a complex variable in standard C, given that clang
does not support ``_Imaginary``.  (Clang also supports the ``__real__`` and
``__imag__`` extensions from gcc, which help in some cases, but are not usable
in static initializers.)

Note that this extension does not allow eliding the braces; the meaning of the
following two lines is different:

.. code-block:: c++

  complex float x[] = { { 1.0f, 1.0f } }; // [0] = (1, 1)
  complex float x[] = { 1.0f, 1.0f }; // [0] = (1, 0), [1] = (1, 0)

This extension also works in C++ mode, as far as that goes, but does not apply
to the C++ ``std::complex``.  (In C++11, list initialization allows the same
syntax to be used with ``std::complex`` with the same meaning.)

Builtin Functions
=================

Clang supports a number of builtin library functions with the same syntax as
GCC, including things like ``__builtin_nan``, ``__builtin_constant_p``,
``__builtin_choose_expr``, ``__builtin_types_compatible_p``,
``__builtin_assume_aligned``, ``__sync_fetch_and_add``, etc.  In addition to
the GCC builtins, Clang supports a number of builtins that GCC does not, which
are listed here.

Please note that Clang does not and will not support all of the GCC builtins
for vector operations.  Instead of using builtins, you should use the functions
defined in target-specific header files like ``<xmmintrin.h>``, which define
portable wrappers for these.  Many of the Clang versions of these functions are
implemented directly in terms of :ref:`extended vector support
<langext-vectors>` instead of builtins, in order to reduce the number of
builtins that we need to implement.

``__builtin_assume``
------------------------------

``__builtin_assume`` is used to provide the optimizer with a boolean
invariant that is defined to be true.

**Syntax**:

.. code-block:: c++

  __builtin_assume(bool)

**Example of Use**:

.. code-block:: c++

  int foo(int x) {
    __builtin_assume(x != 0);

    // The optimizer may short-circuit this check using the invariant.
    if (x == 0)
      return do_something();

    return do_something_else();
  }

**Description**:

The boolean argument to this function is defined to be true. The optimizer may
analyze the form of the expression provided as the argument and deduce from
that information used to optimize the program. If the condition is violated
during execution, the behavior is undefined. The argument itself is never
evaluated, so any side effects of the expression will be discarded.

Query for this feature with ``__has_builtin(__builtin_assume)``.

``__builtin_readcyclecounter``
------------------------------

``__builtin_readcyclecounter`` is used to access the cycle counter register (or
a similar low-latency, high-accuracy clock) on those targets that support it.

**Syntax**:

.. code-block:: c++

  __builtin_readcyclecounter()

**Example of Use**:

.. code-block:: c++

  unsigned long long t0 = __builtin_readcyclecounter();
  do_something();
  unsigned long long t1 = __builtin_readcyclecounter();
  unsigned long long cycles_to_do_something = t1 - t0; // assuming no overflow

**Description**:

The ``__builtin_readcyclecounter()`` builtin returns the cycle counter value,
which may be either global or process/thread-specific depending on the target.
As the backing counters often overflow quickly (on the order of seconds) this
should only be used for timing small intervals.  When not supported by the
target, the return value is always zero.  This builtin takes no arguments and
produces an unsigned long long result.

Query for this feature with ``__has_builtin(__builtin_readcyclecounter)``. Note
that even if present, its use may depend on run-time privilege or other OS
controlled state.

.. _langext-__builtin_shufflevector:

``__builtin_shufflevector``
---------------------------

``__builtin_shufflevector`` is used to express generic vector
permutation/shuffle/swizzle operations.  This builtin is also very important
for the implementation of various target-specific header files like
``<xmmintrin.h>``.

**Syntax**:

.. code-block:: c++

  __builtin_shufflevector(vec1, vec2, index1, index2, ...)

**Examples**:

.. code-block:: c++

  // identity operation - return 4-element vector v1.
  __builtin_shufflevector(v1, v1, 0, 1, 2, 3)

  // "Splat" element 0 of V1 into a 4-element result.
  __builtin_shufflevector(V1, V1, 0, 0, 0, 0)

  // Reverse 4-element vector V1.
  __builtin_shufflevector(V1, V1, 3, 2, 1, 0)

  // Concatenate every other element of 4-element vectors V1 and V2.
  __builtin_shufflevector(V1, V2, 0, 2, 4, 6)

  // Concatenate every other element of 8-element vectors V1 and V2.
  __builtin_shufflevector(V1, V2, 0, 2, 4, 6, 8, 10, 12, 14)

  // Shuffle v1 with some elements being undefined
  __builtin_shufflevector(v1, v1, 3, -1, 1, -1)

**Description**:

The first two arguments to ``__builtin_shufflevector`` are vectors that have
the same element type.  The remaining arguments are a list of integers that
specify the elements indices of the first two vectors that should be extracted
and returned in a new vector.  These element indices are numbered sequentially
starting with the first vector, continuing into the second vector.  Thus, if
``vec1`` is a 4-element vector, index 5 would refer to the second element of
``vec2``. An index of -1 can be used to indicate that the corresponding element
in the returned vector is a don't care and can be optimized by the backend.

The result of ``__builtin_shufflevector`` is a vector with the same element
type as ``vec1``/``vec2`` but that has an element count equal to the number of
indices specified.

Query for this feature with ``__has_builtin(__builtin_shufflevector)``.

.. _langext-__builtin_convertvector:

``__builtin_convertvector``
---------------------------

``__builtin_convertvector`` is used to express generic vector
type-conversion operations. The input vector and the output vector
type must have the same number of elements.

**Syntax**:

.. code-block:: c++

  __builtin_convertvector(src_vec, dst_vec_type)

**Examples**:

.. code-block:: c++

  typedef double vector4double __attribute__((__vector_size__(32)));
  typedef float  vector4float  __attribute__((__vector_size__(16)));
  typedef short  vector4short  __attribute__((__vector_size__(8)));
  vector4float vf; vector4short vs;

  // convert from a vector of 4 floats to a vector of 4 doubles.
  __builtin_convertvector(vf, vector4double)
  // equivalent to:
  (vector4double) { (double) vf[0], (double) vf[1], (double) vf[2], (double) vf[3] }

  // convert from a vector of 4 shorts to a vector of 4 floats.
  __builtin_convertvector(vs, vector4float)
  // equivalent to:
  (vector4float) { (float) vs[0], (float) vs[1], (float) vs[2], (float) vs[3] }

**Description**:

The first argument to ``__builtin_convertvector`` is a vector, and the second
argument is a vector type with the same number of elements as the first
argument.

The result of ``__builtin_convertvector`` is a vector with the same element
type as the second argument, with a value defined in terms of the action of a
C-style cast applied to each element of the first argument.

Query for this feature with ``__has_builtin(__builtin_convertvector)``.

``__builtin_bitreverse``
------------------------

* ``__builtin_bitreverse8``
* ``__builtin_bitreverse16``
* ``__builtin_bitreverse32``
* ``__builtin_bitreverse64``

**Syntax**:

.. code-block:: c++

     __builtin_bitreverse32(x)

**Examples**:

.. code-block:: c++

      uint8_t rev_x = __builtin_bitreverse8(x);
      uint16_t rev_x = __builtin_bitreverse16(x);
      uint32_t rev_y = __builtin_bitreverse32(y);
      uint64_t rev_z = __builtin_bitreverse64(z);

**Description**:

The '``__builtin_bitreverse``' family of builtins is used to reverse
the bitpattern of an integer value; for example ``0b10110110`` becomes
``0b01101101``.

``__builtin_rotateleft``
------------------------

* ``__builtin_rotateleft8``
* ``__builtin_rotateleft16``
* ``__builtin_rotateleft32``
* ``__builtin_rotateleft64``

**Syntax**:

.. code-block:: c++

     __builtin_rotateleft32(x, y)

**Examples**:

.. code-block:: c++

      uint8_t rot_x = __builtin_rotateleft8(x, y);
      uint16_t rot_x = __builtin_rotateleft16(x, y);
      uint32_t rot_x = __builtin_rotateleft32(x, y);
      uint64_t rot_x = __builtin_rotateleft64(x, y);

**Description**:

The '``__builtin_rotateleft``' family of builtins is used to rotate
the bits in the first argument by the amount in the second argument. 
For example, ``0b10000110`` rotated left by 11 becomes ``0b00110100``.
The shift value is treated as an unsigned amount modulo the size of
the arguments. Both arguments and the result have the bitwidth specified
by the name of the builtin.

``__builtin_rotateright``
-------------------------

* ``__builtin_rotateright8``
* ``__builtin_rotateright16``
* ``__builtin_rotateright32``
* ``__builtin_rotateright64``

**Syntax**:

.. code-block:: c++

     __builtin_rotateright32(x, y)

**Examples**:

.. code-block:: c++

      uint8_t rot_x = __builtin_rotateright8(x, y);
      uint16_t rot_x = __builtin_rotateright16(x, y);
      uint32_t rot_x = __builtin_rotateright32(x, y);
      uint64_t rot_x = __builtin_rotateright64(x, y);

**Description**:

The '``__builtin_rotateright``' family of builtins is used to rotate
the bits in the first argument by the amount in the second argument. 
For example, ``0b10000110`` rotated right by 3 becomes ``0b11010000``.
The shift value is treated as an unsigned amount modulo the size of
the arguments. Both arguments and the result have the bitwidth specified
by the name of the builtin.

``__builtin_unreachable``
-------------------------

``__builtin_unreachable`` is used to indicate that a specific point in the
program cannot be reached, even if the compiler might otherwise think it can.
This is useful to improve optimization and eliminates certain warnings.  For
example, without the ``__builtin_unreachable`` in the example below, the
compiler assumes that the inline asm can fall through and prints a "function
declared '``noreturn``' should not return" warning.

**Syntax**:

.. code-block:: c++

    __builtin_unreachable()

**Example of use**:

.. code-block:: c++

  void myabort(void) __attribute__((noreturn));
  void myabort(void) {
    asm("int3");
    __builtin_unreachable();
  }

**Description**:

The ``__builtin_unreachable()`` builtin has completely undefined behavior.
Since it has undefined behavior, it is a statement that it is never reached and
the optimizer can take advantage of this to produce better code.  This builtin
takes no arguments and produces a void result.

Query for this feature with ``__has_builtin(__builtin_unreachable)``.

``__builtin_unpredictable``
---------------------------

``__builtin_unpredictable`` is used to indicate that a branch condition is
unpredictable by hardware mechanisms such as branch prediction logic.

**Syntax**:

.. code-block:: c++

    __builtin_unpredictable(long long)

**Example of use**:

.. code-block:: c++

  if (__builtin_unpredictable(x > 0)) {
     foo();
  }

**Description**:

The ``__builtin_unpredictable()`` builtin is expected to be used with control
flow conditions such as in ``if`` and ``switch`` statements.

Query for this feature with ``__has_builtin(__builtin_unpredictable)``.

``__sync_swap``
---------------

``__sync_swap`` is used to atomically swap integers or pointers in memory.

**Syntax**:

.. code-block:: c++

  type __sync_swap(type *ptr, type value, ...)

**Example of Use**:

.. code-block:: c++

  int old_value = __sync_swap(&value, new_value);

**Description**:

The ``__sync_swap()`` builtin extends the existing ``__sync_*()`` family of
atomic intrinsics to allow code to atomically swap the current value with the
new value.  More importantly, it helps developers write more efficient and
correct code by avoiding expensive loops around
``__sync_bool_compare_and_swap()`` or relying on the platform specific
implementation details of ``__sync_lock_test_and_set()``.  The
``__sync_swap()`` builtin is a full barrier.

``__builtin_addressof``
-----------------------

``__builtin_addressof`` performs the functionality of the built-in ``&``
operator, ignoring any ``operator&`` overload.  This is useful in constant
expressions in C++11, where there is no other way to take the address of an
object that overloads ``operator&``.

**Example of use**:

.. code-block:: c++

  template<typename T> constexpr T *addressof(T &value) {
    return __builtin_addressof(value);
  }

``__builtin_operator_new`` and ``__builtin_operator_delete``
------------------------------------------------------------

``__builtin_operator_new`` allocates memory just like a non-placement non-class
*new-expression*. This is exactly like directly calling the normal
non-placement ``::operator new``, except that it allows certain optimizations
that the C++ standard does not permit for a direct function call to
``::operator new`` (in particular, removing ``new`` / ``delete`` pairs and
merging allocations).

Likewise, ``__builtin_operator_delete`` deallocates memory just like a
non-class *delete-expression*, and is exactly like directly calling the normal
``::operator delete``, except that it permits optimizations. Only the unsized
form of ``__builtin_operator_delete`` is currently available.

These builtins are intended for use in the implementation of ``std::allocator``
and other similar allocation libraries, and are only available in C++.

Multiprecision Arithmetic Builtins
----------------------------------

Clang provides a set of builtins which expose multiprecision arithmetic in a
manner amenable to C. They all have the following form:

.. code-block:: c

  unsigned x = ..., y = ..., carryin = ..., carryout;
  unsigned sum = __builtin_addc(x, y, carryin, &carryout);

Thus one can form a multiprecision addition chain in the following manner:

.. code-block:: c

  unsigned *x, *y, *z, carryin=0, carryout;
  z[0] = __builtin_addc(x[0], y[0], carryin, &carryout);
  carryin = carryout;
  z[1] = __builtin_addc(x[1], y[1], carryin, &carryout);
  carryin = carryout;
  z[2] = __builtin_addc(x[2], y[2], carryin, &carryout);
  carryin = carryout;
  z[3] = __builtin_addc(x[3], y[3], carryin, &carryout);

The complete list of builtins are:

.. code-block:: c

  unsigned char      __builtin_addcb (unsigned char x, unsigned char y, unsigned char carryin, unsigned char *carryout);
  unsigned short     __builtin_addcs (unsigned short x, unsigned short y, unsigned short carryin, unsigned short *carryout);
  unsigned           __builtin_addc  (unsigned x, unsigned y, unsigned carryin, unsigned *carryout);
  unsigned long      __builtin_addcl (unsigned long x, unsigned long y, unsigned long carryin, unsigned long *carryout);
  unsigned long long __builtin_addcll(unsigned long long x, unsigned long long y, unsigned long long carryin, unsigned long long *carryout);
  unsigned char      __builtin_subcb (unsigned char x, unsigned char y, unsigned char carryin, unsigned char *carryout);
  unsigned short     __builtin_subcs (unsigned short x, unsigned short y, unsigned short carryin, unsigned short *carryout);
  unsigned           __builtin_subc  (unsigned x, unsigned y, unsigned carryin, unsigned *carryout);
  unsigned long      __builtin_subcl (unsigned long x, unsigned long y, unsigned long carryin, unsigned long *carryout);
  unsigned long long __builtin_subcll(unsigned long long x, unsigned long long y, unsigned long long carryin, unsigned long long *carryout);

Checked Arithmetic Builtins
---------------------------

Clang provides a set of builtins that implement checked arithmetic for security
critical applications in a manner that is fast and easily expressable in C. As
an example of their usage:

.. code-block:: c

  errorcode_t security_critical_application(...) {
    unsigned x, y, result;
    ...
    if (__builtin_mul_overflow(x, y, &result))
      return kErrorCodeHackers;
    ...
    use_multiply(result);
    ...
  }

Clang provides the following checked arithmetic builtins:

.. code-block:: c

  bool __builtin_add_overflow   (type1 x, type2 y, type3 *sum);
  bool __builtin_sub_overflow   (type1 x, type2 y, type3 *diff);
  bool __builtin_mul_overflow   (type1 x, type2 y, type3 *prod);
  bool __builtin_uadd_overflow  (unsigned x, unsigned y, unsigned *sum);
  bool __builtin_uaddl_overflow (unsigned long x, unsigned long y, unsigned long *sum);
  bool __builtin_uaddll_overflow(unsigned long long x, unsigned long long y, unsigned long long *sum);
  bool __builtin_usub_overflow  (unsigned x, unsigned y, unsigned *diff);
  bool __builtin_usubl_overflow (unsigned long x, unsigned long y, unsigned long *diff);
  bool __builtin_usubll_overflow(unsigned long long x, unsigned long long y, unsigned long long *diff);
  bool __builtin_umul_overflow  (unsigned x, unsigned y, unsigned *prod);
  bool __builtin_umull_overflow (unsigned long x, unsigned long y, unsigned long *prod);
  bool __builtin_umulll_overflow(unsigned long long x, unsigned long long y, unsigned long long *prod);
  bool __builtin_sadd_overflow  (int x, int y, int *sum);
  bool __builtin_saddl_overflow (long x, long y, long *sum);
  bool __builtin_saddll_overflow(long long x, long long y, long long *sum);
  bool __builtin_ssub_overflow  (int x, int y, int *diff);
  bool __builtin_ssubl_overflow (long x, long y, long *diff);
  bool __builtin_ssubll_overflow(long long x, long long y, long long *diff);
  bool __builtin_smul_overflow  (int x, int y, int *prod);
  bool __builtin_smull_overflow (long x, long y, long *prod);
  bool __builtin_smulll_overflow(long long x, long long y, long long *prod);

Each builtin performs the specified mathematical operation on the
first two arguments and stores the result in the third argument.  If
possible, the result will be equal to mathematically-correct result
and the builtin will return 0.  Otherwise, the builtin will return
1 and the result will be equal to the unique value that is equivalent
to the mathematically-correct result modulo two raised to the *k*
power, where *k* is the number of bits in the result type.  The
behavior of these builtins is well-defined for all argument values.

The first three builtins work generically for operands of any integer type,
including boolean types.  The operands need not have the same type as each
other, or as the result.  The other builtins may implicitly promote or
convert their operands before performing the operation.

Query for this feature with ``__has_builtin(__builtin_add_overflow)``, etc.

Floating point builtins
---------------------------------------

``__builtin_canonicalize``
--------------------------

.. code-block:: c

   double __builtin_canonicalize(double);
   float __builtin_canonicalizef(float);
   long double__builtin_canonicalizel(long double);

Returns the platform specific canonical encoding of a floating point
number. This canonicalization is useful for implementing certain
numeric primitives such as frexp. See `LLVM canonicalize intrinsic
<https://llvm.org/docs/LangRef.html#llvm-canonicalize-intrinsic>`_ for
more information on the semantics.

String builtins
---------------

Clang provides constant expression evaluation support for builtins forms of
the following functions from the C standard library ``<string.h>`` header:

* ``memchr``
* ``memcmp``
* ``strchr``
* ``strcmp``
* ``strlen``
* ``strncmp``
* ``wcschr``
* ``wcscmp``
* ``wcslen``
* ``wcsncmp``
* ``wmemchr``
* ``wmemcmp``

In each case, the builtin form has the name of the C library function prefixed
by ``__builtin_``. Example:

.. code-block:: c

  void *p = __builtin_memchr("foobar", 'b', 5);

In addition to the above, one further builtin is provided:

.. code-block:: c

  char *__builtin_char_memchr(const char *haystack, int needle, size_t size);

``__builtin_char_memchr(a, b, c)`` is identical to
``(char*)__builtin_memchr(a, b, c)`` except that its use is permitted within
constant expressions in C++11 onwards (where a cast from ``void*`` to ``char*``
is disallowed in general).

Support for constant expression evaluation for the above builtins be detected
with ``__has_feature(cxx_constexpr_string_builtins)``.

Atomic Min/Max builtins with memory ordering
--------------------------------------------

There are two atomic builtins with min/max in-memory comparison and swap.
The syntax and semantics are similar to GCC-compatible __atomic_* builtins.

* ``__atomic_fetch_min`` 
* ``__atomic_fetch_max`` 

The builtins work with signed and unsigned integers and require to specify memory ordering.
The return value is the original value that was stored in memory before comparison.

Example:

.. code-block:: c

  unsigned int val = __atomic_fetch_min(unsigned int *pi, unsigned int ui, __ATOMIC_RELAXED);

The third argument is one of the memory ordering specifiers ``__ATOMIC_RELAXED``,
``__ATOMIC_CONSUME``, ``__ATOMIC_ACQUIRE``, ``__ATOMIC_RELEASE``,
``__ATOMIC_ACQ_REL``, or ``__ATOMIC_SEQ_CST`` following C++11 memory model semantics.

In terms or aquire-release ordering barriers these two operations are always
considered as operations with *load-store* semantics, even when the original value
is not actually modified after comparison.

.. _langext-__c11_atomic:

__c11_atomic builtins
---------------------

Clang provides a set of builtins which are intended to be used to implement
C11's ``<stdatomic.h>`` header.  These builtins provide the semantics of the
``_explicit`` form of the corresponding C11 operation, and are named with a
``__c11_`` prefix.  The supported operations, and the differences from
the corresponding C11 operations, are:

* ``__c11_atomic_init``
* ``__c11_atomic_thread_fence``
* ``__c11_atomic_signal_fence``
* ``__c11_atomic_is_lock_free`` (The argument is the size of the
  ``_Atomic(...)`` object, instead of its address)
* ``__c11_atomic_store``
* ``__c11_atomic_load``
* ``__c11_atomic_exchange``
* ``__c11_atomic_compare_exchange_strong``
* ``__c11_atomic_compare_exchange_weak``
* ``__c11_atomic_fetch_add``
* ``__c11_atomic_fetch_sub``
* ``__c11_atomic_fetch_and``
* ``__c11_atomic_fetch_or``
* ``__c11_atomic_fetch_xor``

The macros ``__ATOMIC_RELAXED``, ``__ATOMIC_CONSUME``, ``__ATOMIC_ACQUIRE``,
``__ATOMIC_RELEASE``, ``__ATOMIC_ACQ_REL``, and ``__ATOMIC_SEQ_CST`` are
provided, with values corresponding to the enumerators of C11's
``memory_order`` enumeration.

(Note that Clang additionally provides GCC-compatible ``__atomic_*``
builtins and OpenCL 2.0 ``__opencl_atomic_*`` builtins. The OpenCL 2.0
atomic builtins are an explicit form of the corresponding OpenCL 2.0
builtin function, and are named with a ``__opencl_`` prefix. The macros
``__OPENCL_MEMORY_SCOPE_WORK_ITEM``, ``__OPENCL_MEMORY_SCOPE_WORK_GROUP``,
``__OPENCL_MEMORY_SCOPE_DEVICE``, ``__OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES``,
and ``__OPENCL_MEMORY_SCOPE_SUB_GROUP`` are provided, with values
corresponding to the enumerators of OpenCL's ``memory_scope`` enumeration.)

Low-level ARM exclusive memory builtins
---------------------------------------

Clang provides overloaded builtins giving direct access to the three key ARM
instructions for implementing atomic operations.

.. code-block:: c

  T __builtin_arm_ldrex(const volatile T *addr);
  T __builtin_arm_ldaex(const volatile T *addr);
  int __builtin_arm_strex(T val, volatile T *addr);
  int __builtin_arm_stlex(T val, volatile T *addr);
  void __builtin_arm_clrex(void);

The types ``T`` currently supported are:

* Integer types with width at most 64 bits (or 128 bits on AArch64).
* Floating-point types
* Pointer types.

Note that the compiler does not guarantee it will not insert stores which clear
the exclusive monitor in between an ``ldrex`` type operation and its paired
``strex``. In practice this is only usually a risk when the extra store is on
the same cache line as the variable being modified and Clang will only insert
stack stores on its own, so it is best not to use these operations on variables
with automatic storage duration.

Also, loads and stores may be implicit in code written between the ``ldrex`` and
``strex``. Clang will not necessarily mitigate the effects of these either, so
care should be exercised.

For these reasons the higher level atomic primitives should be preferred where
possible.

Non-temporal load/store builtins
--------------------------------

Clang provides overloaded builtins allowing generation of non-temporal memory
accesses.

.. code-block:: c

  T __builtin_nontemporal_load(T *addr);
  void __builtin_nontemporal_store(T value, T *addr);

The types ``T`` currently supported are:

* Integer types.
* Floating-point types.
* Vector types.

Note that the compiler does not guarantee that non-temporal loads or stores
will be used.

C++ Coroutines support builtins
--------------------------------

.. warning::
  This is a work in progress. Compatibility across Clang/LLVM releases is not 
  guaranteed.

Clang provides experimental builtins to support C++ Coroutines as defined by
https://wg21.link/P0057. The following four are intended to be used by the
standard library to implement `std::experimental::coroutine_handle` type.

**Syntax**:

.. code-block:: c

  void  __builtin_coro_resume(void *addr);
  void  __builtin_coro_destroy(void *addr);
  bool  __builtin_coro_done(void *addr);
  void *__builtin_coro_promise(void *addr, int alignment, bool from_promise)

**Example of use**:

.. code-block:: c++

  template <> struct coroutine_handle<void> {
    void resume() const { __builtin_coro_resume(ptr); }
    void destroy() const { __builtin_coro_destroy(ptr); }
    bool done() const { return __builtin_coro_done(ptr); }
    // ...
  protected:
    void *ptr;
  };

  template <typename Promise> struct coroutine_handle : coroutine_handle<> {
    // ...
    Promise &promise() const {
      return *reinterpret_cast<Promise *>(
        __builtin_coro_promise(ptr, alignof(Promise), /*from-promise=*/false));
    }
    static coroutine_handle from_promise(Promise &promise) {
      coroutine_handle p;
      p.ptr = __builtin_coro_promise(&promise, alignof(Promise),
                                                      /*from-promise=*/true);
      return p;
    }
  };


Other coroutine builtins are either for internal clang use or for use during
development of the coroutine feature. See `Coroutines in LLVM
<https://llvm.org/docs/Coroutines.html#intrinsics>`_ for
more information on their semantics. Note that builtins matching the intrinsics
that take token as the first parameter (llvm.coro.begin, llvm.coro.alloc, 
llvm.coro.free and llvm.coro.suspend) omit the token parameter and fill it to
an appropriate value during the emission.

**Syntax**:

.. code-block:: c

  size_t __builtin_coro_size()
  void  *__builtin_coro_frame()
  void  *__builtin_coro_free(void *coro_frame)

  void  *__builtin_coro_id(int align, void *promise, void *fnaddr, void *parts)
  bool   __builtin_coro_alloc()
  void  *__builtin_coro_begin(void *memory)
  void   __builtin_coro_end(void *coro_frame, bool unwind)
  char   __builtin_coro_suspend(bool final)
  bool   __builtin_coro_param(void *original, void *copy)

Note that there is no builtin matching the `llvm.coro.save` intrinsic. LLVM
automatically will insert one if the first argument to `llvm.coro.suspend` is
token `none`. If a user calls `__builin_suspend`, clang will insert `token none`
as the first argument to the intrinsic.

Non-standard C++11 Attributes
=============================

Clang's non-standard C++11 attributes live in the ``clang`` attribute
namespace.

Clang supports GCC's ``gnu`` attribute namespace. All GCC attributes which
are accepted with the ``__attribute__((foo))`` syntax are also accepted as
``[[gnu::foo]]``. This only extends to attributes which are specified by GCC
(see the list of `GCC function attributes
<https://gcc.gnu.org/onlinedocs/gcc/Function-Attributes.html>`_, `GCC variable
attributes <https://gcc.gnu.org/onlinedocs/gcc/Variable-Attributes.html>`_, and
`GCC type attributes
<https://gcc.gnu.org/onlinedocs/gcc/Type-Attributes.html>`_). As with the GCC
implementation, these attributes must appertain to the *declarator-id* in a
declaration, which means they must go either at the start of the declaration or
immediately after the name being declared.

For example, this applies the GNU ``unused`` attribute to ``a`` and ``f``, and
also applies the GNU ``noreturn`` attribute to ``f``.

.. code-block:: c++

  [[gnu::unused]] int a, f [[gnu::noreturn]] ();

Target-Specific Extensions
==========================

Clang supports some language features conditionally on some targets.

ARM/AArch64 Language Extensions
-------------------------------

Memory Barrier Intrinsics
^^^^^^^^^^^^^^^^^^^^^^^^^
Clang implements the ``__dmb``, ``__dsb`` and ``__isb`` intrinsics as defined
in the `ARM C Language Extensions Release 2.0
<http://infocenter.arm.com/help/topic/com.arm.doc.ihi0053c/IHI0053C_acle_2_0.pdf>`_.
Note that these intrinsics are implemented as motion barriers that block
reordering of memory accesses and side effect instructions. Other instructions
like simple arithmetic may be reordered around the intrinsic. If you expect to
have no reordering at all, use inline assembly instead.

X86/X86-64 Language Extensions
------------------------------

The X86 backend has these language extensions:

Memory references to specified segments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Annotating a pointer with address space #256 causes it to be code generated
relative to the X86 GS segment register, address space #257 causes it to be
relative to the X86 FS segment, and address space #258 causes it to be
relative to the X86 SS segment.  Note that this is a very very low-level
feature that should only be used if you know what you're doing (for example in
an OS kernel).

Here is an example:

.. code-block:: c++

  #define GS_RELATIVE __attribute__((address_space(256)))
  int foo(int GS_RELATIVE *P) {
    return *P;
  }

Which compiles to (on X86-32):

.. code-block:: gas

  _foo:
          movl    4(%esp), %eax
          movl    %gs:(%eax), %eax
          ret

PowerPC Language Extensions
------------------------------

Set the Floating Point Rounding Mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PowerPC64/PowerPC64le supports the builtin function ``__builtin_setrnd`` to set
the floating point rounding mode. This function will use the least significant
two bits of integer argument to set the floating point rounding mode.

.. code-block:: c++

  double __builtin_setrnd(int mode);

The effective values for mode are:

    - 0 - round to nearest
    - 1 - round to zero
    - 2 - round to +infinity
    - 3 - round to -infinity

Note that the mode argument will modulo 4, so if the int argument is greater 
than 3, it will only use the least significant two bits of the mode. 
Namely, ``__builtin_setrnd(102))`` is equal to ``__builtin_setrnd(2)``.

PowerPC Language Extensions
------------------------------

Set the Floating Point Rounding Mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PowerPC64/PowerPC64le supports the builtin function ``__builtin_setrnd`` to set
the floating point rounding mode. This function will use the least significant
two bits of integer argument to set the floating point rounding mode.

.. code-block:: c++

  double __builtin_setrnd(int mode);

The effective values for mode are:

    - 0 - round to nearest
    - 1 - round to zero
    - 2 - round to +infinity
    - 3 - round to -infinity

Note that the mode argument will modulo 4, so if the integer argument is greater
than 3, it will only use the least significant two bits of the mode. 
Namely, ``__builtin_setrnd(102))`` is equal to ``__builtin_setrnd(2)``.

PowerPC Language Extensions
------------------------------

Set the Floating Point Rounding Mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PowerPC64/PowerPC64le supports the builtin function ``__builtin_setrnd`` to set
the floating point rounding mode. This function will use the least significant
two bits of integer argument to set the floating point rounding mode.

.. code-block:: c++

  double __builtin_setrnd(int mode);

The effective values for mode are:

    - 0 - round to nearest
    - 1 - round to zero
    - 2 - round to +infinity
    - 3 - round to -infinity

Note that the mode argument will modulo 4, so if the integer argument is greater
than 3, it will only use the least significant two bits of the mode. 
Namely, ``__builtin_setrnd(102))`` is equal to ``__builtin_setrnd(2)``.

Extensions for Static Analysis
==============================

Clang supports additional attributes that are useful for documenting program
invariants and rules for static analysis tools, such as the `Clang Static
Analyzer <https://clang-analyzer.llvm.org/>`_. These attributes are documented
in the analyzer's `list of source-level annotations
<https://clang-analyzer.llvm.org/annotations.html>`_.


Extensions for Dynamic Analysis
===============================

Use ``__has_feature(address_sanitizer)`` to check if the code is being built
with :doc:`AddressSanitizer`.

Use ``__has_feature(thread_sanitizer)`` to check if the code is being built
with :doc:`ThreadSanitizer`.

Use ``__has_feature(memory_sanitizer)`` to check if the code is being built
with :doc:`MemorySanitizer`.

Use ``__has_feature(safe_stack)`` to check if the code is being built
with :doc:`SafeStack`.


Extensions for selectively disabling optimization
=================================================

Clang provides a mechanism for selectively disabling optimizations in functions
and methods.

To disable optimizations in a single function definition, the GNU-style or C++11
non-standard attribute ``optnone`` can be used.

.. code-block:: c++

  // The following functions will not be optimized.
  // GNU-style attribute
  __attribute__((optnone)) int foo() {
    // ... code
  }
  // C++11 attribute
  [[clang::optnone]] int bar() {
    // ... code
  }

To facilitate disabling optimization for a range of function definitions, a
range-based pragma is provided. Its syntax is ``#pragma clang optimize``
followed by ``off`` or ``on``.

All function definitions in the region between an ``off`` and the following
``on`` will be decorated with the ``optnone`` attribute unless doing so would
conflict with explicit attributes already present on the function (e.g. the
ones that control inlining).

.. code-block:: c++

  #pragma clang optimize off
  // This function will be decorated with optnone.
  int foo() {
    // ... code
  }

  // optnone conflicts with always_inline, so bar() will not be decorated.
  __attribute__((always_inline)) int bar() {
    // ... code
  }
  #pragma clang optimize on

If no ``on`` is found to close an ``off`` region, the end of the region is the
end of the compilation unit.

Note that a stray ``#pragma clang optimize on`` does not selectively enable
additional optimizations when compiling at low optimization levels. This feature
can only be used to selectively disable optimizations.

The pragma has an effect on functions only at the point of their definition; for
function templates, this means that the state of the pragma at the point of an
instantiation is not necessarily relevant. Consider the following example:

.. code-block:: c++

  template<typename T> T twice(T t) {
    return 2 * t;
  }

  #pragma clang optimize off
  template<typename T> T thrice(T t) {
    return 3 * t;
  }

  int container(int a, int b) {
    return twice(a) + thrice(b);
  }
  #pragma clang optimize on

In this example, the definition of the template function ``twice`` is outside
the pragma region, whereas the definition of ``thrice`` is inside the region.
The ``container`` function is also in the region and will not be optimized, but
it causes the instantiation of ``twice`` and ``thrice`` with an ``int`` type; of
these two instantiations, ``twice`` will be optimized (because its definition
was outside the region) and ``thrice`` will not be optimized.

Extensions for loop hint optimizations
======================================

The ``#pragma clang loop`` directive is used to specify hints for optimizing the
subsequent for, while, do-while, or c++11 range-based for loop. The directive
provides options for vectorization, interleaving, unrolling and
distribution. Loop hints can be specified before any loop and will be ignored if
the optimization is not safe to apply.

Vectorization and Interleaving
------------------------------

A vectorized loop performs multiple iterations of the original loop
in parallel using vector instructions. The instruction set of the target
processor determines which vector instructions are available and their vector
widths. This restricts the types of loops that can be vectorized. The vectorizer
automatically determines if the loop is safe and profitable to vectorize. A
vector instruction cost model is used to select the vector width.

Interleaving multiple loop iterations allows modern processors to further
improve instruction-level parallelism (ILP) using advanced hardware features,
such as multiple execution units and out-of-order execution. The vectorizer uses
a cost model that depends on the register pressure and generated code size to
select the interleaving count.

Vectorization is enabled by ``vectorize(enable)`` and interleaving is enabled
by ``interleave(enable)``. This is useful when compiling with ``-Os`` to
manually enable vectorization or interleaving.

.. code-block:: c++

  #pragma clang loop vectorize(enable)
  #pragma clang loop interleave(enable)
  for(...) {
    ...
  }

The vector width is specified by ``vectorize_width(_value_)`` and the interleave
count is specified by ``interleave_count(_value_)``, where
_value_ is a positive integer. This is useful for specifying the optimal
width/count of the set of target architectures supported by your application.

.. code-block:: c++

  #pragma clang loop vectorize_width(2)
  #pragma clang loop interleave_count(2)
  for(...) {
    ...
  }

Specifying a width/count of 1 disables the optimization, and is equivalent to
``vectorize(disable)`` or ``interleave(disable)``.

Loop Unrolling
--------------

Unrolling a loop reduces the loop control overhead and exposes more
opportunities for ILP. Loops can be fully or partially unrolled. Full unrolling
eliminates the loop and replaces it with an enumerated sequence of loop
iterations. Full unrolling is only possible if the loop trip count is known at
compile time. Partial unrolling replicates the loop body within the loop and
reduces the trip count.

If ``unroll(enable)`` is specified the unroller will attempt to fully unroll the
loop if the trip count is known at compile time. If the fully unrolled code size
is greater than an internal limit the loop will be partially unrolled up to this
limit. If the trip count is not known at compile time the loop will be partially
unrolled with a heuristically chosen unroll factor.

.. code-block:: c++

  #pragma clang loop unroll(enable)
  for(...) {
    ...
  }

If ``unroll(full)`` is specified the unroller will attempt to fully unroll the
loop if the trip count is known at compile time identically to
``unroll(enable)``. However, with ``unroll(full)`` the loop will not be unrolled
if the loop count is not known at compile time.

.. code-block:: c++

  #pragma clang loop unroll(full)
  for(...) {
    ...
  }

The unroll count can be specified explicitly with ``unroll_count(_value_)`` where
_value_ is a positive integer. If this value is greater than the trip count the
loop will be fully unrolled. Otherwise the loop is partially unrolled subject
to the same code size limit as with ``unroll(enable)``.

.. code-block:: c++

  #pragma clang loop unroll_count(8)
  for(...) {
    ...
  }

Unrolling of a loop can be prevented by specifying ``unroll(disable)``.

Loop Distribution
-----------------

Loop Distribution allows splitting a loop into multiple loops.  This is
beneficial for example when the entire loop cannot be vectorized but some of the
resulting loops can.

If ``distribute(enable))`` is specified and the loop has memory dependencies
that inhibit vectorization, the compiler will attempt to isolate the offending
operations into a new loop.  This optimization is not enabled by default, only
loops marked with the pragma are considered.

.. code-block:: c++

  #pragma clang loop distribute(enable)
  for (i = 0; i < N; ++i) {
    S1: A[i + 1] = A[i] + B[i];
    S2: C[i] = D[i] * E[i];
  }

This loop will be split into two loops between statements S1 and S2.  The
second loop containing S2 will be vectorized.

Loop Distribution is currently not enabled by default in the optimizer because
it can hurt performance in some cases.  For example, instruction-level
parallelism could be reduced by sequentializing the execution of the
statements S1 and S2 above.

If Loop Distribution is turned on globally with
``-mllvm -enable-loop-distribution``, specifying ``distribute(disable)`` can
be used the disable it on a per-loop basis.

Additional Information
----------------------

For convenience multiple loop hints can be specified on a single line.

.. code-block:: c++

  #pragma clang loop vectorize_width(4) interleave_count(8)
  for(...) {
    ...
  }

If an optimization cannot be applied any hints that apply to it will be ignored.
For example, the hint ``vectorize_width(4)`` is ignored if the loop is not
proven safe to vectorize. To identify and diagnose optimization issues use
`-Rpass`, `-Rpass-missed`, and `-Rpass-analysis` command line options. See the
user guide for details.

Extensions to specify floating-point flags
====================================================

The ``#pragma clang fp`` pragma allows floating-point options to be specified
for a section of the source code. This pragma can only appear at file scope or
at the start of a compound statement (excluding comments). When using within a
compound statement, the pragma is active within the scope of the compound
statement.

Currently, only FP contraction can be controlled with the pragma. ``#pragma
clang fp contract`` specifies whether the compiler should contract a multiply
and an addition (or subtraction) into a fused FMA operation when supported by
the target.

The pragma can take three values: ``on``, ``fast`` and ``off``.  The ``on``
option is identical to using ``#pragma STDC FP_CONTRACT(ON)`` and it allows
fusion as specified the language standard.  The ``fast`` option allows fusiong
in cases when the language standard does not make this possible (e.g. across
statements in C)

.. code-block:: c++

  for(...) {
    #pragma clang fp contract(fast)
    a = b[i] * c[i];
    d[i] += a;
  }


The pragma can also be used with ``off`` which turns FP contraction off for a
section of the code. This can be useful when fast contraction is otherwise
enabled for the translation unit with the ``-ffp-contract=fast`` flag.

Specifying an attribute for multiple declarations (#pragma clang attribute)
===========================================================================

The ``#pragma clang attribute`` directive can be used to apply an attribute to
multiple declarations. The ``#pragma clang attribute push`` variation of the
directive pushes a new "scope" of ``#pragma clang attribute`` that attributes
can be added to. The ``#pragma clang attribute (...)`` variation adds an
attribute to that scope, and the ``#pragma clang attribute pop`` variation pops
the scope. You can also use ``#pragma clang attribute push (...)``, which is a
shorthand for when you want to add one attribute to a new scope. Multiple push
directives can be nested inside each other.

The attributes that are used in the ``#pragma clang attribute`` directives
can be written using the GNU-style syntax:

.. code-block:: c++

  #pragma clang attribute push (__attribute__((annotate("custom"))), apply_to = function)

  void function(); // The function now has the annotate("custom") attribute

  #pragma clang attribute pop

The attributes can also be written using the C++11 style syntax:

.. code-block:: c++

  #pragma clang attribute push ([[noreturn]], apply_to = function)

  void function(); // The function now has the [[noreturn]] attribute

  #pragma clang attribute pop

The ``__declspec`` style syntax is also supported:

.. code-block:: c++

  #pragma clang attribute push (__declspec(dllexport), apply_to = function)

  void function(); // The function now has the __declspec(dllexport) attribute

  #pragma clang attribute pop

A single push directive accepts only one attribute regardless of the syntax
used.

Because multiple push directives can be nested, if you're writing a macro that
expands to ``_Pragma("clang attribute")`` it's good hygiene (though not
required) to add a namespace to your push/pop directives. A pop directive with a
namespace will pop the innermost push that has that same namespace. This will
ensure that another macro's ``pop`` won't inadvertently pop your attribute. Note
that an ``pop`` without a namespace will pop the innermost ``push`` without a
namespace. ``push``es with a namespace can only be popped by ``pop`` with the
same namespace. For instance:

.. code-block:: c++

   #define ASSUME_NORETURN_BEGIN _Pragma("clang attribute AssumeNoreturn.push ([[noreturn]], apply_to = function)")
   #define ASSUME_NORETURN_END   _Pragma("clang attribute AssumeNoreturn.pop")

   #define ASSUME_UNAVAILABLE_BEGIN _Pragma("clang attribute Unavailable.push (__attribute__((unavailable)), apply_to=function)")
   #define ASSUME_UNAVAILABLE_END   _Pragma("clang attribute Unavailable.pop")


   ASSUME_NORETURN_BEGIN
   ASSUME_UNAVAILABLE_BEGIN
   void function(); // function has [[noreturn]] and __attribute__((unavailable))
   ASSUME_NORETURN_END
   void other_function(); // function has __attribute__((unavailable))
   ASSUME_UNAVAILABLE_END

Without the namespaces on the macros, ``other_function`` will be annotated with
``[[noreturn]]`` instead of ``__attribute__((unavailable))``. This may seem like
a contrived example, but its very possible for this kind of situation to appear
in real code if the pragmas are spread out across a large file. You can test if
your version of clang supports namespaces on ``#pragma clang attribute`` with
``__has_extension(pragma_clang_attribute_namespaces)``.

Subject Match Rules
-------------------

The set of declarations that receive a single attribute from the attribute stack
depends on the subject match rules that were specified in the pragma. Subject
match rules are specified after the attribute. The compiler expects an
identifier that corresponds to the subject set specifier. The ``apply_to``
specifier is currently the only supported subject set specifier. It allows you
to specify match rules that form a subset of the attribute's allowed subject
set, i.e. the compiler doesn't require all of the attribute's subjects. For
example, an attribute like ``[[nodiscard]]`` whose subject set includes
``enum``, ``record`` and ``hasType(functionType)``, requires the presence of at
least one of these rules after ``apply_to``:

.. code-block:: c++

  #pragma clang attribute push([[nodiscard]], apply_to = enum)

  enum Enum1 { A1, B1 }; // The enum will receive [[nodiscard]]

  struct Record1 { }; // The struct will *not* receive [[nodiscard]]

  #pragma clang attribute pop

  #pragma clang attribute push([[nodiscard]], apply_to = any(record, enum))

  enum Enum2 { A2, B2 }; // The enum will receive [[nodiscard]]

  struct Record2 { }; // The struct *will* receive [[nodiscard]]

  #pragma clang attribute pop

  // This is an error, since [[nodiscard]] can't be applied to namespaces:
  #pragma clang attribute push([[nodiscard]], apply_to = any(record, namespace))

  #pragma clang attribute pop

Multiple match rules can be specified using the ``any`` match rule, as shown
in the example above. The ``any`` rule applies attributes to all declarations
that are matched by at least one of the rules in the ``any``. It doesn't nest
and can't be used inside the other match rules. Redundant match rules or rules
that conflict with one another should not be used inside of ``any``.

Clang supports the following match rules:

- ``function``: Can be used to apply attributes to functions. This includes C++
  member functions, static functions, operators, and constructors/destructors.

- ``function(is_member)``: Can be used to apply attributes to C++ member
  functions. This includes members like static functions, operators, and
  constructors/destructors.

- ``hasType(functionType)``: Can be used to apply attributes to functions, C++
  member functions, and variables/fields whose type is a function pointer. It
  does not apply attributes to Objective-C methods or blocks.

- ``type_alias``: Can be used to apply attributes to ``typedef`` declarations
  and C++11 type aliases.

- ``record``: Can be used to apply attributes to ``struct``, ``class``, and
  ``union`` declarations.

- ``record(unless(is_union))``: Can be used to apply attributes only to
  ``struct`` and ``class`` declarations.

- ``enum``: Can be be used to apply attributes to enumeration declarations.

- ``enum_constant``: Can be used to apply attributes to enumerators.

- ``variable``: Can be used to apply attributes to variables, including
  local variables, parameters, global variables, and static member variables.
  It does not apply attributes to instance member variables or Objective-C
  ivars.

- ``variable(is_thread_local)``: Can be used to apply attributes to thread-local
  variables only.

- ``variable(is_global)``: Can be used to apply attributes to global variables
  only.

- ``variable(is_parameter)``: Can be used to apply attributes to parameters
  only.

- ``variable(unless(is_parameter))``: Can be used to apply attributes to all
  the variables that are not parameters.

- ``field``: Can be used to apply attributes to non-static member variables
  in a record. This includes Objective-C ivars.

- ``namespace``: Can be used to apply attributes to ``namespace`` declarations.

- ``objc_interface``: Can be used to apply attributes to ``@interface``
  declarations.

- ``objc_protocol``: Can be used to apply attributes to ``@protocol``
  declarations.

- ``objc_category``: Can be used to apply attributes to category declarations,
  including class extensions.

- ``objc_method``: Can be used to apply attributes to Objective-C methods,
  including instance and class methods. Implicit methods like implicit property
  getters and setters do not receive the attribute.

- ``objc_method(is_instance)``: Can be used to apply attributes to Objective-C
  instance methods.

- ``objc_property``: Can be used to apply attributes to ``@property``
  declarations.

- ``block``: Can be used to apply attributes to block declarations. This does
  not include variables/fields of block pointer type.

The use of ``unless`` in match rules is currently restricted to a strict set of
sub-rules that are used by the supported attributes. That means that even though
``variable(unless(is_parameter))`` is a valid match rule,
``variable(unless(is_thread_local))`` is not.

Supported Attributes
--------------------

Not all attributes can be used with the ``#pragma clang attribute`` directive.
Notably, statement attributes like ``[[fallthrough]]`` or type attributes
like ``address_space`` aren't supported by this directive. You can determine
whether or not an attribute is supported by the pragma by referring to the
:doc:`individual documentation for that attribute <AttributeReference>`.

The attributes are applied to all matching declarations individually, even when
the attribute is semantically incorrect. The attributes that aren't applied to
any declaration are not verified semantically.

Specifying section names for global objects (#pragma clang section)
===================================================================

The ``#pragma clang section`` directive provides a means to assign section-names
to global variables, functions and static variables.

The section names can be specified as:

.. code-block:: c++

  #pragma clang section bss="myBSS" data="myData" rodata="myRodata" text="myText"

The section names can be reverted back to default name by supplying an empty
string to the section kind, for example:

.. code-block:: c++

  #pragma clang section bss="" data="" text="" rodata=""

The ``#pragma clang section`` directive obeys the following rules:

* The pragma applies to all global variable, statics and function declarations
  from the pragma to the end of the translation unit.

* The pragma clang section is enabled automatically, without need of any flags.

* This feature is only defined to work sensibly for ELF targets.

* If section name is specified through _attribute_((section("myname"))), then
  the attribute name gains precedence.

* Global variables that are initialized to zero will be placed in the named
  bss section, if one is present.

* The ``#pragma clang section`` directive does not does try to infer section-kind
  from the name. For example, naming a section "``.bss.mySec``" does NOT mean
  it will be a bss section name.

* The decision about which section-kind applies to each global is taken in the back-end.
  Once the section-kind is known, appropriate section name, as specified by the user using
  ``#pragma clang section`` directive, is applied to that global.

Specifying Linker Options on ELF Targets
========================================

The ``#pragma comment(lib, ...)`` directive is supported on all ELF targets.
The second parameter is the library name (without the traditional Unix prefix of
``lib``).  This allows you to provide an implicit link of dependent libraries.

Evaluating Object Size Dynamically
==================================

Clang supports the builtin ``__builtin_dynamic_object_size``, the semantics are
the same as GCC's ``__builtin_object_size`` (which Clang also supports), but
``__builtin_dynamic_object_size`` can evaluate the object's size at runtime.
``__builtin_dynamic_object_size`` is meant to be used as a drop-in replacement
for ``__builtin_object_size`` in libraries that support it.

For instance, here is a program that ``__builtin_dynamic_object_size`` will make
safer:

.. code-block:: c

  void copy_into_buffer(size_t size) {
    char* buffer = malloc(size);
    strlcpy(buffer, "some string", strlen("some string"));
    // Previous line preprocesses to:
    // __builtin___strlcpy_chk(buffer, "some string", strlen("some string"), __builtin_object_size(buffer, 0))
  }

Since the size of ``buffer`` can't be known at compile time, Clang will fold
``__builtin_object_size(buffer, 0)`` into ``-1``. However, if this was written
as ``__builtin_dynamic_object_size(buffer, 0)``, Clang will fold it into
``size``, providing some extra runtime safety.
