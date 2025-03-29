.. _code_style_what:

========================================
The LLVM-libc code style reference guide
========================================

This is intended as a quick reference guide to writing code within LLVM-libc.
For the reasoning behind the rules, read :ref:`_code_style_why`.

.. contents::
   :depth: 2
   :local:

Overall style
=============

The LLVM-libc code style is based on the LLVM code style, with some
LLVM-libc specific additions. See `the coding standards of the LLVM project
<https://llvm.org/docs/CodingStandards.html>`_. for the LLVM code style. If
there is a conflict between the LLVM code style and the LLVM-libc code style,
the LLVM-libc code style takes precedence inside the LLVM-libc codebase.

Naming style
------------

#. **Non-const variables** - Lowercase ``snake_case`` style.
   .. TODO: Should we add the rule of "member variables end with an underscore?"
#. **const and constexpr variables** - Capitalized ``SNAKE_CASE``.
#. **Functions and methods** - ``snake_case``.
#. **Internal class/struct names** - ``CaptilizedCamelCase``.
#. **Public names** - Follow the style as prescribed by the standards.

Macro style
-----------

#. Build flags - start with ``LIBC_COPT_``
   * e.g. ``LIBC_COPT_PRINTF_DISABLE_FLOAT``
#. Code defined flags - Defined in ``src/__support/macros`` and start with ``LIBC_``

   * ``src/__support/macros/properties/`` - defines flags based on the build properties.

      * ``architectures.h`` - Target architecture properties.
        e.g., ``LIBC_TARGET_ARCH_IS_ARM``.
      * ``compiler.h`` - Host compiler properties.
        e.g., ``LIBC_COMPILER_IS_CLANG``.
      * ``cpu_features.h`` - Target cpu feature availability.
        e.g., ``LIBC_TARGET_CPU_HAS_AVX2``.
      * ``types.h`` - Type properties and availability.
        e.g., ``LIBC_TYPES_HAS_FLOAT128``.
      * ``os.h`` - Target os properties.
        e.g., ``LIBC_TARGET_OS_IS_LINUX``.

  * ``src/__support/macros/config.h`` - Important compiler and platform
     features. Such macros can be used to produce portable code by
     parameterizing compilation based on the presence or lack of a given
     feature. e.g., ``LIBC_HAS_FEATURE``
  * ``src/__support/macros/attributes.h`` - Attributes for functions, types,
    and variables. e.g., ``LIBC_UNUSED``
  * ``src/__support/macros/optimization.h`` - Portable macros for performance
    optimization. e.g., ``LIBC_LIKELY``, ``LIBC_LOOP_NOUNROLL``

LLVM-libc specific rules
========================

#. Avoid using any C++ feature that requires runtime support.

   * Examples include exceptions, virtual functions, static/global constructors,
     and anything to do with RTTI.
   * For C++ library features that don't require runtime support, see
     ``/src/__support/CPP/`` which may already provide an implementation.

#. Avoid calling public libc functions (even through the ``LIBC_NAMESPACE``) from
   within LLVM-libc.

   * Instead call the internal implementations for those functions.
   * There are some specific exceptions, such as the allocation functions.

#. Avoid including public libc headers directly from within LLVM-libc.

   * Instead use specific headers for types or macros needed from the ``hdr``
     directory (called "Proxy Headers")
   * Some headers are always provided by the compiler, and are allowed but
     discouraged.
     * e.g. ``stdint.h``

   .. TODO: add a doc on proxy headers.

#. Avoid including from ``/include`` directly.

   * Instead use the Proxy Headers in the ``hdr`` directory.
   * It is allowed, though discouraged, to use headers from ``/include`` for
     types or macros that are LLVM-libc specific and only used in fullbuild.
     Proxy headers are preferred.

#. Avoid setting ``errno`` in internal functions.

   * Instead return errors with ``cpp::ErrorOr``, ``cpp::optional``, or a custom
     struct.
   * The only time to set ``errno`` is in the public entrypoints, preferably
     immediately before returning.

#. Avoid writing code in assembly for performance.

   * Instead write C++ code that the compiler can reason about and optimize.
   * Builtins are allowed where relevant, but may need backup code.
   * If optimal performance is required, it is recommended to improve the
     compiler.

#. Mark all functions in headers with ``LIBC_INLINE``

   * This is to avoid ODR violations, but also to allow adding properties to
     the function.
     .. TODO: Fix this phrasing.

#. Use ``LIBC_ASSERT`` for runtime assertions.

   * There are some files where this is impossible due to ``libc_assert.h``
     depending on that file. These just can't use assertions.
     .. TODO: list which files can't use assertions. Also see if we can fix this.

#. Avoid allocating memory where possible.

   * Instead use statically allocated memory when possible, even if the size
     must be large.
   * Some functions require allocation, such as ``strdup``, these should
     allocate as defined by the standard.


#. Avoid calling the public name of libc functions from unit tests.

   * Instead call the namespaced name of the function (e.g.
     ``LIBC_NAMESPACE::sqrt`` instead of ``sqrt``).
   * For tests of public header macros or integration tests, it may be necessary
     to call the public name of the function. These should be in the
     ``test/include`` directory or the ``test/integration`` directory.
