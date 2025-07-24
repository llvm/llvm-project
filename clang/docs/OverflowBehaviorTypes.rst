=====================
OverflowBehaviorTypes
=====================

.. contents::
   :local:

Introduction
============

Clang provides a type attribute that allows developers to have fine-grained control
over the overflow behavior of integer types. The ``overflow_behavior``
attribute can be used to specify how arithmetic operations on a given integer
type should behave upon overflow. This is particularly useful for projects that
need to balance performance and safety, allowing developers to enable or
disable overflow checks for specific types.

The attribute can be enabled using the compiler option
``-foverflow-behavior-types``.

The attribute syntax is as follows:

.. code-block:: c++

  __attribute__((overflow_behavior(behavior)))

Where ``behavior`` can be one of the following:

* ``wrap``: Specifies that arithmetic operations on the integer type should
  wrap on overflow. This is equivalent to the behavior of ``-fwrapv``, but it
  applies only to the attributed type and may be used with both signed and
  unsigned types. When this is enabled, UBSan's integer overflow and integer
  truncation checks (``signed-integer-overflow``,
  ``unsigned-integer-overflow``, ``implicit-signed-integer-truncation``, and
  ``implicit-unsigned-integer-truncation``) are suppressed for the attributed
  type.

* ``no_wrap``: Specifies that arithmetic operations on the integer type should
  be checked for overflow. When using the ``signed-integer-overflow`` sanitizer
  or when using ``-ftrapv`` alongside a signed type, this is the default
  behavior. Using this, one may enforce overflow checks for a type even when
  ``-fwrapv`` is enabled globally.

This attribute can be applied to ``typedef`` declarations and to integer types directly.

Arithmetic operations containing one or more overflow behavior types may have
result types that do not match standard integer promotion rules. The
characteristics of result types across all scenarios are described under `Promotion Rules`_.

Examples
========

Here is an example of how to use the ``overflow_behavior`` attribute with a ``typedef``:

.. code-block:: c++

  typedef unsigned int __attribute__((overflow_behavior(no_wrap))) non_wrapping_uint;

  non_wrapping_uint add_one(non_wrapping_uint a) {
    return a + 1; // Overflow is checked for this operation.
  }

Here is an example of how to use the ``overflow_behavior`` attribute with a type directly:

.. code-block:: c++

  int mul_alot(int n) {
    int __attribute__((overflow_behavior(wrap))) a = n;
    return a * 1337; // Potential overflow is not checked and is well-defined
  }

"Well-defined" overflow is consistent with two's complement wrap-around
semantics and won't be removed via eager compiler optimizations (like some
undefined behavior might).

Overflow behavior types are implicitly convertible to and from built-in
integral types.

Note that C++ overload set formation rules treat promotions to and from
overflow behavior types the same as normal integral promotions and conversions.

Promotion Rules
===============

The promotion rules for overflow behavior types are designed to preserve the
specified overflow behavior throughout an arithmetic expression. They differ
from standard C/C++ integer promotions but in a predictable way, similar to
how ``_Complex`` and ``_BitInt`` have their own promotion rules.

The resulting type characteristics for overflow behavior types (OBTs) across a
variety of scenarios is detailed below.

* **OBT and Standard Integer Type**: In an operation involving an overflow
  behavior type (OBT) and a standard integer type, the result will have the
  type of the OBT, including its overflow behavior, sign, and bit-width. The
  standard integer type is implicitly converted to match the OBT.

  .. code-block:: c++

    typedef char __attribute__((overflow_behavior(no_wrap))) no_wrap_char;
    // The result of this expression is no_wrap_char.
    no_wrap_char c;
    unsigned long ul;
    auto result = c + ul;

* **Two OBTs of the Same Kind**: When an operation involves two OBTs of the
  same kind (e.g., both ``wrap``), the result will have the larger of the two
  bit-widths. If the bit-widths are the same, an unsigned type is favored over
  a signed one.

  .. code-block:: c++

    typedef unsigned char __attribute__((overflow_behavior(wrap))) u8_wrap;
    typedef unsigned short __attribute__((overflow_behavior(wrap))) u16_wrap;
    // The result of this expression is u16_wrap.
    u8_wrap a;
    u16_wrap b;
    auto result = a + b;

* **Two OBTs of Different Kinds**: In an operation between a ``wrap`` and a
  ``no_wrap`` type, a ``no_wrap`` is produced. It is recommended to avoid such
  operations, as Clang may emit a warning for such cases in the future.
  Regardless, the resulting type matches the bit-width, sign and behavior of
  the ``no_wrap`` type.

.. list-table:: Promotion Rules Summary
   :widths: 30 70
   :header-rows: 1

   * - Operation Type
     - Result Type
   * - OBT + Standard Integer
     - OBT type (preserves overflow behavior, sign, and bit-width)
   * - Same Kind OBTs (both ``wrap`` or both ``no_wrap``)
     - Larger bit-width; unsigned favored if same width
   * - Different Kind OBTs (``wrap`` + ``no_wrap``)
     - ``no_wrap`` type (matches ``no_wrap`` operand's characteristics)

Interaction with Command-Line Flags and Sanitizer Special Case Lists
====================================================================

The ``overflow_behavior`` attribute interacts with sanitizers, ``-ftrapv``,
``-fwrapv``, and Sanitizer Special Case Lists (SSCL) by wholly overriding these
global flags. The following table summarizes the interactions:

.. list-table:: Overflow Behavior Precedence
   :widths: 15 15 15 15 20 15
   :header-rows: 1

   * - Behavior
     - Default(No Flags)
     - -ftrapv
     - -fwrapv
     - Sanitizers
     - SSCL
   * - ``overflow_behavior(wrap)``
     - Wraps
     - Wraps
     - Wraps
     - No report
     - Overrides SSCL
   * - ``overflow_behavior(no_wrap)``
     - Traps
     - Traps
     - Traps
     - Reports
     - Overrides SSCL

It is important to note the distinction between signed and unsigned types. For
unsigned integers, which wrap on overflow by default, ``overflow_behavior(no_wrap)``
is particularly useful for enabling overflow checks. For signed integers, whose
overflow behavior is undefined by default, ``overflow_behavior(wrap)`` provides
a guaranteed wrapping behavior.

The ``overflow_behavior`` attribute can be used to override the behavior of
entries from a :doc:`SanitizerSpecialCaseList`. This is useful for allowlisting
specific types into overflow instrumentation.

Diagnostics
===========

Clang provides diagnostics to help developers manage overflow behavior types.

-Woverflow-behavior-conversion
------------------------------

This warning group is issued when an overflow behavior type is implicitly converted
to a standard integer type, which may lead to the loss of the specified
overflow behavior.

.. code-block:: c++

  typedef int __attribute__((overflow_behavior(wrap))) wrapping_int;

  void some_function(int);

  void another_function(wrapping_int w) {
    some_function(w); // warning: implicit conversion from 'wrapping_int' to
                      // 'int' discards overflow behavior
  }

To fix this, you can explicitly cast the overflow behavior type to a standard
integer type.

.. code-block:: c++

  typedef int __attribute__((overflow_behavior(wrap))) wrapping_int;

  void some_function(int);

  void another_function(wrapping_int w) {
    some_function(static_cast<int>(w)); // OK
  }

This warning group includes
``-Wimplicit-overflow-behavior-conversion`` and
``-Wimplicit-overflow-behavior-conversion-pedantic``.

.. note::
   ``-Woverflow-behavior-conversion`` is implied by ``-Wconversion``.

-Wimplicit-overflow-behavior-conversion
---------------------------------------

This warning is issued when an overflow behavior type is implicitly converted
to a standard integer type as part of most conversions, which may lead to the
loss of the specified overflow behavior. This is the main warning in the
``-Woverflow-behavior-conversion`` group.

.. code-block:: c++

  typedef int __attribute__((overflow_behavior(wrap))) wrapping_int;

  void some_function() {
    wrapping_int w = 1;
    int i = w; // warning: implicit conversion from 'wrapping_int' to 'int'
               // during assignment discards overflow behavior
               // [-Wimplicit-overflow-behavior-conversion]
  }

Here's another example showing function parameter conversion with a ``no_wrap`` type:

.. code-block:: c++

  typedef int __attribute__((overflow_behavior(no_wrap))) safe_int;

  void bar(int x); // Function expects standard int

  void foo() {
    safe_int s = 42;
    bar(s); // warning: implicit conversion from 'safe_int' to 'int'
                      // discards overflow behavior
                      // [-Wimplicit-overflow-behavior-conversion]
  }

To fix this, you can explicitly cast the overflow behavior type to a standard
integer type.

.. code-block:: c++

  typedef int __attribute__((overflow_behavior(wrap))) wrapping_int;
  typedef int __attribute__((overflow_behavior(no_wrap))) safe_int;

  void some_function() {
    wrapping_int w = 1;
    int i = static_cast<int>(w); // OK
    int j = (int)w; // C-style OK
  }

  void bar(int x);

  void foo() {
    safe_int s = 42;
    bar(static_cast<int>(s)); // OK
  }


-Wimplicit-overflow-behavior-conversion-pedantic
------------------------------------------------

A less severe version of the warning, ``-Wimplicit-overflow-behavior-conversion-pedantic``,
is issued for implicit conversions from an unsigned wrapping type to a standard
unsigned integer type. This is considered less problematic because both types
have well-defined wrapping behavior, but the conversion still discards the
explicit ``overflow_behavior`` attribute.

.. code-block:: c++

  typedef unsigned int __attribute__((overflow_behavior(wrap))) wrapping_uint;

  void some_function(unsigned int);

  void another_function(wrapping_uint w) {
    some_function(w); // warning: implicit conversion from 'wrapping_uint' to
                      // 'unsigned int' discards overflow behavior
                      // [-Wimplicit-overflow-behavior-conversion-pedantic]
  }

-Woverflow-behavior-attribute-ignored
-------------------------------------

This warning is issued when the ``overflow_behavior`` attribute is applied to
a type that is not an integer type.

.. code-block:: c++

  typedef float __attribute__((overflow_behavior(wrap))) wrapping_float;
  // warning: 'overflow_behavior' attribute only applies to integer types;
  // attribute is ignored [-Woverflow-behavior-attribute-ignored]

  typedef struct S { int i; } __attribute__((overflow_behavior(wrap))) S_t;
  // warning: 'overflow_behavior' attribute only applies to integer types;
  // attribute is ignored [-Woverflow-behavior-attribute-ignored]

