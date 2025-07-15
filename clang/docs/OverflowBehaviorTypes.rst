=====================
OverflowBehaviorTypes
=====================

.. contents::
   :local:

Introduction
============

Clang provides overflow behavior types that allow developers to have fine-grained control
over the overflow behavior of integer types. Overflow behavior can be specified using
either attribute syntax or keyword syntax to control how arithmetic operations on a given integer
type should behave upon overflow. This is particularly useful for projects that
need to balance performance and safety, allowing developers to enable or
disable overflow checks for specific types.

Overflow behavior types can be enabled using the compiler option
``-foverflow-behavior-types``.

There are two syntax options for specifying overflow behavior:

**Attribute syntax:**

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
  type. Similar to ``-fwrapv``, this behavior defines wrapping behavior which
  also disables eager compiler optimizations concerning undefined behaviors.

* ``trap``: Specifies that arithmetic operations on the integer type should
  be checked for overflow. When using the ``signed-integer-overflow`` sanitizer
  or when using ``-ftrapv`` alongside a signed type, this is the default
  behavior. Using this, one may enforce overflow checks for a type even when
  ``-fwrapv`` is enabled globally.

**Keyword syntax:**

.. code-block:: c++

  __ob_wrap       // equivalent to __attribute__((overflow_behavior(wrap)))
  __ob_trap       // equivalent to __attribute__((overflow_behavior(trap)))

Either of these spellings can be applied to ``typedef`` declarations and to
integer types directly.

Arithmetic operations containing one or more overflow behavior types follow
standard C integer promotion and conversion rules while preserving overflow
behavior information.

Examples
========

Here are examples using both syntax options:

**Using attribute syntax with a typedef:**

.. code-block:: c++

  typedef unsigned int __attribute__((overflow_behavior(trap))) non_wrapping_uint;

  non_wrapping_uint add_one(non_wrapping_uint a) {
    return a + 1; // Overflow is checked for this operation.
  }

**Using keyword syntax with a typedef:**

.. code-block:: c++

  typedef unsigned int __ob_trap non_wrapping_uint;

  non_wrapping_uint add_one(non_wrapping_uint a) {
    return a + 1; // Overflow is checked for this operation.
  }

**Using attribute syntax with a type directly:**

.. code-block:: c++

  int mul_alot(int n) {
    int __attribute__((overflow_behavior(wrap))) a = n;
    return a * 1337; // Potential overflow is not checked and is well-defined
  }

**Using keyword syntax with a type directly:**

.. code-block:: c++

  int mul_alot(int n) {
    int __ob_wrap a = n;
    return a * 1337; // Potential overflow is not checked and is well-defined
  }

"Well-defined" overflow is consistent with two's complement wrap-around
semantics and won't be removed via eager compiler optimizations (like some
undefined behavior might).

Promotion Rules
===============

Overflow behavior types (OBTs) follow the traditional C integer promotion and
conversion rules while propagating overflow behavior qualifiers through implicit
casts. This approach ensures compatibility with existing C semantics while
maintaining overflow behavior information throughout arithmetic expressions.

The resulting type characteristics for overflow behavior types (OBTs) across a
variety of scenarios is detailed below.

* **OBT and Standard Integer Type**: The result follows standard C conversion
  rules, with the OBT qualifier applied to the standard result type.

  .. code-block:: c++

    typedef char __ob_trap trap_char;
    trap_char c;
    unsigned long ul;
    auto result = c + ul; // result is __ob_trap unsigned long

* **Two OBTs with Same Behavior**: When both operands have the same overflow
  behavior, the result follows standard C arithmetic conversions with that
  behavior applied.

  .. code-block:: c++

    typedef unsigned char __ob_wrap u8_wrap;
    typedef unsigned short __ob_wrap u16_wrap;
    u8_wrap a;
    u16_wrap b;
    auto result = a + b; // result is __ob_wrap int (C promotes both to int)

* **Two OBTs with Different Behaviors**: ``trap`` behavior dominates ``wrap``
  behavior. The result follows standard C arithmetic conversions with ``trap``
  behavior applied.

  .. code-block:: c++

    typedef unsigned short __ob_wrap u16_wrap;
    typedef int __ob_trap int_trap;
    u16_wrap a;
    int_trap b;
    auto result = a + b; // result is __ob_trap int (C promotes u16->int, dominance gives trap)


.. list-table:: Promotion Rules Summary
   :widths: 30 70
   :header-rows: 1

   * - Operation Type
     - Result Type
   * - OBT + Standard Integer
     - Standard C conversion result with OBT qualifier applied
   * - Same Kind OBTs (both ``wrap`` or both ``trap``)
     - Standard C conversion result with common overflow behavior
   * - Different Kind OBTs (``wrap`` + ``trap``)
     - Standard C conversion result with ``trap`` behavior (dominance)

**Overflow Behavior Dominance Rules:**

1. If either operand has ``trap`` behavior → result has ``trap`` behavior
2. If either operand has ``wrap`` behavior (and neither has ``trap`) → result has ``wrap`` behavior
3. Otherwise → no overflow behavior annotation

This model preserves traditional C semantics while ensuring overflow behavior
information is correctly propagated through arithmetic expressions.

Conversion Semantics
====================

Overflow behavior types are implicitly convertible to and from built-in
integral types with specific semantics for warnings and constant evaluation.

Truncation Semantics
--------------------

Truncation and overflow are related and are both often desirable in some
contexts and undesirable in others. To provide control over these behaviors,
overflow behavior types may also be used to control truncation instrumentation
at the type level.

Note that implicit integer truncation is not an undefined behavior in C. Due to
this, overflow behavior types make no special guarantees about whether implicit
integer truncation is defined or not -- the behavior is simply always defined.
Use of overflow behavior types in these contexts only control instrumentation
related to truncation.

When an overflow behavior type is involved as the source or destination type of
truncation, instrumentation checks behave as follows:

* **One or more types is 'trap'**: Truncation checks are inserted and may issue
  a trap or sanitizer warning based on compiler settings.

* **One or more types is 'wrap'**: No truncation checks are added regardless of
  compiler flags because truncation with wrapping behavior well-defined. If any
  of the types is ``trap`` then the truncation rules match the behavior
  mentioned above.

* **Both types are standard integer types**: Behaviors surrounding standard
  integer types is unchanged.

.. code-block:: c++

  void foo(char a, int __ob_trap b) {
    a = b; // truncation checks are inserted, may trap
  }

  void bar(char a, int __ob_wrap b) {
    a = b; // sanitizer truncation checks disallowed
  }

Note that truncation itself is a form of overflow behavior - when a value
is too large to fit in the destination type, the high-order bits are
discarded, which is the wrapping behavior that ``wrap`` types are
designed to handle predictably.

Implicit Conversions Due to Assignment
--------------------------------------

Like with the basic integral types in C and C++, types on the right-hand side
of an assignment may be implicitly converted to match the left-hand side.

All built-in integral types can be implicitly converted to an overflow behavior
version.

.. code-block:: c++

  char x = 1;
  int __ob_wrap a = x; // x converted to __ob_wrap int

Assigning one overflow behavior type to another is legal only when the two
type's have matching overflow behavior kinds (i.e., `wrap` and `wrap` or `trap`
and `trap`). Assigning overflow behavior types with differing behavior kinds
will result in an error.

.. code-block:: c++

  long __ob_wrap x = __LONG_MAX__;
  int __ob_trap a = x; // x converted to int __ob_trap

C++ Overload Resolution
-----------------------

For the purposes of C++ overload set formation, promotions or conversions to
and from overflow behavior types are of the same rank as normal integer
promotions and conversions. These rules have implications during overload
candidate selection which may lead to unexpected but correct ambiguity errors.

The example below shows potential ambiguity as matching just the underlying
type of an OBT parameter is not enough to precisely pick an overload candidate.

.. code-block:: c++

  void foo(int __ob_trap a);
  void foo(short a);

  void bar(int a) {
    foo(a); // call to 'foo' is ambiguous
  }

Most integral types can be implicitly converted to match OBT parameter types
and this can be done unambiguously in certain cases. Especially, when all other
candidates are not implicitly convertible.

.. code-block:: c++

  void foo(int __ob_trap a);
  void foo(char *a);

  void bar(int a) {
    foo(a); // picks foo(__ob_trap int)
  }


Overflow behavior types with differing kinds may also create ambiguity in
certain contexts.

.. code-block:: c++

  void foo(int __ob_trap a);
  void foo(int __ob_wrap a);

  void bar(int a) {
    foo(a); // call to 'foo' is ambiguous
  }

Overflow behavior types may also be used as template parameters and used within
C ``_Generic`` expressions.

C _Generic Expressions
----------------------

Overflow behavior types may be used within C ``_Generic`` expressions.

Overflow behavior types do not match against their underlying types within C
``_Generic`` expressions. This means that an OBT will not be considered
equivalent to its base type for generic selection purposes. OBTs will match
against exact types considering bitwidth, signedness and overflow
behavior kind.

.. code-block:: c++

  int foo(int __ob_wrap x) {
    return _Generic(x, int: 1, char: 2, default: 3); // returns 3
  }

  int bar(int __ob_wrap x) {
    return _Generic(x, int __ob_wrap: 1, int: 2, default: 3); // returns 1
  }


C++ Template Specializations
-----------------------------

Like with ``_Generic``, each OBT is treated as a distinct type for template
specialization purposes, enabling precise type-based template selection.

.. code-block:: c++

  template<typename T>
  struct TypeProcessor {
    static constexpr int value = 0; // default case
  };

  template<>
  struct TypeProcessor<int> {
    static constexpr int value = 1; // int specialization
  };

  template<>
  struct TypeProcessor<int __ob_wrap> {
    static constexpr int value = 2; // __ob_wrap int specialization
  };

  template<>
  struct TypeProcessor<int __ob_trap> {
    static constexpr int value = 3; // __ob_trap int specialization
  };

When no exact template specialization exists for an OBT, it falls back to the
default template rather than matching the underlying type specialization,
maintaining type safety and avoiding unexpected behavior.

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
   * - ``overflow_behavior(trap)``
     - Traps
     - Traps
     - Traps
     - Reports
     - Overrides SSCL

It is important to note the distinction between signed and unsigned types. For
unsigned integers, which wrap on overflow by default, ``overflow_behavior(trap)``
is particularly useful for enabling overflow checks. For signed integers, whose
overflow behavior is undefined by default, ``overflow_behavior(wrap)`` provides
a guaranteed wrapping behavior.

The ``overflow_behavior`` attribute can be used to override the behavior of
entries from a :doc:`SanitizerSpecialCaseList`. This is useful for allowlisting
specific types into overflow or truncation instrumentation.

Diagnostics
===========

Clang provides diagnostics to help developers manage overflow behavior types.

-Woverflow-behavior-conversion
------------------------------

This warning group is issued when an overflow behavior type is implicitly
converted to a standard integer type, which may lead to the loss of the
specified overflow behavior.

.. code-block:: c++

  typedef int __ob_wrap wrapping_int;

  void some_function(int);

  void another_function(wrapping_int w) {
    some_function(w); // warning: implicit conversion from 'wrapping_int' to
                      // 'int' discards overflow behavior
  }

To fix this, you can explicitly cast the overflow behavior type to a standard
integer type.

.. code-block:: c++

  typedef int __ob_wrap wrapping_int;

  void some_function(int);

  void another_function(wrapping_int w) {
    some_function(static_cast<int>(w)); // OK
  }

This warning group includes
``-Wimplicit-overflow-behavior-conversion``, which includes
``-Wimplicit-overflow-behavior-conversion-assignment`` and
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

  typedef int __ob_wrap wrapping_int;

  void some_function() {
    wrapping_int w = 1;
    int i = w; // warning: implicit conversion from 'wrapping_int' to 'int'
               // during assignment discards overflow behavior
               // [-Wimplicit-overflow-behavior-conversion]
  }

Here's another example showing function parameter conversion with a ``trap`` type:

.. code-block:: c++

  typedef int __ob_trap safe_int;

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

  typedef int __ob_wrap wrapping_int;
  typedef int __ob_trap safe_int;

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


-Wimplicit-overflow-behavior-conversion-assignment
--------------------------------------------------

This warning is issued specifically when an overflow behavior type is implicitly
converted to a standard integer type during assignment operations. This is a
subset of the more general ``-Wimplicit-overflow-behavior-conversion`` warning,
allowing developers to control assignment-specific warnings separately.

.. code-block:: c++

  typedef int __ob_wrap wrapping_int;

  void some_function() {
    wrapping_int w = 1;
    int i = w; // warning: implicit conversion from 'wrapping_int' to 'int'
               // during assignment discards overflow behavior
               // [-Wimplicit-overflow-behavior-conversion-assignment]
  }

This diagnostic can be controlled independently, allowing projects to suppress
assignment-related warnings while still receiving warnings for other types of
implicit conversions (such as function parameter passing).

-Wimplicit-overflow-behavior-conversion-pedantic
------------------------------------------------

A less severe version of the warning, ``-Wimplicit-overflow-behavior-conversion-pedantic``,
is issued for implicit conversions from an unsigned wrapping type to a standard
unsigned integer type. This is considered less problematic because both types
have well-defined wrapping behavior, but the conversion still discards the
explicit ``overflow_behavior`` attribute.

.. code-block:: c++

  typedef unsigned int __ob_wrap wrapping_uint;

  void some_function(unsigned int);

  void another_function(wrapping_uint w) {
    some_function(w); // warning: implicit conversion from 'wrapping_uint' to
                      // 'unsigned int' discards overflow behavior
                      // [-Wimplicit-overflow-behavior-conversion-pedantic]
  }

Format String Functions
=======================

When overflow behavior types are used with format string functions (printf-family
functions like ``printf``, ``fprintf``, ``sprintf``, etc., and scanf-family
functions like ``scanf``, ``fscanf``, ``sscanf``, etc.), they are treated based
on their underlying integer types for format specifier compatibility checking.
More generally, overflow behavior types are ABI-compatible with their underlying
types when passed to any varargs function.

.. code-block:: c++

  #include <cstdio>

  typedef int __ob_wrap wrap_int;
  typedef unsigned int __ob_trap nowrap_uint;

  void example() {
    wrap_int wi = 42;
    nowrap_uint su = 100;

    scanf("%d\n", &wi);   // OK: &wi treated as int* for %d
    printf("%d\n", wi);  // OK: wi treated as int for %d
    printf("%u\n", su);  // OK: su treated as unsigned int for %u
    printf("%s\n", wi);  // Error: int incompatible with %s (same as regular int)
  }

This behavior ensures that overflow behavior types work seamlessly with existing
format string functions without requiring special format specifiers, while
still maintaining their overflow behavior semantics in arithmetic operations.

The format string checker uses the underlying type to determine compatibility,
so ``int __ob_wrap`` is fully compatible with ``%d``, ``%i``, ``%x``, etc.,
just like a regular ``int`` would be.

Using With Non-Integer Types
----------------------------

An error is issued when attempting to create an overflow behavior type from
a non-integer type.

.. code-block:: c++

  typedef float __attribute__((overflow_behavior(wrap))) wrapping_float;
  // error: 'overflow_behavior' attribute cannot be applied to non-integer type 'float'

  typedef struct S { int i; } __attribute__((overflow_behavior(wrap))) S_t;
  // error: 'overflow_behavior' attribute cannot be applied to non-integer type 'struct S'

