=====================
OverflowBehaviorTypes
=====================

.. contents::
   :local:

Introduction
============

Clang provides overflow behavior types that allow developers to have
fine-grained control over the overflow behavior of integer types. Overflow
behavior can be specified using either attribute syntax or keyword syntax to
control how arithmetic operations on a given integer type should behave upon
overflow. This is particularly useful for projects that need to balance
performance and safety, allowing developers to enable or disable overflow
checks for specific types.

Overflow behavior types can be enabled using the ``-cc1`` compiler option
``-fexperimental-overflow-behavior-types``.

.. note::

   This feature is experimental. The flag spelling may change in future
   releases as the feature matures.

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
conversion rules while propagating overflow behavior qualifiers through
implicit casts. This ensures compatibility with existing C semantics while
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

Pointer Semantics
-----------------

Overflow behavior types can be used with pointers, where the overflow behavior
annotation applies to the pointee type. Pointers to overflow behavior types
have specific compatibility and conversion rules.

**Pointer Type Compatibility:**

Pointers to overflow behavior types are treated as distinct types from their
underlying types and from each other when they have different overflow
behaviors. This affects assignment, parameter passing, and other pointer
operations.

**Discarding Overflow Behavior:**

When assigning from a overflow behavior annotated type to a regular integer
type, the compiler issues a warning about discarding overflow behavior. This is
controlled by the ``-Wincompatible-pointer-types-discards-overflow-behavior``
diagnostic.

.. code-block:: c++

  unsigned long *px;
  unsigned long __ob_trap *py;
  unsigned long __ob_wrap *pz;

  // Discarding overflow behavior - warns but allowed
  px = py; // warning: assigning to 'unsigned long *' from
           // '__ob_trap unsigned long *' discards overflow behavior
  py = px; // warning: assigning to '__ob_trap unsigned long *' from
           // 'unsigned long *' discards overflow behavior

Conversion Semantics
====================

Overflow behavior types are implicitly convertible to and from built-in
integral types with specific semantics for warnings and constant evaluation.

**Incompatible Overflow Behaviors:**

Attempting to assign or convert between types with incompatible overflow
behaviors (``trap`` vs ``wrap``) results in a compilation error, as these
represent fundamentally different behavioral contracts.

.. code-block:: c++

  int __ob_trap a;
  int __ob_wrap b;
  a = b; // error: assigning to '__ob_trap int' from '__ob_wrap int' with
         // incompatible overflow behavior types ('__ob_trap' and '__ob_wrap')

The Diagnostics section further below details the exact diagnostics Clang
provides for overflow behavior types.

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
  either a trap or sanitizer warning based on compiler settings.

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

Note that truncation itself is a form of overflow behavior - when a value is
too large to fit in the destination type, the high-order bits are discarded,
which is a wrapping behavior that ``wrap`` types are designed to handle
predictably.

Constant Conversion Semantics
------------------------------

When converting constant values to overflow behavior types, the behavior
depends on the overflow behavior annotation and whether the conversion would
change the value:

**With 'wrap' types:**

Constant conversions that would overflow are accepted without warning when
assigning to or explicitly casting to ``wrap`` types, as wrapping is the
intended behavior:

.. code-block:: c++

  short x1 = (int __ob_wrap)100000;        // OK: explicit wrap cast
  short __ob_wrap x2 = (int)100000;        // OK: wrap destination
  unsigned short __ob_wrap ux = 100000;    // OK: wrapping expected

**With 'trap' types:**

Constant conversions that would change the value generate a warning when the
destination is a ``trap`` type. This matches the behavior of regular non-OBT
constant conversions.

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
types have matching overflow behavior kinds (i.e., ``wrap`` and ``wrap`` or
``trap`` and ``trap``). Assigning overflow behavior types with differing
behavior kinds will result in an error.

.. code-block:: c++

  int __ob_trap trap_var;
  int __ob_wrap wrap_var;

  trap_var = wrap_var; // error: assigning to '__ob_trap int' from
                       // '__ob_wrap int' with incompatible overflow
                       // behavior types ('__ob_trap' and '__ob_wrap')

However, conversions between compatible overflow behavior types (same kind,
different underlying widths or signedness) are allowed:

.. code-block:: c++

  long __ob_wrap x = __LONG_MAX__;
  int __ob_wrap a = x; // OK: both have 'wrap' behavior

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
     - No report, Wraps
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

Specification Errors and Warnings
----------------------------------

When using overflow behavior types, several diagnostics help catch mistakes in
how the types are specified.

**Conflicting Specifications:**

An error is issued when both the keyword syntax and attribute syntax are used
with incompatible overflow behaviors on the same type:

.. code-block:: c++

  int __ob_wrap __attribute__((overflow_behavior(trap))) x;
  // error: conflicting overflow behavior specification; specifier
  // specifies 'wrap' but attribute specifies 'trap'

**Redundant Specifications:**

A warning is issued when both syntax forms specify the same behavior, which is
redundant but allowed:

.. code-block:: c++

  int __ob_wrap __attribute__((overflow_behavior(wrap))) x;
  // warning: redundant overflow behavior specification; both specifier
  // and attribute specify 'wrap'

**Feature Not Enabled:**

When overflow behavior types are used without enabling the feature via
``-fexperimental-overflow-behavior-types``, a warning is issued and the attribute is ignored:

.. code-block:: c++

  // Without -fexperimental-overflow-behavior-types
  int __attribute__((overflow_behavior(wrap))) x;
  // warning: 'overflow_behavior' attribute is ignored because it is
  // not enabled; pass -fexperimental-overflow-behavior-types

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
``-Wimplicit-overflow-behavior-conversion-assignment``,
``-Wimplicit-overflow-behavior-conversion-function-boundary``,
``-Wimplicit-overflow-behavior-conversion-function-boundary-pedantic``, and
``-Wimplicit-overflow-behavior-conversion-pedantic``.

.. note::
   ``-Woverflow-behavior-conversion`` is implied by ``-Wconversion``.

-Wincompatible-pointer-types-discards-overflow-behavior
--------------------------------------------------------

This warning (enabled by default) is issued when converting between pointer
types where overflow behavior information is discarded. This occurs when
assigning between pointers that differ only in their overflow behavior
annotation.

.. code-block:: c++

  void example() {
    unsigned long __ob_trap *trap_ptr;
    unsigned long *regular_ptr;

    regular_ptr = trap_ptr; // warning: assigning to 'unsigned long *'
                            // from '__ob_trap unsigned long *' discards
                            // overflow behavior

    trap_ptr = regular_ptr; // warning: assigning to '__ob_trap unsigned long *'
                            // from 'unsigned long *' discards overflow behavior
  }

This warning is part of the ``-Wincompatible-pointer-types`` group and helps
catch potential bugs where overflow behavior contracts may be inadvertently
lost through pointer conversions.

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

This warning is issued specifically when an overflow behavior type is
implicitly converted to a standard integer type during assignment operations or
variable initialization. This is a subset of the more general
``-Wimplicit-overflow-behavior-conversion`` warning, allowing developers to
control assignment-specific warnings separately. This warning is disabled by
default.

.. code-block:: c++

  typedef int __ob_wrap wrapping_int;

  void some_function() {
    wrapping_int w = 1;
    int i = w; // warning: implicit conversion from 'wrapping_int' to 'int'
               // during assignment discards overflow behavior
               // [-Wimplicit-overflow-behavior-conversion-assignment]
  }

This also applies to variable initialization:

.. code-block:: c++

  void another_example() {
    int __ob_trap safe = 42;
    int regular = safe; // warning: implicit conversion from '__ob_trap int'
                        // to 'int' during assignment discards overflow behavior
                        // [-Wimplicit-overflow-behavior-conversion-assignment]
  }

This diagnostic can be controlled independently, allowing projects to suppress
assignment-related warnings while still receiving warnings for other types of
implicit conversions (such as function parameter passing).

Note that when assigning from a non-OBT type to an OBT type, no warning is
issued as overflow behavior is being added, not discarded:

.. code-block:: c++

  void adding_obt_is_ok() {
    int plain = 42;
    int __ob_wrap wrapped = plain; // OK - adding overflow behavior, no warning
  }

-Wimplicit-overflow-behavior-conversion-assignment-pedantic
-----------------------------------------------------------

A less severe version of
``-Wimplicit-overflow-behavior-conversion-assignment`` which is issued only
when an unsigned ``wrap`` type is implicitly converted to a standard unsigned
integer type during assignment. This is considered less problematic than other
conversions because unsigned integers already have well-defined wrapping
behavior by the C standard.

.. code-block:: c++

  void example() {
    unsigned int __ob_wrap wrapped_uint = 42;
    unsigned int regular = wrapped_uint; // warning: implicit conversion from
                                         // '__ob_wrap unsigned int' to
                                         // 'unsigned int' during assignment
                                         // discards overflow behavior
                                         // [-Wimplicit-overflow-behavior-conversion-assignment-pedantic]
  }

This warning is useful for projects that want to track all overflow behavior
annotations but consider unsigned wrapping conversions to be lower priority
than signed conversions or ``trap`` conversions.

-Wimplicit-overflow-behavior-conversion-function-boundary
----------------------------------------------------------

This warning (disabled by default) is issued when an overflow behavior type is
passed as an argument to a function parameter that does not have the same
overflow behavior annotation. This helps identify situations where overflow
behavior contracts may be lost at function boundaries.

.. code-block:: c++

  void process_value(int x);  // Function expects standard int

  void caller() {
    int __ob_trap safe_value = 42;
    process_value(safe_value); // warning: passing argument of type
                               // '__ob_trap int' to parameter of type 'int'
                               // discards overflow behavior at function boundary
                               // [-Wimplicit-overflow-behavior-conversion-function-boundary]
  }

This warning can be particularly useful for catching cases where overflow
checking behavior is lost when crossing API boundaries. Both the general
version
(``-Wimplicit-overflow-behavior-conversion-function-boundary``) and a pedantic
version (``-Wimplicit-overflow-behavior-conversion-function-boundary-pedantic``)
for unsigned wrapping types are disabled by default.

To fix this warning, you can either:

1. Update the function parameter to accept the overflow behavior type:

.. code-block:: c++

  void process_value(int __ob_trap x);  // Accept trap-annotated int

  void caller() {
    int __ob_trap safe_value = 42;
    process_value(safe_value); // OK
  }

2. Explicitly cast at the call site to acknowledge the loss of overflow behavior:

.. code-block:: c++

  void process_value(int x);

  void caller() {
    int __ob_trap safe_value = 42;
    process_value((int)safe_value); // OK - explicit cast
  }

-Wimplicit-overflow-behavior-conversion-function-boundary-pedantic
------------------------------------------------------------------

A less severe version of the warning above which is issued only in the case of
unsigned ``wrap`` types being passed to a function expecting a standard
unsigned integer type. This is a less problematic issue since the standard and
well-defined overflow procedure for unsigned integer types is essentially
``wrap``.


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

Incompatibility With Non-Integer Types
--------------------------------------

An error is issued when attempting to create an overflow behavior type from
a non-integer type.

.. code-block:: c++

  typedef float __attribute__((overflow_behavior(wrap))) wrapping_float;
  // error: 'overflow_behavior' attribute cannot be applied to non-integer type 'float'

  typedef struct S { int i; } __attribute__((overflow_behavior(wrap))) S_t;
  // error: 'overflow_behavior' attribute cannot be applied to non-integer type 'struct S'

