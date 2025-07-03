.. title:: clang-tidy - modernize-avoid-fundamental-integer-types

modernize-avoid-fundamental-integer-types
==========================================

Finds fundamental integer types and recommends using typedefs or fixed-width types instead.

This check detects fundamental integer types (``int``, ``short``, ``long``, ``long long``, and their
``unsigned`` variants) and warns against their use due to non-standard platform-dependent behavior.
For example, ``long`` is 64 bits on Linux but 32 bits on Windows. There is no standard rationale or
intent for the sizes of these types.

Instead of fundamental types, use fixed-width types such as ``int32_t`` or implementation-defined
types with standard semantics, e.g. ``int_fast32_t`` for the fastest integer type greater than or
equal to 32 bits.

Examples
--------

.. code-block:: c++

  // Bad: platform-dependent fundamental types
  int global_int = 42;
  short global_short = 10;
  long global_long = 100L;
  unsigned long global_unsigned_long = 100UL;
  
  void function_with_int_param(int param) {
    // ...
  }
  
  int function_returning_int() {
    return 42;
  }
  
  struct MyStruct {
    int member_int;
    long member_long;
  };

.. code-block:: c++

  // Good: use fixed-width types or typedefs
  #include <cstdint>
  
  int32_t global_int32 = 42;
  int16_t global_int16 = 10;
  int64_t global_int64 = 100L;
  uint64_t global_uint64 = 100UL;
  
  void function_with_int32_param(int32_t param) {
    // ...
  }
  
  int32_t function_returning_int32() {
    return 42;
  }
  
  struct MyStruct {
    int32_t member_int32;
    int64_t member_int64;
  };

The check will also warn about typedef declarations that use fundamental types as their underlying type:

.. code-block:: c++

  // Bad: typedef using fundamental type
  typedef long long MyLongType;
  using MyIntType = int;

.. code-block:: c++

  // Good: use descriptive names or fixed-width types
  typedef int64_t TimestampType;
  using CounterType = uint32_t;

Rationale
---------

Fundamental integer types have platform-dependent sizes and behavior:

- ``int`` is typically 32 bits on modern platforms but is only guaranteed to be 16 bits by the spec
- ``long int`` is 32 bits on Windows but 64 bits on most Unix systems

The C++ specification does not define these types beyond their minimum sizes. That means they can
communicate intent in non-standard ways and are often needlessly incompatible. For example, ``int``
was traditionally the word size of a given processor in 16-bit and 32-bit computing and was a
reasonable default for performance. This is no longer true on modern 64-bit computers, but the size
of ``int`` remains fixed at 32 bits for backwards compatibility with code that relied on a 32-bit
implementation of ``int``.

If code is explicitly relying on the size of an ``int`` being 32 bits, it is better to say so in
the typename with ``int32_t``. Otherwise, use an appropriate implementation-defined type that
communicates your intent.

Types Not Flagged
-----------------

The following types are intentionally not flagged:

- ``char``, ``signed char``, ``unsigned char`` (character types)
- ``bool`` (boolean type)
- Standard library typedefs like ``size_t``, ``ptrdiff_t``, or ``uint32_t``.
- Already typedef'd types, though the check will flag the typedef itself

``char`` is excluded because it is implementation-defined to always be 1 byte, regardless of the
platform's definition of a byte.

``bool`` is excluded because it can only be true or false, and is not vulnerable to overflow or
narrowing issues that occur as a result of using implementation-defined types.
