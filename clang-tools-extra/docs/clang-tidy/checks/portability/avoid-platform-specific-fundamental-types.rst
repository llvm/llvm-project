.. title:: clang-tidy - portability-avoid-platform-specific-fundamental-types

portability-avoid-platform-specific-fundamental-types
=====================================================

Finds fundamental types (e.g. `int`, `float`) and recommends using typedefs 
or fixed-width types instead to improve portability across different platforms.

This check detects fundamental types (``int``, ``short``, ``long``, ``float``, 
``char`` and their ``unsigned`` or ``signed`` variants) and warns against their 
use due to non-standard platform-dependent behavior. For example, ``long`` is 
64 bits on Linux but 32 bits on Windows. There is no standard rationale or 
intent for the sizes of these types.

Instead of fundamental types, use fixed-width types such as ``int32_t`` or 
implementation-defined types with standard semantics, e.g. ``int_fast32_t`` for 
the fastest integer type greater than or equal to 32 bits.

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

The check will also warn about typedef declarations that use fundamental types
as their underlying type:

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

Fundamental types have platform-dependent sizes and behavior:

- ``int`` is typically 32 bits on modern platforms but is only guaranteed to be
  16 bits by the spec
- ``long int`` is 32 bits on Windows but 64 bits on most Unix systems
- ``double`` is typically 64-bit IEEE754, but on some microcontrollers without
  a 64-bit FPU (e.g. certain Arduinos) it can be 32 bits
- ``char`` is signed on ARM and unsigned on x86

The C++ specification does not define these types beyond their minimum sizes. 
That means they can communicate intent in non-standard ways and are often 
needlessly incompatible. For example, ``int``was traditionally the word size of
a given processor in 16-bit and 32-bit computing and was a reasonable default 
for performance. This is no longer true on modern 64-bit computers, but the 
size of ``int`` remains fixed at 32 bits for backwards compatibility with code 
that relied on a 32-bit implementation of ``int``.

If code is explicitly relying on the size of an ``int`` being 32 bits, it is 
better to say so in the typename with ``int32_t``. Otherwise, use an 
appropriate implementation-defined type such as ``fast_int32_t`` or 
``least_int32_t`` that communicates the appropriate time/space tradeoff.

Likewise, ``float`` and ``double`` should be replaced by ``float32_t`` and
``float64_t`` which are guaranteed to be standard IEEE754 floats for a given
size.

``char`` should be replaced by ``char8_t`` when used in the representation of
Unicode text. When used to represent a byte on a given platform, ``std::byte``
is an appropriate replacement.

Types Not Flagged
-----------------

The following types are intentionally not flagged:

- ``bool`` (boolean type)
- Standard library typedefs like ``size_t``, ``ptrdiff_t``, or ``uint32_t``.
- Already typedef'd types, though the check will flag the typedef itself

``bool`` is excluded because it can only be true or false, and is not vulnerable to overflow or
narrowing issues that occur as a result of using implementation-defined types.

Options
-------

.. option:: WarnOnFloats

   When `true`, the check will warn about floating point types (``float`` and ``double``).
   When `false` (default), floating point types are not flagged.

   Floating point types can have platform-dependent behavior:
   
   - ``float`` is typically 32-bit IEEE754, but can vary on some platforms
   - ``double`` is typically 64-bit IEEE754, but on some microcontrollers without
     a 64-bit FPU it can be 32 bits
   
   When this option is enabled, the check will suggest using ``float32_t`` and ``float64_t``
   instead of ``float`` and ``double`` respectively, when the target platform supports
   standard IEEE754 sizes.

   Example with ``WarnOnFloats`` enabled:

   .. code-block:: c++

     // Bad: platform-dependent floating point types
     float pi = 3.14f;
     double e = 2.71828;

   .. code-block:: c++

     // Good: use fixed-width floating point types
     #include <stdfloat>  // C++23
     
     float32_t pi = 3.14f;
     float64_t e = 2.71828;
