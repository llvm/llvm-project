.. title:: clang-tidy - portability-avoid-platform-specific-fundamental-types

portability-avoid-platform-specific-fundamental-types
=====================================================

Detects fundamental types (``int``, ``short``, ``long``, ``long long``, ``char``
, ``float``, etc) and warns against their use due to platform-dependent 
behaviour. For example, ``long`` is 64 bits on Linux but 32 bits on Windows.
There is no standard rationale or intent for the sizes of these types.

Instead of fundamental types, use fixed-width types such as ``int32_t`` or
implementation-defined types with standard semantics, e.g. ``int_fast32_t`` for
the fastest integer type greater than or equal to 32 bits.

Examples
--------

.. code-block:: c++

  // Bad: platform-dependent fundamental types
  int global_int = 42;
  short global_short = 10;
  // unsigned long is 32 bits on Windows and 64 bits on Mac/Linux, so this will
  // overflow depending on platform.
  unsigned long global_unsigned_long = 1 << 36;
  // On many systems, loading into a register must be done at the processor's
  // word size. On a 64-bit system with 32-bit integers, loading an element from
  // slowVec could take multiple instructions. The first will load two elements,
  // and additional instructions will delete the unneeded element.
  std::vector<int> slowVec;
  // This could overflow and cause undefined behaviour if slowVec is too big.
  for(int i = 0U; i<slowVec.size();i++) {
    slowVec[i];
  }

.. code-block:: c++

  // Good: use fixed-width types or typedefs
  #include <cstdint>

  int32_t global_int32 = 42;
  uint64_t global_uint64 = 100UL;
  // On a 64-bit system, int_fast32_t will probably be 64 bits in size,
  // potentially allowing faster accesses.
  std::vector<int_fast32_t> fastVec;
  // A size_t can hold any index into an array or vector on a given platform,
  // because it can hold the maximum size of any theoretical object, so we will
  // never overflow fastVec's size.
  for(size_t i = 0U; i<fastVec.size();i++) {
    fastVec[i];
  }

Rationale
---------

Examples of platform-dependent behaviour:

- ``int`` is typically 32 bits on modern platforms but is only guaranteed to be
  16 bits by the spec
- ``long int`` is 32 bits on Windows but 64 bits on most Unix systems
- ``double`` is typically 64-bit IEEE754, but on some microcontrollers without
  a 64-bit FPU (e.g. certain Arduinos) it can be 32 bits
- ``char`` is signed on ARM and unsigned on x86

For historical reasons, the C++ standard allows the implementation to define
the size, representation, and purpose of these types. They communicate intent in
non-standard ways and are often needlessly incompatible.

For example, ``int`` was traditionally the word size of a given processor in
16-bit and 32-bit computing and was a reasonable default for performance. This
is no longer true on modern 64-bit computers, but the size of ``int`` remains
fixed at 32 bits for backwards compatibility with code that relied on a 32-bit
implementation of ``int``.

If code is explicitly relying on the size of an ``int`` being 32 bits, it is
better to say so in the typename with ``int32_t``. Otherwise, use an appropriate
implementation-defined type such as ``fast_int32_t`` or ``least_int32_t`` that
communicates the appropriate time/space tradeoff.

Likewise, ``float`` and ``double`` should be replaced by ``float32_t`` and
``float64_t`` which, if they exist, are guaranteed to be standard IEEE754 floats
of the given size.

``char`` should be replaced by ``char8_t`` when used in the representation of
Unicode text. When used to represent a byte on a given platform, ``std::byte``
is the correct replacement. ``char8_t`` and ``std::byte`` are guaranteed to be
implemented with similar behaviour to unsigned char.

Types Not Flagged
-----------------

The following types are intentionally not flagged:

- ``bool`` (boolean type).
- Standard library typedefs like ``size_t``, ``ptrdiff_t``, or ``uint32_t``.
- Already typedef'd types, though the check will flag the typedef itself.

``bool`` is excluded because it can only be true or false, and is not
vulnerable to overflow or narrowing issues that occur as a result of it being an
implementation-defined size.

Options
-------

.. option:: WarnOnInts

   When `true`, the check will warn about fundamental integer types
   (``short``, ``int``, ``long``, ``long long`` and their ``signed`` and 
   ``unsigned`` variants).
   When `false`, integer types are not flagged. 
   
   Default is `true`.

.. option:: WarnOnFloats

   When `true`, the check will warn about floating point types
   (``float`` and ``double``).
   When `false`, floating point types are not flagged.

   Default is `true`

.. option:: WarnOnChars

   When `true`, the check will warn about character types (``char``,
   ``signed char``, and ``unsigned char``).
   When `false`, character types are not flagged.

   Default is `true`
