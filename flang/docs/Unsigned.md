<!--===- docs/Unsigned.md 
  
   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  
-->

# Fortran Extensions supported by Flang

```{contents}
---
local:
---
```

For better compatibility with GNU Fortran and Sun Fortran,
this compiler supports an option (`-funsigned`) that enables
the `UNSIGNED` data type, constants, intrinsic functions,
its use with intrinsic operations and `SELECT CASE`, and C
language interoperability.

## `UNSIGNED` type

`UNSIGNED` is a numeric type with the same kinds as `INTEGER`.
It may appear as a type-spec in any context, including
a type declaration statement, a type-decl in an array
constructor or `ALLOCATE` statement, `IMPLICIT`, or a
function statement's prefix.

`UNSIGNED` constants are nonempty strings of decimal digits
followed by the letter `U` and optionally a kind suffix with
an underscore.

## `UNSIGNED` operations

`UNSIGNED` operands are accepted for unary negation (`-`),
the basic four binary arithmetic intrinsic operations `+`, `-`, `*`, and `/`,
components in complex constructors,
and for numeric relational operators.
The power operator `**` does not accept `UNSIGNED` operands.

Mixed operations with other types are not allowed.
Mixed operations with one `UNSIGNED` operand and one BOZ literal
constant operand are allowed.
When the operands' kinds differ, the smaller operand is zero-extended
to the size of the larger.

The arithmetic operations `u+v`, `-u`, `u-v`, and `u*v` are implemented
modulo `MAX(HUGE(u),HUGE(v))+1`;
informally speaking, they always truncate their results, or are
guaranteed to "wrap".

## `UNSIGNED` intrinsic functions

`UNSIGNED` operands are accepted as operands to,
or may be returned as results from,
several intrinsic procedures.

Bitwise operations:
* `NOT`
* `IAND`, `IOR`, `IEOR`, `IBCLR`, `IBSET`, `IBITS`, `MERGE_BITS`
* `BTEST`
* `ISHFT`, `ISHFTC`
* `SHIFTA`, `SHIFTL`, `SHIFTR`
* `TRANSFER`
* `MVBITS`

The existing unsigned comparisons `BLT`, `BLE`, `BGE`, and `BGT`.

The inquiries `BIT_SIZE`, `DIGITS`, `HUGE`, and `RANGE`.

Homogeneous `MAX` and `MIN`.

`RANDOM_NUMBER`.

The intrinsic array functions:
* `MAXVAL`, `MINVAL`
* `SUM`, `PRODUCT`
* `IALL`, `IANY`, `IPARITY`
* `DOT_PRODUCT`, `MATMUL`

All of the restructuring array transformational intrinsics: `CSHIFT`, `EOSHIFT`,
  `PACK`, `RESHAPE`, `SPREAD`, `TRANSPOSE`, and `UNPACK`.

The location transformationals `FINDLOC`, `MAXLOC`, and `MINLOC`.

There is a new `SELECTED_UNSIGNED_KIND` intrinsic function; it happens
to work identically to the existing `SELECTED_INT_KIND`.

Two new intrinsic functions `UMASKL` and `UMASKR` work just like
`MASKL` and `MASKR`, returning unsigned results instead of integers.

Conversions to `UNSIGNED`, or between `UNSIGNED` kinds, can be done
via the new `UINT` intrinsic.  The `UNSIGNED` intrinsic name is also
supported as an alias.

Support for `UNSIGNED` in the `OUT_OF_RANGE` predicate remains to be implemented.

## Other usage

`UNSIGNED` is allowed in `SELECT CASE`, but not in `DO` loop indices or
limits, or an arithmetic `IF` expression.

`UNSIGNED` array indices are not allowed.

`UNSIGNED` data may be used as data items in I/O statements, including
list-directed and `NAMELIST` I/O.
Format-directed I/O may edit `UNSIGNED` data with `I`, `G`, `B`, `O`, and `Z`
edit descriptors.

## C interoperability

`UNSIGNED` data map to type codes for C's `unsigned` types in the
`type` member of a `cdesc_t` descriptor in the `ISO_Fortran_binding.h`
header file.

## Standard modules

New definitions (`C_UNSIGNED`, `C_UINT8_T`, &c.) were added to ISO_C_BINDING
and new constants (`UINT8`, `UINT16`, &c.) to ISO_FORTRAN_ENV.
