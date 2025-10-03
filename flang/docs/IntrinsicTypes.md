<!--===- docs/IntrinsicTypes.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# Implementation of `Intrinsic` types in f18

```{contents}
---
local:
---
```

Intrinsic types are integer, real, complex, character, and logical.
All intrinsic types have a kind type parameter called KIND,
which determines the representation method for the specified type.
The intrinsic type character also has a length type parameter called LEN,
which determines the length of the character string.

The implementation of `CHARACTER` type in f18 is described
in [Character.md](Character.md).

## Supported TYPES and KINDS

f18 supports the following type and kind combinations:

| Type | Description |
| :--: | :---------: |
| INTEGER(KIND=1) | 8-bit two's-complement integer |
| INTEGER(KIND=2) | 16-bit two's-complement integer | 
| INTEGER(KIND=4) | 32-bit two's-complement integer |
| INTEGER(KIND=8) | 64-bit two's-complement integer |
| INTEGER(KIND=16) | 128-bit two's-complement integer |
| REAL(KIND=2) | 16-bit IEEE 754 binary16 (5e11m) |  
| REAL(KIND=3) | 16-bit upper half of 32-bit IEEE 754 binary32 (8e8m) |
| REAL(KIND=4) | 32-bit IEEE 754 binary32 (8e24m) |
| REAL(KIND=8) | 64-bit IEEE 754 binary64 (11e53m) |
| REAL(KIND=10) | 80-bit extended precision with explicit normalization bit (15e64m) |
| REAL(KIND=16) | 128-bit IEEE 754 binary128 (15e113m) |
| COMPLEX(KIND=2) | Two 16-bit IEEE 754 binary16 |
| COMPLEX(KIND=3) | Two 16-bit upper half of 32-bit IEEE 754 binary32 |
| COMPLEX(KIND=4) | Two 32-bit IEEE 754 binary32 |
| COMPLEX(KIND=8) | Two 64-bit IEEE 754 binary64 | 
| COMPLEX(KIND=10) | Two 80-bit extended precisions values |
| COMPLEX(KIND=16) | Two 128-bit IEEE 754 binary128 |
| LOGICAL(KIND=1) | 8-bit integer |
| LOGICAL(KIND=2) | 16-bit integer | 
| LOGICAL(KIND=4) | 32-bit integer |
| LOGICAL(KIND=8) | 64-bit integer | 

* No [double-double](https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format)
quad precision type is supported.
* No 128-bit logical support.

### Defaults kinds

f18 defaults to the following kinds for these types:

* `INTEGER` 4  
* `REAL` 4 
* `COMPLEX` 4   
* `DOUBLE PRECISION` 8
* `LOGICAL` 4

Modules compiled with different default-real and default-integer kinds
may be freely mixed. Module files encode the kind value for every entity.

#### Modifying the default kind with default-real-8.  

* `REAL` 8  
* `DOUBLE PRECISION` 8   
* `COMPLEX` 8

#### Modifying the default kind with default-integer-8:  

* `INTEGER` 8
* `LOGICAL` 8

## Representation of LOGICAL variables

The Fortran standard specifies that a logical has two values, `.TRUE.` and
`.FALSE.`, but does not specify their internal representation.

Flang specifies that logical literal constants with `KIND=[1|2|4|8]` share the
following characteristics:

* `.TRUE.` is `1_kind`.
* `.FALSE.` is `0_kind`.
* A true test is `<integer value> .NE. 0_kind`.

Programs should not use integer values in LOGICAL contexts or use LOGICAL values
to interface with other languages.

### Representations of LOGICAL variables in other compilers

#### GNU gfortran

* `.TRUE.` is `1_kind`.
* `.FALSE.` is `0_kind`.
* A true test is `<integer value> .NE. 0_kind`.

#### Intel ifort / NVIDA nvfortran / PGI pgf90

* `.TRUE.` is `-1_kind`.
* `.FALSE.` is `0_kind`.
* Any other values result in undefined behavior.  
* Values with a low-bit set are treated as `.TRUE.`.  
* Values with a low-bit clear are treated as `.FALSE.`.  

#### IBM XLF

* `.TRUE.` is `1_kind`.
* `.FALSE.` is `0_kind`.
* Values with a low-bit set are treated as `.TRUE.`.  
* Values with a low-bit clear are treated as `.FALSE.`.  
