<!--===- docs/IntrinsicTypes.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# Implementation of `Intrinsic` types in f18

```eval_rst
.. contents::
   :local:
```

Intrinsic types are integer, real, complex, character, and logical.
All intrinsic types have a kind type parameter called KIND,
which determines the representation method for the specified type.
The intrinsic type character also has a length type parameter called LEN,
which determines the length of the character string.

The implementation of `CHARACTER` type in f18 is described
in [Character.md](Character.md).

## Supported TYPES and KINDS

Here are the type and kind combinations supported in f18:

INTEGER(KIND=1) 8-bit two's-complement integer  
INTEGER(KIND=2) 16-bit two's-complement integer  
INTEGER(KIND=4) 32-bit two's-complement integer  
INTEGER(KIND=8) 64-bit two's-complement integer  
INTEGER(KIND=16) 128-bit two's-complement integer  

REAL(KIND=2) 16-bit IEEE 754 binary16 (5e11m)  
REAL(KIND=3) 16-bit upper half of 32-bit IEEE 754 binary32 (8e8m)  
REAL(KIND=4) 32-bit IEEE 754 binary32 (8e24m)  
REAL(KIND=8) 64-bit IEEE 754 binary64 (11e53m)  
REAL(KIND=10) 80-bit extended precision with explicit normalization bit (15e64m)  
REAL(KIND=16) 128-bit IEEE 754 binary128 (15e113m)  

COMPLEX(KIND=2) Two 16-bit IEEE 754 binary16  
COMPLEX(KIND=3) Two 16-bit upper half of 32-bit IEEE 754 binary32  
COMPLEX(KIND=4) Two 32-bit IEEE 754 binary32  
COMPLEX(KIND=8) Two 64-bit IEEE 754 binary64  
COMPLEX(KIND=10) Two 80-bit extended precisions values  
COMPLEX(KIND=16) Two 128-bit IEEE 754 binary128  

No
[double-double
](https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format)
quad precision type is supported.

LOGICAL(KIND=1) 8-bit integer  
LOGICAL(KIND=2) 16-bit integer  
LOGICAL(KIND=4) 32-bit integer  
LOGICAL(KIND=8) 64-bit integer  

No 128-bit logical support.

### Defaults kinds

INTEGER 4  
REAL 4  
COMPLEX 4  
DOUBLE PRECISION 8  
LOGICAL 4  

#### Modifying the default kind with default-real-8.  
REAL 8  
DOUBLE PRECISION  8  
COMPLEX 8  

#### Modifying the default kind with default-integer-8:  
INTEGER 8

There is no option to modify the default logical kind.

Modules compiled with different default-real and default-integer kinds
may be freely mixed.
Module files encode the kind value for every entity.

## Representation of LOGICAL variables

The default logical is `LOGICAL(KIND=4)`.

Logical literal constants with kind 1, 2, 4, and 8
share the following characteristics:   
.TRUE. is represented as 1_kind  
.FALSE. is represented as 0_kind  

Tests for true is *integer value is not zero*.

The implementation matches gfortran.

Programs should not use integer values in LOGICAL contexts or
use LOGICAL values to interface with other languages.

### Representations of LOGICAL variables in other compilers

##### Intel ifort / NVIDA nvfortran / PGI pgf90
.TRUE. is represented as -1_kind  
.FALSE. is represented as 0_kind  
Any other values result in undefined behavior.  

Values with a low-bit set are treated as .TRUE..  
Values with a low-bit clear are treated as .FALSE..  

##### IBM XLF
.TRUE. is represented as 1_kind  
.FALSE. is represented as 0_kind  

Values with a low-bit set are treated as .TRUE..  
Values with a low-bit clear are treated as .FALSE..  
