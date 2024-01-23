<!--===- docs/Intrinsics.md 
  
   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  
-->

# A categorization of standard (2018) and extended Fortran intrinsic procedures

```{contents}
---
local:
---
```

This note attempts to group the intrinsic procedures of Fortran into categories
of functions or subroutines with similar interfaces as an aid to
comprehension beyond that which might be gained from the standard's
alphabetical list.

A brief status of intrinsic procedure support in f18 is also given at the end.

Few procedures are actually described here apart from their interfaces; see the
Fortran 2018 standard (section 16) for the complete story.

Intrinsic modules are not covered here.

## General rules

1. The value of any intrinsic function's `KIND` actual argument, if present,
   must be a scalar constant integer expression, of any kind, whose value
   resolves to some supported kind of the function's result type.
   If optional and absent, the kind of the function's result is
   either the default kind of that category or to the kind of an argument
   (e.g., as in `AINT`).
1. Procedures are summarized with a non-Fortran syntax for brevity.
   Wherever a function has a short definition, it appears after an
   equal sign as if it were a statement function.  Any functions referenced
   in these short summaries are intrinsic.
1. Unless stated otherwise, an actual argument may have any supported kind
   of a particular intrinsic type.  Sometimes a pattern variable
   can appear in a description (e.g., `REAL(k)`) when the kind of an
   actual argument's type must match the kind of another argument, or
   determines the kind type parameter of the function result.
1. When an intrinsic type name appears without a kind (e.g., `REAL`),
   it refers to the default kind of that type.  Sometimes the word
   `default` will appear for clarity.
1. The names of the dummy arguments actually matter because they can
   be used as keywords for actual arguments.
1. All standard intrinsic functions are pure, even when not elemental.
1. Assumed-rank arguments may not appear as actual arguments unless
   expressly permitted.
1. When an argument is described with a default value, e.g. `KIND=KIND(0)`,
   it is an optional argument.  Optional arguments without defaults,
   e.g. `DIM` on many transformationals, are wrapped in `[]` brackets
   as in the Fortran standard.  When an intrinsic has optional arguments
   with and without default values, the arguments with default values
   may appear within the brackets to preserve the order of arguments
   (e.g., `COUNT`).

## Elemental intrinsic functions

Pure elemental semantics apply to these functions, to wit: when one or more of
the actual arguments are arrays, the arguments must be conformable, and
the result is also an array.
Scalar arguments are expanded when the arguments are not all scalars.

### Elemental intrinsic functions that may have unrestricted specific procedures

When an elemental intrinsic function is documented here as having an
_unrestricted specific name_, that name may be passed as an actual
argument, used as the target of a procedure pointer, appear in
a generic interface, and be otherwise used as if it were an external
procedure.
An `INTRINSIC` statement or attribute may have to be applied to an
unrestricted specific name to enable such usage.

When a name is being used as a specific procedure for any purpose other
than that of a called function, the specific instance of the function
that accepts and returns values of the default kinds of the intrinsic
types is used.
A Fortran `INTERFACE` could be written to define each of
these unrestricted specific intrinsic function names.

Calls to dummy arguments and procedure pointers that correspond to these
specific names must pass only scalar actual argument values.

No other intrinsic function name can be passed as an actual argument,
used as a pointer target, appear in a generic interface, or be otherwise
used except as the name of a called function.
Some of these _restricted specific intrinsic functions_, e.g. `FLOAT`,
provide a means for invoking a corresponding generic (`REAL` in the case of `FLOAT`)
with forced argument and result kinds.
Others, viz. `CHAR`, `ICHAR`, `INT`, `REAL`, and the lexical comparisons like `LGE`,
have the same name as their generic functions, and it is not clear what purpose
is accomplished by the standard by defining them as specific functions.

### Trigonometric elemental intrinsic functions, generic and (mostly) specific
All of these functions can be used as unrestricted specific names.

```
ACOS(REAL(k) X) -> REAL(k)
ASIN(REAL(k) X) -> REAL(k)
ATAN(REAL(k) X) -> REAL(k)
ATAN(REAL(k) Y, REAL(k) X) -> REAL(k) = ATAN2(Y, X)
ATAN2(REAL(k) Y, REAL(k) X) -> REAL(k)
COS(REAL(k) X) -> REAL(k)
COSH(REAL(k) X) -> REAL(k)
SIN(REAL(k) X) -> REAL(k)
SINH(REAL(k) X) -> REAL(k)
TAN(REAL(k) X) -> REAL(k)
TANH(REAL(k) X) -> REAL(k)
```

These `COMPLEX` versions of some of those functions, and the
inverse hyperbolic functions, cannot be used as specific names.
```
ACOS(COMPLEX(k) X) -> COMPLEX(k)
ASIN(COMPLEX(k) X) -> COMPLEX(k)
ATAN(COMPLEX(k) X) -> COMPLEX(k)
ACOSH(REAL(k) X) -> REAL(k)
ACOSH(COMPLEX(k) X) -> COMPLEX(k)
ASINH(REAL(k) X) -> REAL(k)
ASINH(COMPLEX(k) X) -> COMPLEX(k)
ATANH(REAL(k) X) -> REAL(k)
ATANH(COMPLEX(k) X) -> COMPLEX(k)
COS(COMPLEX(k) X) -> COMPLEX(k)
COSH(COMPLEX(k) X) -> COMPLEX(k)
SIN(COMPLEX(k) X) -> COMPLEX(k)
SINH(COMPLEX(k) X) -> COMPLEX(k)
TAN(COMPLEX(k) X) -> COMPLEX(k)
TANH(COMPLEX(k) X) -> COMPLEX(k)
```

### Non-trigonometric elemental intrinsic functions, generic and specific
These functions *can* be used as unrestricted specific names.
```
ABS(REAL(k) A) -> REAL(k) = SIGN(A, 0.0)
AIMAG(COMPLEX(k) Z) -> REAL(k) = Z%IM
AINT(REAL(k) A, KIND=k) -> REAL(KIND)
ANINT(REAL(k) A, KIND=k) -> REAL(KIND)
CONJG(COMPLEX(k) Z) -> COMPLEX(k) = CMPLX(Z%RE, -Z%IM)
DIM(REAL(k) X, REAL(k) Y) -> REAL(k) = X-MIN(X,Y)
DPROD(default REAL X, default REAL Y) -> DOUBLE PRECISION = DBLE(X)*DBLE(Y)
EXP(REAL(k) X) -> REAL(k)
INDEX(CHARACTER(k) STRING, CHARACTER(k) SUBSTRING, LOGICAL(any) BACK=.FALSE., KIND=KIND(0)) -> INTEGER(KIND)
LEN(CHARACTER(k,n) STRING, KIND=KIND(0)) -> INTEGER(KIND) = n
LOG(REAL(k) X) -> REAL(k)
LOG10(REAL(k) X) -> REAL(k)
MOD(INTEGER(k) A, INTEGER(k) P) -> INTEGER(k) = A-P*INT(A/P)
NINT(REAL(k) A, KIND=KIND(0)) -> INTEGER(KIND)
SIGN(REAL(k) A, REAL(k) B) -> REAL(k)
SQRT(REAL(k) X) -> REAL(k) = X ** 0.5
```

These variants, however *cannot* be used as specific names without recourse to an alias
from the following section:
```
ABS(INTEGER(k) A) -> INTEGER(k) = SIGN(A, 0)
ABS(COMPLEX(k) A) -> REAL(k) = HYPOT(A%RE, A%IM)
DIM(INTEGER(k) X, INTEGER(k) Y) -> INTEGER(k) = X-MIN(X,Y)
EXP(COMPLEX(k) X) -> COMPLEX(k)
LOG(COMPLEX(k) X) -> COMPLEX(k)
MOD(REAL(k) A, REAL(k) P) -> REAL(k) = A-P*INT(A/P)
SIGN(INTEGER(k) A, INTEGER(k) B) -> INTEGER(k)
SQRT(COMPLEX(k) X) -> COMPLEX(k)
```

### Unrestricted specific aliases for some elemental intrinsic functions with distinct names

```
ALOG(REAL X) -> REAL = LOG(X)
ALOG10(REAL X) -> REAL = LOG10(X)
AMOD(REAL A, REAL P) -> REAL = MOD(A, P)
CABS(COMPLEX A) = ABS(A)
CCOS(COMPLEX X) = COS(X)
CEXP(COMPLEX A) -> COMPLEX = EXP(A)
CLOG(COMPLEX X) -> COMPLEX = LOG(X)
CSIN(COMPLEX X) -> COMPLEX = SIN(X)
CSQRT(COMPLEX X) -> COMPLEX = SQRT(X)
CTAN(COMPLEX X) -> COMPLEX = TAN(X)
DABS(DOUBLE PRECISION A) -> DOUBLE PRECISION = ABS(A)
DACOS(DOUBLE PRECISION X) -> DOUBLE PRECISION = ACOS(X)
DASIN(DOUBLE PRECISION X) -> DOUBLE PRECISION = ASIN(X)
DATAN(DOUBLE PRECISION X) -> DOUBLE PRECISION = ATAN(X)
DATAN2(DOUBLE PRECISION Y, DOUBLE PRECISION X) -> DOUBLE PRECISION = ATAN2(Y, X)
DCOS(DOUBLE PRECISION X) -> DOUBLE PRECISION = COS(X)
DCOSH(DOUBLE PRECISION X) -> DOUBLE PRECISION = COSH(X)
DDIM(DOUBLE PRECISION X, DOUBLE PRECISION Y) -> DOUBLE PRECISION = X-MIN(X,Y)
DEXP(DOUBLE PRECISION X) -> DOUBLE PRECISION = EXP(X)
DINT(DOUBLE PRECISION A) -> DOUBLE PRECISION = AINT(A)
DLOG(DOUBLE PRECISION X) -> DOUBLE PRECISION = LOG(X)
DLOG10(DOUBLE PRECISION X) -> DOUBLE PRECISION = LOG10(X)
DMOD(DOUBLE PRECISION A, DOUBLE PRECISION P) -> DOUBLE PRECISION = MOD(A, P)
DNINT(DOUBLE PRECISION A) -> DOUBLE PRECISION = ANINT(A)
DSIGN(DOUBLE PRECISION A, DOUBLE PRECISION B) -> DOUBLE PRECISION = SIGN(A, B)
DSIN(DOUBLE PRECISION X) -> DOUBLE PRECISION = SIN(X)
DSINH(DOUBLE PRECISION X) -> DOUBLE PRECISION = SINH(X)
DSQRT(DOUBLE PRECISION X) -> DOUBLE PRECISION = SQRT(X)
DTAN(DOUBLE PRECISION X) -> DOUBLE PRECISION = TAN(X)
DTANH(DOUBLE PRECISION X) -> DOUBLE PRECISION = TANH(X)
IABS(INTEGER A) -> INTEGER = ABS(A)
IDIM(INTEGER X, INTEGER Y) -> INTEGER = X-MIN(X,Y)
IDNINT(DOUBLE PRECISION A) -> INTEGER = NINT(A)
ISIGN(INTEGER A, INTEGER B) -> INTEGER = SIGN(A, B)
```

## Generic elemental intrinsic functions without specific names

(No procedures after this point can be passed as actual arguments, used as
pointer targets, or appear as specific procedures in generic interfaces.)

### Elemental conversions

```
ACHAR(INTEGER(k) I, KIND=KIND('')) -> CHARACTER(KIND,LEN=1)
CEILING(REAL() A, KIND=KIND(0)) -> INTEGER(KIND)
CHAR(INTEGER(any) I, KIND=KIND('')) -> CHARACTER(KIND,LEN=1)
CMPLX(COMPLEX(k) X, KIND=KIND(0.0D0)) -> COMPLEX(KIND)
CMPLX(INTEGER or REAL or BOZ X, INTEGER or REAL or BOZ Y=0, KIND=KIND((0,0))) -> COMPLEX(KIND)
DBLE(INTEGER or REAL or COMPLEX or BOZ A) = REAL(A, KIND=KIND(0.0D0))
EXPONENT(REAL(any) X) -> default INTEGER
FLOOR(REAL(any) A, KIND=KIND(0)) -> INTEGER(KIND)
IACHAR(CHARACTER(KIND=k,LEN=1) C, KIND=KIND(0)) -> INTEGER(KIND)
ICHAR(CHARACTER(KIND=k,LEN=1) C, KIND=KIND(0)) -> INTEGER(KIND)
INT(INTEGER or REAL or COMPLEX or BOZ A, KIND=KIND(0)) -> INTEGER(KIND)
LOGICAL(LOGICAL(any) L, KIND=KIND(.TRUE.)) -> LOGICAL(KIND)
REAL(INTEGER or REAL or COMPLEX or BOZ A, KIND=KIND(0.0)) -> REAL(KIND)
```

### Other generic elemental intrinsic functions without specific names
N.B. `BESSEL_JN(N1, N2, X)` and `BESSEL_YN(N1, N2, X)` are categorized
below with the _transformational_ intrinsic functions.

```
BESSEL_J0(REAL(k) X) -> REAL(k)
BESSEL_J1(REAL(k) X) -> REAL(k)
BESSEL_JN(INTEGER(n) N, REAL(k) X) -> REAL(k)
BESSEL_Y0(REAL(k) X) -> REAL(k)
BESSEL_Y1(REAL(k) X) -> REAL(k)
BESSEL_YN(INTEGER(n) N, REAL(k) X) -> REAL(k)
ERF(REAL(k) X) -> REAL(k)
ERFC(REAL(k) X) -> REAL(k)
ERFC_SCALED(REAL(k) X) -> REAL(k)
FRACTION(REAL(k) X) -> REAL(k)
GAMMA(REAL(k) X) -> REAL(k)
HYPOT(REAL(k) X, REAL(k) Y) -> REAL(k) = SQRT(X*X+Y*Y) without spurious overflow
IMAGE_STATUS(INTEGER(any) IMAGE [, scalar TEAM_TYPE TEAM ]) -> default INTEGER
IS_IOSTAT_END(INTEGER(any) I) -> default LOGICAL
IS_IOSTAT_EOR(INTEGER(any) I) -> default LOGICAL
LOG_GAMMA(REAL(k) X) -> REAL(k)
MAX(INTEGER(k) ...) -> INTEGER(k)
MAX(REAL(k) ...) -> REAL(k)
MAX(CHARACTER(KIND=k) ...) -> CHARACTER(KIND=k,LEN=MAX(LEN(...)))
MERGE(any type TSOURCE, same type FSOURCE, LOGICAL(any) MASK) -> type of FSOURCE
MIN(INTEGER(k) ...) -> INTEGER(k)
MIN(REAL(k) ...) -> REAL(k)
MIN(CHARACTER(KIND=k) ...) -> CHARACTER(KIND=k,LEN=MAX(LEN(...)))
MODULO(INTEGER(k) A, INTEGER(k) P) -> INTEGER(k); P*result >= 0
MODULO(REAL(k) A, REAL(k) P) -> REAL(k) = A - P*FLOOR(A/P)
NEAREST(REAL(k) X, REAL(any) S) -> REAL(k)
OUT_OF_RANGE(INTEGER(any) X, scalar INTEGER or REAL(k) MOLD) -> default LOGICAL
OUT_OF_RANGE(REAL(any) X, scalar REAL(k) MOLD) -> default LOGICAL
OUT_OF_RANGE(REAL(any) X, scalar INTEGER(any) MOLD, scalar LOGICAL(any) ROUND=.FALSE.) -> default LOGICAL
RRSPACING(REAL(k) X) -> REAL(k)
SCALE(REAL(k) X, INTEGER(any) I) -> REAL(k)
SET_EXPONENT(REAL(k) X, INTEGER(any) I) -> REAL(k)
SPACING(REAL(k) X) -> REAL(k)
```

### Restricted specific aliases for elemental conversions &/or extrema with default intrinsic types

```
AMAX0(INTEGER ...) = REAL(MAX(...))
AMAX1(REAL ...) = MAX(...)
AMIN0(INTEGER...) = REAL(MIN(...))
AMIN1(REAL ...) = MIN(...)
DMAX1(DOUBLE PRECISION ...) = MAX(...)
DMIN1(DOUBLE PRECISION ...) = MIN(...)
FLOAT(INTEGER I) = REAL(I)
IDINT(DOUBLE PRECISION A) = INT(A)
IFIX(REAL A) = INT(A)
MAX0(INTEGER ...) = MAX(...)
MAX1(REAL ...) = INT(MAX(...))
MIN0(INTEGER ...) = MIN(...)
MIN1(REAL ...) = INT(MIN(...))
SNGL(DOUBLE PRECISION A) = REAL(A)
```

### Generic elemental bit manipulation intrinsic functions
Many of these accept a typeless "BOZ" literal as an actual argument.
It is interpreted as having the kind of intrinsic `INTEGER` type
as another argument, as if the typeless were implicitly wrapped
in a call to `INT()`.
When multiple arguments can be either `INTEGER` values or typeless
constants, it is forbidden for *all* of them to be typeless
constants if the result of the function is `INTEGER`
(i.e., only `BGE`, `BGT`, `BLE`, and `BLT` can have multiple
typeless arguments).

```
BGE(INTEGER(n1) or BOZ I, INTEGER(n2) or BOZ J) -> default LOGICAL
BGT(INTEGER(n1) or BOZ I, INTEGER(n2) or BOZ J) -> default LOGICAL
BLE(INTEGER(n1) or BOZ I, INTEGER(n2) or BOZ J) -> default LOGICAL
BLT(INTEGER(n1) or BOZ I, INTEGER(n2) or BOZ J) -> default LOGICAL
BTEST(INTEGER(n1) I, INTEGER(n2) POS) -> default LOGICAL
DSHIFTL(INTEGER(k) I, INTEGER(k) or BOZ J, INTEGER(any) SHIFT) -> INTEGER(k)
DSHIFTL(BOZ I, INTEGER(k), INTEGER(any) SHIFT) -> INTEGER(k)
DSHIFTR(INTEGER(k) I, INTEGER(k) or BOZ J, INTEGER(any) SHIFT) -> INTEGER(k)
DSHIFTR(BOZ I, INTEGER(k), INTEGER(any) SHIFT) -> INTEGER(k)
IAND(INTEGER(k) I, INTEGER(k) or BOZ J) -> INTEGER(k)
IAND(BOZ I, INTEGER(k) J) -> INTEGER(k)
IBCLR(INTEGER(k) I, INTEGER(any) POS) -> INTEGER(k)
IBITS(INTEGER(k) I, INTEGER(n1) POS, INTEGER(n2) LEN) -> INTEGER(k)
IBSET(INTEGER(k) I, INTEGER(any) POS) -> INTEGER(k)
IEOR(INTEGER(k) I, INTEGER(k) or BOZ J) -> INTEGER(k)
IEOR(BOZ I, INTEGER(k) J) -> INTEGER(k)
IOR(INTEGER(k) I, INTEGER(k) or BOZ J) -> INTEGER(k)
IOR(BOZ I, INTEGER(k) J) -> INTEGER(k)
ISHFT(INTEGER(k) I, INTEGER(any) SHIFT) -> INTEGER(k)
ISHFTC(INTEGER(k) I, INTEGER(n1) SHIFT, INTEGER(n2) SIZE=BIT_SIZE(I)) -> INTEGER(k)
LEADZ(INTEGER(any) I) -> default INTEGER
MASKL(INTEGER(any) I, KIND=KIND(0)) -> INTEGER(KIND)
MASKR(INTEGER(any) I, KIND=KIND(0)) -> INTEGER(KIND)
MERGE_BITS(INTEGER(k) I, INTEGER(k) or BOZ J, INTEGER(k) or BOZ MASK) = IOR(IAND(I,MASK),IAND(J,NOT(MASK)))
MERGE_BITS(BOZ I, INTEGER(k) J, INTEGER(k) or BOZ MASK) = IOR(IAND(I,MASK),IAND(J,NOT(MASK)))
NOT(INTEGER(k) I) -> INTEGER(k)
POPCNT(INTEGER(any) I) -> default INTEGER
POPPAR(INTEGER(any) I) -> default INTEGER = IAND(POPCNT(I), Z'1')
SHIFTA(INTEGER(k) I, INTEGER(any) SHIFT) -> INTEGER(k)
SHIFTL(INTEGER(k) I, INTEGER(any) SHIFT) -> INTEGER(k)
SHIFTR(INTEGER(k) I, INTEGER(any) SHIFT) -> INTEGER(k)
TRAILZ(INTEGER(any) I) -> default INTEGER
```

### Character elemental intrinsic functions
See also `INDEX` and `LEN` above among the elemental intrinsic functions with
unrestricted specific names.
```
ADJUSTL(CHARACTER(k,LEN=n) STRING) -> CHARACTER(k,LEN=n)
ADJUSTR(CHARACTER(k,LEN=n) STRING) -> CHARACTER(k,LEN=n)
LEN_TRIM(CHARACTER(k,n) STRING, KIND=KIND(0)) -> INTEGER(KIND) = n
LGE(CHARACTER(k,n1) STRING_A, CHARACTER(k,n2) STRING_B) -> default LOGICAL
LGT(CHARACTER(k,n1) STRING_A, CHARACTER(k,n2) STRING_B) -> default LOGICAL
LLE(CHARACTER(k,n1) STRING_A, CHARACTER(k,n2) STRING_B) -> default LOGICAL
LLT(CHARACTER(k,n1) STRING_A, CHARACTER(k,n2) STRING_B) -> default LOGICAL
SCAN(CHARACTER(k,n) STRING, CHARACTER(k,m) SET, LOGICAL(any) BACK=.FALSE., KIND=KIND(0)) -> INTEGER(KIND)
VERIFY(CHARACTER(k,n) STRING, CHARACTER(k,m) SET, LOGICAL(any) BACK=.FALSE., KIND=KIND(0)) -> INTEGER(KIND)
```

`SCAN` returns the index of the first (or last, if `BACK=.TRUE.`) character in `STRING`
that is present in `SET`, or zero if none is.

`VERIFY` is essentially the opposite: it returns the index of the first (or last) character
in `STRING` that is *not* present in `SET`, or zero if all are.

## Transformational intrinsic functions

This category comprises a large collection of intrinsic functions that
are collected together because they somehow transform their arguments
in a way that prevents them from being elemental.
All of them are pure, however.

Some general rules apply to the transformational intrinsic functions:

1. `DIM` arguments are optional; if present, the actual argument must be
   a scalar integer of any kind.
1. When an optional `DIM` argument is absent, or an `ARRAY` or `MASK`
   argument is a vector, the result of the function is scalar; otherwise,
   the result is an array of the same shape as the `ARRAY` or `MASK`
   argument with the dimension `DIM` removed from the shape.
1. When a function takes an optional `MASK` argument, it must be conformable
  with its `ARRAY` argument if it is present, and the mask can be any kind
  of `LOGICAL`.  It can be scalar.
1. The type `numeric` here can be any kind of `INTEGER`, `REAL`, or `COMPLEX`.
1. The type `relational` here can be any kind of `INTEGER`, `REAL`, or `CHARACTER`.
1. The type `any` here denotes any intrinsic or derived type.
1. The notation `(..)` denotes an array of any rank (but not an assumed-rank array).

### Logical reduction transformational intrinsic functions
```
ALL(LOGICAL(k) MASK(..) [, DIM ]) -> LOGICAL(k)
ANY(LOGICAL(k) MASK(..) [, DIM ]) -> LOGICAL(k)
COUNT(LOGICAL(any) MASK(..) [, DIM, KIND=KIND(0) ]) -> INTEGER(KIND)
PARITY(LOGICAL(k) MASK(..) [, DIM ]) -> LOGICAL(k)
```

### Numeric reduction transformational intrinsic functions
```
IALL(INTEGER(k) ARRAY(..) [, DIM, MASK ]) -> INTEGER(k)
IANY(INTEGER(k) ARRAY(..) [, DIM, MASK ]) -> INTEGER(k)
IPARITY(INTEGER(k) ARRAY(..) [, DIM, MASK ]) -> INTEGER(k)
NORM2(REAL(k) X(..) [, DIM ]) -> REAL(k)
PRODUCT(numeric ARRAY(..) [, DIM, MASK ]) -> numeric
SUM(numeric ARRAY(..) [, DIM, MASK ]) -> numeric
```

`NORM2` generalizes `HYPOT` by computing `SQRT(SUM(X*X))` while avoiding spurious overflows.

### Extrema reduction transformational intrinsic functions
```
MAXVAL(relational(k) ARRAY(..) [, DIM, MASK ]) -> relational(k)
MINVAL(relational(k) ARRAY(..) [, DIM, MASK ]) -> relational(k)
```

### Locational transformational intrinsic functions
When the optional `DIM` argument is absent, the result is an `INTEGER(KIND)`
vector whose length is the rank of `ARRAY`.
When the optional `DIM` argument is present, the result is an `INTEGER(KIND)`
array of rank `RANK(ARRAY)-1` and shape equal to that of `ARRAY` with
the dimension `DIM` removed.

The optional `BACK` argument is a scalar LOGICAL value of any kind.
When present and `.TRUE.`, it causes the function to return the index
of the *last* occurence of the target or extreme value.

For `FINDLOC`, `ARRAY` may have any of the five intrinsic types, and `VALUE`
must a scalar value of a type for which `ARRAY==VALUE` or `ARRAY .EQV. VALUE`
is an acceptable expression.

```
FINDLOC(intrinsic ARRAY(..), scalar VALUE [, DIM, MASK, KIND=KIND(0), BACK=.FALSE. ])
MAXLOC(relational ARRAY(..) [, DIM, MASK, KIND=KIND(0), BACK=.FALSE. ])
MINLOC(relational ARRAY(..) [, DIM, MASK, KIND=KIND(0), BACK=.FALSE. ])
```

### Data rearrangement transformational intrinsic functions
The optional `DIM` argument to these functions must be a scalar integer of
any kind, and it takes a default value of 1 when absent.

```
CSHIFT(any ARRAY(..), INTEGER(any) SHIFT(..) [, DIM ]) -> same type/kind/shape as ARRAY
```
Either `SHIFT` is scalar or `RANK(SHIFT) == RANK(ARRAY) - 1` and `SHAPE(SHIFT)` is that of `SHAPE(ARRAY)` with element `DIM` removed.

```
EOSHIFT(any ARRAY(..), INTEGER(any) SHIFT(..) [, BOUNDARY, DIM ]) -> same type/kind/shape as ARRAY
```
* `SHIFT` is scalar or `RANK(SHIFT) == RANK(ARRAY) - 1` and `SHAPE(SHIFT)` is that of `SHAPE(ARRAY)` with element `DIM` removed.
* If `BOUNDARY` is present, it must have the same type and parameters as `ARRAY`.
* If `BOUNDARY` is absent, `ARRAY` must be of an intrinsic type, and the default `BOUNDARY` is the obvious `0`, `' '`, or `.FALSE.` value of `KIND(ARRAY)`.
* If `BOUNDARY` is present, either it is scalar, or `RANK(BOUNDARY) == RANK(ARRAY) - 1` and `SHAPE(BOUNDARY)` is that of `SHAPE(ARRAY)` with element `DIM`
  removed.

```
PACK(any ARRAY(..), LOGICAL(any) MASK(..)) -> vector of same type and kind as ARRAY
```
* `MASK` is conformable with `ARRAY` and may be scalar.
* The length of the result vector is `COUNT(MASK)` if `MASK` is an array, else `SIZE(ARRAY)` if `MASK` is `.TRUE.`, else zero.

```
PACK(any ARRAY(..), LOGICAL(any) MASK(..), any VECTOR(n)) -> vector of same type, kind, and size as VECTOR
```
* `MASK` is conformable with `ARRAY` and may be scalar.
* `VECTOR` has the same type and kind as `ARRAY`.
* `VECTOR` must not be smaller than result of `PACK` with no `VECTOR` argument.
* The leading elements of `VECTOR` are replaced with elements from `ARRAY` as
  if `PACK` had been invoked without `VECTOR`.

```
RESHAPE(any SOURCE(..), INTEGER(k) SHAPE(n) [, PAD(..), INTEGER(k2) ORDER(n) ]) -> SOURCE array with shape SHAPE
```
* If `ORDER` is present, it is a vector of the same size as `SHAPE`, and
  contains a permutation.
* The element(s) of `PAD` are used to fill out the result once `SOURCE`
  has been consumed.

```
SPREAD(any SOURCE, DIM, scalar INTEGER(any) NCOPIES) -> same type as SOURCE, rank=RANK(SOURCE)+1
TRANSFER(any SOURCE, any MOLD) -> scalar if MOLD is scalar, else vector; same type and kind as MOLD
TRANSFER(any SOURCE, any MOLD, scalar INTEGER(any) SIZE) -> vector(SIZE) of type and kind of MOLD
TRANSPOSE(any MATRIX(n,m)) -> matrix(m,n) of same type and kind as MATRIX
```

The shape of the result of `SPREAD` is the same as that of `SOURCE`, with `NCOPIES` inserted
at position `DIM`.

```
UNPACK(any VECTOR(n), LOGICAL(any) MASK(..), FIELD) -> type and kind of VECTOR, shape of MASK
```
`FIELD` has same type and kind as `VECTOR` and is conformable with `MASK`.

### Other transformational intrinsic functions
```
BESSEL_JN(INTEGER(n1) N1, INTEGER(n2) N2, REAL(k) X) -> REAL(k) vector (MAX(N2-N1+1,0))
BESSEL_YN(INTEGER(n1) N1, INTEGER(n2) N2, REAL(k) X) -> REAL(k) vector (MAX(N2-N1+1,0))
COMMAND_ARGUMENT_COUNT() -> scalar default INTEGER
DOT_PRODUCT(LOGICAL(k) VECTOR_A(n), LOGICAL(k) VECTOR_B(n)) -> LOGICAL(k) = ANY(VECTOR_A .AND. VECTOR_B)
DOT_PRODUCT(COMPLEX(any) VECTOR_A(n), numeric VECTOR_B(n)) = SUM(CONJG(VECTOR_A) * VECTOR_B)
DOT_PRODUCT(INTEGER(any) or REAL(any) VECTOR_A(n), numeric VECTOR_B(n)) = SUM(VECTOR_A * VECTOR_B)
MATMUL(numeric ARRAY_A(j), numeric ARRAY_B(j,k)) -> numeric vector(k)
MATMUL(numeric ARRAY_A(j,k), numeric ARRAY_B(k)) -> numeric vector(j)
MATMUL(numeric ARRAY_A(j,k), numeric ARRAY_B(k,m)) -> numeric matrix(j,m)
MATMUL(LOGICAL(n1) ARRAY_A(j), LOGICAL(n2) ARRAY_B(j,k)) -> LOGICAL vector(k)
MATMUL(LOGICAL(n1) ARRAY_A(j,k), LOGICAL(n2) ARRAY_B(k)) -> LOGICAL vector(j)
MATMUL(LOGICAL(n1) ARRAY_A(j,k), LOGICAL(n2) ARRAY_B(k,m)) -> LOGICAL matrix(j,m)
NULL([POINTER/ALLOCATABLE MOLD]) -> POINTER
REDUCE(any ARRAY(..), function OPERATION [, DIM, LOGICAL(any) MASK(..), IDENTITY, LOGICAL ORDERED=.FALSE. ])
REPEAT(CHARACTER(k,n) STRING, INTEGER(any) NCOPIES) -> CHARACTER(k,n*NCOPIES)
SELECTED_CHAR_KIND('DEFAULT' or 'ASCII' or 'ISO_10646' or ...) -> scalar default INTEGER
SELECTED_INT_KIND(scalar INTEGER(any) R) -> scalar default INTEGER
SELECTED_REAL_KIND([scalar INTEGER(any) P, scalar INTEGER(any) R, scalar INTEGER(any) RADIX]) -> scalar default INTEGER
SHAPE(SOURCE, KIND=KIND(0)) -> INTEGER(KIND)(RANK(SOURCE))
TRIM(CHARACTER(k,n) STRING) -> CHARACTER(k)
```

The type and kind of the result of a numeric `MATMUL` is the same as would result from
a multiplication of an element of ARRAY_A and an element of ARRAY_B.

The kind of the `LOGICAL` result of a `LOGICAL` `MATMUL` is the same as would result
from an intrinsic `.AND.` operation between an element of `ARRAY_A` and an element
of `ARRAY_B`.

Note that `DOT_PRODUCT` with a `COMPLEX` first argument operates on its complex conjugate,
but that `MATMUL` with a `COMPLEX` argument does not.

The `MOLD` argument to `NULL` may be omitted only in a context where the type of the pointer is known,
such as an initializer or pointer assignment statement.

At least one argument must be present in a call to `SELECTED_REAL_KIND`.

An assumed-rank array may be passed to `SHAPE`, and if it is associated with an assumed-size array,
the last element of the result will be -1.

### Coarray transformational intrinsic functions
```
FAILED_IMAGES([scalar TEAM_TYPE TEAM, KIND=KIND(0)]) -> INTEGER(KIND) vector
GET_TEAM([scalar INTEGER(?) LEVEL]) -> scalar TEAM_TYPE
IMAGE_INDEX(COARRAY, INTEGER(any) SUB(n) [, scalar TEAM_TYPE TEAM ]) -> scalar default INTEGER
IMAGE_INDEX(COARRAY, INTEGER(any) SUB(n), scalar INTEGER(any) TEAM_NUMBER) -> scalar default INTEGER
NUM_IMAGES([scalar TEAM_TYPE TEAM]) -> scalar default INTEGER
NUM_IMAGES(scalar INTEGER(any) TEAM_NUMBER) -> scalar default INTEGER
STOPPED_IMAGES([scalar TEAM_TYPE TEAM, KIND=KIND(0)]) -> INTEGER(KIND) vector
TEAM_NUMBER([scalar TEAM_TYPE TEAM]) -> scalar default INTEGER
THIS_IMAGE([COARRAY, DIM, scalar TEAM_TYPE TEAM]) -> default INTEGER
```
The result of `THIS_IMAGE` is a scalar if `DIM` is present or if `COARRAY` is absent,
and a vector whose length is the corank of `COARRAY` otherwise.

## Inquiry intrinsic functions
These are neither elemental nor transformational; all are pure.

### Type inquiry intrinsic functions
All of these functions return constants.
The value of the argument is not used, and may well be undefined.
```
BIT_SIZE(INTEGER(k) I(..)) -> INTEGER(k)
DIGITS(INTEGER or REAL X(..)) -> scalar default INTEGER
EPSILON(REAL(k) X(..)) -> scalar REAL(k)
HUGE(INTEGER(k) X(..)) -> scalar INTEGER(k)
HUGE(REAL(k) X(..)) -> scalar of REAL(k)
KIND(intrinsic X(..)) -> scalar default INTEGER
MAXEXPONENT(REAL(k) X(..)) -> scalar default INTEGER
MINEXPONENT(REAL(k) X(..)) -> scalar default INTEGER
NEW_LINE(CHARACTER(k,n) A(..)) -> scalar CHARACTER(k,1) = CHAR(10)
PRECISION(REAL(k) or COMPLEX(k) X(..)) -> scalar default INTEGER
RADIX(INTEGER(k) or REAL(k) X(..)) -> scalar default INTEGER, always 2
RANGE(INTEGER(k) or REAL(k) or COMPLEX(k) X(..)) -> scalar default INTEGER
TINY(REAL(k) X(..)) -> scalar REAL(k)
```

### Bound and size inquiry intrinsic functions
The results are scalar when `DIM` is present, and a vector of length=(co)rank(`(CO)ARRAY`)
when `DIM` is absent.
```
LBOUND(any ARRAY(..) [, DIM, KIND=KIND(0) ]) -> INTEGER(KIND)
LCOBOUND(any COARRAY [, DIM, KIND=KIND(0) ]) -> INTEGER(KIND)
SIZE(any ARRAY(..) [, DIM, KIND=KIND(0) ]) -> INTEGER(KIND)
UBOUND(any ARRAY(..) [, DIM, KIND=KIND(0) ]) -> INTEGER(KIND)
UCOBOUND(any COARRAY [, DIM, KIND=KIND(0) ]) -> INTEGER(KIND)
```

Assumed-rank arrays may be used with `LBOUND`, `SIZE`, and `UBOUND`.

### Object characteristic inquiry intrinsic functions
```
ALLOCATED(any type ALLOCATABLE ARRAY) -> scalar default LOGICAL
ALLOCATED(any type ALLOCATABLE SCALAR) -> scalar default LOGICAL
ASSOCIATED(any type POINTER POINTER [, same type TARGET]) -> scalar default LOGICAL
COSHAPE(COARRAY, KIND=KIND(0)) -> INTEGER(KIND) vector of length corank(COARRAY)
EXTENDS_TYPE_OF(A, MOLD) -> default LOGICAL
IS_CONTIGUOUS(any data ARRAY(..)) -> scalar default LOGICAL
PRESENT(OPTIONAL A) -> scalar default LOGICAL
RANK(any data A) -> scalar default INTEGER = 0 if A is scalar, SIZE(SHAPE(A)) if A is an array, rank if assumed-rank
SAME_TYPE_AS(A, B) -> scalar default LOGICAL
STORAGE_SIZE(any data A, KIND=KIND(0)) -> INTEGER(KIND)
```
The arguments to `EXTENDS_TYPE_OF` must be of extensible derived types or be unlimited polymorphic.

An assumed-rank array may be used with `IS_CONTIGUOUS` and `RANK`.

## Intrinsic subroutines

(*TODO*: complete these descriptions)

### One elemental intrinsic subroutine
```
INTERFACE
  SUBROUTINE MVBITS(FROM, FROMPOS, LEN, TO, TOPOS)
    INTEGER(k1) :: FROM, TO
    INTENT(IN) :: FROM
    INTENT(INOUT) :: TO
    INTEGER(k2), INTENT(IN) :: FROMPOS
    INTEGER(k3), INTENT(IN) :: LEN
    INTEGER(k4), INTENT(IN) :: TOPOS
  END SUBROUTINE
END INTERFACE
```

### Non-elemental intrinsic subroutines
```
CALL CPU_TIME(REAL INTENT(OUT) TIME)
```
The kind of `TIME` is not specified in the standard.

```
CALL DATE_AND_TIME([DATE, TIME, ZONE, VALUES])
```
* All arguments are `OPTIONAL` and `INTENT(OUT)`.
* `DATE`, `TIME`, and `ZONE` are scalar default `CHARACTER`.
* `VALUES` is a vector of at least 8 elements of `INTEGER(KIND >= 2)`.
```
CALL EVENT_QUERY(EVENT, COUNT [, STAT])
CALL EXECUTE_COMMAND_LINE(COMMAND [, WAIT, EXITSTAT, CMDSTAT, CMDMSG ])
CALL GET_COMMAND([COMMAND, LENGTH, STATUS, ERRMSG ])
CALL GET_COMMAND_ARGUMENT(NUMBER [, VALUE, LENGTH, STATUS, ERRMSG ])
CALL GET_ENVIRONMENT_VARIABLE(NAME [, VALUE, LENGTH, STATUS, TRIM_NAME, ERRMSG ])
CALL MOVE_ALLOC(ALLOCATABLE INTENT(INOUT) FROM, ALLOCATABLE INTENT(OUT) TO [, STAT, ERRMSG ])
CALL RANDOM_INIT(LOGICAL(k1) INTENT(IN) REPEATABLE, LOGICAL(k2) INTENT(IN) IMAGE_DISTINCT)
CALL RANDOM_NUMBER(REAL(k) INTENT(OUT) HARVEST(..))
CALL RANDOM_SEED([SIZE, PUT, GET])
CALL SYSTEM_CLOCK([COUNT, COUNT_RATE, COUNT_MAX])
```

### Atomic intrinsic subroutines
```
CALL ATOMIC_ADD(ATOM, VALUE [, STAT=])
CALL ATOMIC_AND(ATOM, VALUE [, STAT=])
CALL ATOMIC_CAS(ATOM, OLD, COMPARE, NEW [, STAT=])
CALL ATOMIC_DEFINE(ATOM, VALUE [, STAT=])
CALL ATOMIC_FETCH_ADD(ATOM, VALUE, OLD [, STAT=])
CALL ATOMIC_FETCH_AND(ATOM, VALUE, OLD [, STAT=])
CALL ATOMIC_FETCH_OR(ATOM, VALUE, OLD [, STAT=])
CALL ATOMIC_FETCH_XOR(ATOM, VALUE, OLD [, STAT=])
CALL ATOMIC_OR(ATOM, VALUE [, STAT=])
CALL ATOMIC_REF(VALUE, ATOM [, STAT=])
CALL ATOMIC_XOR(ATOM, VALUE [, STAT=])
```

### Collective intrinsic subroutines
```
CALL CO_BROADCAST
CALL CO_MAX
CALL CO_MIN
CALL CO_REDUCE
CALL CO_SUM
```

## Non-standard intrinsics
### PGI
```
AND, OR, XOR
LSHIFT, RSHIFT, SHIFT
ZEXT, IZEXT
COSD, SIND, TAND, ACOSD, ASIND, ATAND, ATAN2D
COMPL
DCMPLX
EQV, NEQV
INT8
JINT, JNINT, KNINT
LOC
```

### Intel
```
DCMPLX(X,Y), QCMPLX(X,Y)
DREAL(DOUBLE COMPLEX A) -> DOUBLE PRECISION
DFLOAT, DREAL
QEXT, QFLOAT, QREAL
DNUM, INUM, JNUM, KNUM, QNUM, RNUM - scan value from string
ZEXT
RAN, RANF
ILEN(I) = BIT_SIZE(I)
SIZEOF
MCLOCK, SECNDS
COTAN(X) = 1.0/TAN(X)
COSD, SIND, TAND, ACOSD, ASIND, ATAND, ATAN2D, COTAND - degrees
AND, OR, XOR
LSHIFT, RSHIFT
IBCHNG, ISHA, ISHC, ISHL, IXOR
IARG, IARGC, NARGS, NUMARG
BADDRESS, IADDR
CACHESIZE, EOF, FP_CLASS, INT_PTR_KIND, ISNAN, LOC
MALLOC
```

### Library subroutine 
```
CALL FDATE(TIME)
CALL GETLOG(USRNAME)
```

## Intrinsic Procedure Name Resolution

When the name of a procedure in a program is the same as the one of an intrinsic
procedure, and nothing other than its usage allows to decide whether the procedure
is the intrinsic or not (i.e, it does not appear in an INTRINSIC or EXTERNAL attribute
statement, is not an use/host associated procedure...), Fortran 2018 standard
section 19.5.1.4 point 6 rules that the procedure is established to be intrinsic if it is
invoked as an intrinsic procedure.

In case the invocation would be an error if the procedure were the intrinsic
(e.g. wrong argument number or type), the broad wording of the standard
leaves two choices to the compiler: emit an error about the intrinsic invocation,
or consider this is an external procedure and emit no error.

f18 will always consider this case to be the intrinsic and emit errors, unless the procedure
is used as a function (resp. subroutine) and the intrinsic is a subroutine (resp. function).
The table below gives some examples of decisions made by Fortran compilers in such case.

| What is ACOS ?     | Bad intrinsic call       | External with warning |  External no warning | Other error |
| --- | --- | --- | --- | --- |
| `print*, ACOS()`     | gfortran, nag, xlf, f18  |  ifort                |  nvfortran           | |
| `print*, ACOS(I)`    | gfortran, nag, xlf, f18  |  ifort                |  nvfortran           | |
| `print*, ACOS(X=I)`  | gfortran, nag, xlf, f18  |  ifort                |                      | nvfortran (keyword on implicit extrenal )|
| `print*, ACOS(X, X)` | gfortran, nag, xlf, f18  |  ifort                |  nvfortran           | |
| `CALL ACOS(X)`       |                          |                       |  gfortran, nag, xlf, nvfortran, ifort, f18  | |


The rationale for f18 behavior is that when referring to a procedure with an
argument number or type that does not match the intrinsic specification, it seems safer to block
the rather likely case where the user is using the intrinsic the wrong way.
In case the user wanted to refer to an external function, he can add an explicit EXTERNAL
statement with no other consequences on the program.
However, it seems rather unlikely that a user would confuse an intrinsic subroutine for a
function and vice versa. Given no compiler is issuing an error here, changing the behavior might
affect existing programs that omit the EXTERNAL attribute in such case.

Also note that in general, the standard gives the compiler the right to consider
any procedure that is not explicitly external as a non standard intrinsic (section 4.2 point 4).
So it is highly advised for the programmer to use EXTERNAL statements to prevent any ambiguity.

## Intrinsic Procedure Support in f18
This section gives an overview of the support inside f18 libraries for the
intrinsic procedures listed above.
It may be outdated, refer to f18 code base for the actual support status.

### Semantic Analysis
F18 semantic expression analysis phase detects intrinsic procedure references,
validates the argument types and deduces the return types.
This phase currently supports all the intrinsic procedures listed above but the ones in the table below.

| Intrinsic Category | Intrinsic Procedures Lacking Support |
| --- | --- |
| Coarray intrinsic functions | IMAGE_INDEX, COSHAPE |
| Object characteristic inquiry functions | ALLOCATED, ASSOCIATED, EXTENDS_TYPE_OF, IS_CONTIGUOUS, PRESENT, RANK, SAME_TYPE, STORAGE_SIZE |
| Type inquiry intrinsic functions | BIT_SIZE, DIGITS, EPSILON, HUGE, KIND, MAXEXPONENT, MINEXPONENT, NEW_LINE, PRECISION, RADIX, RANGE, TINY|
| Non-standard intrinsic functions | AND, OR, XOR, SHIFT, ZEXT, IZEXT, COSD, SIND, TAND, ACOSD, ASIND, ATAND, ATAN2D, COMPL, EQV, NEQV, INT8, JINT, JNINT, KNINT, QCMPLX, DREAL, DFLOAT, QEXT, QFLOAT, QREAL, DNUM, NUM, JNUM, KNUM, QNUM, RNUM, RAN, RANF, ILEN, SIZEOF, MCLOCK, SECNDS, COTAN, IBCHNG, ISHA, ISHC, ISHL, IXOR, IARG, IARGC, NARGS, GETPID, NUMARG, BADDRESS, IADDR, CACHESIZE, EOF, FP_CLASS, INT_PTR_KIND, ISNAN, MALLOC |
| Intrinsic subroutines |MVBITS (elemental), CPU_TIME, DATE_AND_TIME, EVENT_QUERY, EXECUTE_COMMAND_LINE, GET_COMMAND, GET_COMMAND_ARGUMENT, GET_ENVIRONMENT_VARIABLE, MOVE_ALLOC, RANDOM_INIT, RANDOM_NUMBER, RANDOM_SEED, SYSTEM_CLOCK |
| Atomic intrinsic subroutines | ATOMIC_ADD |
| Collective intrinsic subroutines | CO_REDUCE |
| Library subroutines | FDATE, GETLOG |


### Intrinsic Function Folding
Fortran Constant Expressions can contain references to a certain number of
intrinsic functions (see Fortran 2018 standard section 10.1.12 for more details).
Constant Expressions may be used to define kind arguments. Therefore, the semantic
expression analysis phase must be able to fold references to intrinsic functions
listed in section 10.1.12.

F18 intrinsic function folding is either performed by implementations directly
operating on f18 scalar types or by using host runtime functions and
host hardware types. F18 supports folding elemental intrinsic functions over
arrays when an implementation is provided for the scalars (regardless of whether
it is using host hardware types or not).
The status of intrinsic function folding support is given in the sub-sections below.

#### Intrinsic Functions with Host Independent Folding Support
Implementations using f18 scalar types enables folding intrinsic functions
on any host and with any possible type kind supported by f18. The intrinsic functions
listed below are folded using host independent implementations.

| Return Type | Intrinsic Functions with Host Independent Folding Support|
| --- | --- |
| INTEGER| ABS(INTEGER(k)), DIM(INTEGER(k), INTEGER(k)), DSHIFTL, DSHIFTR, IAND, IBCLR, IBSET, IEOR, INT, IOR, ISHFT, KIND, LEN, LEADZ, MASKL, MASKR, MERGE_BITS, POPCNT, POPPAR, SHIFTA, SHIFTL, SHIFTR, TRAILZ |
| REAL | ABS(REAL(k)), ABS(COMPLEX(k)), AIMAG, AINT, DPROD, REAL |
| COMPLEX | CMPLX, CONJG |
| LOGICAL | BGE, BGT, BLE, BLT |

#### Intrinsic Functions with Host Dependent Folding Support
Implementations using the host runtime may not be available for all supported
f18 types depending on the host hardware types and the libraries available on the host.
The actual support on a host depends on what the host hardware types are.
The list below gives the functions that are folded using host runtime and the related C/C++ types.
F18 automatically detects if these types match an f18 scalar type. If so,
folding of the intrinsic functions will be possible for the related f18 scalar type,
otherwise an error message will be produced by f18 when attempting to fold related intrinsic functions.

| C/C++ Host Type | Intrinsic Functions with Host Standard C++ Library Based Folding Support |
| --- | --- |
| float, double and long double | ACOS, ACOSH, ASINH, ATAN, ATAN2, ATANH, COS, COSH, ERF, ERFC, EXP, GAMMA, HYPOT, LOG, LOG10, LOG_GAMMA, MOD, SIN, SQRT, SINH, SQRT, TAN, TANH |
| std::complex for float, double and long double| ACOS, ACOSH, ASIN, ASINH, ATAN, ATANH, COS, COSH, EXP, LOG, SIN, SINH, SQRT, TAN, TANH |

On top of the default usage of C++ standard library functions for folding described
in the table above, it is possible to compile f18 evaluate library with
[libpgmath](https://github.com/flang-compiler/flang/tree/master/runtime/libpgmath)
so that it can be used for folding. To do so, one must have a compiled version
of the libpgmath library available on the host and add
`-DLIBPGMATH_DIR=<path to the compiled shared libpgmath library>` to the f18 cmake command.

Libpgmath comes with real and complex functions that replace C++ standard library
float and double functions to fold all the intrinsic functions listed in the table above.
It has no long double versions. If the host long double matches an f18 scalar type,
C++ standard library functions will still be used for folding expressions with this scalar type.
Libpgmath adds the possibility to fold the following functions for f18 real scalar
types related to host float and double types.

| C/C++ Host Type | Additional Intrinsic Function Folding Support with Libpgmath (Optional) |
| --- | --- |
|float and double| BESSEL_J0, BESSEL_J1, BESSEL_JN (elemental only), BESSEL_Y0, BESSEL_Y1, BESSEL_Yn (elemental only), ERFC_SCALED |

Libpgmath comes in three variants (precise, relaxed and fast). So far, only the
precise version is used for intrinsic function folding in f18. It guarantees the greatest numerical precision.

### Intrinsic Functions with Missing Folding Support
The following intrinsic functions are allowed in constant expressions but f18
is not yet able to fold them. Note that there might be constraints on the arguments
so that these intrinsics can be used in constant expressions (see section 10.1.12 of Fortran 2018 standard).

ALL, ACHAR, ADJUSTL, ADJUSTR, ANINT, ANY, BESSEL_JN (transformational only),
BESSEL_YN (transformational only), BTEST, CEILING, CHAR, COUNT, CSHIFT, DOT_PRODUCT,
DIM (REAL only), DOT_PRODUCT, EOSHIFT, FINDLOC, FLOOR, FRACTION, HUGE, IACHAR, IALL,
IANY, IPARITY, IBITS, ICHAR, IMAGE_STATUS, INDEX, ISHFTC, IS_IOSTAT_END,
IS_IOSTAT_EOR, LBOUND, LEN_TRIM, LGE, LGT, LLE, LLT, LOGICAL, MATMUL, MAX, MAXLOC,
MAXVAL, MERGE, MIN, MINLOC, MINVAL, MOD (INTEGER only), MODULO, NEAREST, NINT,
NORM2, NOT, OUT_OF_RANGE, PACK, PARITY, PRODUCT, REPEAT, REDUCE, RESHAPE,
RRSPACING, SCAN, SCALE, SELECTED_CHAR_KIND, SELECTED_INT_KIND, SELECTED_REAL_KIND,
SET_EXPONENT, SHAPE, SIGN, SIZE, SPACING, SPREAD, SUM, TINY, TRANSFER, TRANSPOSE,
TRIM, UBOUND, UNPACK, VERIFY.

Coarray, non standard, IEEE and ISO_C_BINDINGS intrinsic functions that can be
used in constant expressions have currently no folding support at all.

### Standard Intrinsics: EXECUTE_COMMAND_LINE

#### Usage and Info

- **Standard:** Fortran 2008 and later, specified in subclause 16.9.73
- **Class:** Subroutine
- **Syntax:** `CALL EXECUTE_COMMAND_LINE(COMMAND [, WAIT, EXITSTAT, CMDSTAT, CMDMSG ])`
- **Arguments:**

| Argument   | Description                                                           |
|------------|-----------------------------------------------------------------------|
| `COMMAND`  | Shall be a default CHARACTER scalar.                                  |
| `WAIT`     | (Optional) Shall be a default LOGICAL scalar.                         |
| `EXITSTAT` | (Optional) Shall be an INTEGER with kind greater than or equal to 4.  |
| `CMDSTAT`  | (Optional) Shall be an INTEGER with kind greater than or equal to 2.  |
| `CMDMSG`   | (Optional) Shall be a CHARACTER scalar of the default kind.           |

#### Implementation Specifics

##### `COMMAND`:

- Must be preset.

##### `WAIT`:

- If set to `false`, the command is executed asynchronously.
- If not preset or set to `true`, it is executed synchronously.
- Synchronous execution is achieved by passing the command into `std::system` on all systems.
- Asynchronous execution is achieved by calling `fork()` on POSIX-compatible systems or `CreateProcess()` on Windows.

##### `EXITSTAT`:

- Synchronous execution:
  - Inferred by the return value of `std::system(cmd)`.
    - On POSIX-compatible systems: return value is first passed into `WEXITSTATUS(status)`, then assigned to `EXITSTAT`.
    - On Windows, the value is directly assigned as the return value of `std::system()`.
- Asynchronous execution:
  - Value is not modified.

##### `CMDSTAT`:

- Synchronous execution:
  - -2: No error condition occurs, but `WAIT` is present with the value `false`, and the processor does not support asynchronous execution.
  - -1: The processor does not support command line execution.
  - \+ (positive value): An error condition occurs.
    - 1: Fork Error (occurs only on POSIX-compatible systems).
    - 2: Execution Error (command exits with status -1).
    - 3: Invalid Command Error (determined by the exit code depending on the system).
      - On Windows: exit code is 1.
      - On POSIX-compatible systems: exit code is 127 or 126.
    - 4: Signal error (either stopped or killed by signal, occurs only on POSIX-compatible systems).
  - 0: Otherwise.
- Asynchronous execution:
  - 0 will always be assigned.

##### `CMDMSG`:

- Synchronous execution:
  - If an error condition occurs, it is assigned an explanatory message; otherwise, it remains unchanged.
  - If a condition occurs that would assign a nonzero value to `CMDSTAT` but the `CMDSTAT` variable is not present, error termination is initiated (applies to both POSIX-compatible systems and Windows).
- Asynchronous execution:
  - The value is unchanged.
  - If a condition occurs that would assign a nonzero value to `CMDSTAT` but the `CMDSTAT` variable is not present, error termination is initiated.
    - On POSIX-compatible systems, the child process (async process) will be terminated with no effect on the parent process (continues).
    - On Windows, error termination is not initiated.

