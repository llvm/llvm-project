## Implementation of `CHARACTER` types in f18

### Kinds and Character Sets

The f18 compiler and runtime support three kinds of the intrinsic
`CHARACTER` type of Fortran 2018.
The default (`CHARACTER(KIND=1)`) holds 8-bit character codes;
`CHARACTER(KIND=2)` holds 16-bit character codes;
and `CHARACTER(KIND=4)` holds 32-bit character codes.

We assume that code values 0 through 127 correspond to
the 7-bit ASCII character set (ISO-646) in every kind of `CHARACTER`.
This is a valid assumption for Unicode (UCS == ISO/IEC-10646),
ISO-8859, and many legacy character sets and interchange formats.

`CHARACTER` data in memory and unformatted files are not in an
interchange representation (like UTF-8, Shift-JIS, EUC-JP, or a JIS X).
Each character's code in memory occupies a 1-, 2-, or 4- byte
word and substrings can be indexed with simple arithmetic.
In formatted I/O, however, `CHARACTER` data may be assumed to use
the UTF-8 variable-length encoding when it is selected with
`OPEN(ENCODING='UTF-8')`.

`CHARACTER(KIND=1)` literal constants in Fortran source files,
Hollerith constants, and formatted I/O with `ENCODING='DEFAULT'`
are not translated.

For the purposes of non-default-kind `CHARACTER` constants in Fortran
source files, formatted I/O with `ENCODING='UTF-8'` or non-default-kind
`CHARACTER` value, and conversions between kinds of `CHARACTER`,
by default:
* `CHARACTER(KIND=1)` is assumed to be ISO-8859-1 (Latin-1),
* `CHARACTER(KIND=2)` is assumed to be UCS-2 (16-bit Unicode), and
* `CHARACTER(KIND=4)` is assumed to be UCS-4 (full Unicode in a 32-bit word).

In particular, conversions between kinds are assumed to be
simple zero-extensions or truncation, not table look-ups.

We might want to support one or more environment variables to change these
assumptions, especially for `KIND=1` users of ISO-8859 character sets
besides Latin-1.

### Lengths

Allocatable `CHARACTER` objects in Fortran may defer the specification
of their lengths until the time of their allocation or whole (non-substring)
assignment.
Non-allocatable objects (and non-deferred-length allocatables) have
lengths that are fixed or assumed from an actual argument, or,
in the case of assumed-length `CHARACTER` functions, their local
declaration in the calling scope.

The elements of `CHARACTER` arrays have the same length.

Assignments to targets that are not deferred-length allocatables will
truncate or pad the assigned value to the length of the left-hand side
of the assignment.

Lengths and offsets that are used by or exposed to Fortran programs via
declarations, substring bounds, and the `LEN()` intrinsic function are always
represented in units of characters, not bytes.
In generated code, assumed-length arguments, the runtime support library,
and in the `elem_len` field of the interoperable descriptor `cdesc_t`,
lengths are always in units of bytes.
The distinction matters only for kinds other than the default.

Fortran substrings are rather like subscript triplets into a hidden
"zero" dimension of a scalar `CHARACTER` value, but they cannot have
strides.

### Concatenation

Fortran has one `CHARACTER`-valued intrinsic operator, `//`, which
concatenates its operands (10.1.5.3).
The operands must have the same kind type parameter.
One or both of the operands may be arrays; if both are arrays, their
shapes must be identical.
The effective length of the result is the sum of the lengths of the
operands.
Parentheses may be ignored, so any `CHARACTER`-valued expression
may be "flattened" into a single sequence of concatenations.

The result of `//` may be used
* as an operand to another concatenation,
* as an operand of a `CHARACTER` relation,
* as an actual argument,
* as the right-hand side of an assignment,
* as the `SOURCE=` or `MOLD=` of an `ALLOCATE` statemnt,
* as the selector or case-expr of an `ASSOCIATE` or `SELECT` construct,
* as a component of a structure or array constructor,
* as the value of a named constant or initializer,
* as the `NAME=` of a `BIND(C)` attribute,
* as the stop-code of a `STOP` statement,
* as the value of a specifier of an I/O statement,
* or as the value of a statement function.

The f18 compiler has a general (but slow) means of implementing concatenation
and a specialized (fast) option to optimize the most common case.

#### General concatenation

In the most general case, the f18 compiler's generated code and
runtime support library represent the result as a deferred-length allocatable
`CHARACTER` temporary scalar or array variable that is initialized
as a zero-length array by `AllocatableInitCharacter()`
and then progressively augmented in place by the values of each of the
operands of the concatenation sequence in turn with calls to
`CharacterConcatenate()`.
Conformability errors are fatal -- Fortran has no means by which a program
may recover from them.
The result is then used as any other deferred-length allocatable
array or scalar would be, and finally deallocated like any other
allocatable.

The runtime routine `CharacterAssign()` takes care of
truncating, padding, or replicating the value(s) assigned to the left-hand
side, as well as reallocating an nonconforming or deferred-length allocatable
left-hand side.  It takes the descriptors of the left- and right-hand sides of
a `CHARACTER` assignemnt as its arguments.

When the left-hand side of a `CHARACTER` assignment is a deferred-length
allocatable and the right-hand side is a temporary, use of the runtime's
`MoveAlloc()` subroutine instead can save an allocation and a copy.

#### Optimized concatenation

Scalar `CHARACTER(KIND=1)` expressions evaluated as the right-hand sides of
assignments to independent substrings or whole variables that are not
deferred-length allocatables can be optimized into a sequence of
calls to the runtime support library that do not allocate temporary
memory.

The routine `CharacterAppend()` copies data from the right-hand side value
to the remaining space, if any, in the left-hand side object, and returns
the new offset of the reduced remaining space.
It is essentially `memcpy(lhs + offset, rhs, min(lhsLength - offset, rhsLength))`.
It does nothing when `offset > lhsLength`.

`void CharacterPad()`adds any necessary trailing blank characters.
