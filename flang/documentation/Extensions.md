<!--===- documentation/Extensions.md 
  
   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  
-->

As a general principle, this compiler will accept by default and
without complaint many legacy features, extensions to the standard
language, and features that have been deleted from the standard,
so long as the recognition of those features would not cause a
standard-conforming program to be rejected or misinterpreted.

Other non-standard features, which do conflict with the current
standard specification of the Fortran programming language, are
accepted if enabled by command-line options.

Intentional violations of the standard
======================================
* Scalar `INTEGER` actual argument expressions (not variables!)
  are converted to the kinds of scalar `INTEGER` dummy arguments
  when the interface is explicit and the kinds differ.
  This conversion allows the results of the intrinsics like
  `SIZE` that (as mentioned below) may return non-default
  `INTEGER` results by default to be passed.  A warning is
  emitted when truncation is possible.
* We are not strict on the contents of `BLOCK DATA` subprograms
  so long as they contain no executable code, no internal subprograms,
  and allocate no storage outside a named `COMMON` block.  (C1415)

Extensions, deletions, and legacy features supported by default
===============================================================
* Tabs in source
* `<>` as synonym for `.NE.` and `/=`
* `$` and `@` as legal characters in names
* Initialization in type declaration statements using `/values/`
* Kind specification with `*`, e.g. `REAL*4`
* `DOUBLE COMPLEX`
* Signed complex literal constants
* DEC `STRUCTURE`, `RECORD`, `UNION`, and `MAP`
* Structure field access with `.field`
* `BYTE` as synonym for `INTEGER(KIND=1)`
* Quad precision REAL literals with `Q`
* `X` prefix/suffix as synonym for `Z` on hexadecimal literals
* `B`, `O`, `Z`, and `X` accepted as suffixes as well as prefixes
* Triplets allowed in array constructors
* Old-style `PARAMETER pi=3.14` statement without parentheses
* `%LOC`, `%VAL`, and `%REF`
* Leading comma allowed before I/O item list
* Empty parentheses allowed in `PROGRAM P()`
* Missing parentheses allowed in `FUNCTION F`
* Cray based `POINTER(p,x)` and `LOC()` intrinsic (with `%LOC()` as
  an alias)
* Arithmetic `IF`.  (Which branch should NaN take? Fall through?)
* `ASSIGN` statement, assigned `GO TO`, and assigned format
* `PAUSE` statement
* Hollerith literals and edit descriptors
* `NAMELIST` allowed in the execution part
* Omitted colons on type declaration statements with attributes
* COMPLEX constructor expression, e.g. `(x+y,z)`
* `+` and `-` before all primary expressions, e.g. `x*-y`
* `.NOT. .NOT.` accepted
* `NAME=` as synonym for `FILE=`
* Data edit descriptors without width or other details
* `D` lines in fixed form as comments or debug code
* `CONVERT=` on the OPEN and INQUIRE statements
* `DISPOSE=` on the OPEN and INQUIRE statements
* Leading semicolons are ignored before any statement that
  could have a label
* The character `&` in column 1 in fixed form source is a variant form
  of continuation line.
* Character literals as elements of an array constructor without an explicit
  type specifier need not have the same length; the longest literal determines
  the length parameter of the implicit type, not the first.
* Outside a character literal, a comment after a continuation marker (&)
  need not begin with a comment marker (!).
* Classic C-style /*comments*/ are skipped, so multi-language header
  files are easier to write and use.
* $ and \ edit descriptors are supported in FORMAT to suppress newline
  output on user prompts.
* REAL and DOUBLE PRECISION variable and bounds in DO loops
* Integer literals without explicit kind specifiers that are out of range
  for the default kind of INTEGER are assumed to have the least larger kind
  that can hold them, if one exists.
* BOZ literals can be used as INTEGER values in contexts where the type is
  unambiguous: the right hand sides of assigments and initializations
  of INTEGER entities, and as actual arguments to a few intrinsic functions
  (ACHAR, BTEST, CHAR).  But they cannot be used if the type would not
  be known (e.g., `IAND(X'1',X'2')`).
* BOZ literals can also be used as REAL values in some contexts where the
  type is unambiguous, such as initializations of REAL parameters.
* EQUIVALENCE of numeric and character sequences (a ubiquitous extension)
* Values for whole anonymous parent components in structure constructors
  (e.g., `EXTENDEDTYPE(PARENTTYPE(1,2,3))` rather than `EXTENDEDTYPE(1,2,3)`
   or `EXTENDEDTYPE(PARENTTYPE=PARENTTYPE(1,2,3))`).
* Some intrinsic functions are specified in the standard as requiring the
  same type and kind for their arguments (viz., ATAN with two arguments,
  ATAN2, DIM, HYPOT, MAX, MIN, MOD, and MODULO);
  we allow distinct types to be used, promoting
  the arguments as if they were operands to an intrinsic `+` operator,
  and defining the result type accordingly.
* DOUBLE COMPLEX intrinsics DREAL, DCMPLX, DCONJG, and DIMAG.
* INT_PTR_KIND intrinsic returns the kind of c_intptr_t.
* Restricted specific conversion intrinsics FLOAT, SNGL, IDINT, IFIX, DREAL,
  and DCMPLX accept arguments of any kind instead of only the default kind or
  double precision kind. Their result kinds remain as specified.
* Specific intrinsics AMAX0, AMAX1, AMIN0, AMIN1, DMAX1, DMIN1, MAX0, MAX1,
  MIN0, and MIN1 accept more argument types than specified. They are replaced by
  the related generics followed by conversions to the specified result types.
* When a scalar CHARACTER actual argument of the same kind is known to
  have a length shorter than the associated dummy argument, it is extended
  on the right with blanks, similar to assignment.
* When a dummy argument is `POINTER` or `ALLOCATABLE` and is `INTENT(IN)`, we
  relax enforcement of some requirements on actual arguments that must otherwise
  hold true for definable arguments.
* Assignment of `LOGICAL` to `INTEGER` and vice versa (but not other types) is
  allowed.  The values are normalized.
* An effectively empty source file (no program unit) is accepted and
  produces an empty relocatable output file.
* A `RETURN` statement may appear in a main program.
* DATA statement initialization is allowed for procedure pointers outside
  structure constructors.

Extensions supported when enabled by options
--------------------------------------------
* C-style backslash escape sequences in quoted CHARACTER literals
  (but not Hollerith) [-fbackslash]
* Logical abbreviations `.T.`, `.F.`, `.N.`, `.A.`, `.O.`, and `.X.`
  [-flogical-abbreviations]
* `.XOR.` as a synonym for `.NEQV.` [-fxor-operator]
* The default `INTEGER` type is required by the standard to occupy
  the same amount of storage as the default `REAL` type.  Default
  `REAL` is of course 32-bit IEEE-754 floating-point today.  This legacy
  rule imposes an artificially small constraint in some cases
  where Fortran mandates that something have the default `INTEGER`
  type: specifically, the results of references to the intrinsic functions
  `SIZE`, `LBOUND`, `UBOUND`, `SHAPE`, and the location reductions
  `FINDLOC`, `MAXLOC`, and `MINLOC` in the absence of an explicit
  `KIND=` actual argument.  We return `INTEGER(KIND=8)` by default in
  these cases when the `-flarge-sizes` option is enabled.

Extensions and legacy features deliberately not supported
---------------------------------------------------------
* `.LG.` as synonym for `.NE.`
* `REDIMENSION`
* Allocatable `COMMON`
* Expressions in formats
* `ACCEPT` as synonym for `READ *`
* `TYPE` as synonym for `PRINT`
* `ARRAY` as synonym for `DIMENSION`
* `VIRTUAL` as synonym for `DIMENSION`
* `ENCODE` and `DECODE` as synonyms for internal I/O
* `IMPLICIT AUTOMATIC`, `IMPLICIT STATIC`
* Default exponent of zero, e.g. `3.14159E`
* Characters in defined operators that are neither letters nor digits
* `B` suffix on unquoted octal constants
* `Z` prefix on unquoted hexadecimal constants (dangerous)
* `T` and `F` as abbreviations for `.TRUE.` and `.FALSE.` in DATA (PGI/XLF)
* Use of host FORMAT labels in internal subprograms (PGI-only feature)
* ALLOCATE(TYPE(derived)::...) as variant of correct ALLOCATE(derived::...) (PGI only)
* Defining an explicit interface for a subprogram within itself (PGI only)
* USE association of a procedure interface within that same procedure's definition
* NULL() as a structure constructor expression for an ALLOCATABLE component (PGI).
* Conversion of LOGICAL to INTEGER in expressions.
* IF (integer expression) THEN ... END IF  (PGI/Intel)
* Comparsion of LOGICAL with ==/.EQ. rather than .EQV. (also .NEQV.) (PGI/Intel)
* Procedure pointers in COMMON blocks (PGI/Intel)
* Underindexing multi-dimensional arrays (e.g., A(1) rather than A(1,1)) (PGI only)
* Legacy PGI `NCHARACTER` type and `NC` Kanji character literals
* Using non-integer expressions for array bounds (e.g., REAL A(3.14159)) (PGI/Intel)
* Mixing INTEGER types as operands to bit intrinsics (e.g., IAND); only two
  compilers support it, and they disagree on sign extension.
* Module & program names that conflict with an object inside the unit (PGI only).
* When the same name is brought into scope via USE association from
  multiple modules, the name must refer to a generic interface; PGI
  allows a name to be a procedure from one module and a generic interface
  from another.
* Type parameter declarations must come first in a derived type definition;
  some compilers allow them to follow `PRIVATE`, or be intermixed with the
  component declarations.
* Wrong argument types in calls to specific intrinsics that have different names than the
  related generics. Some accepted exceptions are listed above in the allowed extensions.
  PGI, Intel, and XLF support this in ways that are not numerically equivalent.
  PGI converts the arguments while Intel and XLF replace the specific by the related generic.

Preprocessing behavior
======================
* The preprocessor is always run, whatever the filename extension may be.
* We respect Fortran comments in macro actual arguments (like GNU, Intel, NAG;
  unlike PGI and XLF) on the principle that macro calls should be treated
  like function references.  Fortran's line continuation methods also work.
