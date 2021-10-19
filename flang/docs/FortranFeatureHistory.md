<!--===- docs/FortranFeatureHistory.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# A Fortran feature history cheat sheet

```eval_rst
.. contents::
   :local:
```

## Original IBM 704 FORTRAN

Features marked with asterisks `*` were gone by FORTRAN IV.

* Fixed form input with comment and continuation cards
* INTEGER and REAL types, implicit naming conventions
* DIMENSION and EQUIVALENCE statements
* Assignment statements
* Arithmetic (3-way) IF statement
* IF statements for checking exceptions and sense switches, manipulating lights
* GO TO, computed GO TO, ASSIGN, and assigned GO TO statements
* DO loops: positive expressions, 1 trip minimum
* extended DO loop ranges
* PAUSE, STOP, and CONTINUE statements
* Formatted I/O: FORMAT, READ, WRITE, PRINT, PUNCH
   and `*` READ INPUT / WRITE OUTPUT TAPE
* Unformatted I/O: READ/WRITE `*` TAPE/DRUM
* ENDFILE, REWIND, and BACKSPACE statements
* FREQUENCY statement (optimization hint - survived into FORTRAN IV)
* Hollerith constants
* Intrinsic functions (all names ending in F`*`)
* statement functions (names ending in F only`*`)

## FORTRAN II
* SUBROUTINE and FUNCTION subprograms
* END statement (with five Sense Switch override argument`*`)
   (Sense Switch 4, if on: "Causes FORTRAN II to produce a program optimized
    with respect to index registers.")
* CALL and RETURN statements
* COMMON (blank only)
* DOUBLE PRECISION and (single) COMPLEX data types
* 6-character identifiers
* Bitwise assignment statements with 'B' in column 1 (IBM 7090 only)
* Double precision with 'D' in column 1 (ditto); complex with 'I'; funcs with 'F'

## FORTRAN IV
* DATA statement
* labeled COMMON
* BLOCK DATA subprograms
* LOGICAL type and expressions, logical IF statement
* Removal of weird original features (`*` above)
* Quoted character strings
* NAMELIST
* EXTERNAL subprograms for use as actual arguments
* alternate RETURN, ENTRY
* &666 label actual arguments for assigned GO TO alternate return
* implicit RETURN at END

## FORTRAN 66
* max 3 array dimensions; subscripts only like `C*V+K`; lower bounds all 1
* adjustable array dummy arguments (dimension of dummy array is dummy argument)

## FORTRAN 77
* array dimension lower bounds other than 1
* IF THEN / ELSE IF THEN / ELSE / END IF construct
* DO loops with negative expressions and zero trip counts
* OPEN, CLOSE, and INQUIRE statements
* Direct-access I/O
* IMPLICIT statement (was in FORTRAN IV)
* CHARACTER data type (was in FORTRAN IV)
* PARAMETER statement
* SAVE statement
* Generic intrinsic names
* lexical string comparisons
* Obsolescent or deleted features: Hollerith constants and data; H edit descriptors; overindexing;
   extended range DO loops
* (non-standard option) recursion
* .EQV. and .NEQV.
* implicit RETURN at END

## MIL-STD-1753 Fortran (1978)
* DO WHILE, DO / END DO
* INCLUDE statement
* IMPLICIT NONE
* Bit manipulation intrinsics (IAND, IOR, IEOR, ISHFT, ISHFTC, MVBITS, &c.)

## Fortran 90
* ALLOCATABLE attribute/statement, ALLOCATE and DEALLOCATE statements
* keyword= actual arguments
* Derived TYPEs, PRIVATE, SEQUENCE; structure components
* Modules
* POINTER and TARGET attributes, NULLIFY statement
* Free form source
* Long identifiers
* Inline ! comments
* Array expressions and assignments
* WHERE construct
* RECURSIVE procedures
* INTERFACE
* generic procedures
* operator overloading
* new declaration syntax with ::
* EXIT and CYCLE statements
* SELECT CASE construct
* Portable kind specifications
* INTENT on arguments
* Obsolescent features beyond those removed in Fortran 95 below: alternate
   return, computed GO TO, statement functions, intermixed DATA,
   `CHARACTER*x` form, assumed-length `CHARACTER*(*)` functions, fixed form source

## Fortran 95 (acquiring some HPF features)
* FORALL construct
* nested WHERE
* Default initialization of derived type components
* initialization of pointers to NULL()
* (clarification) automatic DEALLOCATE at end of scope
* extended intrinsics, e.g. DIM= arguments
* PURE subprograms
* removed features (obsolescent in Fortran 90): floating-point DO index variables,
   GO TO an END IF from outside, PAUSE statement, ASSIGN statement and
   assigned GO TO and formats, H edit descriptor

## Fortran 2003
* KIND and LEN parameterized derived types (still not widely available with correct implementations)
* PROCEDURE pointers and pointer components
* FINAL subroutines
* type-bound procedures
* GENERIC bindings
* PASS attribute
* type-bound generic OPERATOR(+) and ASSIGNMENT(=)
* EXTENDS(type)
* type-bound procedure overriding; NON_OVERRIDABLE attribute to prevent it
* ENUM / ENUMERATOR :: / END ENUM
* ASSOCIATE / END ASSOCIATE construct
* CLASS polymorphic declarator
* SELECT TYPE / END SELECT construct, TYPE IS and CLASS IS clauses
* Abstract interface allowed on DEFERRED type-bound procedure meant to be overridden
* Structure constructors with keyword=
* ALLOCATE statement now works on scalars
* Assignment to allocatable array with automatic (re)allocation
* CALL MOVE_ALLOC(from, to) intrinsic
* Finer-grained PUBLIC/PRIVATE
* PROTECTED attribute and statement
* USE module, OPERATOR(.foo.) => OPERATOR(.bar.)
* Lower bounds on pointer assignment; expansion of
   vector RHS to multidimensional pointer
* INTENT allowed on POINTER dummy argument, defined
   to pertain to the pointer rather than to its target
* VOLATILE attribute
* IMPORT statement in INTERFACEs
* ISO_FORTRAN_ENV intrinsic module
* Unicode, SELECTED_CHAR_KIND()
* 63-char names and 256-line statements
* BOZ constants in INT/REAL/CMPLX/DBLE intrinsic calls
* [array constant] with optional [type::...] specifier
* Named constants in complex constant values
* SYSTEM_CLOCK(COUNT_RATE=real type) now allowed
* MAX, MAXLOC, MAXVAL, MIN, MINLOC, MINVAL on CHARACTER
* Negative zero on ATAN2, LOG, SQRT
* IEEE underflow control
* Derived type I/O: DT edit, GENERIC READ/WRITE bindings
* ASYNCHRONOUS attribute and I/O, WAIT statement
* FLUSH statement
* IOMSG=str
* OPEN(ACCESS='STREAM')
* OPEN(ROUND=mode), overrides on READ/WRITE; Rx edits
* OPEN(DECIMAL=COMMA/POINT), overrides on READ/WRITE; DC and DP edits
* OPEN(SIGN=)
* KIND= type parameters allowed on specifiers, e.g. NEXTREC=n
   for cases where n is not default kind of INTEGER
* Recursive I/O (also mentioned in Fortran 2008)
* NEW_LINE()
* I/O of IEEE-754 negative zero, infinities and NaNs
* Fortran 66-style optional comma in 2P[,]2E12.4 edit descriptor
* Interoperability with C

## Fortran 2008
* SUBMODULE, MODULE PROCEDURE
* Coarray references and image control statements
* DO CONCURRENT as a non-parallel construct
* CONTIGUOUS attribute and statement, IS_CONTIGUOUS() intrinsic
* Simply contiguous arrays
* Maximum rank now 15
* 64-bit INTEGER required as SELECTED_INT_KIND(18)
* ALLOCATABLE members with recursive types
* Implied-shape array declarations, e.g. `INTEGER :: x(0:*) = [0, 1, 2]`
* Pointer association initialization in declaration with => to SAVE target
* Generalization of expressions allowed in DATA statement subscripts
   and implied DO subexpressions
* FORALL(INTEGER(kind) :: ...) kind specification
* Intrinsic types in TYPE statements, e.g. TYPE(INTEGER)
* Multiple type-bound procedures on one PROCEDURE statement
* Structure constructors can omit ALLOCATABLE components
* ALLOCATE(arr, SOURCE=x or MOLD=x) sets shape without needing
   explicit bounds on arr
* ALLOCATE(polymorphic, MOLD=x) sets type
* z%RE, z%IM
* POINTER-valued functions as variables suitable for LHS of =, &c.
* OPEN(NEWUNIT=u)
* G0 edit descriptor
* `(*(...))` format item unlimited repetition
* Recursive I/O
* BLOCK construct
* EXIT statement for constructs other than DO
* STOP statement constant generalized
* BGE(), BGT(), BLE(), BLT() unsigned integer comparisons
* DSHIFTL(), DSHIFTR()
* LEADZ(), POPCNT(), POPPAR(), TRAILZ()
* MASKL(), MASKR()
* SHIFTL(), SHIFTR(), SHIFTA()
* MERGE_BITS()
* IALL(), IANY(), IPARITY()
* STORAGE_SIZE() in bits
* RADIX argument to SELECTED_REAL_KIND()
* COMPLEX arguments to ACOS et al.
* ACOSH(), ASINH(), ATANH()
* ATAN(x,y) synonym for ATAN2()
* Bessel functions
* ERF(), ERFC(), ERFC_SCALED(), GAMMA(), HYPOT(), LOG_GAMMA()
* NORM2()
* PARITY()
* CALL EXECUTE_COMMAND_LINE()
* MINLOC(BACK=.TRUE.), MAXLOC(BACK=.TRUE.)
* FINDLOC()
* More constants and functions in intrinsic module ISO_FORTRAN_ENV.
* Implicit SAVE attribute assumed for module/submodule variables,
   procedure pointers, and COMMON blocks.
* CONTAINS section can be empty in a procedure or type.
* Internal procedures may be passed as actual arguments and assigned
   to procedure pointers.
* Null pointer or unallocated allocatable may be passed to OPTIONAL dummy
   argument, which then appears to not be present.
* POINTER INTENT(IN) dummy arg may be associated with non-pointer TARGET actual
* Refinement of GENERIC resolution rules on pointer/allocatable, data/procedure
* IMPURE for ELEMENTAL procedures (still PURE by default of course)
* Obsolescence of ENTRY
* A source line can begin with a semicolon.

## Fortran 2018
* Obsolescence of COMMON, EQUIVALENCE, BLOCK DATA, FORALL, labeled DO,
   specific names for generic intrinsics
* Arithmetic IF and non-block DO deleted
* Constant properties of an object can be used in its initialization
* Implied DO variables can be typed in array constructors and DATA
* Assumed-rank arrays with DIMENSION(..), SELECT RANK construct
* A file can be opened on multiple units
* Advancing input with SIZE=
* G0.d for integer, logical, character
* D0.d, E0.d, EN0.d, ES0.d, Ew.dE0, &c.
* EX hex floating-point output; hex acceptable for floating-point input
* Variable stop code allowed in (ERROR) STOP
* new COSHAPE, OUT_OF_RANGE, RANDOM_INIT, REDUCE intrinsics
* minor tweaks to extant intrinsics
* IMPORT statement for BLOCK and contained subprograms
* IMPLICIT NONE can require explicit EXTERNAL
* RECURSIVE becomes default; NON_RECURSIVE added
* DO CONCURRENT locality clauses
