<!--===- docs/Extensions.md 
  
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

As a general principle, this compiler will accept by default and
without complaint many legacy features, extensions to the standard
language, and features that have been deleted from the standard,
so long as the recognition of those features would not cause a
standard-conforming program to be rejected or misinterpreted.

Other non-standard features, which do conflict with the current
standard specification of the Fortran programming language, are
accepted if enabled by command-line options.

## Intentional violations of the standard

* Scalar `INTEGER` actual argument expressions (not variables!)
  are converted to the kinds of scalar `INTEGER` dummy arguments
  when the interface is explicit and the kinds differ.
  This conversion allows the results of the intrinsics like
  `SIZE` that (as mentioned below) may return non-default
  `INTEGER` results by default to be passed.  A warning is
  emitted when truncation is possible.  These conversions
  are not applied in calls to non-intrinsic generic procedures.
* We are not strict on the contents of `BLOCK DATA` subprograms
  so long as they contain no executable code, no internal subprograms,
  and allocate no storage outside a named `COMMON` block.  (C1415)
* Delimited list-directed (and NAMELIST) character output is required
  to emit contiguous doubled instances of the delimiter character
  when it appears in the output value.  When fixed-size records
  are being emitted, as is the case with internal output, this
  is not possible when the problematic character falls on the last
  position of a record.  No two other Fortran compilers do the same
  thing in this situation so there is no good precedent to follow.
  Because it seems least wrong, we emit one copy of the delimiter as
  the last character of the current record and another as the first
  character of the next record.  (The second-least-wrong alternative
  might be to flag a runtime error, but that seems harsh since it's
  not an explicit error in the standard, and the output may not have
  to be usable later as input anyway.)
  Consequently, the output is not suitable for use as list-directed or
  NAMELIST input.  If a later standard were to clarify this case, this
  behavior will change as needed to conform.
```
character(11) :: buffer(3)
character(10) :: quotes = '""""""""""'
write(buffer,*,delim="QUOTE") quotes
print "('>',a10,'<')", buffer
end
```
* The name of the control variable in an implied DO loop in an array
  constructor or DATA statement has a scope over the value-list only,
  not the bounds of the implied DO loop.  It is not advisable to use
  an object of the same name as the index variable in a bounds
  expression, but it will work, instead of being needlessly undefined.
* If both the `COUNT=` and the `COUNT_MAX=` optional arguments are
  present on the same call to the intrinsic subroutine `SYSTEM_CLOCK`,
  we require that their types have the same integer kind, since the
  kind of these arguments is used to select the clock rate.  In common
  with some other compilers, the clock rate varies from tenths of a
  second to nanoseconds depending on argument kind and platform support.
* If a dimension of a descriptor has zero extent in a call to
  `CFI_section`, `CFI_setpointer` or `CFI_allocate`, the lower
  bound on that dimension will be set to 1 for consistency with
  the `LBOUND()` intrinsic function.
* `-2147483648_4` is, strictly speaking, a non-conforming literal
  constant on a machine with 32-bit two's-complement integers as
  kind 4, because the grammar of Fortran expressions parses it as a
  negation of a literal constant, not a negative literal constant.
  This compiler accepts it with a portability warning.
* Construct names like `loop` in `loop: do j=1,n` are defined to
  be "local identifiers" and should be distinct in the "inclusive
  scope" -- i.e., not scoped by `BLOCK` constructs.
  As most (but not all) compilers implement `BLOCK` scoping of construct
  names, so does f18, with a portability warning.
* 15.6.4 paragraph 2 prohibits an implicitly typed statement function
  from sharing the same name as a symbol in its scope's host, if it
  has one.
  We accept this usage with a portability warning.
* A module name from a `USE` statement can also be used as a
  non-global name in the same scope.  This is not conforming,
  but it is useful and unambiguous.
* The argument to `RANDOM_NUMBER` may not be an assumed-size array.
* `NULL()` without `MOLD=` is not allowed to be associated as an
  actual argument corresponding to an assumed-rank dummy argument;
  its rank in the called procedure would not be well-defined.
* When an index variable of a `FORALL` or `DO CONCURRENT` is present
  in the enclosing scope, and the construct does not have an explicit
  type specification for its index variables, some weird restrictions
  in F'2023 subclause 19.4 paragraphs 6 & 8 should apply.  Since this
  compiler properly scopes these names, violations of these restrictions
  elicit only portability warnings by default.
* The standard defines the intrinsic functions `MOD` and `MODULO`
  for real arguments using expressions in terms of `AINT` and `FLOOR`.
  These definitions yield fairly poor results due to floating-point
  cancellation, and every Fortran compiler (including this one)
  uses better algorithms.
* When an index variable of a `FORALL` or `DO CONCURRENT` is present
  in the enclosing scope, and the construct does not have an explicit
  type specification for its index variables, some weird restrictions
  in F'2023 subclause 19.4 paragraphs 6 & 8 should apply.  Since this
  compiler properly scopes these names, violations of these restrictions
  elicit only portability warnings by default.
* The rules for pairwise distinguishing the specific procedures of a
  generic interface are inadequate, as admitted in note C.11.6 of F'2023.
  Generic interfaces whose specific procedures can be easily proven by
  hand to be pairwise distinct (i.e., no ambiguous reference is possible)
  appear in real applications, but are still non-conforming under the
  incomplete tests in F'2023 15.4.3.4.5.
  These cases are compiled with optional portability warnings.

## Extensions, deletions, and legacy features supported by default

* Tabs in source
* `<>` as synonym for `.NE.` and `/=`
* `$` and `@` as legal characters in names
* Initialization in type declaration statements using `/values/`
* Saved variables without explicit or default initializers are zero initialized.
* In a saved entity of a type with a default initializer, components without default
  values are zero initialized.
* Kind specification with `*`, e.g. `REAL*4`
* `DOUBLE COMPLEX` as a synonym for `COMPLEX(KIND(0.D0))` --
  but not when spelled `TYPE(DOUBLECOMPLEX)`.
* Signed complex literal constants
* DEC `STRUCTURE`, `RECORD`, with '%FILL'; but `UNION`, and `MAP`
  are not yet supported throughout compilation, and elicit a
  "not yet implemented" message.
* Structure field access with `.field`
* `BYTE` as synonym for `INTEGER(KIND=1)`; but not when spelled `TYPE(BYTE)`.
* When kind-param is used for REAL literals, allow a matching exponent letter
* Quad precision REAL literals with `Q`
* `X` prefix/suffix as synonym for `Z` on hexadecimal literals
* `B`, `O`, `Z`, and `X` accepted as suffixes as well as prefixes
* Support for using bare `L` in FORMAT statement
* Triplets allowed in array constructors
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
* `CARRIAGECONTROL=` on the OPEN and INQUIRE statements
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
* Tabs in format strings (not `FORMAT` statements) are allowed on output.
* REAL and DOUBLE PRECISION variable and bounds in DO loops
* Integer literals without explicit kind specifiers that are out of range
  for the default kind of INTEGER are assumed to have the least larger kind
  that can hold them, if one exists.
* BOZ literals can be used as INTEGER values in contexts where the type is
  unambiguous: the right hand sides of assignments and initializations
  of INTEGER entities, as actual arguments to a few intrinsic functions
  (ACHAR, BTEST, CHAR), and as actual arguments of references to
  procedures with explicit interfaces whose corresponding dummy
  argument has a numeric type to which the BOZ literal may be
  converted.  BOZ literals are interpreted as default INTEGER only
  when they appear as the first items of array constructors with no
  explicit type.  Otherwise, they generally cannot be used if the type would
  not be known (e.g., `IAND(X'1',X'2')`).
* BOZ literals can also be used as REAL values in some contexts where the
  type is unambiguous, such as initializations of REAL parameters.
* EQUIVALENCE of numeric and character sequences (a ubiquitous extension),
  as well as of sequences of non-default kinds of numeric types
  with each other.
* Values for whole anonymous parent components in structure constructors
  (e.g., `EXTENDEDTYPE(PARENTTYPE(1,2,3))` rather than `EXTENDEDTYPE(1,2,3)`
   or `EXTENDEDTYPE(PARENTTYPE=PARENTTYPE(1,2,3))`).
* Some intrinsic functions are specified in the standard as requiring the
  same type and kind for their arguments (viz., ATAN with two arguments,
  ATAN2, DIM, HYPOT, IAND, IEOR, IOR, MAX, MIN, MOD, and MODULO);
  we allow distinct types to be used, promoting
  the arguments as if they were operands to an intrinsic `+` operator,
  and defining the result type accordingly.
* DOUBLE COMPLEX intrinsics DREAL, DCMPLX, DCONJG, and DIMAG.
* The DFLOAT intrinsic function.
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
  allowed.  The values are normalized to canonical `.TRUE.`/`.FALSE.`.
  The values are also normalized for assignments of `LOGICAL(KIND=K1)` to
  `LOGICAL(KIND=K2)`, when `K1 != K2`.
* Static initialization of `LOGICAL` with `INTEGER` is allowed in `DATA` statements
  and object initializers.
  The results are *not* normalized to canonical `.TRUE.`/`.FALSE.`.
  Static initialization of `INTEGER` with `LOGICAL` is also permitted.
* An effectively empty source file (no program unit) is accepted and
  produces an empty relocatable output file.
* A `RETURN` statement may appear in a main program.
* DATA statement initialization is allowed for procedure pointers outside
  structure constructors.
* Nonstandard intrinsic functions: ISNAN, SIZEOF
* A forward reference to a default INTEGER scalar dummy argument or
  `COMMON` block variable is permitted to appear in a specification
  expression, such as an array bound, in a scope with IMPLICIT NONE(TYPE)
  if the name of the variable would have caused it to be implicitly typed
  as default INTEGER if IMPLICIT NONE(TYPE) were absent.
* OPEN(ACCESS='APPEND') is interpreted as OPEN(POSITION='APPEND')
  to ease porting from Sun Fortran.
* Intrinsic subroutines EXIT([status]) and ABORT()
* The definition of simple contiguity in 9.5.4 applies only to arrays;
  we also treat scalars as being trivially contiguous, so that they
  can be used in contexts like data targets in pointer assignments
  with bounds remapping.
* The `CONTIGUOUS` attribute can be redundantly applied to simply
  contiguous objects, including scalars, with a portability warning.
* We support some combinations of specific procedures in generic
  interfaces that a strict reading of the standard would preclude
  when their calls must nonetheless be distinguishable.
  Specifically, `ALLOCATABLE` dummy arguments are distinguishing
  if an actual argument acceptable to one could not be passed to
  the other & vice versa because exactly one is polymorphic or
  exactly one is unlimited polymorphic).
* External unit 0 is predefined and connected to the standard error output,
  and defined as `ERROR_UNIT` in the intrinsic `ISO_FORTRAN_ENV` module.
* Objects in blank COMMON may be initialized.
* Initialization of COMMON blocks outside of BLOCK DATA subprograms.
* Multiple specifications of the SAVE attribute on the same object
  are allowed, with a warning.
* Specific intrinsic functions BABS, IIABS, JIABS, KIABS, ZABS, and CDABS.
* A `POINTER` component's type need not be a sequence type when
  the component appears in a derived type with `SEQUENCE`.
  (This case should probably be an exception to constraint C740 in
  the standard.)
* Format expressions that have type but are not character and not
  integer scalars are accepted so long as they are simply contiguous.
  This legacy extension supports pre-Fortran'77 usage in which
  variables initialized in DATA statements with Hollerith literals
  as modifiable formats.
* At runtime, `NAMELIST` input will skip over `NAMELIST` groups
  with other names, and will treat text before and between groups
  as if they were comment lines, even if not begun with `!`.
* Commas are required in FORMAT statements and character variables
  only when they prevent ambiguity.
* Legacy names `AND`, `OR`, and `XOR` are accepted as aliases for
  the standard intrinsic functions `IAND`, `IOR`, and `IEOR`
  respectively.
* A digit count of d=0 is accepted in Ew.0, Dw.0, and Gw.0 output
  editing if no nonzero scale factor (kP) is in effect.
* The name `IMAG` is accepted as an alias for the generic intrinsic
  function `AIMAG`.
* The legacy extension intrinsic functions `IZEXT` and `JZEXT`
  are supported; `ZEXT` has different behavior with various older
  compilers, so it is not supported.
* f18 doesn't impose a limit on the number of continuation lines
  allowed for a single statement.
* When a type-bound procedure declaration statement has neither interface
  nor attributes, the "::" before the bindings is optional, even
  if a binding has renaming with "=> proc".
  The colons are not necessary for an unambiguous parse, C768
  notwithstanding.
* A type-bound procedure binding can be passed as an actual
  argument corresponding to a dummy procedure and can be used as
  the target of a procedure pointer assignment statement.
* An explicit `INTERFACE` can declare the interface of a
  procedure pointer even if it is not a dummy argument.
* A `NOPASS` type-bound procedure binding is required by C1529
  to apply only to a scalar data-ref, but most compilers don't
  enforce it and the constraint is not necessary for a correct
  implementation.
* A label may follow a semicolon in fixed form source.
* A scalar logical dummy argument to a `BIND(C)` procedure does
  not have to have `KIND=C_BOOL` since it can be converted to/from
  `_Bool` without loss of information.
* The character length of the `SOURCE=` or `MOLD=` in `ALLOCATE`
  may be distinct from the constant character length, if any,
  of an allocated object.
* When a name is brought into a scope by multiple ways,
  such as USE-association as well as an `IMPORT` from its host,
  it's an error only if the resolution is ambiguous.
* An entity may appear in a `DATA` statement before its explicit
  type declaration under `IMPLICIT NONE(TYPE)`.
* `INCLUDE` lines can start in any column, can be preceded in
  fixed form source by a '0' in column 6, can contain spaces
  between the letters of the word INCLUDE, and can have a
  numeric character literal kind prefix on the file name.
* Intrinsic procedures SIND, COSD, TAND and ATAND. Constant folding
  is currently not supported for these procedures but this is planned.
* When a pair of quotation marks in a character literal are split
  by a line continuation in free form, the second quotation mark
  may appear at the beginning of the continuation line without an
  ampersand, althought one is required by the standard.
* Unrestricted `INTRINSIC` functions are accepted for use in
  `PROCEDURE` statements in generic interfaces, as in some other
  compilers.
* A `NULL()` pointer is treated as an unallocated allocatable
  when associated with an `INTENT(IN)` allocatable dummy argument.
* `READ(..., SIZE=n)` is accepted with `NML=` and `FMT=*` with
  a portability warning.
  The Fortran standard doesn't allow `SIZE=` with formatted input
  modes that might require look-ahead, perhaps to ease implementations.
* When a file included via an `INCLUDE` line or `#include` directive
  has a continuation marker at the end of its last line in free form,
  Fortran line continuation works.
* A `NAMELIST` input group may omit its trailing `/` character if
  it is followed by another `NAMELIST` input group.
* A `NAMELIST` input group may begin with either `&` or `$`.
* A comma in a fixed-width numeric input field terminates the
  field rather than signaling an invalid character error.

### Extensions supported when enabled by options

* C-style backslash escape sequences in quoted CHARACTER literals
  (but not Hollerith) [-fbackslash], including Unicode escapes
  with `\U`.
* Logical abbreviations `.T.`, `.F.`, `.N.`, `.A.`, `.O.`, and `.X.`
  [-flogical-abbreviations]
* `.XOR.` as a synonym for `.NEQV.` [-fxor-operator]
* The default `INTEGER` type is required by the standard to occupy
  the same amount of storage as the default `REAL` type.  Default
  `REAL` is of course 32-bit IEEE-754 floating-point today.  This legacy
  rule imposes an artificially small constraint in some cases
  where Fortran mandates that something have the default `INTEGER`
  type: specifically, the results of references to the intrinsic functions
  `SIZE`, `STORAGE_SIZE`,`LBOUND`, `UBOUND`, `SHAPE`, and the location reductions
  `FINDLOC`, `MAXLOC`, and `MINLOC` in the absence of an explicit
  `KIND=` actual argument.  We return `INTEGER(KIND=8)` by default in
  these cases when the `-flarge-sizes` option is enabled.
  `SIZEOF` and `C_SIZEOF` always return `INTEGER(KIND=8)`.
* Treat each specification-part like is has `IMPLICIT NONE`
  [-fimplicit-none-type-always]
* Ignore occurrences of `IMPLICIT NONE` and `IMPLICIT NONE(TYPE)`
  [-fimplicit-none-type-never]
* Old-style `PARAMETER pi=3.14` statement without parentheses
  [-falternative-parameter-statement]

### Extensions and legacy features deliberately not supported

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
* Use of INTEGER data with the intrinsic logical operators `.NOT.`, `.AND.`, `.OR.`,
  and `.XOR.`.
* IF (integer expression) THEN ... END IF  (PGI/Intel)
* Comparison of LOGICAL with ==/.EQ. rather than .EQV. (also .NEQV.) (PGI/Intel)
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
* VMS listing control directives (`%LIST`, `%NOLIST`, `%EJECT`)
* Continuation lines on `INCLUDE` lines
* `NULL()` actual argument corresponding to an `ALLOCATABLE` dummy data object
* User (non-intrinsic) `ELEMENTAL` procedures may not be passed as actual
  arguments, in accordance with the standard; some Fortran compilers
  permit such usage.
* Constraint C1406, which prohibits the same module name from being used
  in a scope for both an intrinsic and a non-intrinsic module, is implemented
  as a portability warning only, not a hard error.
* IBM @PROCESS directive is accepted but ignored.

## Preprocessing behavior

* The preprocessor is always run, whatever the filename extension may be.
* We respect Fortran comments in macro actual arguments (like GNU, Intel, NAG;
  unlike PGI and XLF) on the principle that macro calls should be treated
  like function references.  Fortran's line continuation methods also work.

## Standard features not silently accepted

* Fortran explicitly ignores type declaration statements when they
  attempt to type the name of a generic intrinsic function (8.2 p3).
  One can declare `CHARACTER::COS` and still get a real result
  from `COS(3.14159)`, for example.  f18 will complain when a
  generic intrinsic function's inferred result type does not
  match an explicit declaration.  This message is a warning.

## Standard features that might as well not be

* f18 supports designators with constant expressions, properly
  constrained, as initial data targets for data pointers in
  initializers of variable and component declarations and in
  `DATA` statements; e.g., `REAL, POINTER :: P => T(1:10:2)`.
  This Fortran 2008 feature might as well be viewed like an
  extension; no other compiler that we've tested can handle
  it yet.
* According to 11.1.3.3p1, if a selector of an `ASSOCIATE` or
  related construct is defined by a variable, it has the `TARGET`
  attribute if the variable was a `POINTER` or `TARGET`.
  We read this to include the case of the variable being a
  pointer-valued function reference.
  No other Fortran compiler seems to handle this correctly for
  `ASSOCIATE`, though NAG gets it right for `SELECT TYPE`.
* The standard doesn't explicitly require that a named constant that
  appears as part of a complex-literal-constant be a scalar, but
  most compilers emit an error when an array appears.
  f18 supports them with a portability warning.
* f18 does not enforce a blanket prohibition against generic
  interfaces containing a mixture of functions and subroutines.
  Apart from some contexts in which the standard requires all of
  a particular generic interface to have only all functions or
  all subroutines as its specific procedures, we allow both to
  appear, unlike several other Fortran compilers.
  This is especially desirable when two generics of the same
  name are combined due to USE association and the mixture may
  be inadvertent.
* Since Fortran 90, `INCLUDE` lines have been allowed to have
  a numeric kind parameter prefix on the file name.  No other
  Fortran compiler supports them that I can find.
* A `SEQUENCE` derived type is required (F'2023 C745) to have
  at least one component.  No compiler enforces this constraint;
  this compiler emits a warning.
* Many compilers disallow a `VALUE` assumed-length character dummy
  argument, which has been standard since F'2008.
  We accept this usage with an optional portability warning.

## Behavior in cases where the standard is ambiguous or indefinite

* When an inner procedure of a subprogram uses the value or an attribute
  of an undeclared name in a specification expression and that name does
  not appear in the host, it is not clear in the standard whether that
  name is an implicitly typed local variable of the inner procedure or a
  host association with an implicitly typed local variable of the host.
  For example:
```
module module
 contains
  subroutine host(j)
    ! Although "m" never appears in the specification or executable
    ! parts of this subroutine, both of its contained subroutines
    ! might be accessing it via host association.
    integer, intent(in out) :: j
    call inner1(j)
    call inner2(j)
   contains
    subroutine inner1(n)
      integer(kind(m)), intent(in) :: n
      m = n + 1
    end subroutine
    subroutine inner2(n)
      integer(kind(m)), intent(out) :: n
      n = m + 2
    end subroutine
  end subroutine
end module

program demo
  use module
  integer :: k
  k = 0
  call host(k)
  print *, k, " should be 3"
end

```

  Other Fortran compilers disagree in their interpretations of this example;
  some seem to treat the references to `m` as if they were host associations
  to an implicitly typed variable (and print `3`), while others seem to
  treat them as references to implicitly typed local variables, and
  load uninitialized values.

  In f18, we chose to emit an error message for this case since the standard
  is unclear, the usage is not portable, and the issue can be easily resolved
  by adding a declaration.

* In subclause 7.5.6.2 of Fortran 2018 the standard defines a partial ordering
  of the final subroutine calls for finalizable objects, their non-parent
  components, and then their parent components.
  (The object is finalized, then the non-parent components of each element,
  and then the parent component.)
  Some have argued that the standard permits an implementation
  to finalize the parent component before finalizing an allocatable component in
  the context of deallocation, and the next revision of the language may codify
  this option.
  In the interest of avoiding needless confusion, this compiler implements what
  we believe to be the least surprising order of finalization.
  Specifically: all non-parent components are finalized before
  the parent, allocatable or not;
  all finalization takes place before any deallocation;
  and no object or subobject will be finalized more than once.

* When `RECL=` is set via the `OPEN` statement for a sequential formatted input
  file, it functions as an effective maximum record length.
  Longer records, if any, will appear as if they had been truncated to
  the value of `RECL=`.
  (Other compilers ignore `RECL=`, signal an error, or apply effective truncation
  to some forms of input in this situation.)
  For sequential formatted output, RECL= serves as a limit on record lengths
  that raises an error when it is exceeded.

* When a `DATA` statement in a `BLOCK` construct could be construed as
  either initializing a host-associated object or declaring a new local
  initialized object, f18 interprets the standard's classification of
  a `DATA` statement as being a "declaration" rather than a "specification"
  construct, and notes that the `BLOCK` construct is defined as localizing
  names that have specifications in the `BLOCK` construct.
  So this example will elicit an error about multiple initialization:
```
subroutine subr
  integer n = 1
  block
    data n/2/
  end block
end subroutine
```

  Other Fortran compilers disagree with each other in their interpretations
  of this example.
  The precedent among the most commonly used compilers
  agrees with f18's interpretation: a `DATA` statement without any other
  specification of the name refers to the host-associated object.

* Many Fortran compilers allow a non-generic procedure to be `USE`-associated
  into a scope that also contains a generic interface of the same name
  but does not have the `USE`-associated non-generic procedure as a
  specific procedure.
```
module m1
 contains
  subroutine foo(n)
    integer, intent(in) :: n
  end subroutine
end module

module m2
  use m1, only: foo
  interface foo
    module procedure noargs
  end interface
 contains
  subroutine noargs
  end subroutine
end module
```

  This case elicits a warning from f18, as it should not be treated
  any differently than the same case with the non-generic procedure of
  the same name being defined in the same scope rather than being
  `USE`-associated into it, which is explicitly non-conforming in the
  standard and not allowed by most other compilers.
  If the `USE`-associated entity of the same name is not a procedure,
  most compilers disallow it as well.

* Fortran 2018 19.3.4p1: "A component name has the scope of its derived-type
  definition.  Outside the type definition, it may also appear ..." which
  seems to imply that within its derived-type definition, a component
  name is in its scope, and at least shadows any entity of the same name
  in the enclosing scope and might be read, thanks to the "also", to mean
  that a "bare" reference to the name could be used in a specification inquiry.
  However, most other compilers do not allow a component to shadow exterior
  symbols, much less appear in specification inquiries, and there are
  application codes that expect exterior symbols whose names match
  components to be visible in a derived-type definition's default initialization
  expressions, and so f18 follows that precedent.

* 19.3.1p1 "Within its scope, a local identifier of an entity of class (1)
  or class (4) shall not be the same as a global identifier used in that scope..."
  is read so as to allow the name of a module, submodule, main program,
  or `BLOCK DATA` subprogram to also be the name of an local entity in its
  scope, with a portability warning, since that global name is not actually
  capable of being "used" in its scope.

* In the definition of the `ASSOCIATED` intrinsic function (16.9.16), its optional
  second argument `TARGET=` is required to be "allowable as the data-target or
  proc-target in a pointer assignment statement (10.2.2) in which POINTER is
  data-pointer-object or proc-pointer-object."  Some Fortran compilers
  interpret this to require that the first argument (`POINTER=`) be a valid
  left-hand side for a pointer assignment statement -- in particular, it
  cannot be `NULL()`, but also it is required to be modifiable.
  As there is  no good reason to disallow (say) an `INTENT(IN)` pointer here,
  or even `NULL()` as a well-defined case that is always `.FALSE.`,
  this compiler doesn't require the `POINTER=` argument to be a valid
  left-hand side for a pointer assignment statement, and we emit a
  portability warning when it is not.

* F18 allows a `USE` statement to reference a module that is defined later
  in the same compilation unit, so long as mutual dependencies do not form
  a cycle.
  This feature forestalls any risk of such a `USE` statement reading an
  obsolete module file from a previous compilation and then overwriting
  that file later.

* F18 allows `OPTIONAL` dummy arguments to interoperable procedures
  unless they are `VALUE` (C865).

* F18 processes the `NAMELIST` group declarations in a scope after it
  has resolved all of the names in that scope.  This means that names
  that appear before their local declarations do not resolve to host
  associated objects and do not elicit errors about improper redeclarations
  of implicitly typed entities.

* Standard Fortran allows forward references to derived types, which
  can lead to ambiguity when combined with host association.
  Some Fortran compilers resolve the type name to the host type,
  others to the forward-referenced local type; this compiler diagnoses
  an error.
```
module m
  type ambiguous; integer n; end type
 contains
  subroutine s
    type(ambiguous), pointer :: ptr
    type ambiguous; real a; end type
  end
end
```

* When an intrinsic procedure appears in the specification part of a module
  only in function references, but not an explicit `INTRINSIC` statement,
  its name is not brought into other scopes by a `USE` statement.

* The subclause on rounding in formatted I/O (13.7.2.3.8 in F'2023)
  only discusses rounding for decimal-to/from-binary conversions,
  omitting any mention of rounding for hexadecimal conversions.
  As other compilers do apply rounding, so does this one.

* For real `MAXVAL`, `MINVAL`, `MAXLOC`, and `MINLOC`, NaN values are
  essentially ignored unless there are some unmasked array entries and
  *all* of them are NaNs.

## De Facto Standard Features

* `EXTENDS_TYPE_OF()` returns `.TRUE.` if both of its arguments have the
  same type, a case that is technically implementation-defined.

* `ENCODING=` is not in the list of changeable modes on an I/O unit,
  but every Fortran compiler allows the encoding to be changed on an
  open unit.

* A `NAMELIST` input item that references a scalar element of a vector
  or contiguous array can be used as the initial element of a storage
  sequence.  For example, "&GRP A(1)=1. 2. 3./" is treated as if had been
  "&GRP A(1:)=1. 2. 3./".
