# Semantics: Resolving Labels and Construct Names

## Overview

After the Fortran input file(s) has been parsed into a syntax tree, the compiler must check that the program checks semantically.  Target labels must be checked and violations of legal semantics should be reported to the user.

This is the detailed design document on how these labels will be semantically checked.  Legal semantics may result in rewrite operations on the syntax tree.  Semantics violations will be reported as errors to the user.

## Requirements

- Input: a parse tree that decomposes the Fortran program unit
- Output:
   * **Success** returns true
     (Additionally, the parse tree may be rewritten on success to capture the nested DO loop structure explicitly from any _label-do-stmt_ type loops.)
   * **Failure** returns false, instantiates (a container of) error message(s) to indicate the problem(s)


### Label generalities (6.2.5)

Enforcement of the general label constraints.  There are three sorts of label usage. Labels can serve 
   1. as a _label-do-stmt_ block range marker
   1. as branching (control flow) targets
   1. as specification annotations (`FORMAT` statements) for data transfer statements (I/O constructs)

Labels are related to the standard definition of inclusive scope.  For example, control-flow arcs are not allowed to originate from one inclusive scope and target statements outside of that inclusive scope.

Inclusive scope is defined as a tree structure of nested scoping constructs. A statement, _s_, is said to be *in* the same inclusive scope as another statement, _t_, if and only if _s_ and _t_ are in the same scope or _t_ is in one of the enclosing scopes of _s_, otherwise _s_ is *not in* the same inclusive scope as _t_. (Inclusive scope is unidirectional and is always from innermost scopes to outermost scopes.)

#### Semantic Checks

- labels range from 1 to 99999, inclusive (6.2.5 note 2)
  * handled automatically by the parser, but add a range check
- labels must be pairwise distinct within their program unit scope (6.2.5 para 2)
  * if redundant labels appear &rarr; error redundant labels
  * the total number of unique statement labels may have a limit


### Labels Used for `DO` Loop Ranging

#### _label-do-stmt_ (R1121)

A _label-do-stmt_ is a control construct that results in the iterative execution of a number of statements.  A _label-do-stmt_ has a (possibly shared, _nonblock-do-construct_) _label_ that will be called the loop target label.  The statements to be executed will be the range from the _label-do-stmt_ to the statement identified by the loop target label, inclusive. This range of statements will be called the loop's body and logically forms a _do-block_.

A _label-do-stmt_ is quite similar to a _block-do-construct_ in semantics, but the parse tree is different in that the parser does not impose a _do-block_ structure on the loop body.

In F18, the nonblock `DO` construct has been removed.  For legacy support (through F08), we will need to handle nonblock `DO` constructs.  In F18, the following legacy code is an error.

```fortran
  DO 100 I = 1, 100
    DO 100 J = 1, 100
      ...
 100 CONTINUE
```

##### Semantic Checks

- the loop body target label must exist in the scope (F18:C1133; F08:C815, C817, C819)
  * if the label does not appear, error of missing label
- the loop body target label must be, lexically, after the _label-do-stmt_ (R1119)
  * if the label appears lexically preceding the `DO`, error of malformed `DO`
- control cannot transfer into the body from outside the _do-block_
  * Exceptions (errors demoted to warnings)
    - some implementations relax enforcement of this and allow `GOTO`s from the loop body to "extended ranges" and back again (PGI & gfortan appear to allow, NAG & Intel do not.)
    - should some form of "extended ranges" for _do-constructs_ be supported, it should still be limited and not include parallel loops such as `DO CONCURRENT` or loops annotated with OpenACC or OpenMP directives.
  * `GOTO`s into the `DO`s inclusive scope, error/warn of invalid transfer of control
- requires that the loop terminating statement for a _label-do-stmt_ be either an `END DO` or a `CONTINUE`
  * Exception
    - earlier standards allowed other statements to be terminators

Semantics for F08 and earlier that support sharing the loop terminating statement in a _nonblock-do-construct_ between multiple loops
- some statements cannot be _do-term-action-stmt_ (F08:C816)
  * a _do-term-action-stmt_ is an _action-stmt_ but does not include _arithmetic-if-stmt_, _continue-stmt_, _cycle-stmt_, _end-function-stmt_, _end-mp-subprogram-stmt_, _end-program-stmt_, _end-subroutine-stmt_, _error-stop-stmt_, _exit-stmt_, _goto-stmt_, _return-stmt_, or _stop-stmt_
    - if the term action statement is forbidden, error invalid statement in `DO` loop term position
- some statements cannot be _do-term-shared-stmt_ (F08:C818)
  * this is the case as in our above example where two different nested loops share the same terminating statement (`100 continue`)
  * a _do-term-shared-stmt_ is an _action-stmt_ with all the same exclusions as a _do-term-action-stmt_ except a _continue-stmt_ **is** allowed
    - if the term shared action statement is forbidden, error invalid statement in term position

If the `DO` loop is a `DO CONCURRENT` construct, there are additional constraints (11.1.7.5).
- a _return-stmt_ is not allowed (C1136)
- image control statements are not allowed (C1137)
- branches must be from a statement and to a statement that both reside within the `DO CONCURRENT` (C1138)
- impure procedures shall not be called (C1139)
- deallocation of polymorphic objects is not allowed (C1140)
- references to `IEEE_GET_FLAG`, `IEEE_SET_HALTING_MODE`, and `IEEE_GET_HALTING_MODE` cannot appear in the body of a `DO CONCURRENT` (C1141)
- the use of the `ADVANCE=` specifier by an I/O statement in the body of a `DO CONCURRENT` is not allowed (11.1.7.5, para 5)

### Labels Used in Branching

#### _goto-stmt_ (11.2.2, R1157)

A `GOTO` statement is a simple, direct transfer of control from the `GOTO` to the labelled statement.

##### Semantic Checks

- the labelled statement that is the target of a `GOTO` (11.2.1 constraints)
  - must refer to a label that is in inclusive scope of the computed `GOTO` statement (C1169)
    * if a label does not exist, error nonexistent label
    * if a label is out of scope, error out of inclusive scope
  - the branch target statement must be valid
    * if the statement is not allowed as a branch target, error not a valid branch target
- the labelled statement must be a branch target statement
  * a branch target statement is any of _action-stmt_, _associate-stmt_, _end-associate-stmt_, _if-then-stmt_, _end-if-stmt_, _select-case-stmt_, _end-select-stmt_, _select-rank-stmt_, _end-select-rank-stmt_, _select-type-stmt_, _end-select-type-stmt_, _do-stmt_, _end-do-stmt_, _block-stmt_, _end-block-stmt_, _critical-stmt_, _end-critical-stmt_, _forall-construct-stmt_, _forall-stmt_, _where-construct-stmt_, _end-function-stmt_, _end-mp-subprogram-stmt_, _end-program-stmt_, or _end-subroutine-stmt_. (11.2.1)
  * Some deleted features that were _action-stmt_ in older standards include _arithmetic-if-stmt_, _assign-stmt_, _assigned-goto-stmt_, and _pause-stmt_. For legacy mode support, these statements should be considered _action-stmt_.


#### _computed-goto-stmt_ (11.2.3, R1158)

The computed `GOTO` statement is analogous to a `switch` statement in C++.

```fortran
  GOTO ( label-list ) [,] scalar-int-expr
```

##### Semantics Checks

- each label in _label-list_ (11.2.1 constraints, same as `GOTO`)
  - must refer to a label that is in inclusive scope of the computed `GOTO` statement (C1170)
    * if a label does not exist, error nonexistent label
    * if a label is out of scope, error out of inclusive scope
  - the branch target statement must be valid
    * if the statement is not allowed as a branch target, error not a valid branch target
- the _scalar-int-expr_ needs to have `INTEGER` type
  * check the type of the expression (type checking done elsewhere)


#### R853 _arithmetic-if-stmt_ (F08:8.2.4)

This control-flow construct is deleted in F18.

```fortran
  IF (scalar-numeric-expr) label1,label2,label3
```

The arithmetic if statement is like a three-way branch operator. If the scalar numeric expression is less than zero goto _label-1_, else if the variable is equal to zero goto _label-2_, else if the variable is greater than zero goto _label-3_.

##### Semantics Checks

- the labels in the _arithmetic-if-stmt_ triple must all be present in the inclusive scope (F08:C848)
  * if a label does not exist, error nonexistent label
  * if a label is out of scope, error out of inclusive scope
- the _scalar-numeric-expr_ must not be `COMPLEX` (F08:C849)
  * check the type of the expression (type checking done elsewhere)


#### _alt-return-spec_ (15.5.1, R1525)

These are a Fortran control-flow construct for combining a return from a subroutine with a branch to a labelled statement in the calling routine all in one operation. A typical implementation is for the subroutine to return a hidden integer, which is used as a key in the calling code to then, possibly, branch to a labelled statement in inclusive scope.

The labels are passed by the calling routine. We want to check those labels at the call-site, that is instances of _alt-return-spec_.

##### Semantics Checks

- each _alt-return-spec_ (11.2.1 constraints, same as `GOTO`)
  - must refer to a label that is in inclusive scope of the `CALL` statement
    * if a label does not exist, error nonexistent label
    * if a label is out of scope, error out of inclusive scope
  - the branch target statement must be valid
    * if the statement is not allowed as a branch target, error not a valid branch target


#### **END**, **EOR**, **ERR** specifiers (12.11)

These specifiers can appear in I/O statements and can transfer control to specific labelled statements under exceptional conditions like end-of-file, end-of-record, and other error conditions.  (The PGI compiler adds code to test the results from the runtime routines to determine if these branches should take place.)

##### Semantics Checks

- each END, EOR, and ERR specifier (11.2.1 constraints, same as `GOTO`)
  - must refer to a label that is in inclusive scope of the I/O statement
    * if a label does not exist, error nonexistent label
    * if a label is out of scope, error out of inclusive scope
  - the branch target statement must be valid
    * if the statement is not allowed as a branch target, error not a valid branch target

#### _assigned-goto-stmt_ and _assign-stmt_ (F90:8.2.4)

Deleted feature since Fortran 95.

The _assigned-goto-stmt_ and _assign-stmt_ were _action-stmt_ in the Fortran 90 standard. They are included here for completeness. This pair of obsolete statements can (will) be enabled as part of the compiler's legacy Fortran support.

The _assign-stmt_ stores a _label_ in an integer variable.  The _assigned-goto-stmt_ will then transfer control to the _label_ stored in that integer variable.

```fortran
  ASSIGN 10 TO i
  ...
  GOTO i (10,20,30)
```

##### Semantic Checks

- an _assigned-goto-stmt_ cannot be a _do-term-action-stmt_ (F90:R829)
- an _assigned-goto-stmt_ cannot be a _do-term-shared-stmt_ (F90:R833)
- constraints from (F90:R839)
  - each _label_ in an optional _label-list_ must be the statement label of a branch target statement that appears in the same scoping unit as the _assigned-goto-stmt_
  - _scalar-int-variable_ (`i` in the example above) must be named and of type default integer
  - an integer variable that has been assigned a label may only be referenced in an _assigned-goto_ or as a format specifier in an I/O statement
  - when an I/O statement with a _format-specifier_ that is an integer variable is executed or when an _assigned-goto_ is executed, the variable must have been assigned a _label_
  - an integer variable can only be assigned a label via the `ASSIGN` statement
  - the label assigned to the variable must be in the same scoping unit as the _assigned-goto_ that branches to the _label_ value
  - if the parameterized list of labels is present, the label value assigned to the integer variable must appear in that _label-list_
  - a distinct _label_ can appear more than once in the _label-list_

Some interpretation is needed as the terms of the older standard are different.

A "scoping unit" is defined as
  - a derived-type definition
  - a procedure interface body, excluding derived-types and interfaces contained within it
  - a program unit or subprogram, excluding derived-types, interfaces, and subprograms contained within it

This is a more lax definition of scope than inclusive scope.

A _named variable_ distinguishes a variable such as, `i`, from an element of an array, `a(i)`, for example.

### Labels used in I/O

#### Data transfer statements

In data transfer (I/O) statements (e.g., `READ`), the user can specify a `FMT=` specifier that can take a label as its argument. (R1215)

##### Semantic Checks

- if the `FMT=` specifier has a label as its argument (C1230)
  - the label must correspond to a `FORMAT` statement
    * if the statement is not a `FORMAT`, error statement must be a `FORMAT`
  - the labelled `FORMAT` statement must be in the same inclusive scope as the originating data transfer statement (also in 2008)
    * if the label statement does not exist, error label does not exist
    * if the label statement is not in scope, error label is not in inclusive scope
  - Exceptions (errors demoted to warnings)
    - PGI extension: referenced `FORMAT` statements may appear in a host procedure
    - Possible relaxation: the scope of the referenced `FORMAT` statement may be ignored, allowing a `FORMAT` to be referenced from any scope in the compilation.

### Construct Name generalities

Various Fortran constructs can have names. These include
  - the `WHERE` construct (10.2.3)
  - the `FORALL` construct (10.2.4)
  - the `ASSOCIATE` construct (11.1.3)
  - the `BLOCK` construct (11.1.4)
  - the `CHANGE TEAM` construct (11.1.5)
  - the `CRITICAL` construct (11.1.6)
  - the `DO` construct (11.1.7)
  - the `IF` construct (11.1.8)
  - the `SELECT CASE` construct (11.1.9)
  - the `SELECT RANK` construct (11.1.10)
  - the `SELECT TYPE` construct (11.1.11)

#### Semantics Checks

A construct name is a name formed under 6.2.2.  A name is an identifier. Identifiers are parsed by the parser.
  - the maximum length of a name is 63 characters (C601)

Names must either not be given for the construct or used throughout when specified.
- if a construct is given a name, the construct's `END` statement must also specify the same name (`WHERE` C1033, `FORALL` C1035, ...)
- `WHERE` has additional `ELSEWHERE` clauses
- `IF` has additional `ELSE IF` and `ELSE` clauses
- `SELECT CASE` has additional `CASE` clauses
- `SELECT RANK` has additional `RANK` clauses
- `SELECT TYPE` has additional _type-guard-stmt_
These additional statements must meet the same constraint as the `END` of the construct. Names must match, if present, or there must be no names for any of the clauses.

### `CYCLE` statement (11.1.7.4.4)

The `CYCLE` statement takes an optional _do-construct-name_.

#### Semantics Checks

- if the `CYCLE` has a _construct-name_, then the `CYCLE` statement must appear within that named _do-construct_ (C1134)
- if the `CYCLE` does not have a _do-construct-name_, the `CYCLE` statement must appear within a _do-construct_ (C1134)

### `EXIT` statement (11.1.12)

The `EXIT` statement takes an optional _construct-name_.

#### Semantics Checks

- if the `EXIT` has a _construct-name_, then the `EXIT` statement must appear within that named construct (C1166)
- if the `EXIT` does not have a _construct-name_, the `EXIT` statement must appear within a _do-construct_ (C1166)
- an _exit-stmt_ must not appear in a `DO CONCURRENT` if the `EXIT` belongs to the `DO CONCURRENT` or an outer construct enclosing the `DO CONCURRENT` (C1167)
- an _exit-stmt_ must not appear in a `CHANGE TEAM` (`CRITICAL`) if the `EXIT` belongs to an outer construct enclosing the `CHANGE TEAM` (`CRITICAL`) (C1168)

