<!--===- docs/ControlFlowGraph.md 
  
   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  
-->

# Control Flow Graph

```{contents}
---
local:
---
```

## Concept
After a Fortran subprogram has been parsed, its names resolved, and all its
semantic constraints successfully checked, the parse tree of its
executable part is translated into another abstract representation,
namely the _control flow graph_ described in this note.

This second representation of the subprogram's executable part is
suitable for analysis and incremental modification as the subprogram
is readied for code generation.
Many high-level Fortran features are implemented by rewriting portions
of a subprogram's control flow graph in place.

### Control Flow Graph
A _control flow graph_ is a collection of simple (_i.e.,_ "non-extended")
basic _blocks_ that comprise straight-line sequences of _actions_ with a
single entry point and a single exit point, and a collection of
directed flow _edges_ (or _arcs_) denoting all possible transitions of
control flow that may take place during execution from the end of
one basic block to the beginning of another (or itself).

A block that has multiple distinct successors in the flow of control
must end with an action that selects its successor.

The sequence of actions that constitutes a basic block may
include references to user and library procedures.
Subprogram calls with implicit control flow afterwards, namely
alternate returns and `END=`/`ERR=` labels on input/output,
will be lowered in translation to a representation that materializes
that control flow into something similar to a computed `GO TO` or
C language `switch` statement.

For convenience in optimization and to simplify the implementation of
data flow confluence functions, we may choose to maintain the
property that each flow arc is the sole outbound arc emanating from
its originating block, the sole inbound arc arriving at its destination,
or both.
Empty blocks would inserted to "split" arcs when necessary to maintain this
invariant property.

Fortran subprograms (other than internal subprograms) can have multiple
entry points by using the obsolescent `ENTRY` statement.
We will implement such subprograms by constructing a union
of their dummy argument lists and using it as part of the definition
of a new subroutine or function that can be called by each of
the entry points, which are then all converted into wrapper routines that
pass a selector value as an additional argument to drive a `switch` on entry
to the new subprogram.

This transformation ensures that every subprogram's control
flow graph has a well-defined `START` node.

Statement labels can be used in Fortran on any statement, but only
the labels that decorate legal destinations of `GO TO` statements
need to be implemented in the control flow graph.
Specifically, non-executable statements like `DATA`, `NAMELIST`, and
`FORMAT` statements will be extracted into data initialization
records before or during the construction of the control flow
graph, and will survive only as synonyms for `CONTINUE`.

Nests of multiple labeled `DO` loops that terminate on the same
label will be have that label rewritten so that `GO TO` within
the loop nest will arrive at the copy that most closely nests
the context.
The Fortran standard does not require us to do this, but XLF
(at least) works this way.

### Expressions and Statements (Operations and Actions)
Expressions are trees, not DAGs, of intrinsic operations,
resolved function references, constant literals, and
data designators.

Expression nodes are represented in the compiler in a type-safe manner.
There is a distinct class or class template for every category of
intrinsic type, templatized over its supported kind type parameter values.

Operands are storage-owning indirections to other instances
of `Expression`, instances of constant values, and to representations
of data and function references.
These indirections are not nullable apart from the situation in which
the operands of an expression are being removed for use elsewhere before
the expression is destructed.

The ranks and the extents of the shapes of the results of expressions
are explicit for constant arrays and recoverable by analysis otherwise.

Parenthesized subexpressions are scrupulously preserved in accordance with
the Fortran standard.

The expression tree is meant to be a representation that is
as equally well suited for use in the symbol table (e.g., for
a bound of an explicit shape array) as it is for an action
in a basic block of the control flow graph (e.g., the right
hand side of an assignment statement).

Each basic block comprises a linear sequence of _actions_.
These are represented as a doubly-linked list so that insertion
and deletion can be done in constant time.

Only the last action in a basic block can represent a change
to the flow of control.

### Scope Transitions
Some of the various scopes of the symbol table are visible in the control flow
graph as `SCOPE ENTRY` and `SCOPE EXIT` actions.
`SCOPE ENTRY` actions are unique for their corresponding scopes,
while `SCOPE EXIT` actions need not be so.
It must be the case that
any flow of control within the subprogram will enter only scopes that are
not yet active, and exit only the most recently entered scope that has not
yet been deactivated; i.e., when modeled by a push-down stack that is
pushed by each traversal of a `SCOPE ENTRY` action,
the entries of the stack are always distinct, only the scope at
the top of the stack is ever popped by `SCOPE EXIT`, and the stack is empty
when the subprogram terminates.
Further, any references to resolved symbols must be to symbols whose scopes
are active.

The `DEALLOCATE` actions and calls to `FINAL` procedures implied by scoped
lifetimes will be explicit in the sequence of actions in the control flow
graph.

Parallel regions might be partially represented by scopes, or by explicit
operations similar to the scope entry and exit operations.

### Data Flow Representation
The subprogram text will be in static single assignment form by the time the
subprogram arrives at the bridge to the LLVM IR builder.
Merge points are actions at the heads of basic blocks whose operands
are definition points; definition points are actions at the ends of
basic blocks whose operands are expression trees (which may refer to
merge points).

### Rewriting Transformations

#### I/O
#### Dynamic allocation
#### Array constructors

#### Derived type initialization, deallocation, and finalization
The machinery behind the complicated semantics of Fortran's derived types
and `ALLOCATABLE` objects will be implemented in large part by the run time
support library.

#### Actual argument temporaries
#### Array assignments, `WHERE`, and `FORALL`

Array operations have shape.

`WHERE` masks have shape.
Their effects on array operations are by means of explicit `MASK` operands that
are part of array assignment operations.

#### Intrinsic function and subroutine calls
