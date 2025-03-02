<!--===- docs/Directives.md 
  
   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  
-->

# Compiler directives supported by Flang

A list of non-standard directives supported by Flang

* `!dir$ fixed` and `!dir$ free` select Fortran source forms.  Their effect
  persists to the end of the current source file.
* `!dir$ ignore_tkr [[(TKRDMAC)] dummy-arg-name]...` in an interface definition
  disables some semantic checks at call sites for the actual arguments that
  correspond to some named dummy arguments (or all of them, by default).
  The directive allow actual arguments that would otherwise be diagnosed
  as incompatible in type (T), kind (K), rank (R), CUDA device (D), or
  managed (M) status.  The letter (A) is a shorthand for all of these,
  and is the default when no letters appear.  The letter (C) checks for
  contiguity for example allowing an element of an assumed-shape array to be
  passed as a dummy argument. For example, if one wanted to call a "set all
  bytes to zero" utility that could be applied to arrays of any type or rank:
```
  interface
    subroutine clear(arr,bytes)
!dir$ ignore_tkr arr
      integer(1), intent(out) :: arr(bytes)
    end
  end interface
```
* `!dir$ assume_aligned desginator:alignment`, where designator is a variable,
  maybe with array indices, and alignment is what the compiler should assume the
  alignment to be. E.g A:64 or B(1,1,1):128. The alignment should be a power of 2,
  and is limited to 256.
  [This directive is currently recognised by the parser, but not
  handled by the other parts of the compiler].
* `!dir$ vector always` forces vectorization on the following loop regardless
  of cost model decisions. The loop must still be vectorizable.
  [This directive currently only works on plain do loops without labels].
* `!dir$ unroll [n]` specifies that the compiler ought to unroll the immediately
  following loop `n` times. When `n` is `0` or `1`, the loop should not be unrolled
  at all. When `n` is `2` or greater, the loop should be unrolled exactly `n`
  times if possible. When `n` is omitted, the compiler should attempt to fully
  unroll the loop. Some compilers accept an optional `=` before the `n` when `n`
  is present in the directive. Flang does not.
* `!dir$ unroll_and_jam [N]` control how many times a loop should be unrolled and
  jammed. It must be placed immediately before a loop that follows. `N` is an optional 
  integer that specifying the unrolling factor. When `N` is `0` or `1`, the loop 
  should not be unrolled at all. If `N` is omitted the optimizer will
  selects the number of times to unroll the loop.

# Directive Details

## Introduction
Directives are commonly used in Fortran programs to specify additional actions 
to be performed by the compiler. The directives are always specified with the 
`!dir$` or `cdir$` prefix.

## Loop Directives

Some directives are associated with the following construct, for example loop
directives. Directives on loops are used to specify additional transformation to
be performed by the compiler like enabling vectorisation, unrolling, interchange
etc.

Currently loop directives are not accepted in the presence of OpenMP or OpenACC
constructs on the loop. This should be implemented as it is used in some
applications.

### Array Expressions
It is to be decided whether loop directives should also be able to be associated
with array expressions.

## Semantics
Directives that are associated with constructs must appear in the same section
as the construct they are associated with, for example loop directives must
appear in the executable section as the loops appear there. To facilitate this
the parse tree is corrected to move such directives that appear in the
specification part into the execution part.

When a directive that must be associated with a construct appears, a search
forward from that directive to the next non-directive construct is performed to
check that that construct matches the expected construct for the directive.
Skipping other intermediate directives allows multiple directives to appear on
the same construct.

## Lowering 
Evaluation is extended with a new field called dirs for representing directives
associated with that Evaluation. When lowering loop directives, the associated
Do Loop's evaluation is found and the directive is added to it. This information
is used only during the lowering of the loop.

### Representation in LLVM
The `llvm.loop` metadata is used in LLVM to provide information to the optimizer
about the loop. For example, the `llvm.loop.vectorize.enable` metadata informs
the optimizer that a loop can be vectorized without considering its cost-model.
This attribute is added to the loop condition branch.

### Representation in MLIR 
The MLIR LLVM dialect models this by an attribute called LoopAnnotation
Attribute. The attribute can be added to the latch of the loop in the cf
dialect and is then carried through lowering to the LLVM dialect.

## Testing
Since directives must maintain a flow from source to LLVM IR, an integration
test is provided that tests the `vector always` directive, as well as individual
lit tests for each of the parsing, semantics and lowering stages.
