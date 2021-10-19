<!--===- docs/Overview.md 
  
   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  
-->

# Intro
This document goes briefly over compiler phases in Flang. It focuses on the
internal implementation and as such, it is intended for Flang developers rather
than end-users.

# Overview of Compiler Phases

```eval_rst
.. contents::
   :local:
```

Each phase produces either correct output or fatal errors.

## Prescan and Preprocess

See: [Preprocessing.md](Preprocessing.md).

**Input:** Fortran source and header files, command line macro definitions,
  set of enabled compiler directives (to be treated as directives rather than
  comments).

**Output:**
- A "cooked" character stream: the entire program as a contiguous stream of
  normalized Fortran source.
  Extraneous whitespace and comments are removed (except comments that are
  compiler directives that are not disabled) and case is normalized.
- Provenance information mapping each character back to the source it came from.
  This is used in subsequent phases to issue errors messages that refer to source locations.

**Entry point:** `parser::Parsing::Prescan`

**Command:** `flang-new -fc1 -E src.f90` dumps the cooked character stream

## Parse

**Input:** Cooked character stream.

**Output:** A parse tree representing a syntactically correct program,
  rooted at a `parser::Program`.
  See: [Parsing.md](Parsing.md) and [ParserCombinators.md](ParserCombinators.md).

**Entry point:** `parser::Parsing::Parse`

**Command:**
  - `flang-new -fc1 -fdebug-dump-parse-tree src.f90` dumps the parse tree
  - `flang-new -fc1 -fdebug-unparse src.f90` converts the parse tree to normalized Fortran

## Validate Labels and Canonicalize Do Statements

**Input:** Parse tree.

**Output:** The parse tree with label constraints and construct names checked,
  and each `LabelDoStmt` converted to a `NonLabelDoStmt`.
  See: [LabelResolution.md](LabelResolution.md).

**Entry points:** `semantics::ValidateLabels`, `parser::CanonicalizeDo`

## Resolve Names

**Input:** Parse tree (without `LabelDoStmt`) and `.mod` files from compilation
  of USEd modules.

**Output:**
- Tree of scopes populated with symbols and types
- Parse tree with some refinements:
  - each `parser::Name::symbol` field points to one of the symbols
  - each `parser::TypeSpec::declTypeSpec` field points to one of the types
  - array element references that were parsed as function references or
    statement functions are corrected

**Entry points:** `semantics::ResolveNames`, `semantics::RewriteParseTree`

**Command:** `flang-new -fc1 -fdebug-dump-symbols src.f90` dumps the
  tree of scopes and symbols in each scope

## Check DO CONCURRENT Constraints

**Input:** Parse tree with names resolved.

**Output:** Parse tree with semantically correct DO CONCURRENT loops.

## Write Module Files

**Input:** Parse tree with names resolved.

**Output:** For each module and submodule, a `.mod` file containing a minimal
  Fortran representation suitable for compiling program units that depend on it.
  See [ModFiles.md](ModFiles.md).

## Analyze Expressions and Assignments

**Input:** Parse tree with names resolved.

**Output:** Parse tree with `parser::Expr::typedExpr` filled in and semantic
  checks performed on all expressions and assignment statements.

**Entry points**: `semantics::AnalyzeExpressions`, `semantics::AnalyzeAssignments`

## Produce the Intermediate Representation

**Input:** Parse tree with names and labels resolved.

**Output:** An intermediate representation of the executable program.
  See [FortranIR.md](FortranIR.md).
