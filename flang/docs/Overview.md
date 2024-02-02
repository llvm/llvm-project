<!--===- docs/Overview.md 
  
   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  
-->

# Overview of Compiler Phases

```{contents}
---
local:
---
```
The Flang compiler transforms Fortran source code into an executable file. 
This transformation proceeds in three high level phases -- analysis, lowering,
and code generation/linking.

The first high level phase (analysis) transforms Fortran source code into a
decorated parse tree and a symbol table.  During this phase, all user
related errors are detected and reported.

The second high level phase (lowering), changes the decorated parse tree and
symbol table into the Fortran Intermediate Representation (FIR), which is a
dialect of LLVM's Multi-Level Intermediate Representation or MLIR.  It then
runs a series of passes on the FIR code which verify its validity, perform a
series of optimizations, and finally transform it into LLVM's Intermediate
Representation, or LLVM IR

The third high level phase generates machine code and invokes a linker to
produce an executable file.

This document describes the first two high level phases.  Each of these is
described in more detailed phases.

Each detailed phase is described -- its inputs and outputs along with how to
produce a readable version of the outputs.

Each detailed phase produces either correct output or fatal errors.

## Analysis

This high level phase validates that the program is correct and creates all of
the information needed for lowering.

### Prescan and Preprocess

See [Preprocessing.md](Preprocessing.md).

**Input:** Fortran source and header files, command line macro definitions,
  set of enabled compiler directives (to be treated as directives rather than
  comments).

**Output:**
- A "cooked" character stream: the entire program as a contiguous stream of
  normalized Fortran source.
  Extraneous whitespace and comments are removed (except comments that are
  compiler directives that are not disabled) and case is normalized.  Also,
  directives are processed and macros expanded.
- Provenance information mapping each character back to the source it came from.
  This is used in subsequent phases that need source locations.  This includes
  error messages, optimization reports, and debugging information.

**Entry point:** `parser::Parsing::Prescan`

**Commands:** 
 - `flang-new -fc1 -E src.f90` dumps the cooked character stream
 - `flang-new -fc1 -fdebug-dump-provenance src.f90` dumps provenance
   information

### Parsing

**Input:** Cooked character stream

**Output:** A parse tree for each Fortran program unit in the source code
representing a syntactically correct program, rooted at the program unit.  See:
[Parsing.md](Parsing.md) and [ParserCombinators.md](ParserCombinators.md).

**Entry point:** `parser::Parsing::Parse`

**Commands:**
  - `flang-new -fc1 -fdebug-dump-parse-tree-no-sema src.f90` dumps the parse tree
  - `flang-new -fc1 -fdebug-unparse src.f90` converts the parse tree to normalized Fortran
  - `flang-new -fc1 -fdebug-dump-parsing-log src.f90` runs an instrumented parse and dumps the log
  - `flang-new -fc1 -fdebug-measure-parse-tree src.f90` measures the parse tree

### Semantic processing

**Input:** the parse tree, the cooked character stream, and provenance
information

**Output:** 
* a symbol table
* modified parse tree
* module files, (see: [ModFiles.md](ModFiles.md))
* the intrinsic procedure table
* the target characteristics
* the runtime derived type derived type tables (see: [RuntimeTypeInfo.md](RuntimeTypeInfo.md))

**Entry point:** `semantics::Semantics::Perform`

For more detail on semantic analysis, see: [Semantics.md](Semantics.md).
Semantic processing performs several tasks: 
* validates labels, see: [LabelResolution.md](LabelResolution.md).
* canonicalizes DO statements, 
* canonicalizes OpenACC and OpenMP code
* resolves names, building a tree of scopes and symbols
* rewrites the parse tree to correct parsing mistakes (when needed) once semantic information is available to clarify the program's meaning
* checks the validity of declarations
* analyzes expressions and statements, emitting error messages where appropriate
* creates module files if the source code contains modules, 
  see [ModFiles.md](ModFiles.md).

In the course of semantic analysis, the compiler:
* creates the symbol table
* decorates the parse tree with semantic information (such as pointers into the symbol table)
* creates the intrinsic procedure table
* folds constant expressions

At the end of semantic processing, all validation of the user's program is complete.  This is the last detailed phase of analysis processing.

**Commands:**
  - `flang-new -fc1 -fdebug-dump-parse-tree src.f90` dumps the parse tree after semantic analysis
  - `flang-new -fc1 -fdebug-dump-symbols src.f90` dumps the symbol table
  - `flang-new -fc1 -fdebug-dump-all src.f90` dumps both the parse tree and the symbol table

## Lowering

Lowering takes the parse tree and symbol table produced by analysis and
produces LLVM IR.

### Create the lowering bridge

**Inputs:** 
  - the parse tree
  - the symbol table
  - The default KINDs for intrinsic types (specified by default or command line option)
  - The intrinsic procedure table (created in semantics processing)
  - The target characteristics (created during semantics processing)
  - The cooked character stream
  - The target triple -- CPU type, vendor, operating system
  - The mapping between Fortran KIND values to FIR KIND values

The lowering bridge is a container that holds all of the information needed for lowering.

**Output:** A container with all of the information needed for lowering

**Entry point:** lower::LoweringBridge::create

### Initial lowering

**Input:** the lowering bridge

**Output:** A Fortran IR (FIR) representation of the program.

**Entry point:** `lower::LoweringBridge::lower`

The compiler then takes the information in the lowering bridge and creates a
pre-FIR tree or PFT.  The PFT is a list of programs and modules.  The programs
and modules contain lists of function-like units.  The function-like units
contain a list of evaluations.  All of these contain pointers back into the
parse tree.  The compiler walks the PFT generating FIR.

**Commands:**
  - `flang-new -fc1 -fdebug-dump-pft src.f90` dumps the pre-FIR tree
  - `flang-new -fc1 -emit-mlir src.f90` dumps the FIR to the files src.mlir

### Transformation passes

**Input:** initial version of the FIR code

**Output:** An LLVM IR representation of the program

**Entry point:** `mlir::PassManager::run`

The compiler then runs a series of passes over the FIR code.  The first is a
verification pass.  It's followed by a series of transformation passes that
perform various optimizations and transformations.  The final pass creates an
LLVM IR representation of the program.

**Commands:**
  - `flang-new -mmlir --mlir-print-ir-after-all -S src.f90` dumps the FIR code after each pass to standard error
  - `flang-new -fc1 -emit-llvm src.f90` dumps the LLVM IR to src.ll

## Object code generation and linking

After the LLVM IR is created, the flang driver invokes LLVM's existing
infrastructure to generate object code and invoke a linker to create the
executable file.
