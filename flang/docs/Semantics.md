# Semantic Analysis

The semantic analysis pass determines if a syntactically correct Fortran
program is is legal by enforcing the constraints of the language.

The input is a parse tree with a `Program` node at the root;
and a "cooked" character stream, a contiguous stream of characters
containing a normalized form of the Fortran source.

The semantic analysis pass takes a parse tree for a syntactically
correct Fortran program and determines whether it is legal by enforcing
the constraints of the language.

If the program is not legal, the results of the semantic pass will be a list of
errors associated with the program.

If the program is legal, the semantic pass will produce a (possibly modified)
parse tree for the semantically correct program with each name mapped to a symbol
and each expression fully analyzed.

All user errors are detected either prior to or during semantic analysis.
After it completes successfully the program should compile with no error messages.
There may still be warnings or informational messages.

## Phases of Semantic Analysis

1. [Validate labels](#validate-labels) -
   Check all constraints on labels and branches
2. [Rewrite DO loops](#rewrite-do-loops) -
   Convert all occurrences of `LabelDoStmt` to `DoConstruct`.
3. [Name resolution](#name-resolution) -
   Analyze names and declarations, build a tree of Scopes containing Symbols,
   and fill in the `Name::symbol` data member in the parse tree
4. [Rewrite parse tree](#rewrite-parse-tree) -
   Fix incorrect parses based on symbol information
5. [Expression analysis](#expression-analysis) -
   Analyze all expressions in the parse tree and fill in `Expr::typedExpr` and
   `Variable::typedExpr` with analyzed expressions; fix incorrect parses
   based on the result of this analysis
6. [Statement semantics](#statement-semantics) -
   Perform remaining semantic checks on the execution parts of subprograms
7. [Write module files](#write-module-files) -
   If no errors have occurred, write out `.mod` files for modules and submodules

If phase 1 or phase 2 encounter an error on any of the program units,
compilation terminates. Otherwise, phases 3-6 are all performed even if
errors occur.
Module files are written (phase 7) only if there are no errors.

### Validate labels

Perform semantic checks related to labels and branches:
- check that any labels that are referenced are defined and in scope
- check branches into loop bodies
- check that labeled `DO` loops are properly nested
- check labels in data transfer statements

### Rewrite DO loops

This phase normalizes the parse tree by removing all unstructured `DO` loops
and replacing them with `DO` constructs.

### Name resolution

The name resolution phase walks the parse tree and constructs the symbol table.

The symbol table consists of a tree of `Scope` objects rooted at the global scope.
The global scope is owned by the `SemanticsContext` object.
It contains a `Scope` for each program unit in the compilation.

Each `Scope` in the scope tree contains child scopes representing other scopes
lexically nested in it.
Each `Scope` also contains a map of `CharBlock` to `Symbol` representing names
declared in that scope. (All names in the symbol table are represented as
`CharBlock` objects, i.e. as substrings of the cooked character stream.)

All `Symbol` objects are owned by the symbol table data structures.
They should be accessed as `Symbol *` or `Symbol &` outside of the symbol
table classes as they can't be created, copied, or moved.
The `Symbol` class has functions and data common across all symbols, and a
`details` field that contains more information specific to that type of symbol.
Many symbols also have types, represented by `DeclTypeSpec`.
Types are also owned by scopes.

Name resolution happens on the parse tree in this order:
1. Process the specification of a program unit:
   1. Create a new scope for the unit
   2. Create a symbol for each contained subprogram containing just the name
   3. Process the opening statement of the unit (`ModuleStmt`, `FunctionStmt`, etc.)
   4. Process the specification part of the unit
2. Apply the same process recursively to nested subprograms
3. Process the execution part of the program unit
4. Process the execution parts of nested subprograms recursively

After the completion of this phase, every `Name` corresponds to a `Symbol`
unless an error occurred.

### Rewrite parse tree

The parser cannot build a completely correct parse tree without symbol information.
This phase corrects mis-parses based on symbols:
- Array element assignments may be parsed as statement functions: `a(i) = ...`
- Namelist group names without `NML=` may be parsed as format expressions
- A file unit number expression may be parsed as a character variable

This phase also produces an internal error if it finds a `Name` that does not
have its `symbol` data member filled in. This error is suppressed if other
errors have occurred because in that case a `Name` corresponding to an erroneous
symbol may not be resolved.

### Expression analysis

Expressions that occur in the specification part are analyzed during name
resolution, for example, initial values, array bounds, type parameters.
Any remaining expressions are analyzed in this phase.

For each `Variable` and top-level `Expr` (i.e. one that is not nested below
another `Expr` in the parse tree) the analyzed form of the expression is saved
in the `typedExpr` data member. After this phase has completed, the analyzed
expression can be accessed using `semantics::GetExpr()`.

This phase also corrects mis-parses based on the result of expression analysis:
- An expression like `a(b)` is parsed as a function reference but may need
  to be rewritten to an array element reference (if `a` is an object entity)
  or to a structure constructor (if `a` is a derive type)
- An expression like `a(b:c)` is parsed as an array section but may need to be
  rewritten as a substring if `a` is an object with type CHARACTER

### Statement semantics

Multiple independent checkers driven by the `SemanticsVisitor` framework
perform the remaining semantic checks.
By this phase, all names and expressions that can be successfully resolved
have been. But there may be names without symbols or expressions without
analyzed form if errors occurred earlier.

### Write module files

Separate compilation information is written out on successful compilation
of modules and submodules. These are used as input to name resolution
in program units that `USE` the modules.

Module files are stripped down Fortran source for the module.
Parts that aren't needed to compile dependent program units (e.g. action statements)
are omitted.

The module file for module `m` is named `m.mod` and the module file for
submodule `s` of module `m` is named `m-s.mod`.
