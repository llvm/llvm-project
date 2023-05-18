<!--===- docs/FortranIR.md 
  
   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  
-->

# Design: Fortran IR

```{contents}
---
local:
---
```

## Introduction

After semantic analysis is complete and it has been determined that the compiler has a legal Fortran program as input, the parse tree will be lowered to an intermediate representation for the purposes of high-level analysis and optimization.  In this document, that intermediate representation will be called Fortran IR or FIR.  The pass that converts from the parse tree and other data structures of the front-end to FIR will be called the "Burnside bridge".

FIR will be an explicit, operational, and strongly-typed representation, which shall encapsulate control-flow as graphs.

## Requirements

### White Paper: [Control Flow Graph](ControlFlowGraph.md)<sup>1</sup>

This is a list of requirements extracted from that document, which will be referred to as CFG-WP.

1. Control flow to be explicit (e.g. ERR= specifiers)
2. May need critical edge splitting
3. Lowering of procedures with ENTRY statements is specified
4. Procedures will have a start node
5. Non-executable statements will be ignored
6. Labeled DO loop execution with GOTO specified
7. Operations and actions (statements) are defined
8. The last statement in a basic block can represent a change in control flow
9. Scope transitions to be made explicit (special actions)
10. The IR will be in SSA form

### Explicit Control Flow

In Fortran, there are a number of statements that result in control flow to statements other than the one immediately subsequent. These can be sorted these into two categories: structured and unstructured.

#### Structured Control Flow

Fortran has executable constructs that imply three basic control flow forms.  The first form is a structured loop (DO construct)<sup>2</sup>. The second form is a structured cascade of conditional branches (IF construct, IF statement,<sup>3</sup> WHERE construct).  The third form is a structured multiway branch (SELECT CASE, SELECT RANK, and SELECT TYPE constructs).  The FORALL construct, while it implies a semantic model of interleaved iterations, can be modeled as a special single-entry single-exit region in FIR perhaps with backstage marker statements.<sup>4</sup>

The CYCLE and EXIT statements interact with the above structured executable constructs by providing structured transfers of control.<sup>5</sup> CYCLE (possibly named) is only valid in DO constructs and creates an alternate backedge in the enclosing loop.  EXIT transfers control out of the enclosing (possibly named) construct, which need not be a DO construct.

#### Unstructured Control Flow

Fortran also has mechanisms of transferring control between a statement and another statement with a corresponding label.  The origin of these edges can be GOTO statements, computed GOTO statements, assigned GOTO statements, arithmetic IF statements, alt-return specifications, and END/EOR/ERR I/O specifiers.  These statements are "unstructured" in the sense that the target of the control-flow has fewer constraints and the labelled statements must be linked to their origins.

Another category of unstructured control flow are statements that terminate execution.  These include RETURN, FAIL IMAGE, STOP and ERROR STOP statements.  The PAUSE statement can be modeled as a call to the runtime.

### Operations

The compiler's to be determined optimization passes will inform us as to the exact composition of FIR at the operations level.  This details here will necessarily change, so please read them with a grain of salt.

The plan (see CFG-WP) is that statements (actions) will be a veneer model of Fortran syntactical executable constructs. Fortran statements will correspond one to one with actions. Actions will be composed of and own objects of Fortran::evaluate::GenericExprWrapper. Values of type GenericExprWrapper will have Fortran types. This implies that actions will not be in an explicit data flow representation and have optional type information.<sup>6</sup> Initially, values will bind to symbols in a context and have an implicit use-def relation. An action statement may entail a "big step" operation with many side-effects. No semantics has been defined at this time.  Actions may reference other non-executable statements from the parse tree in some to be determined manner.

From the CFG-WP, it is stated that the FIR will ultimately be in an SSA form.  It is clear that a later pass can rewrite the values/expressions and construct a factored use-def version of the expressions. This may/should also involve expanding "big step" actions to a series of instructions and introducing typing information for all instructions. Again, the exact "lowered representation" will be informed from the requirements of the optimization passes and is presently to be determined.

### Other

Overall project goals include becoming part of the LLVM ecosystem as well as using LLVM as a backend.

Critical edge splitting can be constructed on-demand and as needed.

Lowering of procedures with ENTRY statements is specified.  The plan is to lower procedures with ENTRY statements as specified in the CFG-WP.

In FIR, a procedure will have a method that returns the start node.

When lowering to FIR statements, non-executable statements will be discarded.

Labeled DO loops are converted to non-labeled DO loops in the semantics processing.

The last statement in a basic block can represent a change in control flow. LLVM-IR and SIL<sup>7</sup> require that basic blocks end with a terminator. FIR will also have terminators.

The CFG-WP states that scope transitions are to be made explicit. We will cover this more below.

LLVM does not require the FIR to be in SSA form. LLVM's mem-to-reg pass does the conversion into SSA form. FIR can support SSA for optimization passes on-demand with its own mem-to-reg and reg-to-mem type passes.

Data objects with process lifetime will be captured indirectly by a reference to the (global) symbol table.

## Exploration

### Construction

Our aim to construct a CFG where all control-flow is explicitly modeled by relations. A basic block will be a sequence of statements for which if the first statement is executed then all other statements in the basic block will also be executed, in order.<sup>8</sup>  A CFG is therefore this set of basic blocks and the control-flow relations between those blocks.

#### Alternative: direct approach

The CFG can be directly constructed by traversing the parse tree, threading contextual state, and building basic blocks along with control-flow relationships.

* Pro: Straightforward implementation when control-flow is well-structured as the contextual state parallels the syntax of the language closely.
* Con: The contextual state needed can become large and difficult to manage in the presence of unstructured control-flow. For example, not every labeled statement in Fortran may be a control-flow destination.
* Con: The contextual state must deal with the recursive nature of the parse tree. 
* Con: Complexity. Since structured constructs cohabitate with unstructured constructs, the context needs to carry information about all combinations until the basic blocks and relations are fully elaborated.

#### Alternative: linearized approach (decomposing the problem)

Instead of constructing the CFG directly from a parse tree traversal, an intermediate form can be constructed to explicitly capture the executable statements, which ones give rise to control-flow graph edge sources, and which are control-flow graph edge targets.  This linearized form flattens the tree structure of the parse tree. The linearized form does not require recursive visitation of nested constructs and can be used to directly identify the entries and exits of basic blocks.

While each control-flow source statement is explicit in the traversal, it can be the case that not all of the targets have been traversed yet (references to forward basic blocks), and those basic blocks will not yet have been created.  These relations can be captured at the time the source is traversed, added to a to do list, and then completed when all the basic blocks for the procedure have been created. Specifically, at the point when we create a terminator all information is known to create the FIR terminator, however all basic blocks that may be referenced may not have been created. Those are resolved in one final "clean up" pass over a list of closures.

* Con: An extra representation must be defined and constructed.  
* Pro: This representation reifies all the information that is referred to as contextual state in the direct approach.
* Pro: Constructing the linearized form can be done with a simple traversal of the parse tree.
* Pro: Once composed the linearized form can be traversed and a CFG directly constructed.  This greatly reduces bookkeeping of contextual state.

### Details

#### Grappling with Control Flow

Above, various Fortran executable constructs were discussed with respect to how they (may) give rise to control flow.  These Fortran statements are mapped to a small number of FIR statements: ReturnStmt, BranchStmt, SwitchStmt, IndirectBrStmt, and UnreachableStmt.

_ReturnStmt_: execution leaves the enclosing Procedure. A ReturnStmt can return an optional value. This would appear for RETURN statements or at END SUBROUTINE.

_BranchStmt_: execution of the current basic block ends. If the branch is unconditional then control transfers to exactly one successor basic block. If the branch is conditional then control transfers to exactly one of two successor blocks depending on the true/false value of the condition. All successors must be in the current Procedure. Unconditional branches would appear for GOTO statements. Conditional branches would appear for IF constructs, IF statements, etc.

_SwitchStmt_: Exactly one of multiple successors is selected based on the control expression. Successors are pairs of case expressions and basic blocks.  If the control expression compares to the case expression and returns true, then that control transfers to that block. There may be one special block, the default block, that is selected if none of the case expressions compares true. This would appear for SELECT CASE, SELECT TYPE, SELECT RANK, COMPUTED GOTO, WRITE with exceptional condition label specificers, alternate return specifiers, etc.

_IndirectBrStmt_: A variable is loaded with the address of a basic block in the containing Procedure. Control is transferred to the contents of this variable. An IndirectBrStmt also requires a complete list of potential basic blocks that may be loaded into the variable. This would appear for ASSIGNED GOTO.

Supporting ASSIGNED GOTO offers a little extra challenge as the ASSIGN GOTO statement's list of target labels is optional.  If that list is not present, then the procedure must be analyzed to find ASSIGN statements. The implementation proactively looks for ASSIGN statements and keeps a dictionary mapping an assigned Symbol to its set of targets. When constructing the CFG, ASSIGNED GOTOs can be processed as to potential targets either from the list provided in the ASSIGNED GOTO or from the analysis pass.

Alternatively, ASSIGNED GOTO could be implemented as a _SwitchStmt_ that tests on a compiler-defined value and fully elaborates all potential target basic blocks.

_UnreachableStmt_: If control reaches an unreachable statement, then an error has occurred. Calls to library routines that do not return should be followed by an UnreachableStmt.  An example would be the STOP statement.

#### Scope

In the CFG-WP, scopes are meant to be captured by a pair of backstage statements for entering and exiting a particular scope. In structured code, these pairs would not be problematic; however, control flow in Fortran is ad hoc, particularly in legacy Fortran. In short, Fortran does not have a clean sense of structure with respect to scope.

To separate concerns, FIR will construct the ad hoc CFG and impose bounding boxes over regions of that graph to demarcate and superimpose scope structures on that CFG. Any GOTO-like statements that are side-entries and side-exits to the region will be explicit.

Once the basic blocks are constructed, CFG edges defined, and the CFG is simplified, a simple pass that analyzes the region bounding boxes can decorate the basic blocks with the SCOPE ENTER and SCOPE EXIT statements and flatten/remove the region structure. It will then be the burden of any optimization passes to guarantee legal orderings of SCOPE ENTER and SCOPE EXIT pairs.

* Pro: Separation of concerns allows for simpler, easier to maintain code
* Pro: Simplification of the CFG can be done without worrying about SCOPE markers
* Pro: Allows a precise superimposing of all Fortran constructs with scoping considerations over an otherwise ad hoc CFG.
* Con: Adds "an extra layer" to FIR as compared to SIL. However, that can be mitigated/made inconsequential by a pass that flattens the Region tree and inserts the backstage SCOPE marker statements.

#### Structure

_Program_: A program instance is the top-level object that contains the representation of all the code being compiled, the compilation unit. It contains a list of procedures and a reference to the global symbol table.

_Procedure_: This is a named Fortran procedure (subroutine or function). It contains a (hierarchical) list of regions. It also owns a list of all basic blocks for the procedure.

_Region_: A region is owned by a procedure or by another region. A region owns a reference to a scope in the symbol table tree. The list of delineated basic blocks can also be requested from a region.

_Basic block_: A basic block is owned by a procedure. A basic block owns a list of statements. The last statement in the list must be a terminator, and no other statement in the list can be a terminator. A basic block owns a list of its predecessors, which are also basic blocks. (Precisely, it is this level of FIR that is the CFG.)

_Statement_: An executable Fortran construct that owns/refers to expressions, symbols, scopes, etc. produced by the front-end.

_Terminator_: A statement that orchestrates control-flow. Terminator statements may reference other basic blocks and can be accessed by their parent basic block to discover successor blocks, if any.

#### Support

Since there is some state that needs to be maintained and forwarded as the FIR is constructed, a FIRBuilder can be used for convenience.  The FIRBuilder constructs statements and updates the CFG accordingly.

To support visualization, there is a support class to dump the FIR to a dotty graph.

### Data Structures

FIR is intentionally similar to SIL from the statement level up to the level of a program.

#### Alternative: LLVM

Program, procedure, region, and basic block all leverage code from LLVM, in much the same way as SIL. These data structures have significant investment and engineering behind their use in compilers, and it makes sense to leverage that work.

* Pro: Uses LLVM data structures, pervasive in compiler projects such as LLVM, SIL, etc.
* Pro: Get used to seeing and using LLVM, as f18 aims to be an LLVM project
* Con: Uses LLVM data structures, which the project has been avoiding

#### Alternative: C++ Standard Template Library

Clearly, the STL can be used to maintain lists, etc.

* Pro: Keeps the number of libraries minimal
* Con: The STL is general purpose and not necessarily tuned to support compiler construction

#### Alternative: Boost, Library XYZ, etc.

* Con: Don't see a strong motivation at present for adding another library.

Statements are a bit of a transition point. Instead of the LLVM-IR approach of strictly using subtype polymorphism (for hash consing, etc.), FIR statements are a hybrid between ad hoc/subtype polymorphism and parametric polymorphism. This gives us a middle ground of genericity through superclassing and the strong and exact type-safety of algebraic data types &mdash; effectively providing type casing and type classing.

The operations (expressions) owned/referenced by a statement, variable references, etc. will be data structures from the Fortran::evaluate, Fortran::semantics, etc. namespaces.



<hr>

<sup>1</sup> CFG paper. https://bit.ly/2q9IRaQ

<sup>2</sup> All labeled DO sequences will have been translated to DO constructs by semantic analysis.

<sup>3</sup> IF statements are handled like IF constructs with no ELSE alternatives.

<sup>4</sup> In a subsequent discussion, we may want to lower FORALL constructs to semantically distinct loops or even another canonical representation.

<sup>5</sup> These statements are only valid in structured constructs and the branches are well-defined by that executable construct.

<sup>6</sup> Unlike SIL and LLVM-IR.

<sup>7</sup> SIL is the Swift (high-level) intermediate language. https://bit.ly/2RHW0DQ

<sup>8</sup> Single-threaded semantics.
