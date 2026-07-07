================================
ClangIR Code Duplication Roadmap
================================

.. contents::
   :local:

Introduction
============

This document describes the general approach to code duplication in the ClangIR
code generation implementation. It acknowledges specific problems with the
current implementation, discusses strategies for mitigating the risk inherent in
the current approach, and describes a general long-term plan for addressing the
issue.

Background
==========

The ClangIR code generation is very closely modeled after Clang's LLVM IR code
generation, and we intend for the CIR produced to eventually be semantically
equivalent to the LLVM IR produced when not going through ClangIR. However, we
acknowledge that as the ClangIR implementation is under development, there will
be differences in semantics, both because we have not yet implemented all
features of the classic codegen and because the CIR dialect is still evolving
and does not yet have a way to represent all of the necessary semantics.

We have chosen to model the ClangIR code generation directly after the classic
codegen, to the point of following identical code structure, using similar names
and often duplicating the logic because this seemed to be the most certain path
to producing equivalent results. Having such nearly identical code allows for
direct comparison between the CIR codegen and the LLVM IR codegen to find what
is missing or incorrect in the CIR implementation.

However, we recognize that this is not a sustainable permanent solution. As
bugs are fixed and new features are added to the classic codegen, the process of
keeping the analogous CIR code up to date will be a purely manual process.

Long term, we need a more sustainable approach.

Current Strategy
================

Practical considerations require that we make steady progress towards a working
implementation of ClangIR. This necessity is directly opposed to the goal of
minimizing code duplication.

For this reason, we have decided to accept a large amount of code duplication
in the short term, even with the explicit understanding that this is producing
a significant amount of technical debt as the project progresses.

As the CIR implementation is developed, we often note small pieces of code that
could be shared with the classic codegen if they were moved to a different part
of the source, such as a shared utility class in some directory available to
both codegen implementations or by moving the function into a related AST class.
It is left to the discretion of the developer and reviewers to decide whether
such refactoring should be done during the CIR development, or if it is
sufficient to leave a comment in the code indicating this as an opportunity for
future improvement. Because much of the current code is likely to change when
the long term code sharing strategy is complete, we will lean towards only
implementing refactorings that make sense independent of the code sharing
problem.

We have discussed various ways that major classes such as CGCXXABI/CIRGenCXXABI
could be refactored to allow parts of there implementation to be shared today
through inheritence and templated base classes. However, this may prove to be
wasted effort when the permanent solution is developed. Also, deferring this
kind of intertwined implementation prevents introducing cross-dependencies that
would make it more difficult to remove one IR code generation implementation
without degrading the quality of the other. Therefore, we have decided that it
is better to accept significant amounts of code duplication now, and defer
this type of refactoring until it is clear what the permanent solution will be.

Mitigation Through Testing
==========================

The most important tactic that we are using to mitigate the risk of CIR diverging
from classic codegen is to incorporate two sets of LLVM IR checks in the CIR
codegen LIT tests. One set checks the LLVM IR that is produced by first
generating CIR and then lowering that to LLVM IR. Another set checks the LLVM IR
that is produced directly by the classic codegen.

At the time that tests are created, we compare the LLVM IR output from these two
paths to verify (manually) that any meaningful differences between them are the
result of known missing features in the current CIR implementation. Whenever
possible, differences are corrected in the same PR that the test is being added,
updating the CIR implementation as it is being developed.

However, these tests serve a second purpose. They also serve as sentinels to
alert us to changes in the classic codegen behavior that will need to be
accounted for in the CIR implementation. While we appreciate any help from
developers contributing to classic codegen, our current expectation is that it
will be the responsibility of the ClangIR contributors to update the CIR
implementation when these tests fail.

As the CIR implementation gets closer to the goal of IR that is semantically
equivalent to the LLVM IR produced by the classic codegen, we would like to
enhance the CIR tests to perform some automatic verification of the equivalence
of the generated LLVM IR, perhaps using a combination of tools such as `opt
-pass-normalize` and Alive2.

Eventually, we would like to be able to run all existing classic codegen tests
using the CIR path as well.

Other Considerations
====================

The close modeling of CIR after classic codegen has also meant that the CIR
dialect often represents language details at a much lower level than it ideally
should.

In the interest of having a complete working implementation of ClangIR as soon
as is practical, we have chosen to take the approach of following the classic
codegen implementation closely in the initial implementation and only raising
the representation in the CIR dialect to a higher level when there is a clear
and immediate benefit to doing so.

Over time, we expect to progressively raise the CIR representation to a higher
level and remove low level details, including ABI-specific handling from the
dialect. (See the "Long Term Vision" section below for more  details.) Having
a working implementation in place makes it easier to verify that the
high-level representation and subsequent lowering are correct.

Mixing With Other Dialects
==========================

Mixing of dialects is a central design feature of MLIR. The CIR dialect is
currently more self-contained than most dialects, but even now we generate
the ACC (OpenACCC) dialect in combination with CIR, and when support for OpenMP
and CUDA are added, similar mixing will occur.

We also expect CIR to be at least partially lowered to other dialects during
the optimization phase to enable features such as data dependence analysis, even
if we will eventually be lowering it to LLVM IR.

Therefore, any plan for generating LLVM IR from CIR must be integrated with the
general MLIR lowering design, which typically involves lowering to the LLVM
dialect, which is then transformed to LLVM IR.

Other Consumers of CIR and MLIR
===============================

We must also consider that we will not always be lowering CIR to LLVM IR. CIR,
usually mixed with other dialects, will also be directed to offload targets
and other code generators through interfaces that are opaque to Clang, such as
SPIR-V and MLIR core dialects. We must still produce semantically correct CIR
for these consumers.

Long Term Vision
================

As the CIR implementation matures, we will eliminate target-specific handling
from the high-level CIR generated by Clang. The high-level CIR will then be
progressively lowered to a form that is closer to LLVM IR, including a pass
that inserts ABI-specific handling, potentially representing the target-specific
details in another dialect. More complex transformations, such as library-aware
idiom recognition or advanced loop representations—may occur later in the
compilation pipeline through additional passes, which can be controlled by
specific compiler flags.

As we raise CIR to this higher level implementation, there will naturally be
less code duplication, and less need to have the same logic repeated in the
CIR generation.

We will continue to use that same basic design and structure for CIR code
generation, with classes like CIRGenModule and CIRGenFunction that serve the
same purpose as their counterparts in classic codegen, but the handling there
will be more closely tied to core semantics and therefore less likely to require
frequent changes to stay in sync with classic codegen.

As the handling of low-level details is moved to later lowering phases, we will
need to move away from the current tight coupling of the CIR and classic codegen
implementations. As this happens, we will look for ways that this handling can
be moved to new classes that are specifically designed to be shared among
clients that are targeting different IR substrates. That is, rather than trying
to overlay reuse onto the existing implementations, we will replace relevant
parts of the existing implementation, piece by piece, as appropriate, with new
implementations that perform the same function but with a more general design.

Example: C Calling Convention Handling
======================================

C calling convention handling is an example of a general purpose redesign that
is already underway. This was started independently of CIR, but it will be
directly useful for lowering from high-level call representation in CIR to a
representation that includes the target- and calling convention-specific details
of function signatures, parameter type coercion, and so on.

The current CIR implementation duplicates most of the classic codegen handling
for function call handling, but it omits several pieces that handle type
coercion. This leads to an implementation that has all of the complexity of the
class codegen without actually achieving the goals of that complexity. It will
be a significant improvement to the CIR implementation to simplify the function
call handling in such a way that it generates a high-level representation of the
call, while preserving all information that will be needed to lower the call to
an ABI-compliant representation in a later phase of compilation.

This provides a clear example where trying to refactor the classic codegen in
some way to be reused by CIR would have been counterproductive. The classic
codegen implementation was tightly coupled with Clang's LLVM IR generation. The
implementation is being completely redesigned to allow general reuse, not just by
CIR, but also by other front ends.

The CIR calling convention lowering will make use of the general purpose C
calling convention library that is being created, but it should create an MLIR
transform pass on top of that library that is general enough to be used by other
dialects, such as FIR, that also need the same calling convention handling.

Significant Areas For Improvement
=================================

The following list enumerates some of the areas where significant restructuring
of the code is needed to enable better code sharing between CIR and classic
codegen. Each of these areas is relatively self-contained in the codegen
implementation, making the path to a shared implementation relatively clear.

- Constant expression evaluation
- Complex multiplication and division expansion
- Builtin function handling
- Exception Handling and C++ Cleanups
- Inline assembly handling
- C++ ABI Handling

  - VTable generation
  - Virtual function calls
  - Constructor and destructor arguments
  - Dynamic casts
  - Base class address calculation
  - Type descriptors
  - Array new and delete

Pervasive Low-Level Issues
==========================

This section lists some of the features where a non-trivial amount of code
is shared between CIR and classic codegen, but the handling of the feature
is distributed across the codegen implementation, making it more difficult
to design an abstraction that can easily be shared.

- Global variable and function linkage
- Alignment management
- Debug information
- TBAA handling
- Sanitizer integration
- Lifetime markers
