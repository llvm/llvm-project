==================
Thread Convergence
==================

.. contents::
   :local:

Revisions
=========

- 2025/04/14 --- Created

Introduction
============

Some languages such as OpenCL, CUDA and HIP execute threads in groups (typically
on a GPU) that allow efficient communication within the group using special
*crosslane* primitives. The outcome of a crosslane communication
is sensitive to the set of threads that execute it "together", i.e.,
`convergently`__. When control flow *diverges*, i.e., threads of the same group
follow different paths through the program, not all threads of the group may be
available to participate in this communication.

__ https://llvm.org/docs/ConvergenceAndUniformity.html

Crosslane Operations
--------------------

A *crosslane operation* is an expression whose evaluation by multiple threads
produces a side-effect visible to all those threads in a manner that does not
depend on volatile objects, library I/O functions or memory. The set of threads
which participate in this communication is implicitly affected by control flow.

For example, in the following GPU compute kernel, communication during the
crosslane operation is expected to occur precisely among an environment-defined
set of threads (such as workgroup or subgroup) for which ``condition`` is true:

.. code-block:: c++
   :caption: A crosslane operation
   :name: convergence-example-crosslane-operation

   void example_kernel() {
      ...
      if (condition)
          crosslane_operation();
      ...
   }

Thread Convergence
------------------

Whether two threads convergently execute an operation is different at every
execution of that operation by those two threads. [Note: This corresponds to
`dynamic instances in LLVM IR`__.] In a structured program, there is often an
intuitive and unambiguous way of determining the threads that are converged at a
particular operation. Threads may *diverge* at a *divergent branch*, and then
*reconverge* at some later point in the program such as the end of an enclosing
statement. However, this intuition does not work very well with unstructured
control flow. In particular, when two threads enter an `irreducible cycle`__ in
the control-flow graph along different paths, whether they converge inside the
cycle and at which point depends on the choices made by the implementation.

__ https://llvm.org/docs/ConvergenceAndUniformity.html#threads-and-dynamic-instances
__ https://llvm.org/docs/CycleTerminology.html

The intuitive picture of *convergence* is built around threads executing in
"lock step" --- a set of threads is thought of as *converged* if they are all
executing "the same sequence of instructions together". But this assumption is
not necessary for describing communication at crosslane operations, and the
convergence defined here *does not* assume that converged threads execute in
lock-step.

This document defines convergence at every evaluation in the program based on
the state of the control-flow reaching that point in the source, including the
iterations being performed by any enclosing loop statements. Convergence merely
relates threads that must participate when a crosslane operation is executed.
Such threads are not required to execute a crosslane operation "at the same
time" or even on the same hardware resources. They may appear to do so in
practice, but that is an implementation detail.

.. _convergent-operation:

Convergent Operations
=====================

A *convergent* operation is an expression marked with the attribute
``convergent``. A *non-convergent* operation is an expression that is not marked
as ``convergent``, and optionally marked with the attribute ``noconvergent``.

In general, an implementation may not modify the set of converged threads
associated with each evaluation of a convergent operation. But such
optimizations are possible where the semantics of the specific convergent
operation allows it. The specification for convergence control tokens in LLVM IR
provides some `examples of correct transforms`__ in the presence of convergent
operations.

__ https://llvm.org/docs/ConvergentOperations.html#examples-for-the-correctness-of-program-transforms

.. _convergence-thread-masks:

Explicit Thread Masks
---------------------

Some languages like CUDA and HIP provide convergent operations that take an
explicit threadmask as an argument. Threads are organized in groups called warps
or waves, and a threadmask passed to a convergent operation specifies the
threads within a warp that must participate in that convergent operation. The
set of threads is explicitly specified by the programmer, rather than being
implied by the control-flow of the program.

The convergence defined in this document is not sufficient for describing the
semantics of explicit threadmasks. The optimization constraints placed by these
operations on the implementation are different from those placed by convergent
operations with implicit threadmasks. At the same time, the convergence
specified here is also not contradictory to that semantics --- it can still be
used to determine the sets of threads that are potentially converged at each
execution of such an operation.

.. code-block:: C++
   :caption: Explicit thread masks
   :name: convergence-example-thread-masks

   void crosslane_operation (unsigned long mask) __attribute__(("convergent"));

   void bar(unsigned long mask) {
     convergent_func(mask);
   }

   void foo() {
     ...
     auto mask = ...;

     if (cond)
       bar(mask); // B
     else
       bar(mask); // C
   }

The interpretation of the mask depends on the implementation:

- On implementations where threads in a warp are assumed to execute in lock-step
  (such as AMDGPU or for PTX specifying a target lower than sm_70), the mask
  argument partitions this set of potentially converged threads into subsets of
  threads that must be converged. In :numref:`convergence-example-thread-masks`,
  threads that reach ``B`` (respectively ``C``) and have the same mask are
  converged with each other when they eventually execute the call to
  ``convergent_func``.
- On implementations that allow full concurrency between threads (such as PTX
  specifying sm_70 or higher targets), the mask argument partitions this set of
  potentially converged threads into subsets of threads that converge with each
  other, as well as with subsets executing other instances of the same
  operation. In :numref:`convergence-example-thread-masks`, threads that execute
  ``convergent_func`` as a result of reaching any one of ``B`` or ``C``
  converged if they have the same mask, irrespective of whether they reached
  ``convergent_func`` via ``B`` or ``C``.

Cycles
======

Convergence is affected by `cycles in the control-flow graph`__ of the program.
These may originate from iteration statements or from ``goto`` statements that
transfer control to a label that occurs earlier in the program source. In
particular, specifying convergence for irreducible cycles is cumbersome and
likely to place unnecessary constraints on the implementation. Hence the
convergence of threads in patterns that can potentially produce irreducible
cycles is left to the implementation.

__ https://llvm.org/docs/CycleTerminology.html

  The *span* of a ``goto`` statement is the inclusive sequence of statements that
  occur between a ``goto`` and its target label.

  A *backwards jump* is a ``goto`` statement that transfers control to a label
  that occurs before the ``goto`` in program source.

  A *goto cycle* is the span of a backwards jump.

  A *cycle* is either a *goto cycle* or an iteration statement.

.. note::

   To define a "backwards" jump, statements are ordered according to their
   appearance in the sequence of tokens in a preprocessed source file. This
   definition of a cycle is only a convenient approximation of a cycle in the
   control-flow graph as defined by LLVM IR. Some instances of cycles defined
   here may result in a different cycle in the corresponding control flow graph,
   or maybe even no cycle at all.

   For example:

   - A backwards ``goto`` statement ``G`` that jumps out of an iteration
     statement ``L`` may result in a control-flow cycle that includes ``L`` as a
     child cycle.
   - A ``goto`` in the ``else`` substatement of an ``if`` that jumps
     to the "then" part of the ``if``.
   - A ``goto`` in a ``switch`` statement that jumps backwards to a ``case``
     that is not a fall-through.
   - A ``goto`` that jumps backwards, but other subsequent jumps ensure that the
     same ``goto`` is not encountered again except as a result of some outer
     loop statement.

   But such situations are rare and do not provide enough justification to
   create a more detailed definition of cycles in the source code.

Convergence
===========

*Converged-with* is a transitive symmetric relation over the evaluations of the
same expression performed by different threads. In general, when two evaluations
of the same expression performed by different threads are converged, they may
communicate through any crosslane communication produced by that evaluation.

*Convergence-before* is a strict partial order over evaluations
performed by multiple threads. It is the transitive closure of:

1. If evaluation ``P`` is sequenced-before ``Q``, then ``P`` is
   *convergence-before* ``Q``.
2. If evaluation ``P`` is sequenced-before ``Q1`` and ``Q1`` is *converged-with*
   ``Q2``, then ``P`` is *convergence-before* ``Q2``.
3. If evaluation ``P1`` is *converged-with* ``P2`` and ``P2`` is
   sequenced-before ``Q``, then ``P1`` is *convergence-before* ``Q``.

*Thread-converged-with* is a transitive symmetric relation over threads. For an
expression ``E``, let ``S`` be the smallest statement that contains ``E``.

- When two threads are converged at the execution of ``S``, they are also
  converged at the evaluation of ``E`` if they both evaluate ``E``.
- When two threads are converged at the evaluation of ``E``, those two
  evaluations of ``E`` are also converged.

Two evaluations are converged only if specified below as converged or
implementation-defined.

Mere convergence does not imply any memory synchronization or control-flow
barriers.

Function body
-------------

Whether two threads are converged at the start of a function body is determined
at each invocation of that function.

- When a function is invoked from outside the scope of the current program,
  whether two threads are converged at this invocation is environment-defined.
  For example:

  - In an OpenCL kernel launch, the maximal set of threads that can communicate
    outside the memory model is a workgroup. Hence, a suitable choice is to
    specify that all the threads from a single OpenCL workgroup are pair-wise
    converged at that launch of the kernel.
  - In a C/C++ program, threads are launched independently and they can
    communicate only through the memory model. Thus, a thread that enters a
    C/C++ program (usually via the ``main`` function) is not converged with any
    other thread.

- When two threads are converged at a *convergent* function call in the program,
  those two threads are converged at the start of the called function.

Two threads that are converged at the beginning of a function are also converged
when they exit the function by executing the same or different occurrences of
the ``return`` statement in that function.

.. _convergence-sequential-execution:

Sequential Execution
--------------------

In C++, statements are executed in sequence unless control is transferred
explicitly. Convergence follows this sequential execution.

When two threads are converged at the execution of a statement ``S``, they are
also converged at any substatement ``S1`` of ``S``, if every cycle that contains
``S1`` also contains ``S`` and if they both reach ``S1`` during that execution
of ``S``.

.. code-block:: C++
   :caption: Sequential execution at a branch
   :name: convergence-example-sequential-branch

   void foo() {
     ... // A1
     ... // A2
     if (cond) {
       ... // B1
       ... // B2
     } else {
       ... // C
     }
     ... // D
   }

In :numref:`convergence-example-sequential-branch`, threads that are converged
at the start of ``foo()`` are also converged at ``A1`` and ``A2``. Out of these,
threads that evaluate ``cond`` to be ``true`` are converged at ``B1`` and
``B2``. On the other hand, threads that evaluate ``cond`` to be ``false`` are
converged at ``C``. All threads are finally converged at ``D`` when they reach
there after finishing the ``if`` statement.

.. code-block:: C++
   :caption: Sequential execution in a loop
   :name: convergence-example-sequential-loop

   void foo() {
     ... // A1
     ... // A2
     while (cond) {
       ... // L1
       ... // L2
     }
     ... // C
   }

In :numref:`convergence-example-sequential-loop`, threads that are converged at
the start of ``foo()`` are converged at the start of the ``while`` loop and
again at ``C``. But whether they are converged at the execution of statements
inside the loop is determined by the rules for convergence inside iteration
statements.

Iteration Statement
-------------------

C++ expresses the semantics of the ``for`` statement and the ``ranged-for``
statement in terms of the ``while`` statement. Similarly, convergence at
different parts of these statements is defined as if that statement is replaced
with the equivalent pattern using the ``while`` statement.

An iteration statement ``S`` is said to be *reducible* if and only if for every
label statement ``L`` that occurs inside ``S``, every ``goto`` or ``switch``
statement that transfers control to ``L`` is also inside ``S``.

The following rules apply to reducible iteration statements:

- When two threads are converged at the execution of a ``do-while`` statement,
  they are also converged at that first execution of the body substatement.
- When two threads are converged at the execution of a ``while`` statement, they
  are also converged at that first execution of the condition.
- When two threads are converged at the execution of the condition, they are
  also converged at the subsequent execution of the body substatement if they
  both reach the body substatement.
- When two threads are converged at the end of the body substatement, they are
  also converged at the subsequent execution of the condition if they both reach
  the condition.

When an iteration statement ``S`` is not reducible, the convergence of threads
at each substatement of ``S`` is implementation-defined.

.. code-block:: C++
   :caption: Iteration statement
   :name: convergence-example-iteration-statement

   void foo() {
     ... // A1
     ... // A2
     while (cond1) {
       ... // L1
       if (cond2)
         continue;
       ... // L2
       if (cond3)
         break;
       ... // L3
     }
     ... // C
   }

Consider the execution of the the function ``foo()`` shown in
:numref:`convergence-example-iteration-statement`.

- All threads that were converged at the start of ``foo()`` continue to be
  converged at points ``A1`` and ``A2``.
- Threads converged at ``A2`` for whom ``cond1`` evaluates to ``true`` execute
  the loop body for the first time, and are converged at ``L1``.
- Threads converged at ``L1`` for whom ``cond2`` evaluates to ``true`` transfer
  control to the end of the loop body, while the remaining threads are converged
  at ``L2``.
- Threads converged at ``L2`` for whom ``cond3`` evaluates to ``true`` exit the
  loop, while the remaining threads are converged at ``L3``.
- All threads that were converged at the start of the loop body and did not exit
  the loop body are converged at the end of the loop body, and at the subsequent
  evaluation of ``cond1``.
- All threads that were converged at the start of the ``while`` statement are
  also converged at ``C``.

.. code-block:: C++
   :caption: Jump into loop
   :name: convergence-example-jump-into-loop

   void foo() {
     ... // A
     if (cond1)
       goto inside_loop; // G1
     ... // B
     while (cond) {
       ... // L1
     inside_loop:
       ... // L2
       if (cond3) { // L3
         ...        // L4
         goto outside_loop; // G2
       }
       ... // L5
     }
     ... // C
     outside_loop:
     ... // D
   }

In :numref:`convergence-example-jump-into-loop`:

- Convergence is implementation defined at the loop condition ``cond``, ``L1``,
  ``L2``, ``L3``, and ``L5``.
- Threads that are converged at ``L3`` are converged at ``L4`` and ``G2`` if
  they enter the branch.
- Threads that are converged at the start of the function are converged at
  ``C``. This includes thread that jumped to ``inside_loop`` as well as threads
  that reached the ``while`` loop after executing ``B``.
- Threads that are converged at the start of the function are converged at
  ``outside_loop``. This includes threads that jumped from ``G2`` as well as
  threads that  reached ``outside_loop`` after executing ``C``.

.. code-block:: C++
   :caption: Duff's device
   :name: convergence-example-duffs-device

   void foo() {
     ... // A
     switch (value) {
       case 1:
         ... // C1
         while (cond) {
           ... // L
           // note the fall-through
       case 2:
           ... // LC2
         }
         ... // C2
         break;
       case 3:
         ... // C3
     }
     ... // D
   }

:numref:`convergence-example-duffs-device` shows how C++ allows the statements
of a ``while`` loop to be interleaved with ``case`` labels of a ``switch``
statement, resulting in irreducible control-flow.

- Threads that are converged at the start of ``foo()`` are converged at the
  start of the switch statement.
- Convergence is implementation-defined at ``L`` and ``LC2``.
- Threads that are converged at the start of the ``switch`` statement are
  converged at ``C2`` if they reach ``C2``.
- Threads that jump to ``case 3`` are converged at ``C3``.
- Threads that are converged at the start of ``foo()`` are converged at ``D``.

Jump Statements
---------------

A jump statement (i.e., ``goto`` or ``switch``) results in
implementation-defined convergence only if it is a backwards jump or it
transfers control into an iteration statement.

- Whether two threads are converged at each statement in a ``goto`` cycle is
  implementation-defined.
- In a "straight-line jump" that does not jump into a loop, threads that make
  the jump and threads that do not make the jump both converge at the target
  label.

.. code-block:: C++
   :caption: Simple goto
   :name: convergence-example-goto

   void foo() {
     ... // A
     while (cond) {
       ... // L1
       if (cond)
         goto label_X;
       ... // L2
     label_X: ...
       ... // L3
     }
     ... // B
   }

Consider the execution of the the function ``foo()`` shown in
:numref:`convergence-example-goto`.

- Threads that are converged at ``L1`` are converged at ``L2`` if they reach
  ``L2``.
- Threads that are converged at ``L2`` are converged at ``label_X``.
- Threads that are converged at the ``goto`` are converged at ``label_X``.
- The body substatement contains ``label_X`` as well as every ``goto`` that
  jumps to it, and is a compound statement that contains ``label_X``. Thus, all
  threads that are converged at the start of the body substatement are converged
  at ``label_X``. This includes the previous two sets of threads converged at
  ``label_X``.
- Threads that are converged at ``label_X`` are converged at ``L3``.

.. code-block:: C++
   :caption: Simple ``switch``
   :name: convergence-example-switch

   void foo() {
     ... // A
     switch (value) {
       case 1:
         ... // C1
         break;
       case 2:
         ... // C2
         [[fall_through]]
       case 3:
         ... // C3
     }
   }

In :numref:`convergence-example-switch`, consider threads that are converged at
the ``switch`` statement:

- Threads that jump to ``case 1`` (respectively, ``case 2`` and ``case 3``) are
  converged at ``C1`` (respectively, ``C2`` and ``C3``).
- Threads that jump to ``case 2`` fall-through to ``case 3``. These threads
  are converged with threads that directly jump to ``case 3``.

.. code-block:: C++
   :caption: Backwards ``goto``
   :name: convergence-example-backwards-goto

   void foo() {
     ... // A
     if (cond1)
       goto inside_loop; // G1
     ... // B
     loop:
       ... // L1
     inside_loop:
       ... // L2
       if (cond3) { // L3
         ... // L4
         goto outside_loop; // G2
       }
       ... // L5
     if (cond) // L6
       goto loop; // G3
     ... // C
     outside_loop:
     ... // D
   }

:numref:`convergence-example-backwards-goto` shows a cycle similar to the one in
:numref:`convergence-example-jump-into-loop`, except this cycle is created by a
backwards ``goto`` instead of a ``while`` statement.

- The convergence of threads is implementation-defined in the span of the
  ``goto`` statement ``G3``, which includes ``L1``, ``L2``, ``L3``, ``L5`` and
  ``L6``.
- Threads that are converged at ``L3`` are converged at ``L4`` and ``G2`` if
  they enter the branch.
- Threads that are converged at the start of the function are converged at
  ``C``. This includes thread that jumped to ``inside_loop`` as well as threads
  that reached ``loop`` after executing ``B``.
- Threads that are converged at the start of the function are converged at
  ``outside_loop``. This includes threads that jumped from ``G2`` as well as
  threads that  reached ``outside_loop`` after executing ``C``.

.. _noconvergent-statement:

The ``noconvergent`` Statement
==============================

When a statement is marked as ``noconvergent`` the convergence of threads at the
start of this statement is not constrained by any convergent operations inside
the statement.

- When two threads execute a statement marked ``noconvergent``, it is
  implementation-defined whether they are converged at that execution. [Note:
  The resulting evaluations must still satisfy the strict partial order imposed
  by convergence-before.]
- When two threads are converged at the start of this statement (as determined
  by the implementation), whether they are converged at each convergent
  operation inside this statement is determined by the usual rules.

For every label statement ``L`` occurring inside a ``noconvergent``
statement, every ``goto`` or ``switch`` statement that transfers control to
``L`` must also occur inside that statement.

.. note::

   Convergence control tokens are necessary for correctly implementing the
   "noconvergent" statement attribute. When tokens are not in use, the legacy
   behaviour is retained, where the only effect of this attribute is that
   ``asm`` calls within the statement are not treated as convergent operations.

Implementation-defined Convergence
==================================

Implementation-defined convergence is in the context of each execution of a
function body, corresponding to a distinct execution of a call to that function.
An implementation may not converge two threads that enter the same function body
by executing distinct calls to that function. If those two function calls were
inlined, the resulting evaluations would correspond to distinct copies of the
same expressions in the inlined function bodies. Note that
implementation-defined convergence is still constrained in two ways:

- The strict partial order imposed by *convergence-before*, and
- The convergence at substatements inside a statement ``S`` imposed by
  :ref:`sequential execution<convergence-sequential-execution>` on threads that
  are converged at ``S``.

`Maximal convergence in LLVM IR`__ is an example of implementation-defined
convergence.

__ https://llvm.org/docs/ConvergenceAndUniformity.html#maximal-convergence

Limitation: Loops in LLVM IR
============================

Reference -- `Evolving "convergent": Lessons from Control Flow in AMDGPU
<https://llvm.org/devmtg/2020-09/program/>`_ - Nicolai Haehnle, LLVM Developers'
Meeting, October 2020.

Ambiguity in a Simplified CFG
-----------------------------

The representation of loops in LLVM IR may lose information about the intended
convergence in a program when the control-flow graph is simplified. This happens
when loop structures in the language source that differ in the implied
convergence, are considered equivalent in the CFG.

.. code-block:: C++
   :caption: Different loops with the same single-threaded execution
   :name: convergence-ambiguity-source

   void loop_continue() {
     ... // A
     for (;;) {
       ... // B
       if (cond1)
         continue;
       ... // C
       if (cond2)
         continue;
       break;
     }
     ... // D
   }

   void loop_nest() {
     ... // A
     do {
       do {
         ... // B
       } while (cond1);
       ... // C
     } while (cond2);
     ... // D
   }

:numref:`convergence-ambiguity-source` shows two different loop statements that
have identical semantics in a single-threaded environment. But in a
multi-threaded environment, the convergence of threads is different for these
two statements.

In function ``loop_continue()``, threads that evaluate either ``cond1`` or
``cond2`` to be ``true`` converge at the start of the ``for`` statement for the
next iteration. An execution may produce the following example trace of
converged evaluations.

.. table::
   :align: left

   +----------+----+----+----+----+----+----+
   |          | 1  | 2  | 3  | 4  | 5  | 6  |
   +----------+----+----+----+----+----+----+
   | Thread 1 | A1 | B1 |    | B3 | C1 | D1 |
   +----------+----+----+----+----+----+----+
   | Thread 2 | A2 | B2 | C2 | B4 | C3 | D2 |
   +----------+----+----+----+----+----+----+

But in function ``loop_nest()``, threads that evaluate ``cond1`` to be true
continue to execute the inner ``do`` statement convergently until the condition
becomes ``false``. All threads then proceed to execute ``C`` and then evaluate
``cond2``. An equivalent execution produces the following different trace of
converged evaluations.

.. table::
   :align: left

   +----------+----+----+----+----+----+----+----+
   |          | 1  | 2  | 3  | 4  | 5  | 6  | 7  |
   +----------+----+----+----+----+----+----+----+
   | Thread 1 | A1 | B1 | B3 | C1 |    |    | D1 |
   +----------+----+----+----+----+----+----+----+
   | Thread 2 | A2 | B2 |    | C2 | B4 | C3 | D2 |
   +----------+----+----+----+----+----+----+----+

But both loop statements can result in the same control-flow graph after
simplification in the LLVM IR as shown in :numref:`convergence-ambiguity-cfg`,
thus making convergence ambiguous in an optimizing compiler.

.. code-block:: none
   :caption: Canonicalized Loops
   :name: convergence-ambiguity-cfg

    +-----+
    | A   |
    +-+---+
      |
      v
    +-----+
    | B   |<---+
    +-+-+-+    |
      |  \-----+
      v        |
    +-----+    |
    | C   |    |
    +-+-+-+    |
      |  \-----+
      v
    +-----+
    | D   |
    +-+---+

SimplifyCFG in the LLVM optimizer is an example transform that can produce this
canonicalization. This can be prevented if there was some way to a mark loop
header that should not be merged into its predecessor or successor.

One way to achieve this is to insert some operation with unknown side-effects so
that the optimizer can no longer merge these blocks. But this is clearly a
workaround for the fundamental problem that LLVM IR does not have sufficient
semantics to represent convergence. A better solution is the use of `convergence
control tokens`__ which are currently an experimental feature in LLVM IR.

__ https://llvm.org/docs/ConvergentOperations.html

Divergent Loop Exits
--------------------

.. code-block:: C++
   :caption: Loop with a conditional break
   :name: convergence-divergent-exit-source

   void loop_continue() {
     ... // A
     for (...) {
       ... // B
       if (cond) {
         ... // C
         break;
       }
     }
     ... // D
   }

:numref:`convergence-divergent-exit-source` shows an iteration statement with a
``break`` that occurs inside a condition. When this condition is `divergent`__,
different threads that are converged within the iteration statement execute
``C`` on different iterations, and then reach ``D``. All such threads are
converged at ``D``, but not at the respective execution of ``C`` in different
iterations. An execution may produce the following example trace of
converged evaluations.

.. table::
   :align: left

   +----------+----+----+----+----+----+----+
   |          | 1  | 2  | 3  | 4  | 5  | 6  |
   +----------+----+----+----+----+----+----+
   | Thread 1 | A1 | B1 |    | C1 |    | D1 |
   +----------+----+----+----+----+----+----+
   | Thread 2 | A2 | B2 | B4 |    | C2 | D2 |
   +----------+----+----+----+----+----+----+

__ https://llvm.org/docs/ConvergenceAndUniformity.html

.. code-block:: none
   :caption: Divergent loop exit in LLVM IR
   :name: convergence-divergent-exit-cfg

    +-----+
    | A   |
    +-+---+
      |
      v
    +-----+
    | B   |<---+
    +-+-+-+    |
      |  \-----+
      v
    +-----+
    | C   |
    +-+-+-+
      |
      v
    +-----+
    | D   |
    +-+---+

:numref:`convergence-divergent-exit-cfg` shows the resulting natural loop in
LLVM IR, where this divergent execution of ``C`` is lost. In the LLVM optimizer
and code generator, the block ``C`` is no longer part of the natural loop headed
by ``B``, although it was lexically inside the corresponding iteration statement
in the source code. As a result, the implementation causes all threads that exit
the loop to converge at ``C``, when in fact they should converge at ``D``. An
equivalent execution produces the following trace of converged evaluations.

.. table::
   :align: left

   +----------+----+----+----+----+----+
   |          | 1  | 2  | 3  | 4  | 5  |
   +----------+----+----+----+----+----+
   | Thread 1 | A1 | B1 |    | C1 | D1 |
   +----------+----+----+----+----+----+
   | Thread 2 | A2 | B2 | B4 | C2 | D2 |
   +----------+----+----+----+----+----+

The only way to represent this correctly is using the experimental feature for
`convergence control tokens`__.

__ https://llvm.org/docs/ConvergentOperations.html
