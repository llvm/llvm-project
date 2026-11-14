<!--===- docs/AliasingInLLVMIR.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# Representing Fortran aliasing rules in LLVM IR: options and costs

Status: draft design, for discussion.

## 1. Summary

Flang encodes Fortran's aliasing guarantees in TBAA metadata, with one TBAA tree per function. The trees mix two kinds of facts:

- facts that hold everywhere, such as "a descriptor field never overlaps array data" or "two different module variables never overlap";
- facts that hold only during one invocation of a procedure: its dummy arguments do not overlap each other or the other variables it accesses (Fortran 2023, 15.5.2.14).

TBAA has no notion of an invocation. The second kind of fact becomes wrong as soon as LLVM lets the accesses of two invocations meet, which happens in two ways:

1. A procedure is inlined twice into one caller. This is [llvm-project#240818][i240818]. One fix makes Flang's TBAA roots `distinct` and has the inliner clone a distinct root's tree at every inline site ([#242095][pr242095] for the inliner, plus a small Flang change). The review of [#242095][pr242095] pointed out that TBAA was never intended to be used with distinct metadata. We agree, so we do not propose that fix (Option 0, Section 7.0).
2. A procedure is inlined into a loop whose iterations pass overlapping arguments, as in `do i = 1, n - 1; call step(x(i + 1), x(i)); end do`. The loop vectorizer then miscompiles the loop at `-O2` (Section 4.2). This is [llvm-project#242533][i242533]. Distinct roots would not help either, because all iterations share one inline site.

This document compares five representations, from today's scheme with distinct roots (Option 0) to the explicit alias graph presented at the 2020 LLVM Developers' Meeting (Option 4). Options 0 and 1 serve only for comparison: Option 1, per-invocation TBAA roots, would need new IR semantics and changes to seven passes, which is too intrusive. **We propose Option 3.** TBAA keeps only the facts that hold everywhere, in one tree for all of Fortran. The per-invocation facts move into one `!alias.scope` domain per procedure. The domain has disjoint scopes ([#218768][pr218768]), declared with `llvm.experimental.noalias.scope.decl`: one scope per dummy argument and one for all other named storage. This is the mechanism LLVM already uses for the `noalias` arguments of inlined calls. The inliner, the passes that duplicate code and the passes that reason across loop iterations already handle it, so **no LLVM change is needed**.

We emulated every encoding except Option 1 on 4,365 modules from llvm-test-suite and CloverLeaf (Section 9):

- The miscompiled loop is correct under Options 2 and 3. It stays wrong under today's encoding, with distinct roots, and under Option 4.
- Before inlining, Options 3a and 3b keep every NoAlias answer of today's encoding. After inlining on the gfortran tests, they drop 750 and 744 of today's answers. Nearly all of them (743) relate data accessed through a dummy argument in one inlined copy to data in another copy, which Fortran does not guarantee across invocations. They also gain about 130,000 answers (1.4% of today's), because one shared TBAA tree separates descriptors from data across procedures. Option 3c, which drops the facts between named variables, loses 1.5% to 3.4% before inlining.
- Under Option 3 the bitcode is 0.94x to 0.99x of today's before optimization, and `opt -O2` stays within 1% of today's time on the corpus. In synthetic stress tests (up to 512 dummy arguments, 10,000 accesses and 8 inline sites in a loop) it stays within 3%. The 2020-style explicit encoding takes up to 1.47x the time and writes up to 10x the bitcode there.

The evaluation also found that when an assignment allocates the buffer of an allocatable scalar, Flang tags the store into the new buffer inconsistently with later accesses to the variable (Section 4.5, [llvm-project#242536][i242536]). This is a separate bug, but it matters for the choice between the variants of Option 3.

Section 11 lists the questions for reviewers. The main ones are the MLIR representation of disjoint domains, the number of scope declarations, whether to keep `noalias` arguments, and the choice between 3a and 3b.

## 2. Background

### 2.1 Fortran rules

- **Dummy arguments** (15.5.2.14). While a procedure runs, if an entity associated with a dummy argument is modified, or its allocation or association status changes, it may be accessed only through that dummy argument. The exceptions are dummies with the TARGET or POINTER attribute and a few related cases. So within one invocation, two dummy arguments do not overlap if one of them is written, and a dummy argument does not overlap a module or COMMON variable that the procedure accesses by name. Nothing is guaranteed across invocations: `call step(x(i + 1), x(i))` in a loop legally passes overlapping elements in consecutive calls.
- **Storage association.** EQUIVALENCE (8.10.1) and COMMON (19.5.3) make different names share storage. Flang tracks this by byte ranges.
- **POINTER and TARGET.** A pointer can become associated only with an object that has the TARGET or POINTER attribute. Data without these attributes cannot be reached through a pointer.
- **Descriptors.** Allocatable, pointer and assumed-shape objects have descriptors: compiler-managed memory that never overlaps user data.

### 2.2 What Flang emits today

[`AddAliasTags`][addaliastags] attaches a TBAA tag to every FIR memory access. The tags come from a per-function tree built by [`TBAAForest`][tbaaforest]. This is the IR for the procedure `step` of Section 4.2:

```text
define void @_QMrecurrencePstep(ptr noalias captures(none) %0, ptr noalias captures(none) %1) {
  %3 = load float, ptr %1, align 4, !tbaa !4
  %4 = fadd contract float %3, 1.000000e+00
  store float %4, ptr %0, align 4, !tbaa !10
  ret void
}
!4 = !{!5, !5, i64 0}
!5 = !{!"dummy arg data/_QMrecurrenceFstepEb", !6, i64 0}
!6 = !{!"dummy arg data", !7, i64 0}
!7 = !{!"any data access", !8, i64 0}
!8 = !{!"any access", !9, i64 0}
!9 = !{!"Flang function root _QMrecurrencePstep"}
!10 = !{!11, !11, i64 0}
!11 = !{!"dummy arg data/_QMrecurrenceFstepEa", !6, i64 0}
```

- `any data access` has the subtrees `dummy arg data`, `global data`, `allocated data`, `direct data` and `target data`. Each named entity gets a leaf `<subtree>/<unique name>`. COMMON and EQUIVALENCE storage gets byte-range leaves under `global data/<name>`. The data of TARGET or POINTER dummies and of POINTERs gets the generic `target data` tag, and TARGET variables get `target data/<name>`.
- `descriptor member`, the sibling of `any data access`, is used by [`TBAABuilder`][tbaabuilder] in CodeGen for descriptor fields. Accesses that `AddAliasTags` leaves untagged get `any data access` there, except derived types that contain descriptors, which `AddAliasTags` skips ([L660-L667][addaliastags-records]) and CodeGen tags `any access`. All of these come from the tree of the *enclosing* function. A TODO in `TBAABuilder` notes that this is wrong for FIR-inlined code ([L54-L55][tbaabuilder-todo]).
- When FIR inlining is enabled (it is off by default, [`-inline-all`][firinliner]), the dummies of an inlined procedure are tagged under a separate root, `Flang function root <func> - Scope <n>`, selected by the procedure's `fir.dummy_scope`.
- Separately, [`FunctionAttr`][functionattr] marks the by-reference dummy arguments of procedure definitions and module procedures `captures(none)`, unless they are TARGET, ASYNCHRONOUS or VOLATILE, and also `noalias` unless they are POINTERs. It marks the descriptor pointer of descriptor arguments the same way, and marks nothing in BIND(C) procedures. The `noalias` part is on by default through [`force-no-alias`][forcenoalias].

### 2.3 The LLVM mechanisms

- **TBAA** ([LangRef][langref-tbaa]). Tags with different roots MayAlias. Within a root, sibling types are NoAlias, and an ancestor MayAliases its descendants. Merging two accesses takes the lowest common ancestor ([`getMostGenericTBAA`][tbaamerge]). A tag is a fact about types and holds everywhere, so the inliner copies tags unchanged into every inline site.
- **Scoped noalias** ([LangRef][langref-scopes]). An access can carry a `!alias.scope` list and a `!noalias` list of scopes, and each scope belongs to a domain. Two accesses are NoAlias if, for some domain, the first access's scopes in that domain are a subset of the second access's `!noalias` scopes. [#218768][pr218768] added **disjoint domains**, marked by `i1 true`. Two accesses that both have scopes of a disjoint domain, and no scope of it in common, are NoAlias ([ScopedNoAliasAA][scopedaa]), so no complement lists are needed. Merging two accesses intersects their domains and unites their scopes ([Metadata.cpp][mdmerge]).
- **Scope declarations** ([LangRef][langref-decl]). `llvm.experimental.noalias.scope.decl` marks where a scope begins. A pass that duplicates code containing a declaration clones its scopes, and for a disjoint domain also the domain ([`cloneNoAliasScopes`][clonescopes]). LoopUnroll, LoopPeel, LoopRotate, JumpThreading and DFAJumpThreading do this. The passes that reason across loop iterations treat declared scopes as local to an iteration: LoopAccessAnalysis drops a scope list if any of its scopes is declared inside the loop ([L1713-L1734][laa-adjust]), and LICM ignores such scopes for store promotion ([L400-L412][licm-local]).
- **Inliner.** The inliner deep-clones the callee's scopes and domains at every inline site, declared or not ([L1115-L1214][inline-clone]). It also converts the callee's `noalias` arguments into one disjoint domain per inline site, with one declared scope per argument ([#218769][pr218769], [L1220-L1463][inline-noalias]).
- **BasicAA** ([L1582-L1614][basicaa]). A `noalias` argument is NoAlias with identified function-local objects. A pointer loaded from memory is NoAlias with it only if the argument has not been captured before the access, so an explicit-shape dummy passed to an external procedure loses its `noalias` facts.

## 3. Prior work

- **The 2020 IBM talk** ([Islam and Li][ibm2020]) represented an arbitrary alias graph with scoped noalias metadata. Each symbol was a scope in one domain. An access carried `!alias.scope` with the symbols it operates on and an explicit `!noalias` list of all unrelated symbols. On four benchmarks, the talk reports 1.8x to 15.5x compile time and 1.4x to 8.7x IR size, for speedups of 1x to 1.9x. It concluded that the representation is expressive but does not scale in that implementation. The quadratic part is the complement lists. Disjoint domains remove them for partitions, and Fortran's facts are a tree of partitions, so this RFC needs no general graph. Kernel k7 (Appendix A) is the talk's example module.
- **[RFC: disjoint scope domains][rfc91537]** (2026) introduced disjoint domains. It leaves one point open: accesses unrelated to the restrict pointers may need a `!noalias` list naming the whole domain. Option 3 answers this with its "other accesses" scope. Unrelated named accesses get a scope of the same disjoint domain, so they need no list.
- **Full restrict** ([D69542][d69542]) proposed a general `noalias` intrinsic and provenance tracking. It never landed.

## 4. Problem statement

### 4.1 Inlined copies share facts ([#240818][i240818])

```fortran
module copies
contains
  subroutine pair(a, b)
    real :: a, b
    a = b * 2.0
  end subroutine pair
end module copies

subroutine twice(x, i, j, k, l)
  use copies
  real :: x(*)
  integer :: i, j, k, l
  call pair(x(i), x(j))
  call pair(x(k), x(l))       ! when j == k, the b of the first call is the a of the second
end subroutine twice
```

After inlining, the load of `b` in the first copy and the store to `a` in the second have the tags `dummy arg data/b` and `dummy arg data/a` under the same root, so TBAA says NoAlias. That is wrong when `j == k`. The issue was reported with LTO on AArch64, but the wrong answer depends on neither (kernel k1, Appendix A). With distinct roots (Option 0, [#242095][pr242095]), each copy gets its own tree and the pair is MayAlias.

### 4.2 Iterations share facts (new)

```fortran
module recurrence
contains
  subroutine step(a, b)
    real :: a, b
    a = b + 1.0
  end subroutine step
end module recurrence

subroutine run(x, n)
  use recurrence
  integer :: n
  real :: x(n)
  integer :: i
  do i = 1, n - 1
    call step(x(i + 1), x(i))   ! iteration i writes what iteration i + 1 reads
  end do
end subroutine run
```

With `x = 0` and `n = 1000`, `x(n)` must be 999. With `flang -O2` or `-O3` it is 8. After `step` is inlined, the loop vectorizer queries the load of `b` and the store to `a` across iterations. Their TBAA tags say NoAlias, so it vectorizes without a runtime check. The result is the same with distinct roots and with FIR inlining (`-mmlir -inline-all`). It is correct with `-mllvm -enable-tbaa=false`, with `-fno-vectorize` and at `-O1` (kernel k2). It is reported as [llvm-project#242533][i242533].

LLVM is not at fault. TBAA facts are meant to hold everywhere, and the vectorizer correctly ignores the inliner's own `noalias` scopes for this loop, because they are declared inside it. The per-invocation fact is simply in the wrong mechanism. Unrolling exposes the same wrong NoAlias in straight-line code: after a forced unroll by 2, `aa-eval` reports NoAlias between iterations for today's and the distinct-root encodings.

### 4.3 Precision lost across inline boundaries

Two accesses tagged in different trees always MayAlias. Today this costs precision whenever code from two procedures meets:

- After inlining, a callee's data accesses MayAlias the caller's descriptor accesses, although descriptors and data never overlap. The same happens inside one function after FIR inlining, because CodeGen roots the descriptor tags in the enclosing function while the inlined data tags have a scope root (kernel k5, and the `TBAABuilder` TODO).
- A caller's store to a module array MayAliases the inlined callee's access to a module allocatable (kernel k3).

Distinct roots (Option 0) would add the case of two inlined copies of the same procedure. Their descriptor-versus-data facts, which today's shared root states correctly, would become MayAlias: 1,943 answers on the gfortran tests (Section 9.4).

### 4.4 Size

Each function carries its own tree. In the unoptimized bitcode of SNAP and CloverLeaf, the metadata blocks (with their attachment blocks) make up 16.6% of the bytes, and 95.6% of the metadata nodes are alias metadata (Section 9.2). With distinct roots, every inline site would add a copy of the callee's tree. After `opt -O2`, the alias metadata would grow by 1.8% on these applications and by 21.5% on the gfortran tests, which inline more.

### 4.5 Reallocated data contradicts its own tag (new)

When an assignment allocates the buffer of an allocatable scalar, `AddAliasTags` tags the store into the new buffer as anonymous allocated data, `allocated data` ([L908-L915][addaliastags-alloc]). Later accesses to the variable through its descriptor are tagged by the variable's name: `direct data/<name>` for a module or SAVE variable and `target data/<name>` for one with TARGET ([L792-L811][addaliastags-direct]), and `dummy arg data/<name>` for an allocatable dummy argument ([L699-L717][addaliastags-dummy]). These tags are siblings, so TBAA reports NoAlias for two accesses to the same memory, within one invocation:

```fortran
subroutine s(n, r)               ! module procedure
  integer, intent(in) :: n
  integer, intent(out) :: r
  integer, allocatable, save :: j
  if (allocated(j)) deallocate(j)
  j = n                          ! allocates a new buffer and stores n into it
  r = j                          ! reads that buffer through j's descriptor
end subroutine
```

```llvm
  %24 = call ptr @malloc(i64 4)
  store i32 %17, ptr %24, align 4, !tbaa !15    ; allocated data
  ...
  %29 = load i32, ptr %28, align 4, !tbaa !12   ; direct data/_QMmFsEj
```

`aa-eval` reports NoAlias for this store and load. It does the same for an allocatable dummy argument, a module variable, a module variable with TARGET, and a `real` scalar (Appendix A). Local allocatables are not affected, because their tag `allocated data/<name>` is a child of `allocated data`. Neither are arrays, whose assignment stores into the new buffer through the updated descriptor, nor derived-type and deferred-length character scalars, which the runtime assigns. We have not seen wrong code from it. In `s`, the optimizer first forwards the new buffer's address through `j`'s descriptor, so the load uses the same pointer as the store. BasicAA then answers MustAlias before TBAA is asked. The same tags occur in the corpus: in the gfortran test `save_5.f90`, today's encoding answers NoAlias for 6 such pairs within one invocation. A similar inconsistency, in the tags of VALUE dummy copies, did let dead store elimination remove stores; [#215366][pr215366] fixed it. The bug is reported as [llvm-project#242536][i242536].

This bug is independent of the representation and should be fixed in Flang (Section 10). It still matters for the choice of option, because the encodings carry the wrong fact to different places. Today's encoding also states it between two inlined copies of a procedure, since they share a root: 4 more such pairs in `save_5.f90`. Distinct roots and Option 3a state it only within one copy, and Option 3b, which keeps storage facts across inline boundaries, states it across copies as today does (Section 9.4).

## 5. Facts to represent

| fact | holds | where it comes from | carried soundly by |
|---|---|---|---|
| descriptor vs data | everywhere | Flang-managed descriptors | TBAA (one tree for all of Fortran) |
| named storage vs named storage (globals, COMMON byte ranges, locals, TARGET variables) | everywhere (see the note on heap data) | storage names are program-wide | a TBAA tree per LLVM module, or undeclared disjoint domains |
| non-TARGET data vs POINTER and TARGET data | everywhere | Section 2.1 | TBAA, or undeclared disjoint domains |
| dummy vs dummy | one invocation | 15.5.2.14 | `noalias` arguments, declared scopes |
| dummy vs other named storage | one invocation | 15.5.2.14 | declared scopes (`noalias` arguments only for objects that are identified and not captured before) |
| TARGET dummy vs non-TARGET named storage | one invocation | 15.5.2.14; the actual argument need not be a TARGET | declared scopes |

Notes:

- "Program-wide" means across all the files of a program. A global is tagged by its mangled name, which encodes its scope starting from a program unit, external procedure, common block or binding label. No two entities of a program share such a global identifier (19.2), so a name stands for the same storage in every file. LTO renames colliding symbols with local linkage, but not the strings in TBAA nodes, and equal TBAA nodes from different files become one node. A collision can therefore only make one tag stand for two objects, which TBAA answers MayAlias. For example, Flang numbers its read-only array constants per file, so constants with different contents in two files can share the tag `global data/_QQro.5xi4.0`. COMMON is the exception. A member is tagged by its byte range in the block, and files may divide a block differently (8.10.2.4): `common /blk/ x, y` gives `blk_/bytes_0_to_3` and `blk_/bytes_4_to_7`, and `common /blk/ z(2)` in another file gives `blk_/bytes_0_to_7`. Under one root these would be siblings, although they overlap. Today's per-function roots keep files apart. Option 3b is the only option that puts storage names under a root shared by several files, so its root name includes a hash of the file's byte-range leaves (Section 7.3).
- Heap data is tagged by the name of the variable that holds it (`allocated data/<name>`, `direct data/<name>`). MOVE_ALLOC and allocatable function results move data between holders. Today this is sound because the descriptor accesses order the data accesses. It is still the one storage fact that is not purely a property of storage. The reallocation tags of Section 4.5 are a case where it goes wrong (Section 11).
- Fortran allows two dummies to overlap if neither is modified. Like today's TBAA tags and the inliner's `noalias` scopes, the encodings below state NoAlias for such read-only pairs. This only matters for read-read pairs, which optimizations do not need to order.

## 6. Requirements

1. Sound under inlining (including the same procedure inlined many times), code duplication (unroll, peel, rotate, jump threading), merging (hoisting, sinking, CSE) and linking (LTO).
2. No loss of precision within a procedure compared with today's encoding.
3. IR size and compile time that grow linearly in the number of accesses and dummies.
4. Deterministic output.
5. Only LLVM changes that are acceptable upstream, preferably none.

## 7. Options

The options differ in where they put the two kinds of facts:

| | context-free facts (descriptor, storage) | per-invocation facts (dummies) |
|---|---|---|
| 0 | per-function TBAA trees | the same trees |
| 1 | per-function TBAA trees, declared per invocation | the same trees |
| 2 | one TBAA tree for Fortran | `noalias` arguments only |
| 3 | one TBAA tree for Fortran, plus storage facts (3a, 3b) or none (3c) | one declared disjoint scope domain per invocation |
| 4 | one non-disjoint scope domain per function, with explicit complements | the same domain |

### 7.0 Option 0: today plus distinct roots (not proposed)

Flang marks each function root `distinct`, and the inliner clones the tree of a distinct root at every inline site.

- **Status.** Not proposed. The review of the inliner change ([#242095][pr242095]) pointed out that TBAA was never intended to be used with distinct metadata, and we agree. The option stays in this document as the measured baseline for per-function trees that are safe under inlining.
- **Soundness.** Fixes inlined copies (k1). It does not fix iterations (k2), or code duplicated by unrolling, peeling or jump threading, because duplicated code keeps one tree.
- **Precision.** As today before inlining. After inlining, it drops the facts that today's shared root states between inlined copies of one procedure: 2,754 NoAlias answers on the gfortran tests (0.03%), in 35 modules. By today's tags, 1,943 of them relate a descriptor field to data, which is valid across copies. 801 relate data accessed through a dummy argument in one copy to data in another copy. They come from per-invocation facts, the class of the miscompile, which Fortran does not guarantee across invocations. The other 10 relate named storage, including the 4 wrong reallocation pairs of Section 4.5 (Section 9.4).
- **Cost.** One tree copy per inline site of a procedure. After `opt -O2`, the alias metadata grows by 1.8% on SNAP and CloverLeaf, by 21.5% on the gfortran tests and by 15.5% on the cray tests. The bitcode grows by 0.03%, 0.23% and 0.19%, and compile time does not change.
- **Changes.** LLVM: 124 lines in `InlineFunction.cpp`, a few lines in `TypeBasedAliasAnalysis.cpp` and `MDBuilder.h`, and a LangRef paragraph. MLIR: a `distinct` flag on the TBAA root attribute. Flang: 7 lines in `TBAAForest.cpp`. The inliner change is [#242095][pr242095].

### 7.1 Option 1: per-invocation TBAA roots in LLVM

Give TBAA the declaration semantics of scopes: a new way to declare a TBAA root local to an invocation. LoopUnroll, LoopPeel, LoopRotate, JumpThreading, DFAJumpThreading, LoopAccessAnalysis and LICM would be taught to clone or ignore it.

- **Soundness.** Could fix both problems, if every duplicating and cross-iteration pass is updated.
- **Precision.** As today; still MayAlias across inline boundaries.
- **Cost.** As Option 0, plus declarations.
- **Changes.** New IR semantics for TBAA, and changes to every pass listed above, duplicating what scoped noalias already provides.
- **Status.** Not proposed, because it is too intrusive. We did not emulate it.

### 7.2 Option 2: one Fortran TBAA tree plus `noalias` arguments

One TBAA tree for all of Fortran, `Flang Fortran TBAA` → `any access` → `any data access` and `descriptor member`, holds the context-free facts. The dummy facts come only from `noalias` arguments: through BasicAA in the procedure itself, and through the inliner's scopes after inlining.

- **Soundness.** Sound (k2 runs correctly), as long as the tree contains only context-free facts.
- **Precision.** It loses the dummy facts that `noalias` arguments cannot carry:
  - the dummies of FIR-inlined procedures (k5);
  - the data of descriptor dummies, because `noalias` applies only to the descriptor pointer (k5);
  - by-reference dummies whose address is captured, for example by a call to an external procedure (k4, where the vectorizer then adds a runtime alias check);
  - accesses that LLVM cannot trace back to the argument.

  Option 2 could keep the storage facts the way 3a or 3b does. The emulation drops them, so it is Option 3c without the restrict domain. Before inlining, it loses 1.9% of today's NoAlias answers on SNAP and CloverLeaf and 3.6% on the gfortran tests. Option 3c loses 1.5% and 3.4%, so the dummy facts that only scopes carry account for 0.4 and 0.2 points of that. After inlining, the shared tree gains 122,925 NoAlias answers on the gfortran tests, for descriptor versus data across procedures.
- **Cost.** The smallest: 0.91x to 0.96x of today's bitcode before optimization.
- **Changes.** Flang only.

### 7.3 Option 3: one Fortran TBAA tree plus declared restrict scopes (proposed)

Each procedure, and each FIR-inlined `fir.dummy_scope`, gets one disjoint scope domain, declared where the procedure body begins. It has:

- one scope per dummy argument that 15.5.2.14 covers (not TARGET, POINTER, VOLATILE or ASYNCHRONOUS), used by every access to the dummy's data, including the data of assumed-shape dummies;
- one "other accesses" scope, used by every access to other named storage: globals, COMMON, locals, and POINTER and TARGET data.

A procedure that would get fewer than two scopes gets no domain, since a single scope makes nothing NoAlias. Accesses that are not to named Fortran storage get no scope (calls into the runtime, Cray pointers). Descriptor accesses need none, because TBAA separates them from data. This is `step` from Section 4.2 in Option 3:

```llvm
define void @_QMrecurrencePstep(ptr %a, ptr %b) {
  call void @llvm.experimental.noalias.scope.decl(metadata !10)
  call void @llvm.experimental.noalias.scope.decl(metadata !11)
  %v = load float, ptr %b, align 4, !tbaa !3, !alias.scope !11
  %w = fadd contract float %v, 1.0
  store float %w, ptr %a, align 4, !tbaa !3, !alias.scope !10
  ret void
}
!0 = !{!"Flang Fortran TBAA"}
!1 = !{!"any access", !0, i64 0}
!2 = !{!"any data access", !1, i64 0}
!3 = !{!2, !2, i64 0}
!5 = distinct !{!5, i1 true, !"_QMrecurrencePstep"}   ; disjoint domain
!6 = distinct !{!6, !5, !"a"}
!7 = distinct !{!7, !5, !"b"}
!10 = !{!6}
!11 = !{!7}
```

`aa-eval` reports NoAlias for the load and the store from the scopes alone; the arguments are not `noalias` in this example. `step` accesses no other named storage, so it has no "other accesses" scope.

- **Soundness.**
  - Within an invocation, the scopes state exactly the facts of 15.5.2.14.
  - Inlined copies: the inliner clones the domain at every inline site, so copies do not share scopes (k1: MayAlias).
  - Iterations: after inlining into a loop, the declarations are inside the loop. LoopAccessAnalysis and LICM's store promotion treat the scopes as local to an iteration, and unrolling and peeling clone them per copy (k2: correct result, and MayAlias between unrolled iterations).
  - Merging: a merged access has the union of the scopes. It stays NoAlias with the other scopes and becomes MayAlias with each original scope (k6).
  - This is how LLVM already treats the `noalias` arguments of inlined calls. Option 3 therefore needs no new soundness argument: after inlining it is exactly as sound as C `restrict`.
- **Storage facts.** Three variants:
  - **3a** puts them in undeclared disjoint domains, one per inner node of today's storage tree (`any data access`, `global data`, ...), with one scope per child. TBAA keeps only descriptor versus data. Within a procedure this keeps every fact (k7). But the inliner deep-clones the callee's domains, so after inlining the callee's storage scopes no longer meet the caller's or another copy's. Storage facts across inline boundaries are lost, as with distinct roots (k3).
  - **3b** keeps today's storage subtree, without the dummy leaves, in one TBAA tree per LLVM module. The root name includes a hash of the module's COMMON and EQUIVALENCE byte-range leaves, so two modules share a root under LTO only if they agree on them (Section 5). TBAA is not cloned, so storage facts survive inlining. That is a gain over today: k3 becomes NoAlias, and after inlining the corpus gains 105 NoAlias answers on SNAP and CloverLeaf and 732 on the cray tests.
  - **3c** drops the storage facts and relies on BasicAA, which decides most of them for globals and locals but not for data reached through descriptors. Before inlining it loses 1.5% of today's NoAlias answers on SNAP and CloverLeaf and 3.4% on the gfortran tests. In SNAP's `sn.f90`, 248 loads and 221 stores remain after `-O2`, against 186 and 193 with storage facts.
- **TARGET dummies.** "TARGET dummy versus non-TARGET named storage" holds only within one invocation, since the actual argument need not be a TARGET. Today it is a TBAA fact (`target data` versus `global data/...`). It must not go into the context-free storage facts of 3a or 3b. It can be dropped, or carried by a second declared disjoint domain with two scopes, one for TARGET dummy accesses and one for non-TARGET named accesses. The emulation keeps today's tag, so its 3a and 3b numbers are slightly optimistic for programs that rely on this fact.
- **Reallocation tags.** 3b carries storage facts across inline boundaries, so, like today's shared roots, it also carries the wrong ones of Section 4.5 between inlined copies. Fixing those tags is a prerequisite for 3b.
- **Precision.** 3a and 3b lose no NoAlias answer before inlining on any measured code. After inlining on the gfortran tests, they lose 750 (3a) and 744 (3b) of today's answers. 743 of them relate data accessed through a dummy argument in one inlined copy to data in another copy, which Fortran does not guarantee across invocations. They keep every NoAlias answer of the distinct-root encoding, with one exception that is a pair-matching artifact (Section 9.1). They gain 129,456 (3a) and 133,766 (3b) answers, for descriptor versus data across procedures and copies.
- **Cost** (formulas in Appendix B). Let a procedure root access D dummies, plus other named storage or not, for U = D or D + 1 scopes. If U ≥ 2, the root adds one domain, U scopes, U single-scope lists and U declarations, and each access to a dummy or other named storage carries one more attachment. The inliner clones these at each inline site, as it clones `noalias` arguments today. On the corpus, the bitcode is 0.94x to 0.99x of today's before optimization, and `opt -O2` time is within 1% of today's. In the synthetic stress tests, it is within 3%.
- **Changes.**
  - LLVM: none.
  - MLIR: a `disjoint` flag on `#llvm.alias_scope_domain` ([LLVMAttrDefs.td][mlir-domain]), translated to the `i1 true` operand. `llvm.intr.experimental.noalias.scope.decl` already exists ([LLVMIntrinsicOps.td][mlir-decl]).
  - Flang: `AddAliasTags`, or a new pass, emits the core TBAA tag plus a scope per access, and the declarations at the start of each procedure body and each `fir.dummy_scope`. `TBAABuilder` uses the core tree. The pass must be deterministic: scopes are created in a stable order, and the 3b hash is computed over sorted names.
- **Risks.**
  - Declarations are instructions. Code that relies on instruction counts or on the shape of the entry block could change.
  - After heavy inlining, a function can carry many declarations: after `opt -O2`, 5,081 on the gfortran tests against 908 today (the latter from `noalias` arguments). InstCombine removes declarations whose scopes cannot make anything NoAlias ([`AliasScopeTracker`][instcombine-decl]).
  - The `noalias` arguments then partly duplicate the restrict scopes (Section 11).

### 7.4 Option 4: explicit alias graph (the 2020 encoding)

One non-disjoint scope domain per function root and one scope per node of today's tree. Each access carries `!alias.scope` with its node and `!noalias` with the explicit list of unrelated nodes. There is no TBAA.

- **Soundness.** Without declarations, the scopes hold everywhere, exactly like today's TBAA. Unrolled iterations of k2 are NoAlias, and k2 is miscompiled once the `noalias` arguments are removed. With them it runs correctly, only because the inliner adds its own declared scopes to the same accesses. Adding declarations would fix this, at the cost of one declaration per scope.
- **Precision.** As today within a procedure; after inlining, the same as distinct roots, because the inliner clones the scopes per inline site.
- **Cost.** Quadratic in the tree size. On SNAP and CloverLeaf it has 92 times the scope-list operands of Option 3c (86,930 against 944). In the synthetic stress tests it takes up to 1.47x today's `opt -O2` time and writes up to 10.3x the bitcode. Shown for comparison only.

### 7.5 Rejected alternatives

- **`separate_storage` assumptions.** These state that two pointers point to different objects. That is stronger than Fortran's rule, which allows overlap if neither is modified, and allows parts of one object to be passed as two dummies. They are also quadratic in the number of dummies, and BasicAA honors them only behind an option.
- **Full restrict** ([D69542][d69542]). It never landed.
- **A Flang-specific alias analysis in LLVM.** Facts that live outside the IR are not maintained by the passes that duplicate, merge or inline code.
- **Optimizing at the FIR level only.** It cannot help the LLVM passes (vectorizer, LICM, GVN) that need the facts.

## 8. Comparison

"Today" is the current encoding. Measured values are from Section 9: "apps" is SNAP and CloverLeaf, and "gfortran" is the gfortran tests.

| criterion | today | 0 | 1 | 2 | 3a | 3b | 3c | 4 |
|---|---|---|---|---|---|---|---|---|
| inlined copies (k1) | **wrong** | sound | sound | sound | sound | sound | sound | sound |
| iterations (k2) | **miscompile** | **miscompile** | sound | sound | sound | sound | sound | **wrong** (see 7.4) |
| NoAlias kept before inlining, apps / gfortran | 100% / 100% | 100% / 100% | as today | 98.1% / 96.4% | 100% / 100% | 100% / 100% | 98.5% / 96.6% | 100% / 100% |
| NoAlias after inlining, gfortran: lost / gained | – | 2,754 / 0 | as 0 | 45,418 / 122,925 | 750 / 129,456 | 744 / 133,766 | 43,753 / 122,931 | 2,754 / 0 |
| facts across inline boundaries | none | none | none | descriptor vs data | descriptor vs data | descriptor vs data, storage | descriptor vs data | none |
| bitcode before optimization, apps / gfortran | 1.00x | 1.00x / 1.00x | about 1x | 0.91x / 0.96x | 0.95x / 0.98x | 0.99x / 0.99x | 0.94x / 0.97x | 1.00x / 0.97x |
| `opt -O2` time, apps / tests | 1.000x | 1.000x / 1.000x | – | 0.999x / 0.999x | 1.003x / 1.006x | 1.003x / 1.003x | 1.008x / 1.002x | 1.009x / 1.008x |
| `opt -O2` time / output bitcode, largest stress test | 1.00x | 1.00x / 1.03x | – | 0.98x / 0.96x | 0.99x / 1.03x | 1.00x / 1.03x | 1.00x / 1.03x | 1.47x / 2.96x |
| LLVM changes | – | inliner ([#242095][pr242095]) | IR semantics, 7 passes | none | none | none | none | none |
| MLIR changes | – | `distinct` TBAA root | new attribute | none | `disjoint` domain flag | `disjoint` domain flag | `disjoint` domain flag | none |
| proposed | – | no | no | no | yes (3a or 3b) | yes (3a or 3b) | no | no |

What each option keeps, compared with today, by fact and context. "=" means as today, "+" more facts, "−" fewer facts.

| fact | context | 0 | 2 | 3a | 3b | 3c | 4 |
|---|---|---|---|---|---|---|---|
| descriptor vs data | within a procedure | = | = | = | = | = | = |
| | across inlined procedures | = | + | + | + | + | = |
| | across inlined copies of one procedure | − | = | = | = | = | − |
| named storage | within a procedure | = | − | = | = | − | = |
| | across inlining | − (copies) | − | − (copies) | + | − | − (copies) |
| | under LTO | = | = | = | = (layout hash) | = | = |
| dummies | within a procedure | = | − (captured, descriptor data, FIR-inlined) | = | = | = | = |
| | across inlined copies (today wrong) | fixed | fixed | fixed | fixed | fixed | fixed |
| | across iterations and duplicated code (today wrong) | wrong | fixed | fixed | fixed | fixed | wrong |
| wrong reallocation tags (Section 4.5; today within a call and across inlined copies) | | within a call | none | within a call | as today | none | within a call |

## 9. Cost evaluation

### 9.1 Method

- **Tools.** Flang's IR before any LLVM pass comes from `flang -O2 -S -emit-llvm -mllvm -opt-bisect-limit=0`, with flang built from llvm-project `034b938cbbaa`. Everything else uses `opt` from `b53e4eea1879` plus the inliner change of [#242095][pr242095] (commit `9960e1408d95`), built as Release without assertions. The patch acts only on TBAA roots marked `distinct`, which only the distinct-root encoding has.
- **Emulation.** A script rewrites the alias metadata of Flang's IR into each encoding, and nothing else in the module changes. This gives exact counts, and lets the same module run through `opt` in every encoding. Its limits:
  - Declarations for FIR-inlined scopes go to the function entry.
  - TARGET dummies keep today's tag (Section 7.3).
  - Option 2 is emulated without storage facts (Section 7.2).
  - Option 1 is not emulated.
- **Alias results.** `aa-eval` with `-evaluate-aa-metadata` (without it, metadata is ignored) and `-aa-pipeline=basic-aa,scoped-noalias-aa,tbaa`. It runs before inlining and after `cgscc(inline),function(sroa,early-cse<memssa>)`. Load and store pairs are matched across encodings by function and by their text without metadata. Only pairs present in both runs are compared. Matching by text can pair different instructions that print alike. In one gfortran test, the only pair that 3a and 3b lose compared with distinct roots is such a mismatch: the address operand has the same name but a different definition in the two runs. `aa-eval` is quadratic, so modules with more than 1,500 loads and stores are left out (300 for the gfortran tests): CloverLeaf's `clover.f90` and 109 gfortran tests. One more gfortran test timed out after inlining.
- **Corpus.** [llvm-test-suite][testsuite] at `19b349c5` and [CloverLeaf_Serial][cloverleaf] at `b9a2b9c4`, 4,365 modules in total:
  - "applications": Fortran/SNAP and CloverLeaf_Serial, 73 modules;
  - "gfortran tests": the gfortran regression and torture tests that llvm-test-suite builds for Flang (not marked xfail or disabled), 4,277 modules;
  - "cray tests": Fortran/cray, 15 modules.
- **Timing.** User plus system time of `opt -passes='default<O2>' -disable-output` on bitcode input, from `wait4`, as the minimum of interleaved repeated runs (5 for the applications, 3 for the tests). Peak RSS comes from `/usr/bin/time`. The host is an x86-64 Linux machine (2 × Intel Xeon Gold 6338).

### 9.2 Corpus and today's metadata

| | modules | defined functions | loads and stores | `!tbaa` attachments | TBAA roots | inline sites at `-O2` (most in one caller) |
|---|---:|---:|---:|---:|---:|---:|
| applications | 73 | 199 | 24,005 | 26,075 | 163 | 18 (6) |
| gfortran tests | 4,277 | 13,477 | 211,510 | 243,885 | 8,527 | 9,837 (192) |
| cray tests | 15 | 86 | 1,115 | 1,265 | 71 | 68 (17) |

None of these modules has FIR-inlined scope roots, since FIR inlining is off by default. Per function root (median / 90th percentile / maximum):

| | tagged accesses | accessed dummies | tree nodes used |
|---|---|---|---|
| applications | 51 / 505 / 1,319 | 2 / 14 / 40 | 12 / 38 / 93 |
| gfortran tests | 8 / 64 / 2,176 | 0 / 2 / 51 | 3 / 7 / 107 |
| cray tests | 9 / 41 / 104 | 0 / 2 / 9 | 4 / 9 / 13 |

Share of today's alias metadata, without debug information, before optimization and after `flang -O3`:

| | alias metadata, of all metadata nodes | metadata and attachment blocks, of the bitcode |
|---|---|---|
| applications | 95.6% / 82.6% | 16.6% / 18.0% |
| gfortran tests | 85.2% / 74.5% | 11.0% / 12.4% |
| cray tests | 93.8% / 88.3% | 12.2% / 10.7% |

### 9.3 Size per encoding

Applications, before optimization, with the bitcode ratio before and after `opt -O2`:

| encoding | alias metadata nodes | attachments | scope-list operands | declarations | bitcode, before / after `-O2` |
|---|---:|---:|---:|---:|---:|
| today | 6,363 | 26,075 | 0 | 0 | 1.00x / 1.00x |
| 0 (distinct roots) | 6,363 | 26,075 | 0 | 0 | 1.00x / 1.00x |
| 2 | 368 | 26,075 | 0 | 0 | 0.91x / 0.92x |
| 3a | 6,698 | 39,199 | 5,648 | 944 | 0.95x / 0.95x |
| 3b | 5,423 | 36,198 | 944 | 944 | 0.99x / 0.98x |
| 3c | 2,358 | 36,198 | 944 | 944 | 0.94x / 0.95x |
| 4 | 8,411 | 52,015 | 86,930 | 0 | 1.00x / 1.00x |

Bitcode ratios of the tests, before / after `opt -O2`:

| encoding | gfortran tests | cray tests |
|---|---:|---:|
| 0 (distinct roots) | 1.00x / 1.00x | 1.00x / 1.00x |
| 2 | 0.96x / 0.97x | 0.93x / 0.96x |
| 3a | 0.98x / 0.99x | 0.97x / 0.98x |
| 3b | 0.99x / 0.99x | 0.98x / 0.99x |
| 3c | 0.97x / 0.98x | 0.95x / 0.98x |
| 4 | 0.97x / 0.99x | 0.96x / 0.97x |

Option 3 adds attachments and declarations, but it replaces the per-function trees with one tree, so the bitcode shrinks overall. Option 4's explicit complements grow quadratically with the tree: 86,930 list operands on the applications and 249,532 on the gfortran tests, against 944 and 6,074 for Option 3c.

Largest growth of a single module, over modules whose alias metadata (nodes, list operands, attachments and declarations) has at least 100 entries today:

| encoding | alias metadata, before / after `-O2` | bitcode, before / after `-O2` |
|---|---:|---:|
| 0 (distinct roots) | – / – | – / 1.07x |
| 2 | 1.00x / 1.25x | 1.00x / 1.14x |
| 3a | 2.29x / 2.65x | 1.08x / 1.15x |
| 3b | 1.97x / 2.38x | 1.05x / 1.16x |
| 3c | 1.96x / 2.38x | 1.01x / 1.17x |
| 4 | 32.15x / 13.69x | 1.74x / 1.25x |

After `-O2` the largest bitcode growth for Options 2 and 3 is mostly code, not metadata. In SNAP's `sn.f90` (Options 2 and 3c), it is the loads and stores that remain without storage facts (Section 7.3). In `elemental_dependency_2.f90` (Options 3a and 3b), less than half the extra bytes are in the metadata block (261 alias metadata nodes against 102). The rest are instructions: 70 more declarations, and different code (61 loads and 29 stores against 76 and 21).

### 9.4 Alias results

NoAlias answers lost compared with today's encoding, before inlining (no encoding gains any):

| | pairs | NoAlias today | 0 | 2 | 3a | 3b | 3c | 4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| applications (72 modules) | 1,098,366 | 1,039,809 | 0 | 19,650 (1.9%) | 0 | 0 | 15,800 (1.5%) | 0 |
| gfortran tests (4,168) | 1,478,788 | 1,392,335 | 0 | 49,531 (3.6%) | 0 | 0 | 46,770 (3.4%) | 0 |
| cray tests (15) | 7,754 | 7,009 | 0 | 137 (2.0%) | 0 | 0 | 135 (1.9%) | 0 |

After inlining, NoAlias answers lost / gained compared with today's encoding:

| | pairs | NoAlias today | 0 | 2 | 3a | 3b | 3c | 4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| applications (72) | 1,019,799 | 824,871 | 0 / 0 | 13,615 / 6 | 0 / 0 | 0 / 105 | 13,524 / 6 | 0 / 0 |
| gfortran tests (4,167) | 9,618,962 | 9,202,388 | 2,754 / 0 | 45,418 / 122,925 | 750 / 129,456 | 744 / 133,766 | 43,753 / 122,931 | 2,754 / 0 |
| cray tests (15) | 54,535 | 49,539 | 0 / 0 | 129 / 28 | 0 / 28 | 0 / 732 | 127 / 28 | 0 / 0 |

- The applications inline little at `-O2` (18 sites), so most of their results are within one procedure.
- Compared with distinct roots, Options 3a and 3b lose 1 answer each (the pair-matching artifact of Section 9.1), and Option 4 loses none.
- The answers that distinct roots lose relate two inlined copies of one procedure. We probed all 35 gfortran modules where this happens and classified the 2,754 answers by today's tags:

| today's tags | answers | NoAlias under 2 | under 3a | under 3b |
|---|---:|---:|---:|---:|
| descriptor field vs data | 1,943 | 1,904 | 1,906 | 1,906 |
| dummy-argument data vs data | 801 | 0 | 0 | 0 |
| other named storage | 10 | 0 | 0 | 6 |

  The descriptor facts that Options 2 and 3 do not restore have no matching pair in their optimized IR. No encoding restores the dummy-argument facts, which do not hold across invocations. Option 3b restores 2 valid facts between a global and allocatable data, and also states again the 4 wrong reallocation answers of Section 4.5 (from `save_5.f90`).
- Most gains come from the shared tree. Option 2 differs from today's encoding only by that tree and by the facts it drops, so its TBAA can add only descriptor-versus-data answers across trees.

Precision projected onto code: in the kernels (Appendix A) and in `sn.f90`, the facts that an option drops show up as loads and stores that stay in loops after `-O2`, or as runtime alias checks.

### 9.5 Compile time and memory

`opt -passes='default<O2>'`, CPU seconds summed over the modules:

| encoding | applications (73 modules) | tests (4,292 modules) |
|---|---:|---:|
| today | 2.997 | 73.868 |
| 0 (distinct roots) | 2.997 (1.000x) | 73.873 (1.000x) |
| 2 | 2.995 (0.999x) | 73.788 (0.999x) |
| 3a | 3.008 (1.003x) | 74.285 (1.006x) |
| 3b | 3.008 (1.003x) | 74.063 (1.003x) |
| 3c | 3.021 (1.008x) | 74.027 (1.002x) |
| 4 | 3.024 (1.009x) | 74.434 (1.008x) |

The peak RSS of the largest module is the same in every encoding (41 MiB for `clover.f90`, 60 MiB for `minlocval_3.f90`). Re-timed alone (10 runs), `minlocval_3.f90` takes 0.997x to 1.010x of today's time in every encoding, with the same instruction count after `-O2`.

### 9.6 Synthetic stress tests

A generated module has one procedure with D `noalias` dummy arguments and access to 8 module arrays. The procedure makes A accesses in statements `d(a)[o] = d(b)[o] + g[o]`, tagged as Flang tags them, and a driver loop calls it K times per iteration. The procedure is `alwaysinline`. Each configuration was measured for D ∈ {8, 64, 512}, A ∈ {1,000, 10,000} and K ∈ {1, 8}.

`opt -O2` CPU time relative to today's encoding, for the configurations with the largest effect:

| D | A | K | today | 0 | 2 | 3a | 3b | 3c | 4 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 10,000 | 8 | 14.64 s | 1.00x | 0.99x | 1.03x | 1.03x | 1.03x | 1.07x |
| 64 | 10,000 | 8 | 12.11 s | 1.00x | 1.00x | 1.00x | 0.99x | 0.98x | 1.04x |
| 512 | 1,000 | 1 | 0.68 s | 1.00x | 0.97x | 0.99x | 0.99x | 0.99x | 1.45x |
| 512 | 1,000 | 8 | 38.11 s | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x | 1.03x |
| 512 | 10,000 | 1 | 4.53 s | 0.98x | 0.98x | 1.00x | 0.99x | 0.99x | 1.35x |
| 512 | 10,000 | 8 | 17.29 s | 1.00x | 0.98x | 0.99x | 1.00x | 1.00x | 1.47x |

Output bitcode relative to today's encoding:

| D | A | K | today | 0 | 2 | 3a | 3b | 3c | 4 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1,000 | 1 | 86,804 B | 1.05x | 0.72x | 0.99x | 1.00x | 0.99x | 7.46x |
| 512 | 1,000 | 8 | 359,988 B | 1.15x | 0.90x | 1.27x | 1.27x | 1.27x | 10.31x |
| 512 | 10,000 | 1 | 492,244 B | 1.02x | 0.91x | 1.01x | 1.01x | 1.01x | 3.06x |
| 512 | 10,000 | 8 | 2,758,856 B | 1.03x | 0.96x | 1.03x | 1.03x | 1.03x | 2.96x |

- Across all 12 configurations, Options 2 and 3 take 0.89x to 1.03x of today's time, and Option 4 up to 1.47x.
- The peak RSS of Option 3 is at most 1.09x of today's, and that of Option 4 at most 1.25x.
- At D = 512, A = 10,000 and K = 8, the output has 21,001 scope-list operands under 3b and 3c, against 8,192 today (from the inliner's `noalias` scopes) and 2,474,560 under Option 4.
- D = 512, A = 1,000, K = 8 is slower than D = 512, A = 10,000, K = 8 in every encoding, including today's. We did not investigate this, since it does not depend on the encoding.

### 9.7 Kernels

Seven kernels, one per fact, plus the reallocation kernel of Section 4.5 (Appendix A). All pass `opt -passes=verify` in every encoding, and the alias results confirm the soundness and precision claims of Sections 7 and 8. k6, which merges two scoped stores, behaves as LangRef specifies for disjoint domains (Section 7.3). The others:

| kernel | fact | today | 0 | 2 | 3a | 3b | 3c | 4 |
|---|---|---|---|---|---|---|---|---|
| k1 | two inlined copies, `b` of copy 1 vs `a` of copy 2 | NoAlias (wrong) | MayAlias | MayAlias | MayAlias | MayAlias | MayAlias | MayAlias |
| k2 | `step(x(i + 1), x(i))` inlined into a loop, iterations unrolled | NoAlias (wrong) | NoAlias (wrong) | MayAlias | MayAlias | MayAlias | MayAlias | NoAlias (wrong) |
| k2 | end to end, `opt -O2`, run | FAIL | FAIL | PASS | PASS | PASS | PASS | PASS (FAIL without `noalias` arguments) |
| k3 | caller's store to a module array vs inlined callee's module allocatable data | MayAlias | MayAlias | MayAlias | MayAlias | NoAlias | MayAlias | MayAlias |
| k4 | captured non-TARGET dummy vs module POINTER data | NoAlias | NoAlias | MayAlias (runtime check added) | NoAlias | NoAlias | NoAlias | NoAlias |
| k5 | FIR-inlined: caller's descriptor loads vs inlined data store | MayAlias | MayAlias | NoAlias | NoAlias | NoAlias | NoAlias | MayAlias |
| k5 | FIR-inlined: dummies of the inlined procedure | NoAlias | NoAlias | MayAlias | NoAlias | NoAlias | NoAlias | NoAlias |
| k7 | the 2020 example: module scalar vs allocatable TARGET data | NoAlias | NoAlias | MayAlias | NoAlias | NoAlias | MayAlias | NoAlias |

## 10. Proposed direction

1. Fix the reallocation tags of Section 4.5 in Flang ([llvm-project#242536][i242536]), independently of the representation.
2. Prototype Option 3 in Flang behind a hidden option (for example `-mmlir -fortran-restrict-scopes`; the name is to be decided), together with the MLIR `disjoint` flag.
3. Choose between 3a and 3b with the prototype. 3b keeps storage facts across inlining and is cheaper; 3a needs no TBAA beyond the core tree but loses storage facts across inlining. Both must handle TARGET dummies as described in Section 7.3.
4. Measure the prototype on full applications with LTO (compile time, size and performance).
5. Make Option 3 the default. This fixes [llvm-project#240818][i240818] and [llvm-project#242533][i242533] without an LLVM change, and Flang no longer needs per-function roots.

We do not propose Option 0 (Section 7.0) or Option 1 (Section 7.1).

## 11. Open questions for reviewers

1. **MLIR.** Is a `disjoint` flag on `#llvm.alias_scope_domain` the right representation, or should disjoint domains be a separate attribute?
2. **Declarations.** Option 3 emits up to D + 1 declarations per procedure, and the inliner adds as many per inline site. Is that acceptable? Or should Flang skip the "other accesses" scope, or the whole domain, when it adds nothing over the `noalias` arguments?
3. **`noalias` arguments.** Should Flang keep them? BasicAA uses them in the procedure itself and for non-Fortran callers. Or should it rely on the restrict scopes after inlining?
4. **Heap data.** Should allocatable data stay tagged by the name of its holder, given MOVE_ALLOC, allocatable function results and the reallocation tags of Section 4.5? Or should it move to the generic `allocated data`, at some loss of precision?
5. **LTO.** Is a layout hash in the root name an acceptable way to key a TBAA tree per LLVM module (3b)?
6. **TARGET dummies.** Should Option 3 drop the per-invocation fact "TARGET dummy versus non-TARGET named storage", or carry it in a second declared domain?

## 12. References

- [llvm-project#240818][i240818]: inlined copies sharing per-function TBAA roots.
- [llvm-project#242533][i242533]: the loop-carried miscompile of Section 4.2.
- [llvm-project#242536][i242536]: the reallocation tags of Section 4.5.
- [#242095][pr242095]: the inliner clones distinct TBAA roots (Option 0). Its review points out that TBAA was never intended to be used with distinct metadata.
- [#218768][pr218768]: alias scope domains with disjoint scopes. [#218769][pr218769]: the inliner uses a disjoint domain for `noalias` arguments.
- [#215366][pr215366]: dead store elimination of VALUE dummy copies, caused by inconsistent TBAA tags.
- [RFC: disjoint scope domains][rfc91537].
- T. Islam and K. Li, [Towards a representation of arbitrary alias graph in LLVM for Fortran][ibm2020], LLVM Developers' Meeting 2020.
- [D69542][d69542]: full restrict support (not landed).
- Fortran 2023 (ISO/IEC 1539-1:2023): 8.10.1 (EQUIVALENCE), 15.5.2.14 (restrictions on entities associated with dummy arguments), 19.5.3 (storage association).
- Code permalinks are at llvm-project `b53e4eea1879`.

## Appendix A. Kernels

Each kernel is new Fortran (k6 is IR), written for one alias fact. The IR comes from `flang -O2` before any LLVM pass (k5 adds `-mmlir -inline-all`). The metadata is rewritten into each encoding, verified, and checked with `aa-eval` after three pipelines: none; `cgscc(inline),function(sroa,early-cse<memssa>)`; and the latter followed by `simplifycfg,loop-mssa(licm),loop-rotate,loop-unroll` with `-unroll-count=2 -unroll-runtime`. Results are in the table in Section 9.7, and `-O2` code counts are given with each kernel.

**k1, inlined copies** (Section 4.1 gives the source). After inlining, today's encoding answers NoAlias for 2 pairs that relate the two copies. Every other encoding answers MayAlias.

**k2, loop-carried dependence** (Section 4.2 gives `step` and `run`). The driver:

```fortran
program main
  integer, parameter :: n = 1000
  real :: x(n)
  x = 0.0
  call run(x, n)
  if (x(n) == real(n - 1)) then
    print *, 'PASS', x(n)
  else
    print *, 'FAIL', x(n), ' expected', real(n - 1)
  end if
end program main
```

After `opt -O2`, the module has 12 loads, 15 stores and 2 vector loops without runtime checks under today's encoding and distinct roots, and the program prints `FAIL 8`. Under Options 2, 3 and 4 it has 6 loads, 9 stores and no vector loop, and prints `PASS 999`. With the `noalias` arguments removed, Option 4 also vectorizes and fails. `flang` alone fails at `-O2`, at `-O3` and with `-mmlir -inline-all`, and passes with `-mllvm -enable-tbaa=false`, with `-fno-vectorize` and at `-O1`.

**k3, cross-inline storage facts.**

```fortran
module state
  real, allocatable :: acc(:)
  real :: weights(1000)
contains
  subroutine accumulate(k)
    integer :: k
    acc(k) = acc(k) + 1.0
  end subroutine accumulate
end module state

subroutine drive(n)
  use state
  integer :: n, i
  do i = 1, n
    weights(i) = weights(i) * 0.5
    call accumulate(i)
  end do
end subroutine drive
```

Only 3b answers NoAlias for `weights(i)` against `acc(k)` after inlining. After `-O2`, 3b gives 11 loads, 5 stores and a vector loop without a runtime check. The others give 15 loads, 9 stores and a vector loop with a runtime check.

**k4, captured dummy.**

```fortran
module ptrs
  real, pointer :: p(:)
contains
  subroutine update(a, n)
    integer :: n
    real :: a(n)
    integer :: i
    external :: observe
    call observe(a)            ! LLVM must assume that the call captures a
    do i = 1, n
      a(i) = a(i) + p(i)
    end do
  end subroutine update
end module ptrs
```

`a` has no TARGET attribute, and `observe` cannot have a TARGET dummy without an explicit interface, so `p` is not associated with `a` during `update`. `noalias` cannot express this after the capture: Option 2 answers MayAlias, and its vector loop needs a runtime check. All other encodings answer NoAlias and need none.

**k5, FIR inlining** (compiled with `-mmlir -inline-all`).

```fortran
module kern
contains
  subroutine scatter(a, b, idx)
    real :: a(:), b(:)
    integer, pointer :: idx(:)
    integer :: i
    do i = 1, size(idx)
      a(idx(i)) = b(idx(i))
    end do
  end subroutine scatter
end module kern

subroutine caller(a, b, idx)
  use kern
  real, target :: a(:), b(:)
  integer, pointer :: idx(:)
  call scatter(a, b, idx)
end subroutine caller
```

In `caller`, today's encoding answers MayAlias between the descriptor loads of `idx` (enclosing function's tree) and the inlined store to `a` (scope root). Options 2 and 3 answer NoAlias. Option 2 loses the facts of the assumed-shape dummy `a` against `b` and against the data of `idx`, both in `caller` and in `scatter` itself.

**k6, merging** (IR). Two stores with different scopes of one disjoint domain are sunk into one store by `simplifycfg<sink-common-insts>`. The merged store carries the union of the two scopes. It stays NoAlias with an access in the "other accesses" scope and becomes MayAlias with an access in either original scope, as specified by LangRef.

**k7, the 2020 talk's example.**

```fortran
module m
  integer, allocatable, target :: a1, a2
  integer, pointer :: ptr
  integer :: b1
contains
  subroutine init_ptr()
    ptr = 0
  end subroutine init_ptr
  subroutine compute(n)
    integer :: idx, n
    do idx = 1, n
      a1 = a1 * b1 + a2 * b1
    end do
  end subroutine compute
end module m
```

Without storage facts, Options 2 and 3c answer MayAlias for 6 pairs. These are the loads of `b1` and `a2` against the store to `a1`, the same loads against the store into the buffer that reallocation on assignment would create for `a1`, and that store against the load and the store of `a1`'s current buffer. 11 loads and 6 stores remain after `-O2` instead of 7 and 2. Alias metadata before optimization:

| | today | 2 | 3a | 3b | 3c | 4 |
|---|---:|---:|---:|---:|---:|---:|
| alias metadata nodes | 27 | 6 | 30 | 24 | 11 | 26 |
| scope lists | 0 | 0 | 8 | 2 | 2 | 15 |
| `!alias.scope` / `!noalias` attachments | 0 / 0 | 0 / 0 | 9 / 0 | 8 / 0 | 8 / 0 | 17 / 17 |

**Reallocation** (Section 4.5 gives `s`). The driver calls `s(7, r)` and checks `r == 7`. It prints PASS at `-O2` and `-O3`, because the read of `j` is forwarded from the store (Section 4.5). The reproducer of [llvm-project#242536][i242536] has `s` and the same assignment for an allocatable dummy argument, a module variable, a module variable with TARGET and a local, with the commands and outputs. At `-O2`, each of its procedures stores `n` into `r` without reading the buffer. We also checked a `real` dummy argument, which is affected in the same way, and an array, a derived-type scalar and a deferred-length character scalar, which are not (Section 4.5).

## Appendix B. Projection formulas

For a function root, let D be the number of named dummies it accesses, O = 1 if it accesses other named storage and 0 otherwise, and U = D + O. Let K be the set of access kinds used in the module, among "any access", descriptor and data.

- **Option 2.** One TBAA root; 1 + \[data ∈ K\] + \[descriptor ∈ K\] type nodes; |K| tags; one `!tbaa` attachment per access, as today; no scopes.
- **Option 3c.** Option 2, plus, for every root with U ≥ 2: one domain, U scopes, U single-scope lists, U declarations, and one `!alias.scope` attachment per access to a dummy or to other named storage.
- **Option 3b.** Option 3c, plus today's storage types without the dummy leaves, in one tree per LLVM module.
- **Option 3a.** Option 3c, plus, per root, one undeclared disjoint domain per inner node of the storage tree it uses, one scope per child edge, and one list per distinct combination of restrict and storage scopes.
- **Option 4.** Per root with N used tree nodes: one domain, N scopes, and for each node an `!alias.scope` list and a `!noalias` list of the unrelated nodes. For N pairwise disjoint nodes, the lists have N² operands (270,400 for the 520 leaves of the D = 512, A = 10,000 stress test).
- **Per inline site.** Distinct roots: a copy of the callee's tree nodes and tags. Option 3: the inliner clones the domain, the U scopes, their lists and declarations (for 3a, also the storage domains). Option 4: the inliner clones the domain and all of its lists, so N² operands per site. In every option, the inliner adds a domain, scopes and declarations for the callee's `noalias` arguments, as today.

The Option 2 and 3c formulas reproduce every count of the rewritten modules on all 4,365 modules. The 3a, 3b and Option 4 numbers in Section 9 are counted on the rewritten modules.

## Appendix C. Commands and revisions

- flang: llvm-project `034b938cbbaa78957e1a94b8ef56bfeb6c3a2021`.
- opt, llvm-as, llvm-bcanalyzer: llvm-project `b53e4eea1879c9a117296b1cf3c1b1c794aa4132` plus the inliner change of [#242095][pr242095] (commit `9960e1408d95`); `CMAKE_BUILD_TYPE=Release`, `LLVM_ENABLE_ASSERTIONS=OFF`, X86 target only.
- llvm-test-suite `19b349c5aa391729e688f9b4b53ded177e1531cb`, CloverLeaf_Serial `b9a2b9c496b5eb1e7e30912d58e32d9dce930a0c`.
- IR before LLVM passes: `flang -O2 -S -emit-llvm -mllvm -opt-bisect-limit=0`.
- Alias results: `opt -passes=<pipeline>,aa-eval -aa-pipeline=basic-aa,scoped-noalias-aa,tbaa -evaluate-aa-metadata -print-all-alias-modref-info -disable-output`.
- Compile time: `opt -passes='default<O2>' -disable-output <module>.bc`.

[i240818]: https://github.com/llvm/llvm-project/issues/240818
[i242533]: https://github.com/llvm/llvm-project/issues/242533
[i242536]: https://github.com/llvm/llvm-project/issues/242536
[pr218768]: https://github.com/llvm/llvm-project/pull/218768
[pr218769]: https://github.com/llvm/llvm-project/pull/218769
[pr215366]: https://github.com/llvm/llvm-project/pull/215366
[pr242095]: https://github.com/llvm/llvm-project/pull/242095
[rfc91537]: https://discourse.llvm.org/t/rfc-disjoint-scope-alias-scope-noalias-domains/91537
[ibm2020]: https://llvm.org/devmtg/2020-09/slides/Islam-Li-Fortran_alias_representation.pdf
[d69542]: https://reviews.llvm.org/D69542
[testsuite]: https://github.com/llvm/llvm-test-suite/tree/19b349c5aa391729e688f9b4b53ded177e1531cb/Fortran
[cloverleaf]: https://github.com/UK-MAC/CloverLeaf_Serial/tree/b9a2b9c496b5eb1e7e30912d58e32d9dce930a0c
[addaliastags]: https://github.com/llvm/llvm-project/blob/b53e4eea1879c9a117296b1cf3c1b1c794aa4132/flang/lib/Optimizer/Transforms/AddAliasTags.cpp#L690-L926
[addaliastags-records]: https://github.com/llvm/llvm-project/blob/b53e4eea1879c9a117296b1cf3c1b1c794aa4132/flang/lib/Optimizer/Transforms/AddAliasTags.cpp#L660-L667
[addaliastags-direct]: https://github.com/llvm/llvm-project/blob/b53e4eea1879c9a117296b1cf3c1b1c794aa4132/flang/lib/Optimizer/Transforms/AddAliasTags.cpp#L792-L811
[addaliastags-dummy]: https://github.com/llvm/llvm-project/blob/b53e4eea1879c9a117296b1cf3c1b1c794aa4132/flang/lib/Optimizer/Transforms/AddAliasTags.cpp#L699-L717
[addaliastags-alloc]: https://github.com/llvm/llvm-project/blob/b53e4eea1879c9a117296b1cf3c1b1c794aa4132/flang/lib/Optimizer/Transforms/AddAliasTags.cpp#L908-L915
[tbaaforest]: https://github.com/llvm/llvm-project/blob/b53e4eea1879c9a117296b1cf3c1b1c794aa4132/flang/lib/Optimizer/Analysis/TBAAForest.cpp#L37-L63
[tbaabuilder]: https://github.com/llvm/llvm-project/blob/b53e4eea1879c9a117296b1cf3c1b1c794aa4132/flang/lib/Optimizer/CodeGen/TBAABuilder.cpp#L50-L88
[tbaabuilder-todo]: https://github.com/llvm/llvm-project/blob/b53e4eea1879c9a117296b1cf3c1b1c794aa4132/flang/lib/Optimizer/CodeGen/TBAABuilder.cpp#L54-L55
[firinliner]: https://github.com/llvm/llvm-project/blob/b53e4eea1879c9a117296b1cf3c1b1c794aa4132/flang/lib/Optimizer/Dialect/Inliner.cpp#L12-L25
[functionattr]: https://github.com/llvm/llvm-project/blob/b53e4eea1879c9a117296b1cf3c1b1c794aa4132/flang/lib/Optimizer/Transforms/FunctionAttr.cpp#L57-L96
[forcenoalias]: https://github.com/llvm/llvm-project/blob/b53e4eea1879c9a117296b1cf3c1b1c794aa4132/flang/lib/Optimizer/Passes/Pipelines.cpp#L483-L495
[langref-tbaa]: https://github.com/llvm/llvm-project/blob/b53e4eea1879c9a117296b1cf3c1b1c794aa4132/llvm/docs/LangRef.md?plain=1#L7527-L7545
[langref-scopes]: https://github.com/llvm/llvm-project/blob/b53e4eea1879c9a117296b1cf3c1b1c794aa4132/llvm/docs/LangRef.md?plain=1#L7710-L7760
[langref-decl]: https://github.com/llvm/llvm-project/blob/b53e4eea1879c9a117296b1cf3c1b1c794aa4132/llvm/docs/LangRef.md?plain=1#L26538-L26605
[tbaamerge]: https://github.com/llvm/llvm-project/blob/b53e4eea1879c9a117296b1cf3c1b1c794aa4132/llvm/lib/Analysis/TypeBasedAliasAnalysis.cpp#L437
[scopedaa]: https://github.com/llvm/llvm-project/blob/b53e4eea1879c9a117296b1cf3c1b1c794aa4132/llvm/lib/Analysis/ScopedNoAliasAA.cpp#L169-L230
[mdmerge]: https://github.com/llvm/llvm-project/blob/b53e4eea1879c9a117296b1cf3c1b1c794aa4132/llvm/lib/IR/Metadata.cpp#L1148-L1178
[clonescopes]: https://github.com/llvm/llvm-project/blob/b53e4eea1879c9a117296b1cf3c1b1c794aa4132/llvm/lib/Transforms/Utils/CloneFunction.cpp#L1158-L1197
[laa-adjust]: https://github.com/llvm/llvm-project/blob/b53e4eea1879c9a117296b1cf3c1b1c794aa4132/llvm/lib/Analysis/LoopAccessAnalysis.cpp#L1713-L1734
[licm-local]: https://github.com/llvm/llvm-project/blob/b53e4eea1879c9a117296b1cf3c1b1c794aa4132/llvm/lib/Transforms/Scalar/LICM.cpp#L400-L412
[inline-clone]: https://github.com/llvm/llvm-project/blob/b53e4eea1879c9a117296b1cf3c1b1c794aa4132/llvm/lib/Transforms/Utils/InlineFunction.cpp#L1115-L1214
[inline-noalias]: https://github.com/llvm/llvm-project/blob/b53e4eea1879c9a117296b1cf3c1b1c794aa4132/llvm/lib/Transforms/Utils/InlineFunction.cpp#L1220-L1463
[basicaa]: https://github.com/llvm/llvm-project/blob/b53e4eea1879c9a117296b1cf3c1b1c794aa4132/llvm/lib/Analysis/BasicAliasAnalysis.cpp#L1582-L1614
[instcombine-decl]: https://github.com/llvm/llvm-project/blob/b53e4eea1879c9a117296b1cf3c1b1c794aa4132/llvm/lib/Transforms/InstCombine/InstructionCombining.cpp#L6039-L6122
[mlir-domain]: https://github.com/llvm/llvm-project/blob/b53e4eea1879c9a117296b1cf3c1b1c794aa4132/mlir/include/mlir/Dialect/LLVMIR/LLVMAttrDefs.td#L1103-L1129
[mlir-decl]: https://github.com/llvm/llvm-project/blob/b53e4eea1879c9a117296b1cf3c1b1c794aa4132/mlir/include/mlir/Dialect/LLVMIR/LLVMIntrinsicOps.td#L398-L405
