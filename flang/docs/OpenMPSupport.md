<!--===- docs/OpenMPSupport.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# Flang OpenMP Support

```{raw} html
<style type="text/css">
   .none { background-color: #FFCCCC }
   .part { background-color: #FFFF99 }
   .progress { background-color: #CCE5FF }
   .good { background-color: #CCFF99 }

   .none,
   .part,
   .progress,
   .good {
      border-radius: 0.25rem;
      color: inherit;
      display: inline-block;
      font-weight: 600;
      padding: 0.1rem 0.4rem;
      white-space: nowrap;
   }

   table.docutils td p,
   table.docutils th p,
   table.docutils td,
   table.docutils th {
      text-align: left;
      text-justify: auto;
      word-spacing: normal;
   }
</style>
```

```{contents}
---
local:
---
```

This document outlines the OpenMP API features supported by Flang. It is
intended as a general reference.  For the most accurate information on
unimplemented features, rely on the compiler’s TODO or “Not Yet Implemented”
messages, which are considered authoritative. Flang provides complete
implementation of the OpenMP 3.1 specification and partial implementation of
OpenMP 4.0.  The sections below summarize support status for OpenMP 4.0, 5.0,
5.1, 5.2, 6.0, and future 6.1.  The table entries are derived from the
information provided in the Version Differences subsection of the Features
History section in the OpenMP standard.

The Status column uses the following values:
- <span class="none">unclaimed</span> : No implementation is known to be underway.
- <span class="part">partial</span> : Some support exists, but important cases are still missing.
- <span class="progress">in progress</span> : Work is actively underway, or the implementation is available only experimentally.
- <span class="good">done</span> : The feature is considered implemented.

The OpenMP 4.0 section uses the historical `Feature/Status/Comments` format with
an added `Claimed By` column. OpenMP 5.0 and newer sections use
`Feature/Status/Claimed By/Notes/Reviews` to capture implementation state,
ownership, and upstream references.  Use GitHub usernames in `Claimed By` (for
example `@alice`). Leave the cell blank when nobody has claimed the work.

This page is for OpenMP features that require changes in Flang compiler
components (parser, semantics, lowering, diagnostics) and that may additionally
entail corresponding runtime support.  OpenMP features whose status does not
involve updates to the Flang compiler are tracked under the Clang OpenMP Support
page (`clang/docs/OpenMPSupport.rst`).

## Updating This Page

- When claiming a feature, set `Status` to <span class="progress">in progress</span> and add the GitHub username in `Claimed By`.
- When a GitHub PR is opened, add the PR link in `Reviews`. Keep the claimant listed until the work is finished or unclaimed.
- When support lands but remains incomplete, set `Status` to <span class="part">partial</span> and summarize the missing parser, semantics, lowering, diagnostics, or test work in `Notes`.
- When support is complete, set `Status` to <span class="good">done</span>. Clear `Claimed By` if no further follow-up is expected.
- When work is abandoned or not yet started, set `Status` to <span class="none">unclaimed</span>, remove the claimant, and keep `Notes` focused on the missing implementation work.

Note: In the OpenMP 4.0 section, no distinction is made between support in
Parser/Semantics, MLIR, Lowering, or the OpenMPIRBuilder.

## OpenMP 4.0

| Feature                                                    | Status                            | Claimed By | Comments                                                |
|:-----------------------------------------------------------|:----------------------------------|:-----------|:--------------------------------------------------------|
| proc_bind clause                                           | <span class="good">done</span> | | |
| simd construct                                             | <span class="part">partial</span> | | Implicit linearization is skipped if iv is a pointer or allocatable|
| declare simd construct                                     | <span class="part">partial</span> | | Semantics coverage exists (for example `flang/test/Semantics/OpenMP/declarative-directive01.f90`) and lowering exists for key forms (for example `flang/test/Lower/OpenMP/declare-simd-interface-body.f90`), but coverage is not yet complete for all variants. |
| do simd construct                                          | <span class="part">partial</span> | | Implicit linearization is skipped if iv is a pointer or allocatable |
| target data construct                                      | <span class="good">done</span> | | |
| target construct                                           | <span class="good">done</span> | | |
| target update construct                                    | <span class="good">done</span> | | |
| declare target directive                                   | <span class="good">done</span> | | |
| teams construct                                            | <span class="good">done</span> | | |
| distribute construct                                       | <span class="good">done</span> | | |
| distribute simd construct                                  | <span class="part">partial</span> | | Implicit linearization is skipped if iv is a pointer or allocatable |
| distribute parallel loop construct                         | <span class="good">done</span> | | |
| distribute parallel loop simd construct                    | <span class="part">partial</span> | | Implicit linearization is skipped if iv is a pointer or allocatable |
| depend clause                                              | <span class="good">done</span> | | |
| declare reduction construct                                | <span class="part">partial</span> | | Partial support, including user-defined reductions with derived types. |
| atomic construct extensions                                | <span class="good">done</span> | | |
| cancel construct                                           | <span class="good">done</span> | | |
| cancellation point construct                               | <span class="good">done</span> | | |
| parallel do simd construct                                 | <span class="part">partial</span> | | Implicit linearization is skipped if iv is a pointer or allocatable |
| target teams construct                                     | <span class="good">done</span> | | |
| teams distribute construct                                 | <span class="good">done</span> | | |
| teams distribute simd construct                            | <span class="part">partial</span> | | Implicit linearization is skipped if iv is a pointer or allocatable |
| target teams distribute construct                          | <span class="good">done</span> | | |
| teams distribute parallel loop construct                   | <span class="good">done</span> | | |
| target teams distribute parallel loop construct            | <span class="good">done</span> | | |
| teams distribute parallel loop simd construct              | <span class="part">partial</span> | | Implicit linearization is skipped if iv is a pointer or allocatable |
| target teams distribute parallel loop simd construct       | <span class="part">partial</span> | | Implicit linearization is completely skipped |

## OpenMP 5.0

| Feature | Status | Claimed By | Notes | Reviews |
|:--------|:-------|:-----------|:------|:--------|
| taskloop and combined taskloop forms | <span class="part">partial</span> | | Covered in semantics/lowering tests including `flang/test/Semantics/OpenMP/taskloop01.f90`, `flang/test/Semantics/OpenMP/taskloop-simd01.f90`, `flang/test/Lower/OpenMP/taskloop.f90`, and `flang/test/Lower/OpenMP/master_taskloop_simd.f90`. Lowering for taskloop simd and combined forms is still TODO. | [llvm/llvm-project#138646](https://github.com/llvm/llvm-project/pull/138646), [llvm/llvm-project#165851](https://github.com/llvm/llvm-project/pull/165851), [llvm/llvm-project#187222](https://github.com/llvm/llvm-project/pull/187222) |
| memory allocators | <span class="part">partial</span> | | Semantics coverage exists in `flang/test/Semantics/OpenMP/allocators01.f90`-`flang/test/Semantics/OpenMP/allocators07.f90`; lowering gaps remain (for example `flang/lib/Lower/OpenMP/OpenMP.cpp` TODO paths). | |
| allocate directive and allocate clause | <span class="part">partial</span> | | Semantics coverage exists in `flang/test/Semantics/OpenMP/allocate-directive.f90` and `flang/test/Semantics/OpenMP/allocate-clause01.f90`; lowering support expanded, with remaining TODO coverage (`flang/test/Lower/OpenMP/Todo/allocate-clause-align.f90`, `flang/test/Lower/OpenMP/Todo/allocate-clause-allocator.f90`). | [llvm/llvm-project#121356](https://github.com/llvm/llvm-project/pull/121356), [llvm/llvm-project#165719](https://github.com/llvm/llvm-project/pull/165719), [llvm/llvm-project#165865](https://github.com/llvm/llvm-project/pull/165865), [llvm/llvm-project#187167](https://github.com/llvm/llvm-project/pull/187167) |
| metadirective | <span class="part">partial</span> | | Semantics coverage exists in `flang/test/Semantics/OpenMP/metadirective-construct.f90`; lowering support exists for several construct-selector paths (`flang/test/Lower/OpenMP/metadirective-construct.f90`, `flang/test/Lower/OpenMP/metadirective-nothing.f90`), but some selector/variant paths remain TODO-tracked in lowering. | [llvm/llvm-project#159945](https://github.com/llvm/llvm-project/pull/159945), [llvm/llvm-project#193664](https://github.com/llvm/llvm-project/pull/193664), [llvm/llvm-project#194402](https://github.com/llvm/llvm-project/pull/194402), [llvm/llvm-project#194424](https://github.com/llvm/llvm-project/pull/194424) |
| support full defaultmap functionality | <span class="part">partial</span> | | Core coverage exists (`flang/test/Lower/OpenMP/defaultmap.f90`, `flang/test/Semantics/OpenMP/defaultmap-clause-v50.f90`), but lowering has known partial paths (for example defaultmap-firstprivate TODO tests). | [llvm/llvm-project#135226](https://github.com/llvm/llvm-project/pull/135226), [llvm/llvm-project#166715](https://github.com/llvm/llvm-project/pull/166715), [llvm/llvm-project#167806](https://github.com/llvm/llvm-project/pull/167806), [llvm/llvm-project#177389](https://github.com/llvm/llvm-project/pull/177389), [llvm/llvm-project#190764](https://github.com/llvm/llvm-project/pull/190764) |
| clause: uses_allocators | <span class="part">partial</span> | | Parsed/checked in semantics allocator tests, but lowering has explicit TODO handling (`cp.processTODO<...UsesAllocators>` in `flang/lib/Lower/OpenMP/OpenMP.cpp`). | |
| clause: in_reduction | <span class="part">partial</span> | | Semantics and lowering coverage exists for several task/taskgroup/taskloop forms (`flang/test/Semantics/OpenMP/in-reduction.f90`, `flang/test/Lower/OpenMP/task-inreduction.f90`, `flang/test/Lower/OpenMP/taskloop-inreduction.f90`, `flang/test/Lower/OpenMP/taskgroup-task_reduction02.f90`); some target-related forms remain TODO (for example `flang/test/Lower/OpenMP/Todo/target-inreduction.f90`). | [llvm/llvm-project#139704](https://github.com/llvm/llvm-project/pull/139704), [llvm/llvm-project#205124](https://github.com/llvm/llvm-project/pull/205124) |
| user-defined mappers | <span class="good">done</span> | | Supported with semantics/lowering/transform coverage. | [llvm/llvm-project#140560](https://github.com/llvm/llvm-project/pull/140560), [llvm/llvm-project#163860](https://github.com/llvm/llvm-project/pull/163860), [llvm/llvm-project#167903](https://github.com/llvm/llvm-project/pull/167903), [llvm/llvm-project#179936](https://github.com/llvm/llvm-project/pull/179936), [llvm/llvm-project#189136](https://github.com/llvm/llvm-project/pull/189136) |
| map array-section with implicit mapper | <span class="part">partial</span> | | Mapper and map coverage exists (`flang/test/Lower/OpenMP/map-mapper.f90`, `flang/test/Lower/OpenMP/target-data-skip-mapper-calls.f90`), with remaining iterator/modifier gaps in lowering (`flang/lib/Lower/OpenMP/ClauseProcessor.cpp` TODOs). | [llvm/llvm-project#175133](https://github.com/llvm/llvm-project/pull/175133), [llvm/llvm-project#177389](https://github.com/llvm/llvm-project/pull/177389) |
| clause: use_device_addr for target data | <span class="good">done</span> | | Supported for core forms. | [llvm/llvm-project#82834](https://github.com/llvm/llvm-project/pull/82834), [llvm/llvm-project#176815](https://github.com/llvm/llvm-project/pull/176815) |
| support non-contiguous array sections for target update | <span class="part">partial</span> | | Target update coverage exists (`flang/test/Semantics/OpenMP/target-update01.f90`, `flang/test/Semantics/OpenMP/target-update-mapper.f90`), with additional corner-case validation still ongoing. | |
| pointer attachment | <span class="part">partial</span> | | Pointer mapping coverage exists (`flang/test/Semantics/OpenMP/use_device_ptr.f90`, `flang/test/Lower/OpenMP/pointer-to-array.f90`), with descriptor and attach-related TODOs remaining in lowering. | |
| hints for the atomic construct | <span class="part">partial</span> | | Semantics coverage exists (`flang/test/Semantics/OpenMP/atomic-hint-clause.f90`) and lowering coverage exists for common atomic forms; compare/fail forms remain partial. | |
| conditional modifier for lastprivate clause | <span class="none">unclaimed</span> | | Lastprivate lowering/semantics coverage exists for base forms, but no dedicated support coverage for the conditional modifier was identified in current Flang tests. | |
| task affinity | <span class="part">partial</span> | | Semantics and lowering coverage exists, including iterator-based affinity in task constructs (`flang/test/Semantics/OpenMP/affinity-clause.f90`, `flang/test/Lower/OpenMP/task-affinity.f90`). | [llvm/llvm-project#179003](https://github.com/llvm/llvm-project/pull/179003), [llvm/llvm-project#182222](https://github.com/llvm/llvm-project/pull/182222) |
| iterator modifier for depend clause | <span class="good">done</span> | | Semantics/lowering coverage exists (`flang/test/Lower/OpenMP/depend-iterator.f90`). | [llvm/llvm-project#189412](https://github.com/llvm/llvm-project/pull/189412) |
| scan directive and inscan modifier for reduction | <span class="part">partial</span> | | Semantics and lowering coverage exists (`flang/test/Semantics/OpenMP/scan1.f90`, `flang/test/Semantics/OpenMP/scan2.f90`, `flang/test/Lower/OpenMP/scan.f90`), but breadth across all reduction combinations should continue to be validated. | [llvm/llvm-project#102792](https://github.com/llvm/llvm-project/pull/102792), [llvm/llvm-project#123254](https://github.com/llvm/llvm-project/pull/123254) |
| reduction and in_reduction clauses on taskloop | <span class="part">partial</span> | | Taskloop reduction and in_reduction lowering coverage exists (`flang/test/Lower/OpenMP/taskloop-reduction.f90`, `flang/test/Lower/OpenMP/taskloop-inreduction.f90`), with remaining TODO coverage in some combinations. | [llvm/llvm-project#205124](https://github.com/llvm/llvm-project/pull/205124) |
| close modifier in map clause | <span class="part">partial</span> | | Close semantics are covered in targeted lowering tests (for example `flang/test/Lower/OpenMP/cptr-usm-close-and-use-device-ptr.f90`), with continued validation needed across broader mapping combinations. | [llvm/llvm-project#163258](https://github.com/llvm/llvm-project/pull/163258) |
| mapping Fortran pointer and allocatable variables | <span class="part">partial</span> | | Pointer/allocatable mapping coverage exists (`flang/test/Semantics/OpenMP/use_device_ptr.f90`, `flang/test/Lower/OpenMP/wsloop-reduction-allocatable.f90`, `flang/test/Lower/OpenMP/wsloop-reduction-pointer.f90`), but not all descriptor/attachment edge cases are complete. | |
| declare variant directive | <span class="part">partial</span> | | Frontend semantics support is substantial (`flang/test/Semantics/OpenMP/declare-variant.f90`, `flang/test/Semantics/OpenMP/declare-variant-match.f90`), while lowering remains TODO-tracked (`flang/test/Lower/OpenMP/Todo/declare-variant.f90`). | [llvm/llvm-project#130578](https://github.com/llvm/llvm-project/pull/130578), [llvm/llvm-project#198799](https://github.com/llvm/llvm-project/pull/198799), [llvm/llvm-project#206714](https://github.com/llvm/llvm-project/pull/206714) |
| implicit declare target directive | <span class="none">unclaimed</span> | | No dedicated Flang parser/semantics/lowering coverage for implicit declare target handling was identified in current OpenMP test coverage. | |
| requires directive | <span class="part">partial</span> | | Frontend and lowering coverage exists (`flang/test/Semantics/OpenMP/requires01.f90`-`requires10.f90`, `flang/test/Lower/OpenMP/requires.f90`), but some clauses are still flagged as unsupported (for example reverse_offload warning path). | [llvm/llvm-project#204647](https://github.com/llvm/llvm-project/pull/204647) |
| teams construct on host | <span class="good">done</span> | | Teams support is established and exercised across semantics/lowering coverage in Flang OpenMP tests. | |
| loop construct and order(concurrent) clause | <span class="part">partial</span> | | Loop and order-related coverage exists (`flang/test/Semantics/OpenMP/compiler-directives-loop.f90`, `flang/test/Semantics/OpenMP/order-clause01.f90`, `flang/test/Lower/OpenMP/loop-directive.f90`, `flang/test/Lower/OpenMP/order-clause.f90`), with some transformations still evolving. | |
| collapsing imperfectly nested loops | <span class="none">unclaimed</span> | | Current checks primarily diagnose non-perfect nests (for example `flang/test/Semantics/OpenMP/do-collapse.f90`), and no dedicated support for imperfect-nest collapsing was identified. | [llvm/llvm-project#202435](https://github.com/llvm/llvm-project/pull/202435) |
| if clause and nontemporal clause on simd | <span class="part">partial</span> | | SIMD nontemporal coverage exists (`flang/test/Semantics/OpenMP/nontemporal.f90`), but complete OpenMP 5.0-level coverage for all if(simd)/nontemporal combinations remains incomplete. | |
| atomic in simd | <span class="none">unclaimed</span> | | No dedicated Flang OpenMP coverage for atomic-in-simd forms was identified in current parser/semantics/lowering tests. | |
| detach clause on task and omp_fulfill_event routine | <span class="good">done</span> | | Flang semantics and lowering coverage exists (`flang/test/Semantics/OpenMP/detach01.f90`, `flang/test/Semantics/OpenMP/detach02.f90`, `flang/test/Lower/OpenMP/task_detach.f90`); runtime routine is available in OpenMP module/runtime. | [llvm/llvm-project#119172](https://github.com/llvm/llvm-project/pull/119172), [llvm/llvm-project#119128](https://github.com/llvm/llvm-project/pull/119128) |
| taskloop construct can be canceled by cancel construct | <span class="part">partial</span> | | Dedicated lowering coverage exists (`flang/test/Lower/OpenMP/taskloop-cancel.f90`), and semantics checks include taskloop nesting constraints for cancel/cancellation-point. | |
| reverse offload | <span class="none">unclaimed</span> | | Flang currently emits an unsupported warning path for reverse_offload in requires handling (`flang/test/Semantics/OpenMP/requires01.f90`). | [llvm/llvm-project#204647](https://github.com/llvm/llvm-project/pull/204647) |
| depend clause on taskwait | <span class="part">partial</span> | | Lowering is explicitly TODO-tracked (`flang/test/Lower/OpenMP/Todo/taskwait-depend.f90`); runtime support exists in OpenMP runtime tests. | [llvm/llvm-project#111562](https://github.com/llvm/llvm-project/pull/111562) |
| acquire/release clauses on atomic and flush | <span class="part">partial</span> | | Atomic acquire/release semantics coverage exists (`flang/test/Semantics/OpenMP/atomic-mem-order.f90`, `flang/test/Semantics/OpenMP/atomic-compare.f90`), but full end-to-end coverage breadth remains in progress. | |
| mutexinoutset on depend clause | <span class="good">done</span> | | Semantics/lowering coverage exists for mutexinoutset depend handling (`flang/test/Semantics/OpenMP/depend06.f90`, `flang/test/Lower/OpenMP/task.f90`). | |
| depobj construct | <span class="part">partial</span> | | Semantics coverage exists for v5.0+ forms (`flang/test/Semantics/OpenMP/depobj-construct-v50.f90`), but lowering still has TODO paths (`flang/test/Lower/OpenMP/Todo/depobj-construct.f90`, `flang/test/Lower/OpenMP/Todo/depend-clause-depobj.f90`). | |
| combined master constructs (master taskloop, parallel master, parallel master taskloop, master taskloop simd, parallel master taskloop simd) | <span class="part">partial</span> | | Frontend/lowering coverage exists for multiple master combined forms (`flang/test/Lower/OpenMP/parallel-master.f90`, `flang/test/Lower/OpenMP/master_taskloop_simd.f90`, `flang/test/Lower/OpenMP/parallel-master-taskloop-simd.f90`) with additional completeness work tracked alongside taskloop combined forms. | [llvm/llvm-project#113893](https://github.com/llvm/llvm-project/pull/113893) |

## OpenMP 5.1

| Feature | Status | Claimed By | Notes | Reviews |
|:--------|:-------|:-----------|:------|:--------|
| compare clause on atomic construct | <span class="part">partial</span> | | Semantics and lowering coverage exist (`flang/test/Semantics/OpenMP/atomic-compare.f90`, `flang/test/Lower/OpenMP/atomic-compare.f90`); remaining gaps are primarily fail/capture combinations and broader type coverage. | [llvm/llvm-project#184761](https://github.com/llvm/llvm-project/pull/184761) |
| fail clause on atomic construct | <span class="part">partial</span> | | Semantics coverage exists (`flang/test/Semantics/OpenMP/atomic-compare.f90`), but lowering for `fail(...)` paths is still TODO (`flang/test/Lower/OpenMP/Todo/atomic-compare-fail.f90`). Complete lowering for compare+fail(+capture), then add non-TODO lowering tests. | [llvm/llvm-project#184761](https://github.com/llvm/llvm-project/pull/184761) |
| interop construct | <span class="part">partial</span> | | Semantics coverage exists (`flang/test/Semantics/OpenMP/interop-construct.f90`), but lowering remains TODO-tracked (`flang/lib/Lower/OpenMP/OpenMP.cpp` TODO: `OpenMPInteropConstruct`; `flang/test/Lower/OpenMP/Todo/interop-construct.f90`). | |
| dispatch construct | <span class="part">partial</span> | | Semantics coverage exists (`flang/test/Semantics/OpenMP/dispatch.f90`), but lowering remains TODO-tracked (`flang/lib/Lower/OpenMP/OpenMP.cpp` TODO: `OpenMPDispatchConstruct`; `flang/test/Lower/OpenMP/Todo/dispatch.f90`). | |
| masked construct | <span class="part">partial</span> | | Covered in semantics/lowering (`flang/test/Semantics/OpenMP/masked.f90`, `flang/test/Lower/OpenMP/masked.f90`). | [llvm/llvm-project#91432](https://github.com/llvm/llvm-project/pull/91432) |
| masked combined constructs | <span class="part">partial</span> | | Covered in lowering tests (`flang/test/Lower/OpenMP/masked_taskloop.f90`, `flang/test/Lower/OpenMP/parallel-masked-taskloop.f90`) with ongoing breadth expansion. | |
| present map type modifier | <span class="part">partial</span> | | Semantics coverage exists (`flang/test/Semantics/OpenMP/present.f90`) and map lowering exists, with ongoing completeness checks. | |
| present modifier in motion clauses | <span class="part">partial</span> | | Motion/update coverage exists (`flang/test/Semantics/OpenMP/target-update01.f90`, `flang/test/Semantics/OpenMP/target-update-mapper.f90`), with ongoing completeness checks. | |
| present in defaultmap clause | <span class="part">partial</span> | | Defaultmap semantics coverage exists (`flang/test/Semantics/OpenMP/defaultmap-clause-v50.f90`), with ongoing completeness checks. | |
| thread_limit clause on target construct | <span class="part">partial</span> | | Lowering/semantics coverage exists for thread-limit dimensions (`flang/test/Lower/OpenMP/thread-limit-dims.f90`), with additional target interactions still under validation. | [llvm/llvm-project#171454](https://github.com/llvm/llvm-project/pull/171454), [llvm/llvm-project#171825](https://github.com/llvm/llvm-project/pull/171825) |
| has_device_addr clause on target construct | <span class="part">partial</span> | | Semantics/lowering coverage exists (`flang/test/Lower/OpenMP/has_device_addr-mapinfo.f90`) with ongoing validation for all mapping combinations. | |
| iterators in map or motion clauses | <span class="part">partial</span> | | Some iterator coverage exists (`flang/test/Lower/OpenMP/depend-iterator.f90`), but lowering still has explicit iterator TODOs in `flang/lib/Lower/OpenMP/ClauseProcessor.cpp`. | |
| omp_all_memory reserved locator for depend clause | <span class="part">partial</span> | | Locator parsing/semantics groundwork exists, with remaining TODO lowering coverage for reserved locator handling (`flang/test/Lower/OpenMP/Todo/locator-reserved.f90`). | [llvm/llvm-project#203910](https://github.com/llvm/llvm-project/pull/203910) |
| align clause on allocate directive and allocator/align modifiers on allocate clause | <span class="part">partial</span> | | Allocate directive/clause support is present, but align-related lowering remains TODO-tracked (`flang/test/Lower/OpenMP/Todo/allocate-clause-align.f90`). | [llvm/llvm-project#121356](https://github.com/llvm/llvm-project/pull/121356), [llvm/llvm-project#165719](https://github.com/llvm/llvm-project/pull/165719) |
| target_device selector | <span class="part">partial</span> | | Semantics coverage exists for target_device selectors in metadirective/declare-variant matching (`flang/test/Semantics/OpenMP/metadirective-device.f90`, `flang/test/Semantics/OpenMP/declare-variant-match.f90`). | [llvm/llvm-project#123243](https://github.com/llvm/llvm-project/pull/123243), [llvm/llvm-project#206714](https://github.com/llvm/llvm-project/pull/206714) |
| adjust_args and append_args on declare variant | <span class="none">unclaimed</span> | | Parsing accepts forms, but semantics currently diagnose both clauses as not yet implemented (`flang/test/Semantics/OpenMP/declare-variant-match.f90`, `flang/test/Semantics/OpenMP/declare-variant.f90`). | [llvm/llvm-project#206714](https://github.com/llvm/llvm-project/pull/206714) |
| indirect clause on declare target | <span class="part">partial</span> | | Parser coverage exists (`flang/test/Parser/OpenMP/declare-target-indirect-tree.f90`), while lowering remains TODO-tracked (`flang/test/Lower/OpenMP/Todo/omp-clause-indirect.f90`). | [llvm/llvm-project#143505](https://github.com/llvm/llvm-project/pull/143505) |
| error directive | <span class="part">partial</span> | | Semantics coverage exists (`flang/test/Semantics/OpenMP/error.f90`), but lowering still has explicit TODO coverage (`flang/test/Lower/OpenMP/Todo/error.f90`). | [llvm/llvm-project#121509](https://github.com/llvm/llvm-project/pull/121509), [llvm/llvm-project#206175](https://github.com/llvm/llvm-project/pull/206175) |
| nothing directive | <span class="good">done</span> | | Parser and lowering coverage exists for standalone and metadirective-selected forms (`flang/test/Parser/OpenMP/nothing.f90`, `flang/test/Lower/OpenMP/nothing.f90`, `flang/test/Lower/OpenMP/metadirective-nothing.f90`). | [llvm/llvm-project#193664](https://github.com/llvm/llvm-project/pull/193664), [llvm/llvm-project#202679](https://github.com/llvm/llvm-project/pull/202679) |
| tile and unroll constructs | <span class="part">partial</span> | | Semantics coverage exists across tile/unroll and loop-transformation tests (`flang/test/Semantics/OpenMP/tile01.f90`, `flang/test/Semantics/OpenMP/tile09.f90`, `flang/test/Semantics/OpenMP/loop-transformation-construct01.f90`), with additional lowering/transform completeness work ongoing. | [llvm/llvm-project#160298](https://github.com/llvm/llvm-project/pull/160298), [llvm/llvm-project#185296](https://github.com/llvm/llvm-project/pull/185296), [llvm/llvm-project#188025](https://github.com/llvm/llvm-project/pull/188025) |
| scope construct | <span class="part">partial</span> | | Scope construct support is available, with follow-on completeness work still in progress for some combinations (see also OpenMP 5.2 scope-related rows). | [llvm/llvm-project#113700](https://github.com/llvm/llvm-project/pull/113700), [llvm/llvm-project#193098](https://github.com/llvm/llvm-project/pull/193098) |
| assumes directives | <span class="none">unclaimed</span> | | Lowering still has explicit TODO (`flang/lib/Lower/OpenMP/OpenMP.cpp` TODO: `OpenMP ASSUMES declaration`; `flang/test/Lower/OpenMP/Todo/assumes.f90`). | [llvm/llvm-project#102008](https://github.com/llvm/llvm-project/pull/102008) |
| assume directive | <span class="none">unclaimed</span> | | Lowering still has explicit TODO (`flang/lib/Lower/OpenMP/OpenMP.cpp` TODO: `OpenMP ASSUME construct`; `flang/test/Lower/OpenMP/Todo/assume.f90`). | [llvm/llvm-project#102008](https://github.com/llvm/llvm-project/pull/102008) |
| default(firstprivate) | <span class="part">partial</span> | | Semantics coverage exists (`flang/test/Semantics/OpenMP/default02.f90`), but defaultmap-firstprivate lowering remains partial (`flang/test/Lower/OpenMP/Todo/defaultmap-clause-firstprivate.f90`). | |
| default(private) | <span class="part">partial</span> | | Semantics/default-clause coverage exists (`flang/test/Semantics/OpenMP/default.f90`, `flang/test/Semantics/OpenMP/default02.f90`) with ongoing lowering completeness checks. | |
| inoutset in depend clause | <span class="part">partial</span> | | Semantics coverage exists (`flang/test/Semantics/OpenMP/depend06.f90`) with broader lowering/runtime verification ongoing. | |
| nowait clause on taskwait | <span class="part">partial</span> | | Taskwait coverage exists (`flang/test/Semantics/OpenMP/taskwait.f90`, `flang/test/Lower/OpenMP/taskwait.f90`), with nowait+depend combinations still partial (`flang/test/Lower/OpenMP/Todo/taskwait-nowait.f90`). | |
| seq_cst clause on flush | <span class="good">done</span> | | OpenMP 5.1 `flush seq_cst` semantics coverage exists (`flang/test/Semantics/OpenMP/flush02.f90`). | [llvm/llvm-project#114072](https://github.com/llvm/llvm-project/pull/114072) |
| omp_atv_serialized and omp_atv_default values for alloctrait_key | <span class="none">unclaimed</span> | | No dedicated Flang parser/semantics/lowering coverage for these alloctrait values was identified in current OpenMP tests. | |
| strict modifier for taskloop construct | <span class="none">unclaimed</span> | | No dedicated parser/semantics/lowering coverage for the taskloop strict modifier was identified in current Flang OpenMP tests. | |

## OpenMP 5.2

| Feature | Status | Claimed By | Notes | Reviews |
|:--------|:-------|:-----------|:------|:--------|
| if clause on teams construct | <span class="part">partial</span> | | Semantics coverage exists for teams-specific `if` forms (for example `flang/test/Semantics/OpenMP/if-clause.f90` with `if(teams: ...)` cases). | |
| ompx and omx sentinel for implementation extensions in free/fixed source | <span class="part">partial</span> | | Extension semantics coverage exists for `ompx` forms (`flang/test/Semantics/OpenMP/ompx-bare.f90`), while broader sentinel coverage (including fixed-form `omx`) remains limited. | [llvm/llvm-project#111106](https://github.com/llvm/llvm-project/pull/111106) |
| allow copyprivate and nowait clause on starting directive of construct | <span class="good">done</span> | | Semantics coverage includes valid starting-directive uses (`!$omp single nowait`, `!$omp single copyprivate(...)`) in `flang/test/Semantics/OpenMP/single04.f90`. | [llvm/llvm-project#204339](https://github.com/llvm/llvm-project/pull/204339), [llvm/llvm-project#205607](https://github.com/llvm/llvm-project/pull/205607) |
| step modifier | <span class="none">unclaimed</span> | | Add parser+semantics acceptance/diagnostics for OpenMP 5.2 step-modifier forms, then add lowering coverage showing emitted loop metadata/ops for accepted cases. | |
| declare mapper iterator modifier | <span class="none">unclaimed</span> | | Parser and semantics should accept iterator-modified DECLARE MAPPER forms, then lowering must thread iterator bounds through map info generation (see related TODO test `flang/test/Lower/OpenMP/Todo/declare-mapper-iterator.f90`). | |
| present modifier in map clauses on declare mapper | <span class="none">unclaimed</span> | | Iterator coverage is tracked separately above; no dedicated present-modifier support coverage was identified for DECLARE MAPPER map clauses. | |
| enter clause replaces to clause on declare target | <span class="good">done</span> | | Semantics coverage exists for `declare target enter(...)` forms (`flang/test/Semantics/OpenMP/declare-target01.f90`, `flang/test/Semantics/OpenMP/requires05.f90`). | [llvm/llvm-project#110015](https://github.com/llvm/llvm-project/pull/110015) |
| otherwise clause on metadirectives | <span class="none">unclaimed</span> | | BEGIN/END METADIRECTIVE frontend support exists, but metadirective lowering remains TODO (`flang/lib/Lower/OpenMP/OpenMP.cpp`). Implement lowering selection for OTHERWISE and add dedicated lowering tests for construct selection. | [llvm/llvm-project#194402](https://github.com/llvm/llvm-project/pull/194402) |
| doacross with omp_cur_iteration | <span class="none">unclaimed</span> | | Implement parser/semantics validation for `omp_cur_iteration` placement and lowering of doacross dependence tokens, then add semantics+lowering tests for source/sink combinations. | |
| implicit map type for target enter and exit data | <span class="none">unclaimed</span> | | Define implicit-map behavior for enter/exit data in lowering map finalization and add dedicated tests covering pointer/allocatable/component cases and mapper interactions. | |
| allocate and firstprivate on scope directive | <span class="part">partial</span> | | Scope lowering coverage includes `allocate` and `firstprivate` on scope directives (`flang/test/Lower/OpenMP/scope.f90`, `flang/test/Lower/OpenMP/target-scope.f90`). | [llvm/llvm-project#193098](https://github.com/llvm/llvm-project/pull/193098) |
| loop consistency changes for order clause | <span class="none">unclaimed</span> | | Extend semantic loop-consistency checks for updated ORDER rules and add diagnostics tests for invalid nest/ordering combinations. | |
| keep original base pointer on map without matched candidate | <span class="none">unclaimed</span> | | Update map finalization so unmatched candidates preserve original base-pointer mapping semantics; add lowering tests for pointer-member mapping regressions. | |
| pure procedure support for certain directives | <span class="none">unclaimed</span> | | Codify semantics restrictions for PURE procedures with these directives and add lowering tests ensuring no illegal side-effecting ops are introduced. | |
| ALLOCATE statement support for allocators | <span class="none">unclaimed</span> | | Wire OpenMP allocator semantics into Fortran ALLOCATE statement handling and add semantics+lowering tests for allocator traits and error paths. | |
| dispatch extension supporting end directive | <span class="none">unclaimed</span> | | Base dispatch support is partial, and no dedicated support coverage for the OpenMP 5.2 dispatch end-directive extension was identified. | |
| minus operator deprecation handling | <span class="part">partial</span> | | Partially handled in semantic diagnostics. | |
| linear clause syntax deprecation | <span class="none">unclaimed</span> | | Add semantic deprecation diagnostics with fix-it guidance and tests for accepted/deprecated spellings. | |
| map clause modifiers without commas (deprecation) | <span class="none">unclaimed</span> | | Add parser/semantics deprecation diagnostics for comma-less map modifiers and tests that verify warning text and accepted replacements. | |
| uses_allocators list syntax (deprecation) | <span class="none">unclaimed</span> | | Add diagnostics for deprecated uses_allocators list syntax and ensure lowering still handles canonical replacements. | |
| default clause on metadirectives (deprecation) | <span class="none">unclaimed</span> | | Add dedicated metadirective deprecation diagnostics and tests for legacy/default-clause usage. | |
| destroy clause syntax on depobj (deprecation) | <span class="part">partial</span> | | Deprecation diagnostics are implemented (for example warning coverage in `flang/test/Semantics/OpenMP/depobj-construct-v52.f90`). | |
| source and sink task-dependence modifiers (deprecation) | <span class="part">partial</span> | | Deprecation diagnostics are implemented (for example warning coverage in `flang/test/Semantics/OpenMP/depobj-construct-v52.f90`). | |
| interop type position on init clause (deprecation) | <span class="none">unclaimed</span> | | Add parser+semantics deprecation diagnostics for legacy interop init type-position forms and tests showing canonical replacement. | |

## OpenMP 6.0

| Feature | Status | Claimed By | Notes | Reviews |
|:--------|:-------|:-----------|:------|:--------|
| threadset clause | <span class="part">partial</span> | | Semantics coverage exists (`flang/test/Semantics/OpenMP/threadset-clause.f90`) with ongoing lowering/runtime validation. | [llvm/llvm-project#169856](https://github.com/llvm/llvm-project/pull/169856) |
| groupprivate directive | <span class="part">partial</span> | | Semantics and lowering coverage exists (`flang/test/Semantics/OpenMP/groupprivate.f90`, `flang/test/Lower/OpenMP/groupprivate.f90`, `flang/test/Lower/OpenMP/groupprivate-modfile.f90`), with remaining limitations in lowering scope/placement (for example currently materialized for `teams`-based paths). | [llvm/llvm-project#166199](https://github.com/llvm/llvm-project/pull/166199), [llvm/llvm-project#166214](https://github.com/llvm/llvm-project/pull/166214), [llvm/llvm-project#180934](https://github.com/llvm/llvm-project/pull/180934) |
| recording of task graphs | <span class="progress">in progress</span> | | Semantics coverage exists (`flang/test/Semantics/OpenMP/taskgraph.f90`). | |
| workdistribute construct | <span class="part">partial</span> | | Semantics/lowering/transform coverage exists (`flang/test/Semantics/OpenMP/workdistribute01.f90`, `flang/test/Lower/OpenMP/workdistribute.f90`, `flang/test/Transforms/OpenMP/lower-workdistribute-fission.mlir`) including `target teams` placement updates; some team-nesting combinations still intentionally diagnose as unsupported. | [llvm/llvm-project#154377](https://github.com/llvm/llvm-project/pull/154377), [llvm/llvm-project#154378](https://github.com/llvm/llvm-project/pull/154378), [llvm/llvm-project#140523](https://github.com/llvm/llvm-project/pull/140523), [llvm/llvm-project#199006](https://github.com/llvm/llvm-project/pull/199006) |
| map clause updates (v6.0 forms) | <span class="part">partial</span> | | Semantics coverage exists (`flang/test/Semantics/OpenMP/map-clause-v60.f90`) with ongoing lowering completeness work. | |
| map modifier updates (v6.0 and v6.1 forms) | <span class="part">partial</span> | | Semantics coverage exists (`flang/test/Semantics/OpenMP/map-modifiers-v60.f90`, `flang/test/Semantics/OpenMP/map-modifiers-v61.f90`) with ongoing lowering completeness work. | [llvm/llvm-project#172080](https://github.com/llvm/llvm-project/pull/172080), [llvm/llvm-project#176810](https://github.com/llvm/llvm-project/pull/176810) |
| loop fusion transformation | <span class="good">done</span> | | Supported. | [llvm/llvm-project#161213](https://github.com/llvm/llvm-project/pull/161213) |
| loop construct with DO CONCURRENT | <span class="progress">in progress</span> | | Experimental support is available and under active development. | [llvm/llvm-project#178138](https://github.com/llvm/llvm-project/pull/178138), [llvm/llvm-project#190990](https://github.com/llvm/llvm-project/pull/190990) |
| optional argument for all clauses | <span class="part">partial</span> | | Semantics coverage exists across clause tests (for example `flang/test/Semantics/OpenMP/if-clause-50.f90`). | |
| canonical loop sequences | <span class="part">partial</span> | | Related loop/transform coverage exists (`flang/test/Semantics/OpenMP/loop-transformation-construct01.f90`, `flang/test/Lower/OpenMP/loop-directive.f90`). | [llvm/llvm-project#161213](https://github.com/llvm/llvm-project/pull/161213), [llvm/llvm-project#168884](https://github.com/llvm/llvm-project/pull/168884), [llvm/llvm-project#170734](https://github.com/llvm/llvm-project/pull/170734), [llvm/llvm-project#170735](https://github.com/llvm/llvm-project/pull/170735) |
| pure directives in DO CONCURRENT | <span class="none">unclaimed</span> | | Define exact PURE+DO CONCURRENT directive legality in semantics and add lowering tests proving accepted forms remain side-effect safe. | |
| extensions to depobj construct | <span class="none">unclaimed</span> | | Semantics and deprecation diagnostics exist for several depobj forms, but extension support remains incomplete in lowering (for example `flang/test/Lower/OpenMP/Todo/depobj-construct.f90`). Implement lowering for extension operands/modifiers and add non-TODO lowering tests. | |
| extensions to atomic construct | <span class="part">partial</span> | | Atomic compare lowering is now available (`flang/test/Lower/OpenMP/atomic-compare.f90`), but fail/capture-related extension paths remain incomplete (`flang/test/Lower/OpenMP/Todo/atomic-compare-fail.f90`). | [llvm/llvm-project#184761](https://github.com/llvm/llvm-project/pull/184761) |
| clarifications to Fortran map semantics | <span class="none">unclaimed</span> | | Document each 6.0 clarification point against existing map finalization behavior, then add focused semantics/lowering regression tests for unresolved points. | |

## OpenMP 6.1 (Future / Experimental)

| Feature | Status | Claimed By | Notes | Reviews |
|:--------|:-------|:-----------|:------|:--------|
| dyn_groupprivate clause | <span class="progress">in progress</span> | | Experimental and in progress (`flang/test/Semantics/OpenMP/dyn-groupprivate.f90`). | [llvm/llvm-project#166199](https://github.com/llvm/llvm-project/pull/166199), [llvm/llvm-project#166214](https://github.com/llvm/llvm-project/pull/166214) |
| dims strict behavior (multidimensional teams/leagues) | <span class="progress">in progress</span> | | Experimental and in progress (`flang/test/Lower/OpenMP/thread-limit-dims.f90`). | [llvm/llvm-project#171454](https://github.com/llvm/llvm-project/pull/171454), [llvm/llvm-project#171767](https://github.com/llvm/llvm-project/pull/171767), [llvm/llvm-project#171825](https://github.com/llvm/llvm-project/pull/171825) |
| attach map-type modifier | <span class="part">partial</span> | | Parser/semantics/lowering coverage exists (`flang/test/Parser/OpenMP/map-modifiers-v61.f90`, `flang/test/Semantics/OpenMP/map-modifiers-v61.f90`, `flang/test/Lower/OpenMP/attach-and-ref-modifier.f90`); broader integration coverage across additional mapping contexts is still in progress. | [llvm/llvm-project#177715](https://github.com/llvm/llvm-project/pull/177715), [llvm/llvm-project#177301](https://github.com/llvm/llvm-project/pull/177301), [llvm/llvm-project#177302](https://github.com/llvm/llvm-project/pull/177302) |
| need_device_ptr modifier for adjust_args clause | <span class="none">unclaimed</span> | | Add parser+semantics support for modifier placement/rules, then lower adjust_args with device-pointer selection semantics and add end-to-end tests. | |
| fallback modifier for use_device_ptr clause | <span class="none">unclaimed</span> | | Add parser+semantics diagnostics and lowering behavior for fallback selection; add tests for fallback and non-fallback resolution paths. | |
| loop flatten transformation | <span class="none">unclaimed</span> | | Implement parser/semantics acceptance and lowering transform plumbing, then add transform/lowering tests demonstrating flattened loop mapping. | |
| loop grid and tile modifiers for sizes clause | <span class="none">unclaimed</span> | | Extend clause parsing/semantics for grid/tile size modifiers and add lowering tests showing generated loop partitioning metadata/ops. | |

## Extensions
### ATOMIC construct
The implementation of the ATOMIC construct follows OpenMP 6.0 with the following extensions:
- `x = x` is an allowed form of ATOMIC UPDATE.
This is motivated by the fact that the equivalent forms `x = x+0` or `x = x*1` are allowed.
- Explicit type conversions are allowed in ATOMIC READ, WRITE or UPDATE constructs, and in the capture statement in ATOMIC UPDATE CAPTURE.
The OpenMP spec requires intrinsic- or pointer-assignments, which include (as per the Fortran standard) implicit type conversions.  Since such conversions need to be handled, allowing explicit conversions comes at no extra cost.
- A literal `.true.` or `.false.` is an allowed condition in ATOMIC UPDATE COMPARE. [1]
- A logical variable is an allowed form of the condition even if its value is not computed within the ATOMIC UPDATE COMPARE construct [1].
- `expr equalop x` is an allowed condition in ATOMIC UPDATE COMPARE. [1]

[1] Code generation for ATOMIC UPDATE COMPARE is not implemented yet.

