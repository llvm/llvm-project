<!--===- docs/MeetingNotes/2026/2026-05-06.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->
# Community Call 2026-05-06

## Agenda

* Design docs and/or RFCs
  * [\[RFC\]\[flang\] Add Floating-point trap handling support](https://discourse.llvm.org/t/rfc-flang-add-floating-point-trap-handling-support/90544)
    * No opposition. Valuable discussion about Fortran standard’s details on the topic.
    * Is the Nvidia option for runtime flag \-Ktrap=option\[,option\]...? ([https://docs.nvidia.com/hpc-sdk/compilers/hpc-compilers-ref-guide/index.html\#k-flag](https://docs.nvidia.com/hpc-sdk/compilers/hpc-compilers-ref-guide/index.html#k-flag))
      * Does Nvidia plan to upstream it?
      * Key difference: Nvidia’s option enforces strict compiler behavior. Proposal is to allow different levels of compiler strictness and decouple compiler behavior from runtime trap handling.
  * [\[RFC\] Aggressive inlinging for OpenMP SIMD loops](https://discourse.llvm.org/t/rfc-aggressive-inlinging-for-openmp-simd-loops/90558)
    * Discussion is ongoing, no updates on the call today
  * [RFC: Make Outlinable OpenMP Operations IsolatedFromAbove](https://discourse.llvm.org/t/rfc-make-outlinable-openmp-operations-isolatedfromabove/90565)
    * Tom has a way forward on this, testing now
  * [\[RFC\]\[Flang-rt\]Allow multiple units to open the same file path](https://discourse.llvm.org/t/rfc-flang-rt-allow-multiple-units-to-open-the-same-file-path/90591)
    * One comment opposing this change, any other input?
  * [\[RFC\] Alignment of global arrays](https://discourse.llvm.org/t/rfc-alignment-of-global-arrays/90397)
    * Related issue: [\[Flang\] Bad alignment of globals in modules · Issue \#189069](https://github.com/llvm/llvm-project/issues/189069)
    * PR [\#194969](https://github.com/llvm/llvm-project/pull/194969)
    * Reviewers needed on the PR
  * [\[RFC\] Flang lowering for OpenMP metadirective](https://discourse.llvm.org/t/rfc-flang-lowering-for-openmp-metadirective/90338)
    * PR [\#193664](https://github.com/llvm/llvm-project/pull/193664)
    * PR [\#194424](https://github.com/llvm/llvm-project/pull/194424)
    * First PR is basic lowering, the second is for dynamic user resolution, the loop and construct is still a work in progress
  * [\[RFC\] Fortran 2023 ENUMERATION TYPEs](https://discourse.llvm.org/t/rfc-fortran-2023-enumeration-types/90255)
    * Related to Issue [\#183789](https://github.com/llvm/llvm-project/issues/183789)
    * PR stack beginning here: [\#192651](https://github.com/llvm/llvm-project/pull/192651)
    * All PRs in stack are still open for review/feedback
  * [Changes to builtin modules](https://discourse.llvm.org/t/changes-to-builtin-modules/89072)
    * Additional PR [\#171610](https://github.com/llvm/llvm-project/pull/171610) has been merged
    * Main PR (from which the other two were extracted): [\#171515](https://github.com/llvm/llvm-project/pull/171515)
    * Related Issue on Discourse: [OpenMP Module Installation Paths in Flang](https://discourse.llvm.org/t/openmp-module-installation-paths-in-flang/90614)
    * No update at this time
  * [\[RFC\] Flang Dependency Scanning for CMake](https://discourse.llvm.org/t/rfc-flang-dependency-scanning-for-cmake/90620)
    * No update at this time

* PRs of Note
  * *Discussion needed*: Remove DO CONCURRENT mapping experimental warning
    * [PR \#195009](https://github.com/llvm/llvm-project/pull/195009#pullrequestreview-4220723299)
    * CPU side is working well, still a few GPU issues to be worked on, but there is now good documentation about what support is available, so the thought is that the warning isn’t needed any more
    * Have been validating it with SPEC
      * [https://www.spec.org/cpu2026/results/res2026q2/cpu2026-20260210-00032.txt](https://www.spec.org/cpu2026/results/res2026q2/cpu2026-20260210-00032.txt)
    * Need more input from users who need the GPU side (one person on the call says it does not yet work for their code and they would still consider this “experimental”)
  * *Reviewers needed*: Support for rank-1 integer array expressions in declarations (assumed-shape and explicit-shape) and allocate statements.
  * *Reviewers needed*: Conditional args support \- semantics, [PR \#195345](https://github.com/llvm/llvm-project/pull/195345)
    * Parser support has been merged, this is the next step of the process and needs reviewers
  * *Reviewers/contributors needed*: Document RFC best practices
    * [PR \#190836](https://github.com/llvm/llvm-project/pull/190836)
    * The intent is to provide better and/or more consistent guidance to contributors about how decisions are made within the Flang community, and in so doing to smooth out kinks in the current decision-making process
    * Since this is a Flang community practice document, the author would like to gather as much insight from community members broadly as is feasible
  * *Discussion needed*: \[openmp\]\[flang\] To add a cmake option to build OPENMP Fortran modules only without building the libomp lib
    * [PR \#195576](https://github.com/llvm/llvm-project/pull/195576#issuecomment-4374663237)

* Issues of Note
  * None today

* FYI
  * Google Summer of Code project to add Fortran support to LLDB accepted
    * [https://discourse.llvm.org/t/gsoc-2026-accepted-adding-native-fortran-support-in-lldb/90688](https://discourse.llvm.org/t/gsoc-2026-accepted-adding-native-fortran-support-in-lldb/90688)
  * [2026 EuroLLVM Slide Decks Available Now](https://discourse.llvm.org/t/2026-eurollvm-slide-decks-available-now/90658)
  * [2026 US LLVM Developers’ Meeting \- October 26-28 | Call for Workshops & Program Committee Volunteers](https://discourse.llvm.org/t/2026-us-llvm-developers-meeting-october-26-28-call-for-workshops-program-committee-volunteers/90715)
  * [LLVM 22.1.5 Released\!](https://urldefense.com/v3/__https://discourse.llvm.org/t/llvm-22-1-5-released/90734/1__;!!Bt8fGhp8LhKGRg!DyyAqxwj-RabnoIpF3FE2u3NaSGg9Q9X6wx7VZpkKjgG2J98uFS5jtmBSUXDC4EI8OL0Gkpg2OXkhm-I5-o7PeInrxtR8TTn$)

* Other topics as time allows

## Details

* Consists of over **758,000** lines of code, documentation, build files, and test
* To date, over **12,269** commits have been made to Flang
