<!--===- docs/MeetingNotes/2026/2026-04-22.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->
# Community Call 2026-04-22

## Agenda

* Design docs and/or RFCs  
  * [\[RFC\]\[Flang\]\[MLIR\] Add opt-in affine loop optimization pipeline](https://discourse.llvm.org/t/rfc-flang-mlir-add-opt-in-affine-loop-optimization-pipeline/90526)   
    * No updates at this time  
  * [\[RFC\]\[flang\] Add Floating-point trap handling support](https://discourse.llvm.org/t/rfc-flang-add-floating-point-trap-handling-support/90544)   
    * More input is needed on the RFC, thank you to those who have participated so far  
    * There are two distinct proposals contained in this RFC  
      * Runtime piece  
      * Flag for the compiler to change behavior for exception handling  
    * Some work has been done at NVIDIA on this, but it hasn’t made it upstream.  Steve Scalpone will check on the status and post relevant work for the author to look over.  
  * [\[RFC\] Aggressive inlinging for OpenMP SIMD loops](https://discourse.llvm.org/t/rfc-aggressive-inlinging-for-openmp-simd-loops/90558)   
    * Michael Kruse \- Seems like low-hanging fruit that could be quite useful
    * PR coming soon  
  * [RFC: Make Outlinable OpenMP Operations IsolatedFromAbove](https://discourse.llvm.org/t/rfc-make-outlinable-openmp-operations-isolatedfromabove/90565)  
    * MLIR CSE is leading to sharing of fir.embox operations between different OpenMP contexts, but fir.embox creates allocas and it matters what context these are in  
    * RFC proposes to block OpenMP operations from hoisting  
    * There was a PR a few months ago that was not accepted by the FIR maintainers that used a different approach (possibly less invasive)  
    * Is it ok for a pure operation to allocate memory, in MLIR? If so, this problem could occur in other dialects besides OpenMP  
    * Relationship between fir.embox and OpenMP privatization?  
    * Broader problem with CSE than just fir.embox?  
    * Should fir.embox not be labelled as pure?  
  * [\[RFC\]\[Flang-rt\]Allow multiple units to open the same file path](https://discourse.llvm.org/t/rfc-flang-rt-allow-multiple-units-to-open-the-same-file-path/90591)  
    * No updates at this time  
  * [\[RFC\] Alignment of global arrays](https://discourse.llvm.org/t/rfc-alignment-of-global-arrays/90397)  
    * Related issue: [\[Flang\] Bad alignment of globals in modules · Issue \#189069](https://github.com/llvm/llvm-project/issues/189069)  
    * Prototype branch with the architecture change is out there, but there are still some issues due to assumption of default alignment of 16\. Other compilers default to either 32 or 64  
    * Current idea: add a flag to opt out of the default to 16  
    * Roughly 30% performance difference between default 16 and default 64 in the author’s test cases  
    * Ideally, this would be architecture dependent, but that significantly increases complexity and potential issues  
      * At least one person agreed with the idea of changing the default to 64
      * However, you might end up with higher memory usage for modules (but that is likely only in the case of scalars, and this is for arrays)  
    * Do we have any backwards compatibility promises between Flang releases?  
    * Michael Klemm \- change the default, don’t do the flag  
    * Themos \- many compilers have been very careful about alerting users of backwards compatibility issues, and usually go several releases without breaking compatibility  
    * Will need wrapper for malloc in the runtime  
  * [\[RFC\] Flang lowering for OpenMP metadirective](https://discourse.llvm.org/t/rfc-flang-lowering-for-openmp-metadirective/90338)  
    * PR coming soon for static resolution  
    * Do you need the begin and end metadirectives to be implemented, or at least parsed?  
  * [\[RFC\] Fortran 2023 ENUMERATION TYPEs](https://discourse.llvm.org/t/rfc-fortran-2023-enumeration-types/90255)  
    * Related to Issue [\#183789](https://github.com/llvm/llvm-project/issues/183789)   
    * PR stack beginning here: [https://github.com/llvm/llvm-project/pull/192651](https://github.com/llvm/llvm-project/pull/192651)   
    * The last of the main case of this change will be PR 4 of the stack  
  * [Changes to builtin modules](https://discourse.llvm.org/t/changes-to-builtin-modules/89072)   
    * Community could help by commenting and highlighting how important this change is.  Would relieve a lot of annoyance and this is holding up Flang community progress.  
    * Initial PR has been merged. Additional PR: [https://github.com/llvm/llvm-project/pull/171610](https://github.com/llvm/llvm-project/pull/171610)  
    * Main PR (from which the other two were extracted): [https://github.com/llvm/llvm-project/pull/171515](https://github.com/llvm/llvm-project/pull/171515)
    * Related Issue on Discourse: [OpenMP Module Installation Paths in Flang](https://discourse.llvm.org/t/openmp-module-installation-paths-in-flang/90614)   
    * There has been progress on the PRs  
  * [\[RFC\] Flang Dependency Scanning for CMake](https://discourse.llvm.org/t/rfc-flang-dependency-scanning-for-cmake/90620)   
    * No updates at this time  
  * [MemRefDataFlowOpt](https://discourse.llvm.org/t/memrefdataflowopt/74711/3)
    * Asked for status update, needs response on Discourse  
  * [\[OpenMP\] Misspelling of \-Wopen-mp](https://discourse.llvm.org/t/openmp-misspelling-of-wopen-mp/90196)  
    * PR: [https://github.com/llvm/llvm-project/pull/188434](https://github.com/llvm/llvm-project/pull/188434)   
    * PR is ready to land, will leave up for two more weeks in case there are any “nay” votes

* PRs of Note  
  * Support for rank-1 integer array expressions in declarations (assumed-shape and explicit-shape) and allocate statements  
    * Has been broken up, reviewers needed.  
    * These should now be self-contained parser support PRs. Short design description is given in the GitHub Issue, [https://github.com/llvm/llvm-project/issues/178071](https://github.com/llvm/llvm-project/issues/178071)   
      * [https://github.com/llvm/llvm-project/pull/188445](https://github.com/llvm/llvm-project/pull/188445)  
      * [https://github.com/llvm/llvm-project/pull/188446](https://github.com/llvm/llvm-project/pull/188446)  
      * [https://github.com/llvm/llvm-project/pull/188447](https://github.com/llvm/llvm-project/pull/188447)  
    * PR related to pointer assignment coming  
  * Document RFC best practices \- [PR \#190836](https://github.com/llvm/llvm-project/pull/190836)   
  * \[flang\]\[OpenMP\] Support user-defined declare reduction with derived types [https://github.com/llvm/llvm-project/pull/190288](https://github.com/llvm/llvm-project/pull/190288) \- received one approval  
  * F2023 Conditional args support  
    * [https://github.com/llvm/llvm-project/pull/191303](https://github.com/llvm/llvm-project/pull/191303) \- Looking for any more comments before merging.  
    * Will open PRs for semantics support shortly  
        
* Issues of Note  
  * None other than already alluded to in RFCs

* FYI  
  * [LLVM 22.1.4 Released\!](https://discourse.llvm.org/t/llvm-22-1-4-released/90622) 

* Other topics as time allows  
  * AMD and NVIDIA are hiring Fortran compiler people

## Details

* Consists of over **752,000** lines of code, documentation, build files, and test  
* To date, over **12,173** commits have been made to Flang

