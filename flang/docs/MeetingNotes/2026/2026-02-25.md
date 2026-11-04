<!--===- docs/MeetingNotes/2026/2026-02-25.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->
# Combined Call 2026-02-25

## Agenda

* Design docs and/or RFCs  
  * [\[RFC\] Proposed wording for a description of Flang for the LLVM home page](https://discourse.llvm.org/t/rfc-proposed-wording-for-a-description-of-flang-for-the-llvm-home-page/89794)  
    * This came out of a suggestion on the last call.  
    * The initial wording has been tweaked some, now it is last call for comments and tweaks before Tarun submits the wording.  
  * [\[RFC\] Runtime for Parameterized Derived Types with LEN type parameters](https://discourse.llvm.org/t/rfc-runtime-for-parameterized-derived-types-with-len-type-parameters/89796)   
    * Related PR: [https://github.com/llvm/llvm-project/pull/181008](https://github.com/llvm/llvm-project/pull/181008)   
    * The PR received feedback, some open questions remain and the authors are working through them  
    * Dan Bonachea: What does this imply for multi-image?  
  * [\[Flang\]\[Affine\] Linearized array access in \-promote-to-affine causes false dependence and blocks loop tiling](https://discourse.llvm.org/t/flang-affine-linearized-array-access-in-promote-to-affine-causes-false-dependence-and-blocks-loop-tiling/89927)   
    * Several paths forward have been suggested, please comment on the post to help guide this new contributor  
  * [Make hlfir.forall can contain PureOp](https://discourse.llvm.org/t/make-hlfir-forall-can-contain-pureop/89805)  
    * Related PR: [https://github.com/llvm/llvm-project/pull/180556](https://github.com/llvm/llvm-project/pull/180556)   
  * [\[RFC\] Adding conditional expressions in Flang (F2023)](https://discourse.llvm.org/t/rfc-adding-conditional-expressions-in-flang-f2023/89869)   
    * PR should be up for review shortly  
  * [\[RFC\] Warning suppression policy](https://discourse.llvm.org/t/rfc-warning-suppression-policy/89676)   
    * Related PR: [https://github.com/llvm/llvm-project/pull/174918](https://github.com/llvm/llvm-project/pull/174918)   
    * Regarding a false positive from gcc and how to address an issue related to zero initialization and possible undefined behavior  
    * Proposed change of disabling warnings for faulty gcc versions globally, this could reduce the maintenance burden  
  * [Cross-compilation of real(kind=16)](https://discourse.llvm.org/t/cross-compilation-of-real-kind-16/89161)   
    * Related PR: [https://github.com/llvm/llvm-project/pull/182230](https://github.com/llvm/llvm-project/pull/182230)   
    * Reviewers needed  
  * [Changes to builtin modules](https://discourse.llvm.org/t/changes-to-builtin-modules/89072)   
    * Waiting on more details from reviewer  
  * [\`-ffp-contract=fast\` Violates the Fortran Standard](https://discourse.llvm.org/t/ffp-contract-fast-violates-the-fortran-standard/88897)  
    * Peter Klausler has “requested a formal interpretation to clarify the extent to which the long-standing guarantee of ‘the integrity of parentheses’ applies to a parenthesized real intrinsic multiplication in ISO Fortran.”  Awaiting response.  
    * The document in question is [https://j3-fortran.org/doc/year/26/26-115r2.txt](https://j3-fortran.org/doc/year/26/26-115r2.txt)   
      * Someone needs to champion the paper on the floor of the committee to force a discussion  
      * Would be best if someone from the US or another convenient time zone could take this on. NVIDIA’s Mark LeAir? Ted Johnson as alternate.  

* PRs of Note  
  * Pass for support for OpenMP and NVIDIA offload  
    * [https://github.com/llvm/llvm-project/pull/180060](https://github.com/llvm/llvm-project/pull/180060)   
    * In need of reviewers on the Flang side  
    * MLIR PR has been merged  
  * [\[flang\] Add runtime trampoline pool for W^X compliance\#183108](https://github.com/llvm/llvm-project/pull/183108)  
    * Slava will take a look  
  * Support for rank-1 integer array expressions in declarations and allocate statements  
    * [https://github.com/llvm/llvm-project/pull/183193](https://github.com/llvm/llvm-project/pull/183193)  
    * Looking at breaking the PR into smaller patches, but needs discussion about how best to design it. Reviewers needed.  

* Issues of Note  
  * None at this time  

* FYI  
  * [2026 EuroLLVM \- Early Bird Pricing Ends Soon](https://discourse.llvm.org/t/2026-eurollvm-early-bird-pricing-ends-soon/89831) (March 1st)   
  * New Flang Liason report to J3: [https://discourse.llvm.org/t/flang-liaison-report-to-j3/68468/11](https://discourse.llvm.org/t/flang-liaison-report-to-j3/68468/11)   
  * [LLVM 22.1.0 Released\!](https://discourse.llvm.org/t/llvm-22-1-0-released/89950)   

* Other topics as time allows  
  * 

## Details

* Consists of over **727,000** lines of code, documentation, build files, and test  
* To date, over **11,828** commits have been made to Flang
