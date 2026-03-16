<!--===- docs/MeetingNotes/2026/2026-03-11.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->
# Combined Call 2026-03-11

## New call location

Microsoft Teams meeting  
Join: [https://teams.microsoft.com/meet/23836454854154?p=ImtxJRnSgszrn5tSeG](https://teams.microsoft.com/meet/23836454854154?p=ImtxJRnSgszrn5tSeG)  
Meeting ID: 238 364 548 541 54  
Passcode: zg9Bd3Zp

## Agenda

* Design docs and/or RFCs  
  * [\[flang\] canonical and optimizable representation for min/max](https://discourse.llvm.org/t/flang-canonical-and-optimizable-representation-for-min-max/90037)   
    * Initial related PRs have been merged  
    * More changes are needed to make things fully consistent, but Slava will pause development for the moment and come back to it.  If others are interested in working on this, that would be helpful (compiler folding and flang-rt in particular).  
    * There is no user visible option to control it at the moment  
    * Will this interfere with OpenMP reduction?  
  * [\[RFC\] Region-based control-flow with early exits in MLIR](https://discourse.llvm.org/t/rfc-region-based-control-flow-with-early-exits-in-mlir/76998)   
    * This is related to a flang PR (\#[184234](https://github.com/llvm/llvm-project/pull/184234)) that was merged, then reverted, that involved using the SCF dialect for while loops.  
    * This came up in the OpenMP call, might have impacts for Flang but it is unclear at the moment.  
    * Discussion about the potential role of using the SCF dialect in Flang, concerns with OpenMP offload and relationship to the current intricate implementation of control flow  
  * [\[RFC\] Runtime for Parameterized Derived Types with LEN type parameters](https://discourse.llvm.org/t/rfc-runtime-for-parameterized-derived-types-with-len-type-parameters/89796)   
    * Related PR: [https://github.com/llvm/llvm-project/pull/181008](https://github.com/llvm/llvm-project/pull/181008)   
    * No updates at this time  
  * [\[Flang\]\[Affine\] Linearized array access in \-promote-to-affine causes false dependence and blocks loop tiling](https://discourse.llvm.org/t/flang-affine-linearized-array-access-in-promote-to-affine-causes-false-dependence-and-blocks-loop-tiling/89927)   
    * Several paths forward have been suggested, please comment on the post to help guide this new contributor  
  * [\[RFC\] Warning suppression policy](https://discourse.llvm.org/t/rfc-warning-suppression-policy/89676)   
    * Related PR: [https://github.com/llvm/llvm-project/pull/174918](https://github.com/llvm/llvm-project/pull/174918)   
    * Regarding a false positive from gcc and how to address an issue related to zero initialization and possible undefined behavior  
    * Proposed change of disabling warnings for faulty gcc versions globally, this could reduce the maintenance burden  
  * [Changes to builtin modules](https://discourse.llvm.org/t/changes-to-builtin-modules/89072)   
    * Met with reviewer to review system changes, reached an agreement but he did not officially approve the PR and asked further questions.  Discussion ongoing.  
    * Community could help by commenting and highlighting how important this change is.  Would relieve a lot of annoyance and this is holding up Flang community progress.  
    * PR: [https://github.com/llvm/llvm-project/pull/177953](https://github.com/llvm/llvm-project/pull/177953)  
    * Additional PR: [https://github.com/llvm/llvm-project/pull/171610](https://github.com/llvm/llvm-project/pull/171610)  
    * Main PR (from which the other two were extracted): [https://github.com/llvm/llvm-project/pull/171515](https://github.com/llvm/llvm-project/pull/171515)  
  * [\`-ffp-contract=fast\` Violates the Fortran Standard](https://discourse.llvm.org/t/ffp-contract-fast-violates-the-fortran-standard/88897)  
    * Newer updated response from the committee: [https://j3-fortran.org/doc/year/26/26-126r1.txt](https://j3-fortran.org/doc/year/26/26-126r1.txt)   
    * J3 will investigate clarifying the text in the standard.  
    * Did not change the position of the standards body, but they are going to investigate adding clarifying wording  
  * [\[RFC\] Adding conditional expressions in Flang (F2023)](https://discourse.llvm.org/t/rfc-adding-conditional-expressions-in-flang-f2023/89869)   
    * PR should be up for review shortly

* PRs of Note  
  * [https://github.com/llvm/llvm-project/pull/180556](https://github.com/llvm/llvm-project/pull/180556)   
    * Related Discourse post: [Make hlfir.forall can contain PureOp](https://discourse.llvm.org/t/make-hlfir-forall-can-contain-pureop/89805)  
  * Support for rank-1 integer array expressions in declarations and allocate statements  
    * [https://github.com/llvm/llvm-project/pull/183193](https://github.com/llvm/llvm-project/pull/183193)  
    * Looking at breaking the PR into smaller patches, but needs discussion about how best to design it. Reviewers needed.  
    * Implementation changes needed to account for additional cases

* Issues of Note  
  * None at this time

* FYI  
  * [2026 EuroLLVM \- Early Bird Pricing Ends Soon \- Extended through Friday, March 13th](https://discourse.llvm.org/t/2026-eurollvm-early-bird-pricing-ends-soon-extended-through-friday-march-13th/90120)   
  * [LLVM 22.1.1 Released\!](https://urldefense.com/v3/__https://discourse.llvm.org/t/llvm-22-1-1-released/90150/1__;!!Bt8fGhp8LhKGRg!DbDWN0bWA4xEdMcUn1p6MemRAapLosX60Je94-Ez1V_PEl2PJkWDAtVSTN37IvazUqXyV8-iG12byGXzxvCF-urVqFF9D_X4$)

* Other topics as time allows  
  * 

## Details

* Consists of over **733,000** lines of code, documentation, build files, and test  
* To date, over **11,923** commits have been made to Flang
