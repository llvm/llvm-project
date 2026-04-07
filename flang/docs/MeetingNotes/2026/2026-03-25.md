<!--===- docs/MeetingNotes/2026/2026-03-25.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->
# Combined Call 2026-03-25

## New call location

Microsoft Teams meeting  
Join: [https://teams.microsoft.com/meet/23836454854154?p=ImtxJRnSgszrn5tSeG](https://teams.microsoft.com/meet/23836454854154?p=ImtxJRnSgszrn5tSeG)  
Meeting ID: 238 364 548 541 54  
Passcode: zg9Bd3Zp

## Agenda

* Design docs and/or RFCs  
  * [\[RFC\] Helping the new contributors to create better contributions](https://discourse.llvm.org/t/rfc-helping-the-new-contributors-to-create-better-contributions/90213)  
    * Related to the uptick in AI-usage to create fixes to issues in the queue  
    * Create a new document with more guidance? Could include how best to start, how to add reviewers properly to a PR, reference to the AI policy, etc.  
    * Flang file to help auto-add reviewers? (like the LLVM one)  
      * Create a CODEOWNERS file?   
      * Use flang’s [Maintainers.md](https://github.com/llvm/llvm-project/blob/main/flang/Maintainers.md)?  
    * Either reference or repeat sections from the GettingStarted document in LLVM  
    * Will start with editing flang/docs/GettingStarted.md to be more explicit, will sort out the automated reviewer addition angle in the background, will make sure Maintainers.md is properly up-to-date  
    * Subscribe to flang labels here: [https://github.com/orgs/llvm/teams?query=flang](https://github.com/orgs/llvm/teams?query=flang)  
  * [\[RFC\] UDR-Based Lowering for OpenMP Conditional Lastprivate in Flang](https://discourse.llvm.org/t/rfc-udr-based-lowering-for-openmp-conditional-lastprivate-in-flang/90222)  
    * Consensus is this is a good idea and should proceed  
  * [\[RFC\] Fortran 2023 ENUMERATION TYPEs](https://discourse.llvm.org/t/rfc-fortran-2023-enumeration-types/90255)  
    * Related to Issue [\#183789](https://github.com/llvm/llvm-project/issues/183789)   
    * Related PR: [https://github.com/llvm/llvm-project/pull/185497](https://github.com/llvm/llvm-project/pull/185497)   
    * Second PR: [https://github.com/llvm/llvm-project/pull/185975](https://github.com/llvm/llvm-project/pull/185975)   
    * Had initial prototype, refactoring now after RFC discussion and PR feedback, more input is welcome  
    * Note that enums are being [extended in F2028](https://j3-fortran.org/doc/year/26/26-128r2.txt)  
  * [\[RFC\] Qualifying syntax rule numbers with Fortran standard version in comments](https://discourse.llvm.org/t/rfc-qualifying-syntax-rule-numbers-with-fortran-standard-version-in-comments/90167)   
    * F2018 and F2023 numbers do not necessarily match up, need to clarify comments in the code to direct people to the right part of the particular standard  
    * Possible solution: Default numbers refer to F2018, later standards need to be specified  
    * Should we put this in the coding standards document? Seems a good idea  
  * What are the best practices and/or community policy on how to close out an RFC?  
    * Once we discuss RFC, we should add a comment on RFC referencing the meeting notes to indicate that a decision was made  
    * Put the onus on the creator of the RFC to put in the final comment summarizing what the decision is  
    * Add this to GettingStarted.md? Need to make sure that document doesn’t become overwhelming though.  
      * Split into two documents, one for users and one for developers?  
    * If the RFC results in a PR, link the PR to the RFC  
  * Creating a new document "OpenMP-extensions"?  
    * [https://github.com/llvm/llvm-project/pull/186696\#discussion\_r2939508874](https://github.com/llvm/llvm-project/pull/186696#discussion_r2939508874)  
    * Similar to how Fortran extensions are documented in Flang  
    * Comment on the PR linked above to provide feedback on the idea  
    * This particular case in the PR may not quite be an “extension” per se, discussion is ongoing, but the idea for the document still stands  
  * [\[OpenMP\] Misspelling of \-Wopen-mp](https://discourse.llvm.org/t/openmp-misspelling-of-wopen-mp/90196)  
    *  PR: [https://github.com/llvm/llvm-project/pull/188434](https://github.com/llvm/llvm-project/pull/188434)   
    * Also affects OpenACC  
    * There is a bug with how Flang handles CamelCase warnings/errors  
    * Please comment on either the RFC or the PR  
    * How long should the deprecation period be? Two releases?  
      * Should definitely add comment to the release notes about this  
  * [\[flang\] canonical and optimizable representation for min/max](https://discourse.llvm.org/t/flang-canonical-and-optimizable-representation-for-min-max/90037)   
    * Initial related PRs have been merged  
    * More changes are needed to make things fully consistent, but Slava will pause development for the moment and come back to it.  If others are interested in working on this, that would be helpful (compiler folding and flang-rt in particular).  
    * There is no user visible option to control it at the moment  
    * Will this interfere with OpenMP reduction?  
  * [\[RFC\] Runtime for Parameterized Derived Types with LEN type parameters](https://discourse.llvm.org/t/rfc-runtime-for-parameterized-derived-types-with-len-type-parameters/89796)   
    * Related PR: [https://github.com/llvm/llvm-project/pull/181008](https://github.com/llvm/llvm-project/pull/181008)   
    * PR needs changes, but will have to go on a back burner until \~May 1 (can remove from agenda until then)  
  * [\[Flang\]\[Affine\] Linearized array access in \-promote-to-affine causes false dependence and blocks loop tiling](https://discourse.llvm.org/t/flang-affine-linearized-array-access-in-promote-to-affine-causes-false-dependence-and-blocks-loop-tiling/89927)   
    * Several paths forward have been suggested, **please comment on the post to help guide this new contributor**  
  * [\[RFC\] Warning suppression policy](https://discourse.llvm.org/t/rfc-warning-suppression-policy/89676)   
    * Related PR: [https://github.com/llvm/llvm-project/pull/174918](https://github.com/llvm/llvm-project/pull/174918)   
    * Regarding a false positive from gcc and how to address an issue related to zero initialization and possible undefined behavior  
    * Proposed change of disabling warnings for faulty gcc versions globally, this could reduce the maintenance burden  
    * Remove from future agenda  
  * [Changes to builtin modules](https://discourse.llvm.org/t/changes-to-builtin-modules/89072)   
    * Met with reviewer to review system changes, reached an agreement but he did not officially approve the PR and asked further questions.  Discussion ongoing.  
    * Community could help by commenting and highlighting how important this change is.  Would relieve a lot of annoyance and this is holding up Flang community progress.  
    * PR (has been approved once): [https://github.com/llvm/llvm-project/pull/177953](https://github.com/llvm/llvm-project/pull/177953)  
    * Additional PR: [https://github.com/llvm/llvm-project/pull/171610](https://github.com/llvm/llvm-project/pull/171610)  
    * Main PR (from which the other two were extracted): [https://github.com/llvm/llvm-project/pull/171515](https://github.com/llvm/llvm-project/pull/171515)  
  * [\[RFC\] Adding conditional expressions in Flang (F2023)](https://discourse.llvm.org/t/rfc-adding-conditional-expressions-in-flang-f2023/89869)   
    * Has been split into two PRs  
      * Parsing/Semantics: [https://github.com/llvm/llvm-project/pull/186489](https://github.com/llvm/llvm-project/pull/186489)  
      * Lowering: [https://github.com/llvm/llvm-project/pull/186490](https://github.com/llvm/llvm-project/pull/186490)  
    * Tests added to test suite: [https://github.com/llvm/llvm-test-suite/pull/369](https://github.com/llvm/llvm-test-suite/pull/369) 

* PRs of Note  
  * Support for rank-1 integer array expressions in declarations (assumed-shape and explicit-shape) and allocate statements  
    * Has been broken up, reviewers needed.  
    * These should now be self-contained parser support PRs. Short design description is given in the GitHub Issue, [https://github.com/llvm/llvm-project/issues/178071](https://github.com/llvm/llvm-project/issues/178071)   
      * [https://github.com/llvm/llvm-project/pull/188445](https://github.com/llvm/llvm-project/pull/188445)  
      * [https://github.com/llvm/llvm-project/pull/188446](https://github.com/llvm/llvm-project/pull/188446)  
      * [https://github.com/llvm/llvm-project/pull/188447](https://github.com/llvm/llvm-project/pull/188447)

* Issues of Note  
  * None at this time

* FYI  
  * [LLVM 22.1.2 Released\!](https://urldefense.com/v3/__https://discourse.llvm.org/t/llvm-22-1-2-released/90308/1__;!!Bt8fGhp8LhKGRg!AJBYD1kumJNeMowID7MBqKQOmgEIUSH2zQNSdKNAbtZ4XAySuvjjbPKSFvZtaAFD44-b3UpQGfwMDPaj4DW7tSMxsJQSgvty$)  
  * [https://llvm.org/](https://llvm.org/) finally lists flang\!

* Other topics as time allows  
  * 
## Details

* Consists of over **738,000** lines of code, documentation, build files, and test  
* To date, over **12,016** commits have been made to Flang
