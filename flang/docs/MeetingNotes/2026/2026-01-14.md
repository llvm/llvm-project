<!--===- docs/MeetingNotes/2026/2026-01-14.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->
# Combined Call 2026-01-14

## Agenda

* Design docs and/or RFCs  
  * Fujitsu test suite ease of use  
    * It’s not easily usable as-is right now: very large, includes C++ tests, lacks the expected flang result, etc.  
    * Issues have been opened in the Fujitsu test repo to make a subset of the suite for Flang-only tests  
      * [https://github.com/fujitsu/compiler-test-suite/issues/53](https://github.com/fujitsu/compiler-test-suite/issues/53)   
      * Another Fujitsu issue to request not to build C/C++ when testing Fortran: [https://github.com/fujitsu/compiler-test-suite/issues/52](https://github.com/fujitsu/compiler-test-suite/issues/52)   
    * Perhaps eventually bring this inside the LLVM test suite directly, after fixes are added  
    * Does Linaro have a bot that is running this currently?  
  * [Flang-tidy](https://discourse.llvm.org/t/rfc-flang-flang-tidy-a-new-tool-for-fortran-static-analysis/87579)  
    * Issue seems to be that many of the checks could be done in Flang itself rather than the tool  
    * What about in-place correction of the source code (ala clang-tidy)?  
    * Could we make some of the checks the tool uses available to the Flang driver as well?  
    * Idea: split the checks up and give the ones that have a repair mechanism to flang-tidy and the rest go to the Flang driver  
    * Do we bring the tool in as-is, then start a process of refactoring to create a separate library of checks?  
    * Need to talk to the original contributors to make sure they are still invested, also need to check-in with Peter Klausler and need to find a maintainer for the tool once merged  
  * [\[RFC\] What to do regarding the Flang Call Notes document](https://discourse.llvm.org/t/rfc-what-to-do-regarding-the-flang-call-notes-document/89450)  
    *   
  * [\[RFC\] Support classic flang driver options in flang](https://discourse.llvm.org/t/rfc-support-classic-flang-driver-options-in-flang/89380)  
    *   
  * [Cross-compilation of real(kind=16)](https://discourse.llvm.org/t/cross-compilation-of-real-kind-16/89161)    
    *   
  * [\[RFC\] Support \-fstrict-aliasing and \-fno-strict-aliasing](https://discourse.llvm.org/t/rfc-support-fstrict-aliasing-and-fno-strict-aliasing/89135)   
    * Related issue: [https://github.com/llvm/llvm-project/issues/171912](https://github.com/llvm/llvm-project/issues/171912)   
  * [\`-ffp-contract=fast\` Violates the Fortran Standard](https://discourse.llvm.org/t/ffp-contract-fast-violates-the-fortran-standard/88897)  
    * Still under debate  
  * [Changes to builtin modules](https://discourse.llvm.org/t/changes-to-builtin-modules/89072)   
    * PR was reverted due to buildbot failure involving Windows  
    * Undergoing another review process, will not try landing again until after the first of the year  
  * [\[RFC\] Use pre-compiled headers to speed up LLVM build by \~1.5-2x](https://discourse.llvm.org/t/rfc-use-pre-compiled-headers-to-speed-up-llvm-build-by-1-5-2x/89345/24)  
    * Related to what has already been done in Flang, attempt to use this approach more generally   
    * Draft PR: [https://github.com/llvm/llvm-project/pull/173868](https://github.com/llvm/llvm-project/pull/173868)   
* PRs of Note  
  * [\[Flang\]\[FIR\] Introduce FIRToCoreMLIR pass. \#168703](https://github.com/llvm/llvm-project/pull/168703)    
* Issues of Note  
  * [Building Flang with offload support](https://discourse.llvm.org/t/building-flang-with-offload-support/89100)   
* FYI  
  * LLVM 22.x has branched, first RC expected Friday 1/16: [https://discourse.llvm.org/t/llvm-22-x-has-branched/89447](https://discourse.llvm.org/t/llvm-22-x-has-branched/89447)   
  * [2026 LLVM Community Area Team Elections \- Call for Nominations](https://discourse.llvm.org/t/2026-llvm-community-area-team-elections-call-for-nominations/89439)  
  * Added LoopInvariantCodeMotion pass for \[HL\]FIR. (PR \#[173438](https://github.com/llvm/llvm-project/pull/173438))    
* Other topics as time allows  
  * One-off additional call next Wednesday at the same start time.  Meeting link to be posted on slack and here in the notes.  
    * [https://lanl-us.webex.com/lanl-us/j.php?MTID=mf9e0b3ff7ba2ba9d5dbdeb510296cf5e](https://lanl-us.webex.com/lanl-us/j.php?MTID=mf9e0b3ff7ba2ba9d5dbdeb510296cf5e)   
    * Wednesday, January 21, 2026 9:30 AM | 30 minutes | (UTC-07:00) Mountain Time (US & Canada)  
    * Meeting number: 2488 038 4796  
    * Password: mY9p2pMx6tm

    * Join by video system  
    * Dial 24880384796@lanl-us.webex.com  
    * You can also dial 173.243.2.68 and enter your meeting number.

    * Join by phone  
    * \+1-415-655-0002 US Toll

    * Access code: 248 803 84796

## Details

* Consists of over **713,000** lines of code, documentation, build files, and test  
* To date, over **11,617** commits have been made to Flang
