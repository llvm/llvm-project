<!--===- docs/MeetingNotes/2026/2026-02-11.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->
# Combined Call 2026-02-11

## Agenda

* Design docs and/or RFCs  
  * [\[RFC\] Warning suppression policy \- Flang \- LLVM Discussion Forums](https://discourse.llvm.org/t/rfc-warning-suppression-policy/89676)   
    * Related PR: [https://github.com/llvm/llvm-project/pull/174918](https://github.com/llvm/llvm-project/pull/174918)   
  * [\[RFC\] Support for \-std=f2023](https://discourse.llvm.org/t/rfc-support-for-std-f2023/89608/24)   
    * SYSTEM\_CLOCK changed with F2023 and the discussion centers on what the default behavior should be and how best to allow users to select the behavior they need.  There are compatibility breaking issues between standards regarding this intrinsic.  
    * MattPD \- re gcc/gfortran: The default value for std is 'gnu', which specifies a superset of the latest Fortran standard that includes all of the extensions supported by GNU Fortran, although warnings are given for obsolete extensions not recommended for use in new code. The 'legacy' value is equivalent but without the warnings for obsolete extensions, and may be useful for old nonstandard programs.  
      * [https://gcc.gnu.org/onlinedocs/gfortran/Fortran-Dialect-Options.html](https://gcc.gnu.org/onlinedocs/gfortran/Fortran-Dialect-Options.html)   
  * [Cross-compilation of real(kind=16)](https://discourse.llvm.org/t/cross-compilation-of-real-kind-16/89161)   
    * PR will come out with something concrete to evaluate, remaining issues can be resolved via that discussion  
  * [Changes to builtin modules](https://discourse.llvm.org/t/changes-to-builtin-modules/89072)   
    * PR was reverted due to buildbot failure involving Windows  
    * Reviewers asked for big changes, working on a compromise involving putting TODOs in other runtimes  
  * [\[RFC\] Support classic flang driver options in flang](https://discourse.llvm.org/t/rfc-support-classic-flang-driver-options-in-flang/89380)  
    * Suggestion was to accept the alternate spellings but give a deprecation warning immediately to encourage migrating to the regular spelling  
    * Tarun plans to put out a broader RFC to address the central issue in this and the following RFC in the next few weeks  
  * [\[RFC\] Support \-fstrict-aliasing and \-fno-strict-aliasing](https://discourse.llvm.org/t/rfc-support-fstrict-aliasing-and-fno-strict-aliasing/89135)   
    * Related issue: [https://github.com/llvm/llvm-project/issues/171912](https://github.com/llvm/llvm-project/issues/171912)   
    * We can’t do anything reasonable with it without violating the Fortran standard, will leave it open for a little while longer but likely will not support this  
  * [\[RFC\] Automatic static promotion of large local variables in Flang](https://discourse.llvm.org/t/rfc-automatic-static-promotion-of-large-local-variables-in-flang/89539)  
    * No updates on the call today  
  * Flang-tidy  
    * Meeting with authors has been deferred, will create meeting including Tarun Prabhu when Michael Klemm is back from travel  
  * [\`-ffp-contract=fast\` Violates the Fortran Standard](https://discourse.llvm.org/t/ffp-contract-fast-violates-the-fortran-standard/88897)  
    * Reply from the standards committee: [https://mailman.j3-fortran.org/pipermail/j3/2026-January/015531.html](https://mailman.j3-fortran.org/pipermail/j3/2026-January/015531.html)   
    * The J3 opinion is now unanimous that parentheses and FMAs are perfectly compatible with the standard  
      * See [recently submitted J3 paper](http://26-115r1.txt)  
    * Remaining question is how to let users control the behavior that they want, implementation still needs to happen  
    * \-f(no)protect-parens is already an option in Flang  
      * Is this too heavy-handed for this situation?  
    * Question about LLVM backend implementation for this control, there are open issues on the LLVM side about it  
  * Call Notes questions  
    * These came from review in PR [\#180287](https://github.com/llvm/llvm-project/pull/180287)   
    * Should they appear in the Flang documentation website (i.e. should they be included in the HTML build)?  
    * What to do about references to Classic Flang?  
      * Ted \- in favor of removing them entirely  
      * Tarun \- include one single reference that states it is an entirely separate compiler and points to one website with further information	  
* PRs of Note  
  * Tests from Cray’s CCE internal test suite in llvm-test-suite \[[PR](https://github.com/llvm/llvm-test-suite/pull/326)\]  
    * Proposed directory structure with one test in linked PR  
    * This will be merged soon  
  * Pass for support for OpenMP and NVIDIA offload  
    * [https://github.com/llvm/llvm-project/pull/180058](https://github.com/llvm/llvm-project/pull/180058)  
    * [https://github.com/llvm/llvm-project/pull/180060](https://github.com/llvm/llvm-project/pull/180060)   
    * In need of reviewers on the Flang side, particularly for the second PR.  MLIR side for the first PR.  
* Issues of Note  
* FYI  
  * [2026 EuroLLVM Developers' Meeting \- Agenda](https://discourse.llvm.org/t/2026-eurollvm-developers-meeting-agenda/89725)  
    * Michael Kruse is presenting “Creating a runtime using the LLVM\_ENABLE\_RUNTIMES system”  
      * The LLVM build system has a mechanism designed for building runtime libraries targeting the platform that compiler’s (be it Clang, Flang, Rust, etc. ) output will run on. For instance, since Clang intrinsically is a cross-compiler, such runtime libraries need to be compiled for each targeted platform. The mechanism originates from splitting target-side runtimes from host-side subprojects such as Clang, Polly, BOLT.   
  * [LLVM 22.1.0-rc3 Released\! \- Announcements](https://discourse.llvm.org/t/llvm-22-1-0-rc3-released/89769)   
    * Who is writing the [flang release notes](https://github.com/llvm/llvm-project/blob/release/22.x/flang/docs/ReleaseNotes.md)?  
  * [CFP: MLIR Workshop at the EuroLLVM Developer Meeting (Apr 13, 2026\)](https://discourse.llvm.org/t/cfp-mlir-workshop-at-the-eurollvm-developer-meeting-apr-13-2026/89790)   
* Other topics as time allows  
  * Should these calls be longer and/or more frequent?  
    * Extending the call to 45 minutes  
    * Is there a way to add the event to [https://llvm.org/docs/GettingInvolved.html\#llvm-community-calendar](https://llvm.org/docs/GettingInvolved.html#llvm-community-calendar) ?  
  * Why doesn’t flang appear on the main page for [https://llvm.org/](https://llvm.org/) ?  
    * Who do we contact about this?  General Discourse post?

## Details

* Consists of over **725,000** lines of code, documentation, build files, and test  
* To date, over **11,767** commits have been made to Flang
