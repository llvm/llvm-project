<!--===- docs/MeetingNotes/2026/2026-09-09.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->
# Community Call 2026-09-09

## Agenda

* Presentation by Iasonas Karaprodromidis from National and Kapodistrian University of Athens on *Adding Native Fortran Support to LLDB*
  * Slides posted in Slack: [https://flang-compiler.slack.com/files/U693Z04F7/F0C1HM0PRDW/lldb-fortran-support-slides.pdf](https://flang-compiler.slack.com/files/U693Z04F7/F0C1HM0PRDW/lldb-fortran-support-slides.pdf)
* Design docs and/or RFCs
  * [Runtime unformatted I/O conversion between big and little endian formats](https://discourse.llvm.org/t/runtime-unformatted-i-o-conversion-between-big-and-little-endian-formats/91751)
    * Discussion ongoing, please see Discourse post
  * [\[RFC\] Reduce \`ExternalFileUnit::Emit()\` call overhead for formatted sequential output](https://discourse.llvm.org/t/rfc-reduce-externalfileunit-emit-call-overhead-for-formatted-sequential-output/91674)
    * Discussion ongoing, please see Discourse post
  * [\[RFC\] .mod by-reference dependencies recorded versus enforced in flang-new](https://discourse.llvm.org/t/rfc-mod-by-reference-dependencies-recorded-versus-enforced-in-flang-new/91745)
    * This is being actively discussed, there do not appear to be many more checks that can be added
  * [Removal of \--fortran-common option](https://discourse.llvm.org/t/removal-of-fortran-common-option/91553)
    * Discussion ongoing, please see Discourse post
  * [\[RFC\]\[flang\] reasssociate within real and complex expressions by default](https://discourse.llvm.org/t/rfc-flang-reasssociate-within-real-and-complex-expressions-by-default/91626)
    * PR has been merged: [https://github.com/llvm/llvm-project/pull/218746](https://github.com/llvm/llvm-project/pull/218746)
    * Floating point results may change now that this has been upstreamed
  * [\[RFC\] Improve Fortran Error Output](https://discourse.llvm.org/t/rfc-improve-fortran-error-output/91648)
    * Related issue (closed): [https://github.com/llvm/llvm-project/issues/220696](https://github.com/llvm/llvm-project/issues/220696)
    * Related PR (merged): [https://github.com/llvm/llvm-project/pull/220988](https://github.com/llvm/llvm-project/pull/220988)
  * [RFC: Add \`-finit-local=\` to Flang for Initializing Automatic Variables](https://discourse.llvm.org/t/rfc-add-finit-local-to-flang-for-initializing-automatic-variables/91545)
    * New draft PR: [https://github.com/llvm/llvm-project/pull/216164](https://github.com/llvm/llvm-project/pull/216164)
    * Related existing PR (but only covers the initialization to zero case): [https://github.com/llvm/llvm-project/pull/159788](https://github.com/llvm/llvm-project/pull/159788)
    * Author of the pre-existing PR has been contacted, but still awaiting response.  Michael Klemm will ping Nimish.
    * Have been addressing review comments and improving the test coverage. Sorry I (Daniel) have to miss today’s meeting.
  * [\[RFC\]\[Flang\] Improving temporary and local array allocations \- unify StackArray/MemoryAllocation](https://discourse.llvm.org/t/rfc-flang-improving-temporary-and-local-array-allocations-unify-stackarray-memoryallocation/91353)
    * 3 of 5 PRs in stack have been merged, these remain:
      * [\#210766 — \[flang\] Enable allocation-placement pass by default](https://github.com/llvm/llvm-project/pull/210766)
      * [\#210930 — \[flang\] Remove legacy stack-arrays and memory-allocation-opt passes](https://github.com/llvm/llvm-project/pull/210930)
    * Status update (should we keep this on the agenda going forward?  Will remove for now and re-add when more information is available)
  * [\[RFC\] add support for collapsed io calls for implicit do loops](https://discourse.llvm.org/t/rfc-add-support-for-collapsed-io-calls-for-implicit-do-loops/91416)
    * Related PR: [https://github.com/llvm/llvm-project/pull/212646](https://github.com/llvm/llvm-project/pull/212646)
    * Needs reviewer feedback
  * [\[RFC\] The Cost of Templates when Building Flang](https://discourse.llvm.org/t/rfc-the-cost-of-templates-when-building-flang/91197)
    * New PR stack (first has been merged, continues here): [https://github.com/llvm/llvm-project/pull/216958](https://github.com/llvm/llvm-project/pull/216958)
    * 2nd PR received a review yesterday, comments are being addressed, but more feedback is strongly desired
  * [Choosing the default summation algorithm for Flang's SUM intrinsic](https://discourse.llvm.org/t/choosing-the-default-summation-algorithm-for-flangs-sum-intrinsic/91300)
    * Proposal is to use “the ordinary summation algorithm as the default” and create a flag for users to opt-in to Kahan summation
    * The author has run into some issues and is looking into alternative solutions.  We should remove from the agenda for now and re-add when alternatives are available.
  * [\[RFC\] Support for \-std=f2023](https://discourse.llvm.org/t/rfc-support-for-std-f2023/89608)
    * New PR: [https://github.com/llvm/llvm-project/pull/205938](https://github.com/llvm/llvm-project/pull/205938)
    * Related Issue: [https://github.com/llvm/llvm-project/issues/196409](https://github.com/llvm/llvm-project/issues/196409)
    * Chris will wrap up addressing review comments today and update the PR
* PRs of Note
  * *Reviewers needed:*
    * PR 2 of 3 has been merged for a stack PR implementing rank-1 integer arrays with constant extent being used as explicit shape bounds.
    * PR 3 of 3: [https://github.com/llvm/llvm-project/pull/215403](https://github.com/llvm/llvm-project/pull/215403)
    * Associated llvm-test-suite PR: [https://github.com/llvm/llvm-test-suite/pull/411](https://github.com/llvm/llvm-test-suite/pull/411)
    * Waiting on author to address comments from two weeks ago
  * *Feedback needed:* Unroll inner do loops when vectorization directive is present: [https://github.com/llvm/llvm-project/pull/210820](https://github.com/llvm/llvm-project/pull/210820)
    * Received feedback from Scott Manley, and there is now a blocker, alternative solutions are being investigated
    * Custom attribute? How does the community feel about this?
  * [Optionally turn module checksum check into warning \- \#219181](https://github.com/llvm/llvm-project/pull/219181)
    * Tarun \- Reviewer said one could potentially use hermetic modules for this, and that same idea appeared on another RFC.  Is this a broader issue that needs to be discussed?
    * Michael Klemm \- would a tool help to look through the existing modules to figure out if anything relevant has changed?
    * Relationship to building projects that have numerous other dependencies that also rely on the modules from Flang
    * How does gfortran handle backwards compatibility for their modules?
      * They do not use a checksum, but details unclear beyond that
      * They do not use Fortran for modules?
    * Do we want to continue with this approach? Or do we want to extend the hermetic modules?
      * Power user solution vs. long-term solution
* FYI
  * [LLVM 23.1.1 Released\!](https://discourse.llvm.org/t/llvm-23-1-1-released/91760)
    * macOS Binary still missing flang \- how do we fix that?
    * From Tom Eccles \- “Unfortunately \[the patch to fix\] was reverted because flang took too long to build [https://github.com/llvm/llvm-project/pull/218978](https://github.com/llvm/llvm-project/pull/218978). I don't think this is fixable unless somebody donates better darwin build infrastructure”
  * [Announcing the 2026 LLVM Developers' Meeting Program](https://discourse.llvm.org/t/announcing-the-2026-llvm-developers-meeting-program/91470)
    * Early bird registration ends Sept. 9
  * OpenMP 6.1 Public Comment Draft: [https://www.openmp.org/wp-content/uploads/openmp-TR15.pdf](https://www.openmp.org/wp-content/uploads/openmp-TR15.pdf)
    * Email [info@openmp.org](mailto:info@openmp.org) or [michael.klemm@openmp.org](mailto:michael.klemm@openmp.org) with any feedback
    * This will be up for comment until the end of September
* Other topics as time allows
  * [\[flang\] Module files incorrectly expose implicitly recognized intrinsics](https://github.com/llvm/llvm-project/issues/219570)
    * Need feedback as other compilers do a variety of things, so what should Flang be consistent with?

## Details

* Consists of over **820,000** lines of code, documentation, build files, and test
* To date, over **13,077** commits have been made to Flang
