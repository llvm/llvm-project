//==--- DiagnosticBoundsSafetyTrapKinds.td --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// These are the trap diagnostics used by -fbounds-safety
//===----------------------------------------------------------------------===//

let Component = "Trap" in {
let CategoryName = "Bounds check failed" in {

// Used for falling back to the legacy infrastructure for constructing
// trap reason strings.
def trap_bs_fallback : Trap<"%0">;

// TODO: Split these up once we know what the diagnostics strings needs to be.
// Ideally we want to be able to generate trap codes and opt-remark strings
// from these definitions too so we'll need to split them up when we do that.
def trap_bs_upper_lower_overflow_bound : Trap<
  "%enum_select<BoundsCheckContextKind>{"
    "%Indexing{indexing}|"
    "%TODO{TODO}"
  "}0 %enum_select<BoundsSafetyPtrCheckKind>{"
    "%Upper{above upper bound}|"
    "%Overflow{overflows address space}|"
    "%Lower{below lower bound"
  "}}2 in %1">;

}
}
