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

}
}
