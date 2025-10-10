//==--- DiagnosticTrapKinds.td ------------------------ -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Trap Diagnostics
//
// These are diagnostics that are emitted into `TrapReason` objects using the
// `TrapReasonBuilder` class. These `TrapReason` objects are then encoded into
// debug info during codegen, rather than to the traditional diagnostic
// consumers like the terminal. Their primary purpose is to make debugging traps
// (e.g. `-fsanitize-trap=undefined`) easier by attaching a trap category and
// message to the trap instruction that tools like a debugger can show.
//
//===----------------------------------------------------------------------===//
let Component = "Trap" in {
let CategoryName = "Undefined Behavior Sanitizer" in {

def trap_ubsan_arith_overflow : Trap<
  "%select{unsigned|signed}0 integer "
  "%enum_select<UBSanArithKind>{"
    "%Add{addition}|"
    "%Sub{subtraction}|"
    "%Mul{multiplication}"
  "}1 overflow in %2">;

}
}

// TO_UPSTREAM(BoundsSafety) ON
include "DiagnosticBoundsSafetyTrapKinds.td"
// TO_UPSTREAM(BoundsSafety) OFF
