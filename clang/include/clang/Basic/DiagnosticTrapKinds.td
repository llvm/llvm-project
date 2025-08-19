//==--- DiagnosticTrapKinds.td - CodeGen Diagnostics -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Trap Diagnostics
//
// These are diagnostics that are emitted into Debug Info, rather than to the
// traditional consumers like the terminal. Their primary purpose is to make
// debugging traps (e.g. `-fsanitize-trap=undefined`) easier by attaching
// a trap category and reason to the trap instruction that tools like a debugger
// can show.
//===----------------------------------------------------------------------===//
let Component = "Trap" in {
let CategoryName = "Undefined Behavior Sanitizer" in {

def trap_ubsan_arith_overflow : Trap<
  "%select{unsigned|signed}0 integer "
  "%select{addition|subtraction|multiplication}1 overflow in %2">;

}
}
