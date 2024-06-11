//==-- x86.h - Definitions common to all x86 ABI variants ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions common to any X86 ABI implementation.
//
//===----------------------------------------------------------------------===//

#ifndef CIR_X86_H
#define CIR_X86_H

namespace cir {

// Possible argument classifications according to the x86 ABI documentation.
enum X86ArgClass {
  Integer = 0,
  SSE,
  SSEUp,
  X87,
  X87Up,
  ComplexX87,
  NoClass,
  Memory
};

} // namespace cir

#endif // CIR_X86_H
