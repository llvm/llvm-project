//===- RISCVRelocationHandler.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_TARGET_RISCV_RISCVRELOCATIONHANDLER_H
#define BOLT_TARGET_RISCV_RISCVRELOCATIONHANDLER_H

#include <memory>

namespace llvm {
namespace bolt {

class RelocationHandler;

std::unique_ptr<RelocationHandler> createRISCVRelocationHandler(bool Is64Bit);

} // namespace bolt
} // namespace llvm

#endif
