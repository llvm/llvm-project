//===- CIROpsClassMembers.h - CIR ops, `ClassMembers` partition -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the CIR ops for C++ classes: vtables, bit-fields,
// base and derived class addresses, member and member function pointers,
// dynamic casts and array constructors and destructors. The ops of the other
// partitions are forward declared; include their headers, or CIRDialect.h
// for all ops, to use them.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_CIROPSCLASSMEMBERS_H
#define CLANG_CIR_DIALECT_IR_CIROPSCLASSMEMBERS_H

#include "clang/CIR/Dialect/IR/CIRDialectBase.h"

#define GET_OP_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsClassMembers.h.inc"

#endif // CLANG_CIR_DIALECT_IR_CIROPSCLASSMEMBERS_H
