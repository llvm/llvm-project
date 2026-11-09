//===- CIROpsControlFlow.h - CIR ops in partition `ControlFlow` -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the CIR control flow ops: structured scopes, loops and
// switches, their terminators, branches, gotos and labels. The ops of the
// other partitions are forward declared; include their headers, or
// CIRDialect.h for all ops, to use them.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_CIROPSCONTROLFLOW_H
#define CLANG_CIR_DIALECT_IR_CIROPSCONTROLFLOW_H

#include "clang/CIR/Dialect/IR/CIRDialectBase.h"

#define GET_OP_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsControlFlow.h.inc"

#endif // CLANG_CIR_DIALECT_IR_CIROPSCONTROLFLOW_H
