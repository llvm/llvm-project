//===- CIRDialect.h - CIR dialect -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the CIR dialect with all of its ops. The ops are split
// into partitions (see CIRDialect.td), each with its own CIROps<Partition>.h
// that declares the ops of that partition and forward declares the others;
// code that uses only a few ops can include just the partition headers it
// needs, together with CIRDialectBase.h for the dialect, attributes and types.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_CIRDIALECT_H
#define CLANG_CIR_DIALECT_IR_CIRDIALECT_H

#include "clang/CIR/Dialect/IR/CIRDialectBase.h"
#include "clang/CIR/Dialect/IR/CIROpsArith.h"
#include "clang/CIR/Dialect/IR/CIROpsBuiltins.h"
#include "clang/CIR/Dialect/IR/CIROpsClassMembers.h"
#include "clang/CIR/Dialect/IR/CIROpsComplex.h"
#include "clang/CIR/Dialect/IR/CIROpsControlFlow.h"
#include "clang/CIR/Dialect/IR/CIROpsCore.h"
#include "clang/CIR/Dialect/IR/CIROpsCoroutine.h"
#include "clang/CIR/Dialect/IR/CIROpsEH.h"
#include "clang/CIR/Dialect/IR/CIROpsMath.h"
#include "clang/CIR/Dialect/IR/CIROpsSymbols.h"
#include "clang/CIR/Dialect/IR/CIROpsVectorMatrix.h"

#endif // CLANG_CIR_DIALECT_IR_CIRDIALECT_H
