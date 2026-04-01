//===- PythonTestDialect.h - PythonTest dialect definition ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TEST_PYTHON_LIB_PYTHONTESTDIALECT_H
#define AIIR_TEST_PYTHON_LIB_PYTHONTESTDIALECT_H

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/Interfaces/InferTypeOpInterface.h"

#include "PythonTestDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "PythonTestAttributes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "PythonTestTypes.h.inc"

#define GET_OP_CLASSES
#include "PythonTestOps.h.inc"

#endif // AIIR_TEST_PYTHON_LIB_PYTHONTESTDIALECT_H
