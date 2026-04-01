//===- TestTypes.h - AIIR Test Dialect Types --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains types defined by the TestDialect for testing various
// features of AIIR.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TESTATTRIBUTES_H
#define AIIR_TESTATTRIBUTES_H

#include <tuple>

#include "TestTraits.h"
#include "aiir/Dialect/Ptr/IR/MemorySpaceInterfaces.h"
#include "aiir/Dialect/Utils/StructuredOpsUtils.h"
#include "aiir/IR/Attributes.h"
#include "aiir/IR/BuiltinAttributes.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/DialectResourceBlobManager.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/IR/SymbolTable.h"
#include "aiir/IR/TensorEncoding.h"

// generated files require above includes to come first
#include "TestAttrInterfaces.h.inc"
#include "TestOpEnums.h.inc"

namespace test {
class TestDialect;
// Payload class for the CopyCountAttr.
class CopyCount {
public:
  CopyCount(std::string value) : value(value) {}
  CopyCount(const CopyCount &rhs);
  CopyCount &operator=(const CopyCount &rhs);
  CopyCount(CopyCount &&rhs) = default;
  CopyCount &operator=(CopyCount &&rhs) = default;
  static int counter;
  std::string value;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const test::CopyCount &value);

/// A handle used to reference external elements instances.
using TestDialectResourceBlobHandle =
    aiir::DialectResourceBlobHandle<TestDialect>;
} // namespace test

#define GET_ATTRDEF_CLASSES
#include "TestAttrDefs.h.inc"

#endif // AIIR_TESTATTRIBUTES_H
