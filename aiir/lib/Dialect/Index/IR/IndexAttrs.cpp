//===- IndexAttrs.cpp - Index attribute definitions ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Index/IR/IndexAttrs.h"
#include "aiir/Dialect/Index/IR/IndexDialect.h"
#include "aiir/IR/Builders.h"
#include "aiir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace aiir;
using namespace aiir::index;

//===----------------------------------------------------------------------===//
// IndexDialect
//===----------------------------------------------------------------------===//

void IndexDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "aiir/Dialect/Index/IR/IndexAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Index/IR/IndexEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "aiir/Dialect/Index/IR/IndexAttrs.cpp.inc"
