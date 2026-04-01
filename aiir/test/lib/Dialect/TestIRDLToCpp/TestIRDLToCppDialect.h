//===- TestIRDLToCppDialect.h - AIIR Test Dialect Types -----------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file includes TestIRDLToCpp dialect headers.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TEST_LIB_DIALECT_TESTIRDLTOCPP_TESTIRDLTOCPPDIALECT_H
#define AIIR_TEST_LIB_DIALECT_TESTIRDLTOCPP_TESTIRDLTOCPPDIALECT_H

#define GEN_DIALECT_DECL_HEADER
#include "test_irdl_to_cpp.irdl.aiir.cpp.inc"

namespace test {
void registerConvertTestDialectPass();
}

#endif // AIIR_TEST_LIB_DIALECT_TESTIRDLTOCPP_TESTIRDLTOCPPDIALECT_H
