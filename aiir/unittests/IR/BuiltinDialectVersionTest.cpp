//===- BuiltinDialectVersionTest.cpp - Test builtin dialect versioning ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "aiir/Bytecode/BytecodeWriter.h"
#include "aiir/IR/BuiltinDialect.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/IR/Diagnostics.h"
#include "aiir/IR/AIIRContext.h"
#include "aiir/Parser/Parser.h"

using namespace aiir;

TEST(BuiltinDialectVersionTest, RejectFutureVersion) {
  AIIRContext ctx;
  auto module = ModuleOp::create(UnknownLoc::get(&ctx));

  auto writeBytecode = [&](std::string &out,
                           std::optional<int64_t> version =
                               std::nullopt) -> LogicalResult {
    BytecodeWriterConfig config;
    if (version)
      config.setDialectVersion<BuiltinDialect>(
          std::make_unique<BuiltinDialectVersion>(*version));
    llvm::raw_string_ostream os(out);
    return writeBytecodeToFile(module, os, config);
  };

  std::string bytecode;
  ASSERT_TRUE(succeeded(writeBytecode(bytecode)));
  std::string bytecodeWithFutureVersion;
  ASSERT_TRUE(succeeded(writeBytecode(bytecodeWithFutureVersion, 99)));

  EXPECT_NE(bytecode, bytecodeWithFutureVersion);

  std::string error;
  ScopedDiagnosticHandler handler(&ctx, [&](Diagnostic &diag) {
    if (diag.getSeverity() == DiagnosticSeverity::Error)
      error = diag.str();
    return success();
  });

  auto parsed =
      parseSourceString(StringRef(bytecodeWithFutureVersion),
                        ParserConfig(&ctx, /*verifyAfterParse=*/true));
  EXPECT_TRUE(error.find("reading newer builtin dialect version") !=
              std::string::npos);

  module->erase();
}
