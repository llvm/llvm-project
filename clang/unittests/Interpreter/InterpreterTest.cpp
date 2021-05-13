//===- unittests/Interpreter/InterpreterTest.cpp --- Interpreter tests ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for Clang's Interpreter library.
//
//===----------------------------------------------------------------------===//

#include "clang/Interpreter/Interpreter.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"

#include "llvm/ADT/ArrayRef.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace clang;

namespace {
using Args = std::vector<const char *>;
static std::unique_ptr<Interpreter>
createInterpreter(const Args &ExtraArgs = {},
                  DiagnosticConsumer *Client = nullptr) {
  Args ClangArgs = {"-Xclang", "-emit-llvm-only"};
  ClangArgs.insert(ClangArgs.end(), ExtraArgs.begin(), ExtraArgs.end());
  auto CI = cantFail(clang::IncrementalCompilerBuilder::create(ClangArgs));
  if (Client)
    CI->getDiagnostics().setClient(Client, /*ShouldOwnClient=*/false);
  return cantFail(clang::Interpreter::create(std::move(CI)));
}

TEST(InterpreterTest, Sanity) {
  std::unique_ptr<Interpreter> Interp = createInterpreter();
  Transaction &R1(cantFail(Interp->Parse("void g(); void g() {}")));
  EXPECT_EQ(2U, R1.Decls.size());

  Transaction &R2(cantFail(Interp->Parse("int i;")));
  EXPECT_EQ(1U, R2.Decls.size());
}

static std::string DeclToString(DeclGroupRef DGR) {
  return llvm::cast<NamedDecl>(DGR.getSingleDecl())->getQualifiedNameAsString();
}

TEST(InterpreterTest, IncrementalInputTopLevelDecls) {
  std::unique_ptr<Interpreter> Interp = createInterpreter();
  auto R1OrErr = Interp->Parse("int var1 = 42; int f() { return var1; }");
  // gtest doesn't expand into explicit bool conversions.
  EXPECT_TRUE(!!R1OrErr);
  auto R1 = R1OrErr->Decls;
  EXPECT_EQ(2U, R1.size());
  EXPECT_EQ("var1", DeclToString(R1[0]));
  EXPECT_EQ("f", DeclToString(R1[1]));

  auto R2OrErr = Interp->Parse("int var2 = f();");
  EXPECT_TRUE(!!R2OrErr);
  auto R2 = R2OrErr->Decls;
  EXPECT_EQ(1U, R2.size());
  EXPECT_EQ("var2", DeclToString(R2[0]));
}

TEST(InterpreterTest, Errors) {
  Args ExtraArgs = {"-Xclang", "-diagnostic-log-file", "-Xclang", "-"};

  // Create the diagnostic engine with unowned consumer.
  std::string DiagnosticOutput;
  llvm::raw_string_ostream DiagnosticsOS(DiagnosticOutput);
  auto DiagPrinter = std::make_unique<TextDiagnosticPrinter>(
      DiagnosticsOS, new DiagnosticOptions());

  auto Interp = createInterpreter(ExtraArgs, DiagPrinter.get());
  auto Err = Interp->Parse("intentional_error v1 = 42; ").takeError();
  using ::testing::HasSubstr;
  EXPECT_THAT(DiagnosticsOS.str(),
              HasSubstr("error: unknown type name 'intentional_error'"));
  EXPECT_EQ("Parsing failed.", llvm::toString(std::move(Err)));

#ifdef GTEST_HAS_DEATH_TEST
  EXPECT_DEATH((void)Interp->Parse("int var1 = 42;"), "");
#endif
}

// Here we test whether the user can mix declarations and statements. The
// interpreter should be smart enough to recognize the declarations from the
// statements and wrap the latter into a declaration, producing valid code.
TEST(InterpreterTest, DeclsAndStatements) {
  Args ExtraArgs = {"-Xclang", "-diagnostic-log-file", "-Xclang", "-"};

  // Create the diagnostic engine with unowned consumer.
  std::string DiagnosticOutput;
  llvm::raw_string_ostream DiagnosticsOS(DiagnosticOutput);
  auto DiagPrinter = std::make_unique<TextDiagnosticPrinter>(
      DiagnosticsOS, new DiagnosticOptions());

  auto Interp = createInterpreter(ExtraArgs, DiagPrinter.get());
  auto R1OrErr = Interp->Parse(
      "int var1 = 42; extern \"C\" int printf(const char*, ...);");
  // gtest doesn't expand into explicit bool conversions.
  EXPECT_TRUE(!!R1OrErr);

  auto R1 = R1OrErr->Decls;
  EXPECT_EQ(2U, R1.size());

  // FIXME: Add support for wrapping and running statements.
  auto R2OrErr = Interp->Parse("var1++; printf(\"var1 value %d\\n\", var1);");
  EXPECT_FALSE(!!R2OrErr);
  using ::testing::HasSubstr;
  EXPECT_THAT(DiagnosticsOS.str(),
              HasSubstr("error: unknown type name 'var1'"));
  auto Err = R2OrErr.takeError();
  EXPECT_EQ("Parsing failed.", llvm::toString(std::move(Err)));
}

} // end anonymous namespace
