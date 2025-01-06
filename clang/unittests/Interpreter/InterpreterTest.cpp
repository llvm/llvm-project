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

#include "InterpreterTestFixture.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/Mangle.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Interpreter/Interpreter.h"
#include "clang/Interpreter/Value.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace clang;

int Global = 42;
// JIT reports symbol not found on Windows without the visibility attribute.
REPL_EXTERNAL_VISIBILITY int getGlobal() { return Global; }
REPL_EXTERNAL_VISIBILITY void setGlobal(int val) { Global = val; }

namespace {

class InterpreterTest : public InterpreterTestBase {
  // TODO: Collect common variables and utility functions here
};

using Args = std::vector<const char *>;
static std::unique_ptr<Interpreter>
createInterpreter(const Args &ExtraArgs = {},
                  DiagnosticConsumer *Client = nullptr) {
  Args ClangArgs = {"-Xclang", "-emit-llvm-only"};
  ClangArgs.insert(ClangArgs.end(), ExtraArgs.begin(), ExtraArgs.end());
  auto CB = clang::IncrementalCompilerBuilder();
  CB.SetCompilerArgs(ClangArgs);
  auto CI = cantFail(CB.CreateCpp());
  if (Client)
    CI->getDiagnostics().setClient(Client, /*ShouldOwnClient=*/false);
  return cantFail(clang::Interpreter::create(std::move(CI)));
}

static size_t DeclsSize(TranslationUnitDecl *PTUDecl) {
  return std::distance(PTUDecl->decls().begin(), PTUDecl->decls().end());
}

TEST_F(InterpreterTest, Sanity) {
  std::unique_ptr<Interpreter> Interp = createInterpreter();

  using PTU = PartialTranslationUnit;

  PTU &R1(cantFail(Interp->Parse("void g(); void g() {}")));
  EXPECT_EQ(2U, DeclsSize(R1.TUPart));

  PTU &R2(cantFail(Interp->Parse("int i;")));
  EXPECT_EQ(1U, DeclsSize(R2.TUPart));
}

static std::string DeclToString(Decl *D) {
  return llvm::cast<NamedDecl>(D)->getQualifiedNameAsString();
}

TEST_F(InterpreterTest, IncrementalInputTopLevelDecls) {
  std::unique_ptr<Interpreter> Interp = createInterpreter();
  auto R1 = Interp->Parse("int var1 = 42; int f() { return var1; }");
  // gtest doesn't expand into explicit bool conversions.
  EXPECT_TRUE(!!R1);
  auto R1DeclRange = R1->TUPart->decls();
  EXPECT_EQ(2U, DeclsSize(R1->TUPart));
  EXPECT_EQ("var1", DeclToString(*R1DeclRange.begin()));
  EXPECT_EQ("f", DeclToString(*(++R1DeclRange.begin())));

  auto R2 = Interp->Parse("int var2 = f();");
  EXPECT_TRUE(!!R2);
  auto R2DeclRange = R2->TUPart->decls();
  EXPECT_EQ(1U, DeclsSize(R2->TUPart));
  EXPECT_EQ("var2", DeclToString(*R2DeclRange.begin()));
}

TEST_F(InterpreterTest, Errors) {
  Args ExtraArgs = {"-Xclang", "-diagnostic-log-file", "-Xclang", "-"};

  // Create the diagnostic engine with unowned consumer.
  std::string DiagnosticOutput;
  llvm::raw_string_ostream DiagnosticsOS(DiagnosticOutput);
  auto DiagPrinter = std::make_unique<TextDiagnosticPrinter>(
      DiagnosticsOS, new DiagnosticOptions());

  auto Interp = createInterpreter(ExtraArgs, DiagPrinter.get());
  auto Err = Interp->Parse("intentional_error v1 = 42; ").takeError();
  using ::testing::HasSubstr;
  EXPECT_THAT(DiagnosticOutput,
              HasSubstr("error: unknown type name 'intentional_error'"));
  EXPECT_EQ("Parsing failed.", llvm::toString(std::move(Err)));

  auto RecoverErr = Interp->Parse("int var1 = 42;");
  EXPECT_TRUE(!!RecoverErr);
}

// Here we test whether the user can mix declarations and statements. The
// interpreter should be smart enough to recognize the declarations from the
// statements and wrap the latter into a declaration, producing valid code.

TEST_F(InterpreterTest, DeclsAndStatements) {
  Args ExtraArgs = {"-Xclang", "-diagnostic-log-file", "-Xclang", "-"};

  // Create the diagnostic engine with unowned consumer.
  std::string DiagnosticOutput;
  llvm::raw_string_ostream DiagnosticsOS(DiagnosticOutput);
  auto DiagPrinter = std::make_unique<TextDiagnosticPrinter>(
      DiagnosticsOS, new DiagnosticOptions());

  auto Interp = createInterpreter(ExtraArgs, DiagPrinter.get());
  auto R1 = Interp->Parse(
      "int var1 = 42; extern \"C\" int printf(const char*, ...);");
  // gtest doesn't expand into explicit bool conversions.
  EXPECT_TRUE(!!R1);

  auto *PTU1 = R1->TUPart;
  EXPECT_EQ(2U, DeclsSize(PTU1));

  auto R2 = Interp->Parse("var1++; printf(\"var1 value %d\\n\", var1);");
  EXPECT_TRUE(!!R2);
}

TEST_F(InterpreterTest, UndoCommand) {
  Args ExtraArgs = {"-Xclang", "-diagnostic-log-file", "-Xclang", "-"};

  // Create the diagnostic engine with unowned consumer.
  std::string DiagnosticOutput;
  llvm::raw_string_ostream DiagnosticsOS(DiagnosticOutput);
  auto DiagPrinter = std::make_unique<TextDiagnosticPrinter>(
      DiagnosticsOS, new DiagnosticOptions());

  auto Interp = createInterpreter(ExtraArgs, DiagPrinter.get());

  // Fail to undo.
  auto Err1 = Interp->Undo();
  EXPECT_EQ("Operation failed. Too many undos",
            llvm::toString(std::move(Err1)));
  auto Err2 = Interp->Parse("int foo = 42;");
  EXPECT_TRUE(!!Err2);
  auto Err3 = Interp->Undo(2);
  EXPECT_EQ("Operation failed. Too many undos",
            llvm::toString(std::move(Err3)));

  // Succeed to undo.
  auto Err4 = Interp->Parse("int x = 42;");
  EXPECT_TRUE(!!Err4);
  auto Err5 = Interp->Undo();
  EXPECT_FALSE(Err5);
  auto Err6 = Interp->Parse("int x = 24;");
  EXPECT_TRUE(!!Err6);
  auto Err7 = Interp->Parse("#define X 42");
  EXPECT_TRUE(!!Err7);
  auto Err8 = Interp->Undo();
  EXPECT_FALSE(Err8);
  auto Err9 = Interp->Parse("#define X 24");
  EXPECT_TRUE(!!Err9);

  // Undo input contains errors.
  auto Err10 = Interp->Parse("int y = ;");
  EXPECT_FALSE(!!Err10);
  EXPECT_EQ("Parsing failed.", llvm::toString(Err10.takeError()));
  auto Err11 = Interp->Parse("int y = 42;");
  EXPECT_TRUE(!!Err11);
  auto Err12 = Interp->Undo();
  EXPECT_FALSE(Err12);
}

static std::string MangleName(NamedDecl *ND) {
  ASTContext &C = ND->getASTContext();
  std::unique_ptr<MangleContext> MangleC(C.createMangleContext());
  std::string mangledName;
  llvm::raw_string_ostream RawStr(mangledName);
  MangleC->mangleName(ND, RawStr);
  return mangledName;
}

TEST_F(InterpreterTest, FindMangledNameSymbol) {
  std::unique_ptr<Interpreter> Interp = createInterpreter();

  auto &PTU(cantFail(Interp->Parse("int f(const char*) {return 0;}")));
  EXPECT_EQ(1U, DeclsSize(PTU.TUPart));
  auto R1DeclRange = PTU.TUPart->decls();

  NamedDecl *FD = cast<FunctionDecl>(*R1DeclRange.begin());
  // Lower the PTU
  if (llvm::Error Err = Interp->Execute(PTU)) {
    // We cannot execute on the platform.
    consumeError(std::move(Err));
    return;
  }

  std::string MangledName = MangleName(FD);
  auto Addr = Interp->getSymbolAddress(MangledName);
  EXPECT_FALSE(!Addr);
  EXPECT_NE(0U, Addr->getValue());
  GlobalDecl GD(FD);
  EXPECT_EQ(*Addr, cantFail(Interp->getSymbolAddress(GD)));
  cantFail(
      Interp->ParseAndExecute("extern \"C\" int printf(const char*,...);"));
  Addr = Interp->getSymbolAddress("printf");
  EXPECT_FALSE(!Addr);

  // FIXME: Re-enable when we investigate the way we handle dllimports on Win.
#ifndef _WIN32
  EXPECT_EQ((uintptr_t)&printf, Addr->getValue());
#endif // _WIN32
}

static Value AllocateObject(TypeDecl *TD, Interpreter &Interp) {
  std::string Name = TD->getQualifiedNameAsString();
  Value Addr;
  // FIXME: Consider providing an option in clang::Value to take ownership of
  // the memory created from the interpreter.
  // cantFail(Interp.ParseAndExecute("new " + Name + "()", &Addr));

  // The lifetime of the temporary is extended by the clang::Value.
  cantFail(Interp.ParseAndExecute(Name + "()", &Addr));
  return Addr;
}

static NamedDecl *LookupSingleName(Interpreter &Interp, const char *Name) {
  Sema &SemaRef = Interp.getCompilerInstance()->getSema();
  ASTContext &C = SemaRef.getASTContext();
  DeclarationName DeclName = &C.Idents.get(Name);
  LookupResult R(SemaRef, DeclName, SourceLocation(), Sema::LookupOrdinaryName);
  SemaRef.LookupName(R, SemaRef.TUScope);
  assert(!R.empty());
  return R.getFoundDecl();
}

TEST_F(InterpreterTest, InstantiateTemplate) {
  // FIXME: We cannot yet handle delayed template parsing. If we run with
  // -fdelayed-template-parsing we try adding the newly created decl to the
  // active PTU which causes an assert.
  std::vector<const char *> Args = {"-fno-delayed-template-parsing"};
  std::unique_ptr<Interpreter> Interp = createInterpreter(Args);

  llvm::cantFail(Interp->Parse("extern \"C\" int printf(const char*,...);"
                               "class A {};"
                               "struct B {"
                               "  template<typename T>"
                               "  static int callme(T) { return 42; }"
                               "};"));
  auto &PTU = llvm::cantFail(Interp->Parse("auto _t = &B::callme<A*>;"));
  auto PTUDeclRange = PTU.TUPart->decls();
  EXPECT_EQ(1, std::distance(PTUDeclRange.begin(), PTUDeclRange.end()));

  // Lower the PTU
  if (llvm::Error Err = Interp->Execute(PTU)) {
    // We cannot execute on the platform.
    consumeError(std::move(Err));
    return;
  }

  TypeDecl *TD = cast<TypeDecl>(LookupSingleName(*Interp, "A"));
  Value NewA = AllocateObject(TD, *Interp);

  // Find back the template specialization
  VarDecl *VD = static_cast<VarDecl *>(*PTUDeclRange.begin());
  UnaryOperator *UO = llvm::cast<UnaryOperator>(VD->getInit());
  NamedDecl *TmpltSpec = llvm::cast<DeclRefExpr>(UO->getSubExpr())->getDecl();

  std::string MangledName = MangleName(TmpltSpec);
  typedef int (*TemplateSpecFn)(void *);
  auto fn =
      cantFail(Interp->getSymbolAddress(MangledName)).toPtr<TemplateSpecFn>();
  EXPECT_EQ(42, fn(NewA.getPtr()));
}

TEST_F(InterpreterTest, Value) {
  std::vector<const char *> Args = {"-fno-sized-deallocation"};
  std::unique_ptr<Interpreter> Interp = createInterpreter(Args);

  Value V1;
  llvm::cantFail(Interp->ParseAndExecute("int x = 42;"));
  llvm::cantFail(Interp->ParseAndExecute("x", &V1));
  EXPECT_TRUE(V1.isValid());
  EXPECT_TRUE(V1.hasValue());
  EXPECT_EQ(V1.getInt(), 42);
  EXPECT_EQ(V1.convertTo<int>(), 42);
  EXPECT_TRUE(V1.getType()->isIntegerType());
  EXPECT_EQ(V1.getKind(), Value::K_Int);
  EXPECT_FALSE(V1.isManuallyAlloc());

  Value V1b;
  llvm::cantFail(Interp->ParseAndExecute("char c = 42;"));
  llvm::cantFail(Interp->ParseAndExecute("c", &V1b));
  EXPECT_TRUE(V1b.getKind() == Value::K_Char_S ||
              V1b.getKind() == Value::K_Char_U);

  Value V2;
  llvm::cantFail(Interp->ParseAndExecute("double y = 3.14;"));
  llvm::cantFail(Interp->ParseAndExecute("y", &V2));
  EXPECT_TRUE(V2.isValid());
  EXPECT_TRUE(V2.hasValue());
  EXPECT_EQ(V2.getDouble(), 3.14);
  EXPECT_EQ(V2.convertTo<double>(), 3.14);
  EXPECT_TRUE(V2.getType()->isFloatingType());
  EXPECT_EQ(V2.getKind(), Value::K_Double);
  EXPECT_FALSE(V2.isManuallyAlloc());

  Value V3;
  llvm::cantFail(Interp->ParseAndExecute(
      "struct S { int* p; S() { p = new int(42); } ~S() { delete p; }};"));
  llvm::cantFail(Interp->ParseAndExecute("S{}", &V3));
  EXPECT_TRUE(V3.isValid());
  EXPECT_TRUE(V3.hasValue());
  EXPECT_TRUE(V3.getType()->isRecordType());
  EXPECT_EQ(V3.getKind(), Value::K_PtrOrObj);
  EXPECT_TRUE(V3.isManuallyAlloc());

  Value V4;
  llvm::cantFail(Interp->ParseAndExecute("int getGlobal();"));
  llvm::cantFail(Interp->ParseAndExecute("void setGlobal(int);"));
  llvm::cantFail(Interp->ParseAndExecute("getGlobal()", &V4));
  EXPECT_EQ(V4.getInt(), 42);
  EXPECT_TRUE(V4.getType()->isIntegerType());

  Value V5;
  // Change the global from the compiled code.
  setGlobal(43);
  llvm::cantFail(Interp->ParseAndExecute("getGlobal()", &V5));
  EXPECT_EQ(V5.getInt(), 43);
  EXPECT_TRUE(V5.getType()->isIntegerType());

  // Change the global from the interpreted code.
  llvm::cantFail(Interp->ParseAndExecute("setGlobal(44);"));
  EXPECT_EQ(getGlobal(), 44);

  Value V6;
  llvm::cantFail(Interp->ParseAndExecute("void foo() {}"));
  llvm::cantFail(Interp->ParseAndExecute("foo()", &V6));
  EXPECT_TRUE(V6.isValid());
  EXPECT_FALSE(V6.hasValue());
  EXPECT_TRUE(V6.getType()->isVoidType());
  EXPECT_EQ(V6.getKind(), Value::K_Void);
  EXPECT_FALSE(V2.isManuallyAlloc());

  Value V7;
  llvm::cantFail(Interp->ParseAndExecute("foo", &V7));
  EXPECT_TRUE(V7.isValid());
  EXPECT_TRUE(V7.hasValue());
  EXPECT_TRUE(V7.getType()->isFunctionProtoType());
  EXPECT_EQ(V7.getKind(), Value::K_PtrOrObj);
  EXPECT_FALSE(V7.isManuallyAlloc());

  Value V8;
  llvm::cantFail(Interp->ParseAndExecute("struct SS{ void f() {} };"));
  llvm::cantFail(Interp->ParseAndExecute("&SS::f", &V8));
  EXPECT_TRUE(V8.isValid());
  EXPECT_TRUE(V8.hasValue());
  EXPECT_TRUE(V8.getType()->isMemberFunctionPointerType());
  EXPECT_EQ(V8.getKind(), Value::K_PtrOrObj);
  EXPECT_TRUE(V8.isManuallyAlloc());

  Value V9;
  llvm::cantFail(Interp->ParseAndExecute("struct A { virtual int f(); };"));
  llvm::cantFail(
      Interp->ParseAndExecute("struct B : A { int f() { return 42; }};"));
  llvm::cantFail(Interp->ParseAndExecute("int (B::*ptr)() = &B::f;"));
  llvm::cantFail(Interp->ParseAndExecute("ptr", &V9));
  EXPECT_TRUE(V9.isValid());
  EXPECT_TRUE(V9.hasValue());
  EXPECT_TRUE(V9.getType()->isMemberFunctionPointerType());
  EXPECT_EQ(V9.getKind(), Value::K_PtrOrObj);
  EXPECT_TRUE(V9.isManuallyAlloc());
}

TEST_F(InterpreterTest, TranslationUnit_CanonicalDecl) {
  std::vector<const char *> Args;
  std::unique_ptr<Interpreter> Interp = createInterpreter(Args);

  Sema &sema = Interp->getCompilerInstance()->getSema();

  llvm::cantFail(Interp->ParseAndExecute("int x = 42;"));

  TranslationUnitDecl *TU =
      sema.getASTContext().getTranslationUnitDecl()->getCanonicalDecl();

  llvm::cantFail(Interp->ParseAndExecute("long y = 84;"));

  EXPECT_EQ(TU,
            sema.getASTContext().getTranslationUnitDecl()->getCanonicalDecl());

  llvm::cantFail(Interp->ParseAndExecute("char z = 'z';"));

  EXPECT_EQ(TU,
            sema.getASTContext().getTranslationUnitDecl()->getCanonicalDecl());
}

} // end anonymous namespace
