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
#include "clang/AST/Mangle.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"

#include "llvm/Support/TargetSelect.h"

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

static size_t DeclsSize(TranslationUnitDecl *PTUDecl) {
  return std::distance(PTUDecl->decls().begin(), PTUDecl->decls().end());
}

TEST(InterpreterTest, Sanity) {
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

TEST(InterpreterTest, IncrementalInputTopLevelDecls) {
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

  auto RecoverErr = Interp->Parse("int var1 = 42;");
  EXPECT_TRUE(!!RecoverErr);
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
  auto R1 = Interp->Parse(
      "int var1 = 42; extern \"C\" int printf(const char*, ...);");
  // gtest doesn't expand into explicit bool conversions.
  EXPECT_TRUE(!!R1);

  auto *PTU1 = R1->TUPart;
  EXPECT_EQ(2U, DeclsSize(PTU1));

  // FIXME: Add support for wrapping and running statements.
  auto R2 = Interp->Parse("var1++; printf(\"var1 value %d\\n\", var1);");
  EXPECT_FALSE(!!R2);
  using ::testing::HasSubstr;
  EXPECT_THAT(DiagnosticsOS.str(),
              HasSubstr("error: unknown type name 'var1'"));
  auto Err = R2.takeError();
  EXPECT_EQ("Parsing failed.", llvm::toString(std::move(Err)));
}

TEST(InterpreterTest, UndoCommand) {
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
  return RawStr.str();
}

struct LLVMInitRAII {
  LLVMInitRAII() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  }
  ~LLVMInitRAII() { llvm::llvm_shutdown(); }
} LLVMInit;

#ifdef _AIX
TEST(IncrementalProcessing, DISABLED_FindMangledNameSymbol) {
#else
TEST(IncrementalProcessing, FindMangledNameSymbol) {
#endif

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
  auto Addr = cantFail(Interp->getSymbolAddress(MangledName));
  EXPECT_NE(0U, Addr);
  GlobalDecl GD(FD);
  EXPECT_EQ(Addr, cantFail(Interp->getSymbolAddress(GD)));
}

static void *AllocateObject(TypeDecl *TD, Interpreter &Interp) {
  std::string Name = TD->getQualifiedNameAsString();
  const clang::Type *RDTy = TD->getTypeForDecl();
  clang::ASTContext &C = Interp.getCompilerInstance()->getASTContext();
  size_t Size = C.getTypeSize(RDTy);
  void *Addr = malloc(Size);

  // Tell the interpreter to call the default ctor with this memory. Synthesize:
  // new (loc) ClassName;
  static unsigned Counter = 0;
  std::stringstream SS;
  SS << "auto _v" << Counter++ << " = "
     << "new ((void*)"
     // Windows needs us to prefix the hexadecimal value of a pointer with '0x'.
     << std::hex << std::showbase << (size_t)Addr << ")" << Name << "();";

  auto R = Interp.ParseAndExecute(SS.str());
  if (!R) {
    free(Addr);
    return nullptr;
  }

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

#ifdef _AIX
TEST(IncrementalProcessing, DISABLED_InstantiateTemplate) {
#else
TEST(IncrementalProcessing, InstantiateTemplate) {
#endif
  // FIXME: We cannot yet handle delayed template parsing. If we run with
  // -fdelayed-template-parsing we try adding the newly created decl to the
  // active PTU which causes an assert.
  std::vector<const char *> Args = {"-fno-delayed-template-parsing"};
  std::unique_ptr<Interpreter> Interp = createInterpreter(Args);

  llvm::cantFail(Interp->Parse("void* operator new(__SIZE_TYPE__, void* __p);"
                               "extern \"C\" int printf(const char*,...);"
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
  void *NewA = AllocateObject(TD, *Interp);

  // Find back the template specialization
  VarDecl *VD = static_cast<VarDecl *>(*PTUDeclRange.begin());
  UnaryOperator *UO = llvm::cast<UnaryOperator>(VD->getInit());
  NamedDecl *TmpltSpec = llvm::cast<DeclRefExpr>(UO->getSubExpr())->getDecl();

  std::string MangledName = MangleName(TmpltSpec);
  typedef int (*TemplateSpecFn)(void *);
  auto fn = (TemplateSpecFn)cantFail(Interp->getSymbolAddress(MangledName));
  EXPECT_EQ(42, fn(NewA));
  free(NewA);
}

} // end anonymous namespace
