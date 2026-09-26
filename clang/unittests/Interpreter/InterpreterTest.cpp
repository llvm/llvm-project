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

#include "llvm/TargetParser/Host.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <set>

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
  llvm::append_range(ClangArgs, ExtraArgs);
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
  DiagnosticOptions DiagOpts;
  auto DiagPrinter =
      std::make_unique<TextDiagnosticPrinter>(DiagnosticsOS, DiagOpts);

  auto Interp = createInterpreter(ExtraArgs, DiagPrinter.get());
  auto Err = Interp->Parse("intentional_error v1 = 42; ").takeError();
  using ::testing::HasSubstr;
  EXPECT_THAT(DiagnosticOutput,
              HasSubstr("error: unknown type name 'intentional_error'"));
  EXPECT_EQ("Parsing failed.", llvm::toString(std::move(Err)));

  auto RecoverErr = Interp->Parse("int var1 = 42;");
  EXPECT_TRUE(!!RecoverErr);

  Err = Interp->Parse("try { throw 1; } catch { 0; }").takeError();
  EXPECT_THAT(DiagnosticOutput, HasSubstr("error: expected '('"));
  EXPECT_EQ("Parsing failed.", llvm::toString(std::move(Err)));

  RecoverErr = Interp->Parse("var1 = 424;");
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
  DiagnosticOptions DiagOpts;
  auto DiagPrinter =
      std::make_unique<TextDiagnosticPrinter>(DiagnosticsOS, DiagOpts);

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

TEST_F(InterpreterTest, TranslationUnitRedeclChainAcrossManyPTUs) {
  std::unique_ptr<Interpreter> Interp = createInterpreter();

  // One partial translation unit per input, as an interop layer doing a
  // type-probe per lookup would produce.
  for (unsigned I = 0; I != 200; ++I)
    cantFail(Interp->Parse("using probe_" + std::to_string(I) + " = int;"));

  TranslationUnitDecl *TU = Interp->getASTContext().getTranslationUnitDecl();

  unsigned Nodes = 0, Decls = 0;
  for (auto *R : TU->redecls()) {
    ++Nodes;
    for (auto *D : cast<DeclContext>(R)->decls()) {
      ++Decls;
      // Walking up from the decl is what faults in a long-lived session.
      EXPECT_EQ(&D->getASTContext(), &Interp->getASTContext());
      if (auto *ND = dyn_cast<NamedDecl>(D))
        (void)ND->getQualifiedNameAsString();
    }
  }
  EXPECT_GT(Nodes, 1u);
  EXPECT_GT(Decls, 200u);
}

TEST_F(InterpreterTest, UndoLeavesDeclsInTranslationUnitChain) {
  std::unique_ptr<Interpreter> Interp = createInterpreter();

  cantFail(Interp->Parse("struct Kept {};"));
  cantFail(Interp->Parse("struct Withdrawn {};"));
  cantFail(Interp->Undo());

  // A partial translation unit gets its own TranslationUnitDecl, so collect
  // names across the whole redeclaration chain.
  std::set<std::string> Names;
  TranslationUnitDecl *TU = Interp->getASTContext().getTranslationUnitDecl();
  for (auto *R : TU->redecls())
    for (auto *D : cast<DeclContext>(R)->decls())
      if (auto *ND = dyn_cast<NamedDecl>(D))
        Names.insert(ND->getNameAsString());

  EXPECT_TRUE(Names.count("Kept"));
  // Undo withdrew this input, so its declaration should not still be reachable.
  EXPECT_FALSE(Names.count("Withdrawn"));
}

TEST_F(InterpreterTest, UndoCommand) {
// FIXME : This test doesn't current work for Emscripten builds.
// It should be possible to make it work.For details on how it fails and
// the current progress to enable this test see
// the following Github issue https: //
// github.com/llvm/llvm-project/issues/153461
#ifdef __EMSCRIPTEN__
  GTEST_SKIP() << "Test fails for Emscipten builds";
#endif
  Args ExtraArgs = {"-Xclang", "-diagnostic-log-file", "-Xclang", "-"};

  // Create the diagnostic engine with unowned consumer.
  std::string DiagnosticOutput;
  llvm::raw_string_ostream DiagnosticsOS(DiagnosticOutput);
  DiagnosticOptions DiagOpts;
  auto DiagPrinter =
      std::make_unique<TextDiagnosticPrinter>(DiagnosticsOS, DiagOpts);

  auto Interp = createInterpreter(ExtraArgs, DiagPrinter.get());

  // Fail to undo.
  auto Err1 = Interp->Undo();
  EXPECT_EQ("Operation failed. No input left to undo",
            llvm::toString(std::move(Err1)));
  auto Err2 = Interp->Parse("int foo = 42;");
  EXPECT_TRUE(!!Err2);
  auto Err3 = Interp->Undo(2);
  EXPECT_EQ("Operation failed. Wanted to undo 2 inputs, only have 1.",
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

// A failed input is not a PTU, so Undo acts on the last good input.
TEST_F(InterpreterTest, UndoAfterFailedInput) {
#ifdef __EMSCRIPTEN__
  GTEST_SKIP() << "Test fails for Emscipten builds";
#endif
  std::unique_ptr<Interpreter> Interp = createInterpreter();

  // Nothing to undo: a failed input does not count.
  auto Err1 = Interp->Parse("int bad = ;").takeError();
  EXPECT_EQ("Parsing failed.", llvm::toString(std::move(Err1)));
  auto Err2 = Interp->Undo();
  EXPECT_EQ("Operation failed. No input left to undo",
            llvm::toString(std::move(Err2)));

  // Undo after a failed input removes the last good input.
  cantFail(Interp->Parse("int kept = 1;"));
  auto Err3 = Interp->Parse("int bad = kept + ;").takeError();
  EXPECT_EQ("Parsing failed.", llvm::toString(std::move(Err3)));
  cantFail(Interp->Undo());
  auto Err4 = Interp->Parse("int use = kept;").takeError();
  EXPECT_EQ("Parsing failed.", llvm::toString(std::move(Err4)));
  auto Err5 = Interp->Undo();
  EXPECT_EQ("Operation failed. No input left to undo",
            llvm::toString(std::move(Err5)));

  // The name is free again.
  cantFail(Interp->Parse("double kept = 2.0;"));
}

// Undo frees a C-linkage name, so the same definition can come back.
TEST_F(InterpreterTest, UndoExternCDefinition) {
#ifdef __EMSCRIPTEN__
  GTEST_SKIP() << "Test fails for Emscipten builds";
#endif
  std::unique_ptr<Interpreter> Interp = createInterpreter();

  cantFail(Interp->Parse("extern \"C\" int f() { return 1; }"));
  cantFail(Interp->Undo());
  cantFail(Interp->Parse("extern \"C\" int f() { return 1; }"));

  cantFail(Interp->ParseAndExecute("extern \"C\" int g() { return 1; }"));
  cantFail(Interp->Undo());
  cantFail(Interp->ParseAndExecute("extern \"C\" int g() { return 2; }"));
  auto G = cantFail(Interp->getSymbolAddress("g")).toPtr<int (*)()>();
  EXPECT_EQ(2, G());

  cantFail(Interp->Parse("extern \"C\" { int h() { return 1; } }"));
  cantFail(Interp->Undo());
  cantFail(Interp->Parse("extern \"C\" { int h() { return 1; } }"));
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
  EXPECT_EQ(llvm::orc::ExecutorAddr::fromPtr(&printf), *Addr);
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

// A failed input must not leave its implicit instantiations behind. A later
// input that uses them must instantiate them again, without the declarations
// of the failed input.
struct FailedInputInstantiationTest : InterpreterTest {
  std::string DiagText;
  llvm::raw_string_ostream DiagOS{DiagText};
  DiagnosticOptions DiagOpts;
  TextDiagnosticPrinter DiagPrinter{DiagOS, DiagOpts};
  std::unique_ptr<Interpreter> Interp;

  void SetUp() override {
    InterpreterTest::SetUp();
    // FIXME: We cannot yet handle delayed template parsing.
    Interp = createInterpreter({"-fno-delayed-template-parsing"}, &DiagPrinter);
  }

  /// The diagnostics of a failed input.
  std::string ParseFails(llvm::StringRef Code) {
    DiagText.clear();
    llvm::Error Err = Interp->Parse(Code).takeError();
    EXPECT_EQ("Parsing failed.", llvm::toString(std::move(Err)));
    return DiagText;
  }

  void ExpectParseFails(llvm::StringRef Code) { ParseFails(Code); }

  template <typename Fn> Fn *Lookup(llvm::StringRef Name) {
    return cantFail(Interp->getSymbolAddress(Name)).toPtr<Fn *>();
  }
};

TEST_F(FailedInputInstantiationTest, FunctionTemplate) {
  cantFail(Interp->ParseAndExecute(
      "template <class T> T twice(T x) { return x + x; }"));
  ExpectParseFails("extern \"C\" double twice_fail(double *p) {"
                   "  return twice(*p) + no_such_name; }");
  cantFail(Interp->ParseAndExecute(
      "extern \"C\" double twice_after(double x) { return twice(x); }"));
  EXPECT_EQ(6.0, Lookup<double(double)>("twice_after")(3.0));
}

TEST_F(FailedInputInstantiationTest, MemberOfClassTemplate) {
  cantFail(Interp->ParseAndExecute(
      "template <class T> struct Box { T v; T twice() { return v + v; } };"));
  ExpectParseFails("extern \"C\" int box_fail(Box<int> *b) {"
                   "  return b->twice() + no_such_name; }");
  cantFail(Interp->ParseAndExecute("extern \"C\" int box_after(int v) { "
                                   "Box<int> b{v}; return b.twice(); }"));
  EXPECT_EQ(8, Lookup<int(int)>("box_after")(4));
}

TEST_F(FailedInputInstantiationTest, StaticDataMemberOfClassTemplate) {
  cantFail(Interp->ParseAndExecute(
      "template <class T> struct Holder { static T value; };"
      "template <class T> T Holder<T>::value = T(7);"));
  ExpectParseFails("extern \"C\" int holder_fail() {"
                   "  return Holder<int>::value + no_such_name; }");
  cantFail(Interp->ParseAndExecute(
      "extern \"C\" int holder_after() { return Holder<int>::value; }"));
  EXPECT_EQ(7, Lookup<int()>("holder_after")());
}

TEST_F(FailedInputInstantiationTest, VariableTemplate) {
  cantFail(Interp->ParseAndExecute("template <class T> T five = T(5);"));
  ExpectParseFails(
      "extern \"C\" int five_fail() { return five<int> + no_such_name; }");
  cantFail(Interp->ParseAndExecute(
      "extern \"C\" int five_after() { return five<int>; }"));
  EXPECT_EQ(5, Lookup<int()>("five_after")());
}

TEST_F(FailedInputInstantiationTest, VTable) {
  cantFail(
      Interp->ParseAndExecute("struct V { virtual int f() { return 3; } };"));
  ExpectParseFails(
      "extern \"C\" int vtable_fail() { V v; return v.f() + no_such_name; }");
  cantFail(Interp->ParseAndExecute(
      "extern \"C\" int vtable_after() { V v; return v.f(); }"));
  EXPECT_EQ(3, Lookup<int()>("vtable_after")());
}

TEST_F(FailedInputInstantiationTest, UndoThenReuseTemplate) {
  cantFail(Interp->ParseAndExecute(
      "template <class T> T thrice(T x) { return x + x + x; }"));
  cantFail(Interp->ParseAndExecute(
      "extern \"C\" int thrice_a(int x) { return thrice(x); }"));
  ExpectParseFails("extern \"C\" int thrice_fail(int x) { return thrice(x) + "
                   "no_such_name; }");
  // The failed input is not a PTU, so this undoes thrice_a.
  cantFail(Interp->Undo());
  ExpectParseFails("extern \"C\" int thrice_b(int x) { return thrice_a(x); }");
  cantFail(Interp->ParseAndExecute(
      "extern \"C\" int thrice_a(int x) { return thrice(x) + 1; }"));
  EXPECT_EQ(7, Lookup<int(int)>("thrice_a")(2));
}

// The instantiations below bind a dependent call to a function of the failed
// input. The failed input withdraws the function, so the instantiation must
// go too. A later use then finds nothing, or a new overload that the stale
// instantiation would not call: it takes its argument by reference.
using ::testing::HasSubstr;

TEST_F(FailedInputInstantiationTest, PoisonedFunctionTemplate) {
  // The unevaluated use declares readv<S> before the failed input.
  cantFail(Interp->ParseAndExecute(
      "template <class T> int readv(T t) { return value(t); } struct S {};"
      "using R = decltype(readv(S{}));"));
  ExpectParseFails("int value(S) { return no_such_name; }"
                   "extern \"C\" int bad() { return readv(S{}); }");
  EXPECT_THAT(ParseFails("extern \"C\" int still_bad() { return readv(S{}); }"),
              HasSubstr("undeclared identifier 'value'"));
  cantFail(Interp->ParseAndExecute(
      "int value(const S &) { return 7; }"
      "extern \"C\" int good() { return readv(S{}); }"));
  EXPECT_EQ(7, Lookup<int()>("good")());
}

// A constexpr function is instantiated at once, not at the end of the input.
TEST_F(FailedInputInstantiationTest, PoisonedConstexprFunctionTemplate) {
  cantFail(Interp->ParseAndExecute(
      "template <class T> constexpr int readk(T t) { return kval(t) + 1; }"
      "struct S {};"));
  ExpectParseFails("constexpr int kval(S) { return no_such_name; }"
                   "constexpr int bad = readk(S{});");
  EXPECT_THAT(ParseFails("extern \"C\" int still_bad() { return readk(S{}); }"),
              HasSubstr("undeclared identifier 'kval'"));
  cantFail(Interp->ParseAndExecute(
      "constexpr int kval(const S &) { return 4; }"
      "extern \"C\" int good() { return readk(S{}); }"));
  EXPECT_EQ(5, Lookup<int()>("good")());
}

TEST_F(FailedInputInstantiationTest, PoisonedDeducedReturnType) {
  cantFail(Interp->ParseAndExecute("template <class T> auto deduced(T t) { "
                                   "return value(t); } struct S {};"));
  ExpectParseFails("int value(S) { return no_such_name; }"
                   "extern \"C\" int bad() { return deduced(S{}); }");
  // A stale 'int' return type would truncate the result.
  cantFail(Interp->ParseAndExecute(
      "double value(const S &) { return 2.5; }"
      "extern \"C\" double good() { return deduced(S{}); }"));
  EXPECT_EQ(2.5, Lookup<double()>("good")());
}

TEST_F(FailedInputInstantiationTest, PoisonedMemberFunction) {
  cantFail(Interp->ParseAndExecute(
      "template <class T> struct Box { int get() { return init(T{}); } };"
      "struct S {};"));
  ExpectParseFails("int init(S) { return no_such_name; }"
                   "extern \"C\" int bad() { return Box<S>{}.get(); }");
  EXPECT_THAT(
      ParseFails("extern \"C\" int still_bad() { return Box<S>{}.get(); }"),
      HasSubstr("undeclared identifier 'init'"));
  cantFail(Interp->ParseAndExecute(
      "int init(const S &) { return 3; }"
      "extern \"C\" int good() { return Box<S>{}.get(); }"));
  EXPECT_EQ(3, Lookup<int()>("good")());
}

// The first use of a vtable instantiates all virtual functions of the class.
TEST_F(FailedInputInstantiationTest, PoisonedVirtualFunction) {
  cantFail(Interp->ParseAndExecute(
      "template <class T> struct VB { virtual int f() { return init(T{}); } };"
      "struct S {};"));
  ExpectParseFails("int init(S) { return no_such_name; }"
                   "extern \"C\" int bad() { VB<S> v; return v.f(); }");
  EXPECT_THAT(
      ParseFails("extern \"C\" int still_bad() { VB<S> v; return v.f(); }"),
      HasSubstr("undeclared identifier 'init'"));
  cantFail(Interp->ParseAndExecute(
      "int init(const S &) { return 4; }"
      "extern \"C\" int good() { VB<S> v; return v.f(); }"));
  EXPECT_EQ(4, Lookup<int()>("good")());
}

TEST_F(FailedInputInstantiationTest, PoisonedStaticDataMember) {
  cantFail(Interp->ParseAndExecute(
      "template <class T> struct Holder { static int value; };"
      "template <class T> int Holder<T>::value = init(T{}); struct S {};"));
  ExpectParseFails("int init(S) { return no_such_name; }"
                   "extern \"C\" int bad() { return Holder<S>::value; }");
  EXPECT_THAT(
      ParseFails("extern \"C\" int still_bad() { return Holder<S>::value; }"),
      HasSubstr("undeclared identifier 'init'"));
  cantFail(Interp->ParseAndExecute(
      "int init(const S &) { return 9; }"
      "extern \"C\" int good() { return Holder<S>::value; }"));
  EXPECT_EQ(9, Lookup<int()>("good")());
}

TEST_F(FailedInputInstantiationTest, PoisonedVariableTemplate) {
  cantFail(Interp->ParseAndExecute(
      "template <class T> int vt = init(T{}); struct S {};"));
  ExpectParseFails("int init(S) { return no_such_name; }"
                   "extern \"C\" int bad() { return vt<S>; }");
  EXPECT_THAT(ParseFails("extern \"C\" int still_bad() { return vt<S>; }"),
              HasSubstr("undeclared identifier 'init'"));
  cantFail(
      Interp->ParseAndExecute("int init(const S &) { return 8; }"
                              "extern \"C\" int good() { return vt<S>; }"));
  EXPECT_EQ(8, Lookup<int()>("good")());
}

// A type of the failed input as template argument. The new type is a new
// argument, so a later input gets a new specialization. The old one reached
// CodeGen before the error, with the same mangled name.
TEST_F(FailedInputInstantiationTest, TypeOfFailedInputAsArgument) {
  cantFail(Interp->ParseAndExecute(
      "template <class T> constexpr int tag(T) { return T::id; }"));
  ExpectParseFails("struct Tg { static constexpr int id = 1; };"
                   "constexpr int old_tag = tag(Tg{}); int e = no_such_name;");
  cantFail(
      Interp->ParseAndExecute("struct Tg { static constexpr int id = 2; };"
                              "extern \"C\" int good() { return tag(Tg{}); }"));
  EXPECT_EQ(2, Lookup<int()>("good")());
}

// A request without a definition. The template gets its definition later, so
// the next use must request the instantiation again.
TEST_F(FailedInputInstantiationTest, DeclaredThenDefinedTemplate) {
  cantFail(Interp->ParseAndExecute("template <class T> T later(T);"));
  ExpectParseFails(
      "extern \"C\" int bad() { return later(1) + no_such_name; }");
  cantFail(Interp->ParseAndExecute(
      "template <class T> T later(T x) { return x + 1; }"
      "extern \"C\" int good() { return later(1); }"));
  EXPECT_EQ(2, Lookup<int()>("good")());
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

  Value V10;
  llvm::cantFail(Interp->ParseAndExecute(
      "enum D : unsigned int {Zero = 0, One}; One", &V10));

  std::string prettyType;
  llvm::raw_string_ostream OSType(prettyType);
  V10.printType(OSType);
  EXPECT_STREQ(prettyType.c_str(), "D");

  // FIXME: We should print only the value or the constant not the type.
  std::string prettyData;
  llvm::raw_string_ostream OSData(prettyData);
  V10.printData(OSData);
  EXPECT_STREQ(prettyData.c_str(), "(One) : unsigned int 1");

  std::string prettyPrint;
  llvm::raw_string_ostream OSPrint(prettyPrint);
  V10.print(OSPrint);
  EXPECT_STREQ(prettyPrint.c_str(), "(D) (One) : unsigned int 1\n");
}

// Regression: Value::setRawBits's NBytes parameter must be interpreted as a
// byte count end-to-end. Before this was fixed, the parameter was named
// NBits and the memcpy divided by 8, so a caller passing sizeof(T) (the
// natural byte count) ended up copying only sizeof(T)/8 bytes -- leaving
// the upper bytes uninitialised. The only in-tree caller compensated by
// multiplying by 8, hiding the bug.
TEST_F(InterpreterTest, ValueSetRawBitsCopiesByteCount) {
  std::vector<const char *> Args;
  std::unique_ptr<Interpreter> Interp = createInterpreter(Args);

  // Explicit byte count: writing sizeof(long long) bytes must round-trip
  // every byte. Pre-fix this copied 1 byte (8 / 8) and left the upper 7
  // bytes stale.
  Value V;
  llvm::cantFail(Interp->ParseAndExecute("long long x = 0; x", &V));
  ASSERT_EQ(V.getKind(), Value::K_LongLong);
  long long Src = 0x0123456789ABCDEFLL;
  V.setRawBits(&Src, sizeof(Src));
  EXPECT_EQ(V.getLongLong(), Src);

  // Default NBytes argument copies sizeof(Storage). Pre-fix this copied
  // sizeof(Storage) / 8 bytes, dropping the high half of an 8-byte payload.
  Value V2;
  llvm::cantFail(Interp->ParseAndExecute("long long y = 0; y", &V2));
  ASSERT_EQ(V2.getKind(), Value::K_LongLong);
  unsigned char Buf[sizeof(long double)] = {};
  std::memcpy(Buf, &Src, sizeof(Src));
  V2.setRawBits(Buf);
  EXPECT_EQ(V2.getLongLong(), Src);
}

// Regression: Value's move ctor and move-assign must transfer ownership of
// the manually-allocated storage without changing the storage refcount.
// Earlier the move ctor called Release() on the just-moved-into storage,
// double-releasing on the next read.
TEST_F(InterpreterTest, ValueMoveSemantics) {
  std::vector<const char *> Args = {"-fno-sized-deallocation"};
  std::unique_ptr<Interpreter> Interp = createInterpreter(Args);

  llvm::cantFail(
      Interp->ParseAndExecute("struct MoveT { int v = 7; ~MoveT() {} };"));

  // Move-construct: source becomes empty, destination owns the storage.
  Value Src;
  llvm::cantFail(Interp->ParseAndExecute("MoveT{}", &Src));
  ASSERT_EQ(Src.getKind(), Value::K_PtrOrObj);
  ASSERT_TRUE(Src.isManuallyAlloc());
  void *Payload = Src.getPtr();

  Value Moved(std::move(Src));
  EXPECT_EQ(Moved.getKind(), Value::K_PtrOrObj);
  EXPECT_TRUE(Moved.isManuallyAlloc());
  EXPECT_EQ(Moved.getPtr(), Payload);
  EXPECT_EQ(Src.getKind(), Value::K_Unspecified);
  EXPECT_FALSE(Src.isManuallyAlloc());

  // Move-assign over a populated Value: previous storage released, new
  // storage adopted with refcount unchanged.
  Value Other;
  llvm::cantFail(Interp->ParseAndExecute("MoveT{}", &Other));
  Other = std::move(Moved);
  EXPECT_EQ(Other.getKind(), Value::K_PtrOrObj);
  EXPECT_EQ(Other.getPtr(), Payload);
  EXPECT_EQ(Moved.getKind(), Value::K_Unspecified);

  // Copy-construct still works (Retain bumps refcount; both share storage).
  Value Copy(Other);
  EXPECT_EQ(Copy.getKind(), Value::K_PtrOrObj);
  EXPECT_EQ(Copy.getPtr(), Payload);
  EXPECT_EQ(Other.getPtr(), Payload);

  // Force destruction order Copy -> Other -> Interp inside the test body so
  // any latent corruption from a buggy move surfaces here. Pre-fix the move
  // ctor leaves Other holding a dangling pointer; the subsequent Release in
  // ~Copy / ~Other reads or asserts on freed memory. Without explicit
  // teardown the abort happened during global cleanup, after gtest already
  // recorded the test as OK.
  Copy.clear();
  Other.clear();
  Interp.reset();
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
