//===- clang/unittest/CodeGen/RISCVLpadLabelGenTest.cpp - lpad unittests --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements unit tests for the RISC-V cf-protection=branch func-sig label scheme
//
//===----------------------------------------------------------------------===//

#include "RISCVLpadLabelGen.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MD5.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Diagnostic.h"

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/Tooling.h"

#include "gtest/gtest.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::CodeGen;

namespace {

// Parses Code, locates the single Decl matching Matcher, and returns the
// psABI signature string computed for it. Fails the test if the match
// isn't unique.
template <typename MatcherT>
std::string sigFor(llvm::StringRef Code, MatcherT Matcher,
                   std::vector<std::string> Args = {"-fsyntax-only",
                                                    "-std=c++17", "-triple",
                                                    "riscv32"},
                   llvm::StringRef FileName = "input.cc") {
  auto AST = tooling::buildASTFromCodeWithArgs(Code, Args, FileName);
  ASTContext &Ctx = AST->getASTContext();
  auto MC = ItaniumMangleContext::create(Ctx, Ctx.getDiagnostics());

  auto Results = match(functionDecl(Matcher).bind("fn"), Ctx);
  if (Results.size() != 1) {
    ADD_FAILURE() << "expected exactly 1 match, got " << Results.size();
    return {};
  }
  const auto *FD = Results[0].template getNodeAs<FunctionDecl>("fn");
  return getRISCVLpadSignatureString(Ctx, *MC, FD);
}

} // namespace

// ---------------------------------------------------------------------
// Decl-level special cases
// ---------------------------------------------------------------------

TEST(RISCVLpadSignature, Main) {
  EXPECT_EQ("FiiPPcE", sigFor("int main(int argc, char **argv) { return 0; }",
                              hasName("main")));
}

TEST(RISCVLpadSignature, MainEmptyParams) {
  EXPECT_EQ("FiiPPcE", sigFor("int main() { return 0; }", hasName("main")));
}

TEST(RISCVLpadSignature, DlRuntimeResolveIsZeroNotHashed) {
  // Special-cased to the literal label 0, bypassing signature/hash
  // computation entirely -- test createRISCVEntryLpadTypeId directly here,
  // not getRISCVLpadSignatureString.
  auto AST =
      tooling::buildASTFromCode("extern \"C\" void _dl_runtime_resolve();");
  ASTContext &Ctx = AST->getASTContext();
  auto MC = ItaniumMangleContext::create(Ctx, Ctx.getDiagnostics());
  auto Results =
      match(functionDecl(hasName("_dl_runtime_resolve")).bind("fn"), Ctx);
  ASSERT_EQ(Results.size(), 1u);
  const auto *FD = Results[0].getNodeAs<FunctionDecl>("fn");

  EXPECT_EQ(0u, createRISCVEntryLpadTypeId(Ctx, *MC, FD));
}

TEST(RISCVLpadSignature, DlRuntimeResolveEntryAndCallSiteLabels) {
  auto AST =
      tooling::buildASTFromCode("extern \"C\" void _dl_runtime_resolve();");
  ASTContext &Ctx = AST->getASTContext();
  auto MC = ItaniumMangleContext::create(Ctx, Ctx.getDiagnostics());

  auto Results =
      match(functionDecl(hasName("_dl_runtime_resolve")).bind("fn"), Ctx);
  ASSERT_EQ(Results.size(), 1u);
  const auto *FD = Results[0].getNodeAs<FunctionDecl>("fn");

  EXPECT_EQ(0u, createRISCVEntryLpadTypeId(Ctx, *MC, FD));

  uint32_t CallSiteLabel = createRISCVCallSiteLpadTypeId(Ctx, *MC, FD);
  EXPECT_NE(0u, CallSiteLabel);
  EXPECT_EQ(computeRISCVLpadLabel(getRISCVLpadSignatureString(Ctx, *MC, FD)),
            CallSiteLabel);
}

TEST(RISCVLpadSignature, StaticDlRuntimeResolveNameCollisionIsNotExempt) {
  auto AST = tooling::buildASTFromCodeWithArgs(
      "static void _dl_runtime_resolve(void) {}",
      {"-fsyntax-only", "-std=c17", "-x", "c"},
      /*FileName=*/"input.c");
  ASTContext &Ctx = AST->getASTContext();
  auto MC = ItaniumMangleContext::create(Ctx, Ctx.getDiagnostics());
  auto Results =
      match(functionDecl(hasName("_dl_runtime_resolve")).bind("fn"), Ctx);
  ASSERT_EQ(Results.size(), 1u);
  const auto *FD = Results[0].getNodeAs<FunctionDecl>("fn");

  // A static function sharing the name must not receive the
  // _dl_runtime_resolve exemption.
  EXPECT_NE(0u, createRISCVEntryLpadTypeId(Ctx, *MC, FD));
}

TEST(RISCVLpadSignature, Destructor) {
  EXPECT_EQ("FvPvE",
            sigFor("class Foo { public: ~Foo(); };", cxxDestructorDecl()));
}

TEST(RISCVLpadSignature, BlockCopyHelperSignature) {
  // void(void*, void*) -- expect second void* to compress via Itanium
  // substitution rather than being re-spelled.
  auto AST = tooling::buildASTFromCode("");
  ASTContext &Ctx = AST->getASTContext();
  auto MC = ItaniumMangleContext::create(Ctx, Ctx.getDiagnostics());
  EXPECT_EQ("FvPvS_E", getRISCVLpadSignatureString(
                           Ctx, *MC, buildBlockCopyHelperSigTy(Ctx)));
}

// ---------------------------------------------------------------------
// Member pointer class substitution + static exemption
// ---------------------------------------------------------------------

TEST(RISCVLpadSignature, PlainMemberFunction) {
  EXPECT_EQ("M1vFvvE",
            sigFor("class Foo { public: void bar(); };", hasName("bar")));
}

TEST(RISCVLpadSignature, StaticMemberTreatedAsFreeFunction) {
  EXPECT_EQ("FviE", sigFor("class Foo { public: static void bar(int); };",
                           hasName("bar")));
}

TEST(RISCVLpadSignature, UnprototypedCFunctionTreatedAsVoid) {
  EXPECT_EQ("FvvE", sigFor("void foo();", hasName("foo"),
                           {"-fsyntax-only", "-std=c17", "-x", "c"},
                           /*FileName=*/"input.c"));
}

// ---------------------------------------------------------------------
// substituteCovariantReturn
// ---------------------------------------------------------------------

TEST(RISCVLpadSignature, PointerToClassReturnSubstituted) {
  EXPECT_EQ("M1vFPviE",
            sigFor("class Foo { public: Foo *bar(int); };", hasName("bar")));
}

TEST(RISCVLpadSignature, ReferenceToClassReturnSubstituted) {
  EXPECT_EQ("M1vFRvvE",
            sigFor("class Foo { public: Foo &bar(); };", hasName("bar")));
}

TEST(RISCVLpadSignature, RvalueReferenceToClassReturnSubstituted) {
  EXPECT_EQ("M1vFOvvE",
            sigFor("class Foo { public: Foo &&bar(); };", hasName("bar")));
}

TEST(RISCVLpadSignature, ValueClassReturnNotSubstituted) {
  EXPECT_EQ("M1vF3BarvE", sigFor("class Bar; class Foo { public: Bar baz(); };",
                                 hasName("baz")));
}

TEST(RISCVLpadSignature, NonClassPointerReturnUnaffected) {
  EXPECT_EQ("M1vFPivE",
            sigFor("class Foo { public: int *bar(); };", hasName("bar")));
}

TEST(RISCVLpadSignature, MultiLevelPointerExempted) {
  // Pointee of the outer pointer is Foo*, not a record -- rule shouldn't
  // fire; inner Foo stays as a real mangled class name.
  EXPECT_EQ("M1vFPP3FoovE",
            sigFor("class Foo { public: Foo **bar(); };", hasName("bar")));
}

TEST(RISCVLpadSignature, CovariantReturnConsistentAcrossCallSiteAndOverride) {
  const char *Code = R"cpp(
    class Base { public: virtual Base *f(int); };
    class Derived : public Base { public: virtual Derived *f(int) override; };
  )cpp";
  EXPECT_EQ(
      sigFor(Code, cxxMethodDecl(hasName("f"), ofClass(hasName("Base")))),
      sigFor(Code, cxxMethodDecl(hasName("f"), ofClass(hasName("Derived")))));
}

TEST(RISCVLpadSignature, CompressionAppliesSubstitution) {
  // Foo appears twice (param + another param of the same pointer-to-Foo
  // type) -- Itanium compression should back-reference the second
  // occurrence rather than re-spelling it.
  EXPECT_EQ("FvP3FooS0_E",
            sigFor("class Foo; void bar(Foo*, Foo*);", hasName("bar")));
}

// ---------------------------------------------------------------------
// stripExceptionSpec
// ---------------------------------------------------------------------

TEST(RISCVLpadSignature, NoexceptMemberFunctionIgnored) {
  EXPECT_EQ("M1vFvvE", sigFor("class Foo { public: void bar() noexcept; };",
                              hasName("bar")));
}

TEST(RISCVLpadSignature, NoexceptFreeFunctionIgnored) {
  EXPECT_EQ("FviE", sigFor("void foo(int) noexcept;", hasName("foo")));
}

TEST(RISCVLpadSignature, NestedExceptionSpecStripped) {
  EXPECT_EQ("FvPFvvEE",
            sigFor("void foo(void (*)() noexcept);", hasName("foo")));
}

TEST(RISCVLpadSignature,
     ExceptionSpecInFunctionPointerTemplateArgumentIsHardError) {
  auto AST =
      tooling::buildASTFromCode("template<typename T> class Wrapper {};"
                                "void bar(Wrapper<void (*)() noexcept>);");
  ASTContext &Ctx = AST->getASTContext();
  auto MC = ItaniumMangleContext::create(Ctx, Ctx.getDiagnostics());
  auto Results = match(functionDecl(hasName("bar")).bind("fn"), Ctx);
  ASSERT_EQ(Results.size(), 1u);
  const auto *FD = Results[0].getNodeAs<FunctionDecl>("fn");

  DiagnosticsEngine &Diags = Ctx.getDiagnostics();
  Diags.setClient(new IgnoringDiagConsumer(), /*ShouldOwnClient=*/true);
  DiagnosticErrorTrap Trap(Diags);

  (void)getRISCVLpadSignatureString(Ctx, *MC, FD);
  EXPECT_TRUE(Trap.hasErrorOccurred());
}

TEST(RISCVLpadSignature,
     ExceptionSpecInMemberFunctionPointerTemplateArgumentIsHardError) {
  // Covers the reviewer's concern more fully: the member-pointer pointee
  // recursion is a structurally distinct path from the plain-pointer one
  // above (see MemberPointerType handling in the recursive scanner).
  auto AST =
      tooling::buildASTFromCode("template<typename T> class Wrapper {};"
                                "struct C {};"
                                "void bar(Wrapper<void (C::*)() noexcept>);");
  ASTContext &Ctx = AST->getASTContext();
  auto MC = ItaniumMangleContext::create(Ctx, Ctx.getDiagnostics());
  auto Results = match(functionDecl(hasName("bar")).bind("fn"), Ctx);
  ASSERT_EQ(Results.size(), 1u);
  const auto *FD = Results[0].getNodeAs<FunctionDecl>("fn");

  DiagnosticsEngine &Diags = Ctx.getDiagnostics();
  Diags.setClient(new IgnoringDiagConsumer(), /*ShouldOwnClient=*/true);
  DiagnosticErrorTrap Trap(Diags);

  (void)getRISCVLpadSignatureString(Ctx, *MC, FD);
  EXPECT_TRUE(Trap.hasErrorOccurred());
}

TEST(RISCVLpadSignature,
     NoexceptFunctionPointerOutsideTemplateArgumentIsIgnored) {
  // Contrast with ExceptionSpecInFunctionPointerTemplateArgumentIsHardError:
  // the same shape, without a template argument boundary, must remain
  // unaffected
  EXPECT_EQ("FvPFvvEE",
            sigFor("void foo(void (*)() noexcept);", hasName("foo")));
}

TEST(RISCVLpadSignature, ExceptionSpecInNestedTemplateArgumentIsHardError) {
  auto AST =
      tooling::buildASTFromCode("template<typename T> class Inner {};"
                                "template<typename T> class Outer {};"
                                "void bar(Outer<Inner<void (*)() noexcept>>);");
  ASTContext &Ctx = AST->getASTContext();
  auto MC = ItaniumMangleContext::create(Ctx, Ctx.getDiagnostics());
  auto Results = match(functionDecl(hasName("bar")).bind("fn"), Ctx);
  ASSERT_EQ(Results.size(), 1u);
  const auto *FD = Results[0].getNodeAs<FunctionDecl>("fn");

  DiagnosticsEngine &Diags = Ctx.getDiagnostics();
  Diags.setClient(new IgnoringDiagConsumer(), /*ShouldOwnClient=*/true);
  DiagnosticErrorTrap Trap(Diags);

  (void)getRISCVLpadSignatureString(Ctx, *MC, FD);
  EXPECT_TRUE(Trap.hasErrorOccurred());
}

// ---------------------------------------------------------------------
// rewriteWCharT
// ---------------------------------------------------------------------

TEST(RISCVLpadSignature, WCharTParam) {
  EXPECT_EQ("FviE", sigFor("void foo(wchar_t);", hasName("foo")));
}

TEST(RISCVLpadSignature, WCharTReturn) {
  EXPECT_EQ("FivE", sigFor("wchar_t foo();", hasName("foo")));
}

TEST(RISCVLpadSignature, WCharTPointer) {
  EXPECT_EQ("FvPiE", sigFor("void foo(wchar_t*);", hasName("foo")));
}

TEST(RISCVLpadSignature, WCharTConstReference) {
  EXPECT_EQ("FvRKiE", sigFor("void foo(const wchar_t&);", hasName("foo")));
}

TEST(RISCVLpadSignature, WCharTDoublePointerFullyRecurses) {
  EXPECT_EQ("FvPPiE", sigFor("void foo(wchar_t**);", hasName("foo")));
}

TEST(RISCVLpadSignature, WCharTInsideMemberPointerPointee) {
  EXPECT_EQ("M1vFviE", sigFor("class Foo { public: void bar(wchar_t); };",
                              hasName("bar")));
}

TEST(RISCVLpadSignature, WCharTPointerToArray) {
  EXPECT_EQ("FvPA4_iE", sigFor("void foo(wchar_t (*)[4]);", hasName("foo")));
}

TEST(RISCVLpadSignature, WCharTThroughTypedefAlias) {
  EXPECT_EQ("FviE", sigFor("using W = wchar_t; void f(W);", hasName("f")));
}

TEST(RISCVLpadSignature, WCharTInsideNestedMemberPointerParam) {
  EXPECT_EQ("FvM3FooFviEE", sigFor("class Foo { public: void bar(wchar_t); };"
                                   "void baz(void (Foo::*)(wchar_t));",
                                   hasName("baz")));
}

TEST(RISCVLpadSignature, WCharTInsideBlockPointer) {
  EXPECT_EQ(
      "FvU13block_pointerFviEE", // confirm exact mangling empirically
      sigFor("void foo(void (^)(wchar_t));", hasName("foo"),
             {"-fsyntax-only", "-std=c++17", "-fblocks", "--target=riscv32"}));
}

/// FIXME: the signatureNormalizer does not work on wchar_t template arguments.
TEST(RISCVLpadSignature, WCharTInTemplateArgumentIsHardError) {
  auto AST = tooling::buildASTFromCode(
      "template<typename T> class Foo {}; void bar(Foo<wchar_t>);");
  ASTContext &Ctx = AST->getASTContext();

  DiagnosticsEngine &Diags = Ctx.getDiagnostics();
  Diags.setClient(new IgnoringDiagConsumer(), /*ShouldOwnClient=*/true);

  DiagnosticErrorTrap Trap(Diags);

  auto MC = ItaniumMangleContext::create(Ctx, Ctx.getDiagnostics());
  auto Results = match(functionDecl(hasName("bar")).bind("fn"), Ctx);
  ASSERT_EQ(Results.size(), 1u);
  const auto *FD = Results[0].getNodeAs<FunctionDecl>("fn");

  (void)getRISCVLpadSignatureString(Ctx, *MC, FD);

  EXPECT_TRUE(Trap.hasErrorOccurred());
}

TEST(RISCVLpadSignature, WCharTNestedTwoLevelsInTemplateArgumentIsHardError) {
  // Foo<Bar<wchar_t>> -- confirms the recursive (not just one-level) case.
  auto AST = tooling::buildASTFromCode("template<typename T> class Bar {};"
                                       "template<typename T> class Foo {};"
                                       "void baz(Foo<Bar<wchar_t>>);");
  ASTContext &Ctx = AST->getASTContext();

  DiagnosticsEngine &Diags = Ctx.getDiagnostics();
  Diags.setClient(new IgnoringDiagConsumer(), /*ShouldOwnClient=*/true);

  DiagnosticErrorTrap Trap(Diags);

  auto MC = ItaniumMangleContext::create(Ctx, Ctx.getDiagnostics());
  auto Results = match(functionDecl(hasName("baz")).bind("fn"), Ctx);
  ASSERT_EQ(Results.size(), 1u);
  const auto *FD = Results[0].getNodeAs<FunctionDecl>("fn");

  (void)getRISCVLpadSignatureString(Ctx, *MC, FD);

  EXPECT_TRUE(Trap.hasErrorOccurred());
}

TEST(RISCVLpadSignature, ShortWCharUsesUnsignedShort) {
  EXPECT_EQ("FvtE", sigFor("void foo(wchar_t);", hasName("foo"),
                           {"-fsyntax-only", "-std=c++17", "-fshort-wchar",
                            "--target=riscv32"}));
}

TEST(RISCVLpadSignature, PointerToVariableLengthArray) {
  EXPECT_EQ("FviPA_iE", sigFor("void foo(int n, int (*p)[n]);", hasName("foo"),
                               {"-fsyntax-only", "-std=c17", "-x", "c",
                                "--target=riscv32"}));
}

// ---------------------------------------------------------------------
// End-to-end: confirms signature-building and hashing are wired together
// correctly.
// ---------------------------------------------------------------------

TEST(RISCVLpadLabel, EndToEndMatchesIndependentHash) {
  auto AST = tooling::buildASTFromCode("class Foo { public: void bar(); };");
  ASTContext &Ctx = AST->getASTContext();
  auto MC = ItaniumMangleContext::create(Ctx, Ctx.getDiagnostics());
  auto Results = match(functionDecl(hasName("bar")).bind("fn"), Ctx);
  ASSERT_EQ(Results.size(), 1u);
  const auto *FD = Results[0].getNodeAs<FunctionDecl>("fn");

  std::string Sig = getRISCVLpadSignatureString(Ctx, *MC, FD);
  EXPECT_EQ(computeRISCVLpadLabel(llvm::StringRef(Sig)),
            createRISCVEntryLpadTypeId(Ctx, *MC, FD));
}

// Clang's Itanium mangler does not currently emit the 'Y' (language
// linkage) marker from CWG1259 for these constructs. Since we delegate
// to Clang's own mangler for <function-type> encoding, there is nothing
// for RISCVLpadLabelGen to strip. These tests pin that behaviour
/// FIXME: if Clang's mangler changes and starts emitting 'Y',
/// getRISCVLpadCanonicalSignature will need an explicit stripping step added.

TEST(RISCVLpadSignature, NoLanguageLinkageMarkerForExternCFunction) {
  EXPECT_EQ("FviE", sigFor(R"cpp(
    extern "C" void c_function(int);
  )cpp",
                           hasName("c_function")));
}

TEST(RISCVLpadSignature, NoLanguageLinkageMarkerForCFunctionPointerParam) {
  EXPECT_EQ("FvPFviEE", sigFor(R"cpp(
    extern "C" void realfunc(int);
    void bar(decltype(realfunc) *p);
  )cpp",
                               hasName("bar")));
}

TEST(RISCVLpadSignature, NoLanguageLinkageMarkerForCFunctionPointerTypedef) {
  EXPECT_EQ("FvPFviEE", sigFor(R"cpp(
    extern "C" typedef void Callback(int);
    void takes_callback(Callback *);
  )cpp",
                               hasName("takes_callback")));
}
