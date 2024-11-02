//===- unittest/Format/QualifierFixerTest.cpp - Formatting unit tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Format/Format.h"

#include "FormatTestUtils.h"
#include "TestLexer.h"
#include "gtest/gtest.h"

#include "../../lib/Format/QualifierAlignmentFixer.h"

#define DEBUG_TYPE "format-qualifier-fixer-test"

using testing::ScopedTrace;

namespace clang {
namespace format {
namespace {

#define CHECK_PARSE(TEXT, FIELD, VALUE)                                        \
  EXPECT_NE(VALUE, Style.FIELD) << "Initial value already the same!";          \
  EXPECT_EQ(0, parseConfiguration(TEXT, &Style).value());                      \
  EXPECT_EQ(VALUE, Style.FIELD) << "Unexpected value after parsing!"

#define FAIL_PARSE(TEXT, FIELD, VALUE)                                         \
  EXPECT_NE(0, parseConfiguration(TEXT, &Style).value());                      \
  EXPECT_EQ(VALUE, Style.FIELD) << "Unexpected value after parsing!"

class QualifierFixerTest : public ::testing::Test {
protected:
  enum StatusCheck { SC_ExpectComplete, SC_ExpectIncomplete, SC_DoNotCheck };

  TokenList annotate(llvm::StringRef Code,
                     const FormatStyle &Style = getLLVMStyle()) {
    return TestLexer(Allocator, Buffers, Style).annotate(Code);
  }
  llvm::SpecificBumpPtrAllocator<FormatToken> Allocator;
  std::vector<std::unique_ptr<llvm::MemoryBuffer>> Buffers;

  std::string format(llvm::StringRef Code,
                     const FormatStyle &Style = getLLVMStyle(),
                     StatusCheck CheckComplete = SC_ExpectComplete) {
    LLVM_DEBUG(llvm::errs() << "---\n");
    LLVM_DEBUG(llvm::errs() << Code << "\n\n");
    std::vector<tooling::Range> Ranges(1, tooling::Range(0, Code.size()));
    FormattingAttemptStatus Status;
    tooling::Replacements Replaces =
        reformat(Style, Code, Ranges, "<stdin>", &Status);
    if (CheckComplete != SC_DoNotCheck) {
      bool ExpectedCompleteFormat = CheckComplete == SC_ExpectComplete;
      EXPECT_EQ(ExpectedCompleteFormat, Status.FormatComplete)
          << Code << "\n\n";
    }
    ReplacementCount = Replaces.size();
    auto Result = applyAllReplacements(Code, Replaces);
    EXPECT_TRUE(static_cast<bool>(Result));
    LLVM_DEBUG(llvm::errs() << "\n" << *Result << "\n\n");
    return *Result;
  }

  FormatStyle getStyleWithColumns(FormatStyle Style, unsigned ColumnLimit) {
    Style.ColumnLimit = ColumnLimit;
    return Style;
  }

  FormatStyle getLLVMStyleWithColumns(unsigned ColumnLimit) {
    return getStyleWithColumns(getLLVMStyle(), ColumnLimit);
  }

  void _verifyFormat(const char *File, int Line, llvm::StringRef Expected,
                     llvm::StringRef Code,
                     const FormatStyle &Style = getLLVMStyle()) {
    ScopedTrace t(File, Line, ::testing::Message() << Code.str());
    EXPECT_EQ(Expected.str(), format(Expected, Style))
        << "Expected code is not stable";
    EXPECT_EQ(Expected.str(), format(Code, Style));
    if (Style.Language == FormatStyle::LK_Cpp) {
      // Objective-C++ is a superset of C++, so everything checked for C++
      // needs to be checked for Objective-C++ as well.
      FormatStyle ObjCStyle = Style;
      ObjCStyle.Language = FormatStyle::LK_ObjC;
      EXPECT_EQ(Expected.str(), format(test::messUp(Code), ObjCStyle));
    }
  }

  void _verifyFormat(const char *File, int Line, llvm::StringRef Code,
                     const FormatStyle &Style = getLLVMStyle()) {
    _verifyFormat(File, Line, Code, test::messUp(Code), Style);
  }

  void _verifyIncompleteFormat(const char *File, int Line, llvm::StringRef Code,
                               const FormatStyle &Style = getLLVMStyle()) {
    ScopedTrace t(File, Line, ::testing::Message() << Code.str());
    EXPECT_EQ(Code.str(),
              format(test::messUp(Code), Style, SC_ExpectIncomplete));
  }

  void _verifyIndependentOfContext(const char *File, int Line,
                                   llvm::StringRef Text,
                                   const FormatStyle &Style = getLLVMStyle()) {
    _verifyFormat(File, Line, Text, Style);
    _verifyFormat(File, Line, llvm::Twine("void f() { " + Text + " }").str(),
                  Style);
  }

  /// \brief Verify that clang-format does not crash on the given input.
  void verifyNoCrash(llvm::StringRef Code,
                     const FormatStyle &Style = getLLVMStyle()) {
    format(Code, Style, SC_DoNotCheck);
  }

  int ReplacementCount;
};

#define verifyFormat(...) _verifyFormat(__FILE__, __LINE__, __VA_ARGS__)

} // namespace

TEST_F(QualifierFixerTest, RotateTokens) {
  // TODO add test
  EXPECT_EQ(LeftRightQualifierAlignmentFixer::getTokenFromQualifier("const"),
            tok::kw_const);
  EXPECT_EQ(LeftRightQualifierAlignmentFixer::getTokenFromQualifier("volatile"),
            tok::kw_volatile);
  EXPECT_EQ(LeftRightQualifierAlignmentFixer::getTokenFromQualifier("inline"),
            tok::kw_inline);
  EXPECT_EQ(LeftRightQualifierAlignmentFixer::getTokenFromQualifier("static"),
            tok::kw_static);
  EXPECT_EQ(LeftRightQualifierAlignmentFixer::getTokenFromQualifier("restrict"),
            tok::kw_restrict);
}

TEST_F(QualifierFixerTest, FailQualifierInvalidConfiguration) {
  FormatStyle Style = {};
  Style.Language = FormatStyle::LK_Cpp;
  FAIL_PARSE("QualifierAlignment: Custom\n"
             "QualifierOrder: [const, volatile, apples, type]",
             QualifierOrder,
             std::vector<std::string>({"const", "volatile", "apples", "type"}));
}

TEST_F(QualifierFixerTest, FailQualifierDuplicateConfiguration) {
  FormatStyle Style = {};
  Style.Language = FormatStyle::LK_Cpp;
  FAIL_PARSE("QualifierAlignment: Custom\n"
             "QualifierOrder: [const, volatile, const, type]",
             QualifierOrder,
             std::vector<std::string>({"const", "volatile", "const", "type"}));
}

TEST_F(QualifierFixerTest, FailQualifierMissingType) {
  FormatStyle Style = {};
  Style.Language = FormatStyle::LK_Cpp;
  FAIL_PARSE("QualifierAlignment: Custom\n"
             "QualifierOrder: [const, volatile ]",
             QualifierOrder,
             std::vector<std::string>({
                 "const",
                 "volatile",
             }));
}

TEST_F(QualifierFixerTest, FailQualifierEmptyOrder) {
  FormatStyle Style = {};
  Style.Language = FormatStyle::LK_Cpp;
  FAIL_PARSE("QualifierAlignment: Custom\nQualifierOrder: []", QualifierOrder,
             std::vector<std::string>({}));
}

TEST_F(QualifierFixerTest, FailQualifierMissingOrder) {
  FormatStyle Style = {};
  Style.Language = FormatStyle::LK_Cpp;
  FAIL_PARSE("QualifierAlignment: Custom", QualifierOrder,
             std::vector<std::string>());
}

TEST_F(QualifierFixerTest, QualifierLeft) {
  FormatStyle Style = {};
  Style.Language = FormatStyle::LK_Cpp;
  CHECK_PARSE("QualifierAlignment: Left", QualifierOrder,
              std::vector<std::string>({"const", "volatile", "type"}));
}

TEST_F(QualifierFixerTest, QualifierRight) {
  FormatStyle Style = {};
  Style.Language = FormatStyle::LK_Cpp;
  CHECK_PARSE("QualifierAlignment: Right", QualifierOrder,
              std::vector<std::string>({"type", "const", "volatile"}));
}

TEST_F(QualifierFixerTest, QualifiersCustomOrder) {
  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Left;
  Style.QualifierOrder = {"inline", "constexpr", "static",
                          "const",  "volatile",  "type"};

  verifyFormat("const volatile int a;", "const volatile int a;", Style);
  verifyFormat("const volatile int a;", "volatile const int a;", Style);
  verifyFormat("const volatile int a;", "int const volatile a;", Style);
  verifyFormat("const volatile int a;", "int volatile const a;", Style);
  verifyFormat("const volatile int a;", "const int volatile a;", Style);

  verifyFormat("static const volatile int a;", "const static int volatile a;",
               Style);
  verifyFormat("inline static const volatile int a;",
               "const static inline int volatile a;", Style);

  verifyFormat("constexpr static int a;", "static constexpr int a;", Style);
  verifyFormat("constexpr static int A;", "static constexpr int A;", Style);
  verifyFormat("constexpr static int Bar;", "static constexpr int Bar;", Style);
  verifyFormat("constexpr static LPINT Bar;", "static constexpr LPINT Bar;",
               Style);
  verifyFormat("const const int a;", "const int const a;", Style);
}

TEST_F(QualifierFixerTest, LeftRightQualifier) {
  FormatStyle Style = getLLVMStyle();

  // keep the const style unaltered
  verifyFormat("const int a;", Style);
  verifyFormat("const int *a;", Style);
  verifyFormat("const int &a;", Style);
  verifyFormat("const int &&a;", Style);
  verifyFormat("int const b;", Style);
  verifyFormat("int const *b;", Style);
  verifyFormat("int const &b;", Style);
  verifyFormat("int const &&b;", Style);
  verifyFormat("int const *b const;", Style);
  verifyFormat("int *const c;", Style);

  verifyFormat("const Foo a;", Style);
  verifyFormat("const Foo *a;", Style);
  verifyFormat("const Foo &a;", Style);
  verifyFormat("const Foo &&a;", Style);
  verifyFormat("Foo const b;", Style);
  verifyFormat("Foo const *b;", Style);
  verifyFormat("Foo const &b;", Style);
  verifyFormat("Foo const &&b;", Style);
  verifyFormat("Foo const *b const;", Style);

  verifyFormat("LLVM_NODISCARD const int &Foo();", Style);
  verifyFormat("LLVM_NODISCARD int const &Foo();", Style);

  verifyFormat("volatile const int *restrict;", Style);
  verifyFormat("const volatile int *restrict;", Style);
  verifyFormat("const int volatile *restrict;", Style);
}

TEST_F(QualifierFixerTest, RightQualifier) {
  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Right;
  Style.QualifierOrder = {"type", "const", "volatile"};

  verifyFormat("int const a;", Style);
  verifyFormat("int const *a;", Style);
  verifyFormat("int const &a;", Style);
  verifyFormat("int const &&a;", Style);
  verifyFormat("int const b;", Style);
  verifyFormat("int const *b;", Style);
  verifyFormat("int const &b;", Style);
  verifyFormat("int const &&b;", Style);
  verifyFormat("int const *b const;", Style);
  verifyFormat("int *const c;", Style);

  verifyFormat("Foo const a;", Style);
  verifyFormat("Foo const *a;", Style);
  verifyFormat("Foo const &a;", Style);
  verifyFormat("Foo const &&a;", Style);
  verifyFormat("Foo const b;", Style);
  verifyFormat("Foo const *b;", Style);
  verifyFormat("Foo const &b;", Style);
  verifyFormat("Foo const &&b;", Style);
  verifyFormat("Foo const *b const;", Style);
  verifyFormat("Foo *const b;", Style);
  verifyFormat("Foo const *const b;", Style);
  verifyFormat("auto const v = get_value();", Style);
  verifyFormat("long long const &a;", Style);
  verifyFormat("unsigned char const *a;", Style);
  verifyFormat("int main(int const argc, char const *const *const argv)",
               Style);

  verifyFormat("LLVM_NODISCARD int const &Foo();", Style);
  verifyFormat("SourceRange getSourceRange() const override LLVM_READONLY",
               Style);
  verifyFormat("void foo() const override;", Style);
  verifyFormat("void foo() const override LLVM_READONLY;", Style);
  verifyFormat("void foo() const final;", Style);
  verifyFormat("void foo() const final LLVM_READONLY;", Style);
  verifyFormat("void foo() const LLVM_READONLY;", Style);

  verifyFormat(
      "template <typename Func> explicit Action(Action<Func> const &action);",
      Style);
  verifyFormat(
      "template <typename Func> explicit Action(Action<Func> const &action);",
      "template <typename Func> explicit Action(const Action<Func>& action);",
      Style);
  verifyFormat(
      "template <typename Func> explicit Action(Action<Func> const &action);",
      "template <typename Func>\nexplicit Action(const Action<Func>& action);",
      Style);

  verifyFormat("int const a;", "const int a;", Style);
  verifyFormat("int const *a;", "const int *a;", Style);
  verifyFormat("int const &a;", "const int &a;", Style);
  verifyFormat("foo(int const &a)", "foo(const int &a)", Style);
  verifyFormat("unsigned char *a;", "unsigned char *a;", Style);
  verifyFormat("unsigned char const *a;", "const unsigned char *a;", Style);
  verifyFormat("vector<int, int const, int &, int const &> args1",
               "vector<int, const int, int &, const int &> args1", Style);
  verifyFormat("unsigned int const &get_nu() const",
               "const unsigned int &get_nu() const", Style);
  verifyFormat("Foo<int> const &a", "const Foo<int> &a", Style);
  verifyFormat("Foo<int>::iterator const &a", "const Foo<int>::iterator &a",
               Style);
  verifyFormat("::Foo<int>::iterator const &a", "const ::Foo<int>::iterator &a",
               Style);

  verifyFormat("Foo(int a, "
               "unsigned b, // c-style args\n"
               "    Bar const &c);",
               "Foo(int a, "
               "unsigned b, // c-style args\n"
               "    const Bar &c);",
               Style);

  verifyFormat("int const volatile;", "volatile const int;", Style);
  verifyFormat("int const volatile;", "const volatile int;", Style);
  verifyFormat("int const volatile;", "const int volatile;", Style);
  verifyFormat("int const volatile *restrict;", "volatile const int *restrict;",
               Style);
  verifyFormat("int const volatile *restrict;", "const volatile int *restrict;",
               Style);
  verifyFormat("int const volatile *restrict;", "const int volatile *restrict;",
               Style);

  verifyFormat("static int const bat;", "static const int bat;", Style);
  verifyFormat("static int const bat;", "static int const bat;", Style);

  verifyFormat("int const Foo<int>::bat = 0;", "const int Foo<int>::bat = 0;",
               Style);
  verifyFormat("int const Foo<int>::bat = 0;", "int const Foo<int>::bat = 0;",
               Style);
  verifyFormat("void fn(Foo<T> const &i);", "void fn(const Foo<T> &i);", Style);
  verifyFormat("int const Foo<int>::fn() {", "int const Foo<int>::fn() {",
               Style);
  verifyFormat("Foo<Foo<int>> const *p;", "const Foo<Foo<int>> *p;", Style);
  verifyFormat(
      "Foo<Foo<int>> const *p = const_cast<Foo<Foo<int>> const *>(&ffi);",
      "const Foo<Foo<int>> *p = const_cast<const Foo<Foo<int>> *>(&ffi);",
      Style);

  verifyFormat("void fn(Foo<T> const &i);", "void fn(const Foo<T> &i);", Style);
  verifyFormat("void fns(ns::S const &s);", "void fns(const ns::S &s);", Style);
  verifyFormat("void fns(::ns::S const &s);", "void fns(const ::ns::S &s);",
               Style);
  verifyFormat("void fn(ns::Foo<T> const &i);", "void fn(const ns::Foo<T> &i);",
               Style);
  verifyFormat("void fns(ns::ns2::S const &s);",
               "void fns(const ns::ns2::S &s);", Style);
  verifyFormat("void fn(ns::Foo<Bar<T>> const &i);",
               "void fn(const ns::Foo<Bar<T>> &i);", Style);
  verifyFormat("void fn(ns::ns2::Foo<Bar<T>> const &i);",
               "void fn(const ns::ns2::Foo<Bar<T>> &i);", Style);
  verifyFormat("void fn(ns::ns2::Foo<Bar<T, U>> const &i);",
               "void fn(const ns::ns2::Foo<Bar<T, U>> &i);", Style);

  verifyFormat("LocalScope const *Scope = nullptr;",
               "const LocalScope* Scope = nullptr;", Style);
  verifyFormat("struct DOTGraphTraits<Stmt const *>",
               "struct DOTGraphTraits<const Stmt *>", Style);

  verifyFormat(
      "bool tools::addXRayRuntime(ToolChain const &TC, ArgList const &Args) {",
      "bool tools::addXRayRuntime(const ToolChain&TC, const ArgList &Args) {",
      Style);
  verifyFormat("Foo<Foo<int> const> P;", "Foo<const Foo<int>> P;", Style);
  verifyFormat("Foo<Foo<int> const> P;\n", "Foo<const Foo<int>> P;\n", Style);
  verifyFormat("Foo<Foo<int> const> P;\n#if 0\n#else\n#endif",
               "Foo<const Foo<int>> P;\n#if 0\n#else\n#endif", Style);

  verifyFormat("auto const i = 0;", "const auto i = 0;", Style);
  verifyFormat("auto const &ir = i;", "const auto &ir = i;", Style);
  verifyFormat("auto const *ip = &i;", "const auto *ip = &i;", Style);

  verifyFormat("Foo<Foo<int> const> P;\n#if 0\n#else\n#endif",
               "Foo<const Foo<int>> P;\n#if 0\n#else\n#endif", Style);

  verifyFormat("Bar<Bar<int const> const> P;\n#if 0\n#else\n#endif",
               "Bar<Bar<const int> const> P;\n#if 0\n#else\n#endif", Style);

  verifyFormat("Baz<Baz<int const> const> P;\n#if 0\n#else\n#endif",
               "Baz<const Baz<const int>> P;\n#if 0\n#else\n#endif", Style);

  // verifyFormat("#if 0\nBoo<Boo<int const> const> P;\n#else\n#endif",
  //             "#if 0\nBoo<const Boo<const int>> P;\n#else\n#endif", Style);

  verifyFormat("int const P;\n#if 0\n#else\n#endif",
               "const int P;\n#if 0\n#else\n#endif", Style);

  verifyFormat("unsigned long const a;", "const unsigned long a;", Style);
  verifyFormat("unsigned long long const a;", "const unsigned long long a;",
               Style);

  // don't adjust macros
  verifyFormat("const INTPTR a;", "const INTPTR a;", Style);
}

TEST_F(QualifierFixerTest, LeftQualifier) {
  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Left;
  Style.QualifierOrder = {"inline", "static", "const", "volatile", "type"};

  verifyFormat("const int a;", Style);
  verifyFormat("const int *a;", Style);
  verifyFormat("const int &a;", Style);
  verifyFormat("const int &&a;", Style);
  verifyFormat("const int b;", Style);
  verifyFormat("const int *b;", Style);
  verifyFormat("const int &b;", Style);
  verifyFormat("const int &&b;", Style);
  verifyFormat("const int *b const;", Style);
  verifyFormat("int *const c;", Style);

  verifyFormat("const Foo a;", Style);
  verifyFormat("const Foo *a;", Style);
  verifyFormat("const Foo &a;", Style);
  verifyFormat("const Foo &&a;", Style);
  verifyFormat("const Foo b;", Style);
  verifyFormat("const Foo *b;", Style);
  verifyFormat("const Foo &b;", Style);
  verifyFormat("const Foo &&b;", Style);
  verifyFormat("const Foo *b const;", Style);
  verifyFormat("Foo *const b;", Style);
  verifyFormat("const Foo *const b;", Style);

  verifyFormat("LLVM_NODISCARD const int &Foo();", Style);

  verifyFormat("const char a[];", Style);
  verifyFormat("const auto v = get_value();", Style);
  verifyFormat("const long long &a;", Style);
  verifyFormat("const unsigned char *a;", Style);
  verifyFormat("const unsigned char *a;", "unsigned char const *a;", Style);
  verifyFormat("const Foo<int> &a", "Foo<int> const &a", Style);
  verifyFormat("const Foo<int>::iterator &a", "Foo<int>::iterator const &a",
               Style);
  verifyFormat("const ::Foo<int>::iterator &a", "::Foo<int>::iterator const &a",
               Style);

  verifyFormat("const int a;", "int const a;", Style);
  verifyFormat("const int *a;", "int const *a;", Style);
  verifyFormat("const int &a;", "int const &a;", Style);
  verifyFormat("foo(const int &a)", "foo(int const &a)", Style);
  verifyFormat("unsigned char *a;", "unsigned char *a;", Style);
  verifyFormat("const unsigned int &get_nu() const",
               "unsigned int const &get_nu() const", Style);

  verifyFormat("const volatile int;", "volatile const int;", Style);
  verifyFormat("const volatile int;", "const volatile int;", Style);
  verifyFormat("const volatile int;", "const int volatile;", Style);

  verifyFormat("const volatile int *restrict;", "volatile const int *restrict;",
               Style);
  verifyFormat("const volatile int *restrict;", "const volatile int *restrict;",
               Style);
  verifyFormat("const volatile int *restrict;", "const int volatile *restrict;",
               Style);

  verifyFormat("SourceRange getSourceRange() const override LLVM_READONLY;",
               Style);

  verifyFormat("void foo() const override;", Style);
  verifyFormat("void foo() const override LLVM_READONLY;", Style);
  verifyFormat("void foo() const final;", Style);
  verifyFormat("void foo() const final LLVM_READONLY;", Style);
  verifyFormat("void foo() const LLVM_READONLY;", Style);

  verifyFormat(
      "template <typename Func> explicit Action(const Action<Func> &action);",
      Style);
  verifyFormat(
      "template <typename Func> explicit Action(const Action<Func> &action);",
      "template <typename Func> explicit Action(Action<Func> const &action);",
      Style);

  verifyFormat("static const int bat;", "static const int bat;", Style);
  verifyFormat("static const int bat;", "static int const bat;", Style);

  verifyFormat("static const int Foo<int>::bat = 0;",
               "static const int Foo<int>::bat = 0;", Style);
  verifyFormat("static const int Foo<int>::bat = 0;",
               "static int const Foo<int>::bat = 0;", Style);

  verifyFormat("void fn(const Foo<T> &i);");

  verifyFormat("const int Foo<int>::bat = 0;", "const int Foo<int>::bat = 0;",
               Style);
  verifyFormat("const int Foo<int>::bat = 0;", "int const Foo<int>::bat = 0;",
               Style);
  verifyFormat("void fn(const Foo<T> &i);", "void fn( Foo<T> const &i);",
               Style);
  verifyFormat("const int Foo<int>::fn() {", "int const Foo<int>::fn() {",
               Style);
  verifyFormat("const Foo<Foo<int>> *p;", "Foo<Foo<int>> const *p;", Style);
  verifyFormat(
      "const Foo<Foo<int>> *p = const_cast<const Foo<Foo<int>> *>(&ffi);",
      "const Foo<Foo<int>> *p = const_cast<Foo<Foo<int>> const *>(&ffi);",
      Style);

  verifyFormat("void fn(const Foo<T> &i);", "void fn(Foo<T> const &i);", Style);
  verifyFormat("void fns(const ns::S &s);", "void fns(ns::S const &s);", Style);
  verifyFormat("void fns(const ::ns::S &s);", "void fns(::ns::S const &s);",
               Style);
  verifyFormat("void fn(const ns::Foo<T> &i);", "void fn(ns::Foo<T> const &i);",
               Style);
  verifyFormat("void fns(const ns::ns2::S &s);",
               "void fns(ns::ns2::S const &s);", Style);
  verifyFormat("void fn(const ns::Foo<Bar<T>> &i);",
               "void fn(ns::Foo<Bar<T>> const &i);", Style);
  verifyFormat("void fn(const ns::ns2::Foo<Bar<T>> &i);",
               "void fn(ns::ns2::Foo<Bar<T>> const &i);", Style);
  verifyFormat("void fn(const ns::ns2::Foo<Bar<T, U>> &i);",
               "void fn(ns::ns2::Foo<Bar<T, U>> const &i);", Style);

  verifyFormat("const auto i = 0;", "auto const i = 0;", Style);
  verifyFormat("const auto &ir = i;", "auto const &ir = i;", Style);
  verifyFormat("const auto *ip = &i;", "auto const *ip = &i;", Style);

  verifyFormat("Foo<const Foo<int>> P;\n#if 0\n#else\n#endif",
               "Foo<Foo<int> const> P;\n#if 0\n#else\n#endif", Style);

  verifyFormat("Foo<Foo<const int>> P;\n#if 0\n#else\n#endif",
               "Foo<Foo<int const>> P;\n#if 0\n#else\n#endif", Style);

  verifyFormat("const int P;\n#if 0\n#else\n#endif",
               "int const P;\n#if 0\n#else\n#endif", Style);

  verifyFormat("const unsigned long a;", "unsigned long const a;", Style);
  verifyFormat("const unsigned long long a;", "unsigned long long const a;",
               Style);

  verifyFormat("const long long unsigned a;", "long const long unsigned a;",
               Style);

  verifyFormat("const std::Foo", "const std::Foo", Style);
  verifyFormat("const std::Foo<>", "const std::Foo<>", Style);
  verifyFormat("const std::Foo < int", "const std::Foo<int", Style);
  verifyFormat("const std::Foo<int>", "const std::Foo<int>", Style);

  // don't adjust macros
  verifyFormat("INTPTR const a;", "INTPTR const a;", Style);
}

TEST_F(QualifierFixerTest, ConstVolatileQualifiersOrder) {
  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Left;
  Style.QualifierOrder = {"inline", "static", "const", "volatile", "type"};

  // The Default
  EXPECT_EQ(Style.QualifierOrder.size(), (size_t)5);

  verifyFormat("const volatile int a;", "const volatile int a;", Style);
  verifyFormat("const volatile int a;", "volatile const int a;", Style);
  verifyFormat("const volatile int a;", "int const volatile a;", Style);
  verifyFormat("const volatile int a;", "int volatile const a;", Style);
  verifyFormat("const volatile int a;", "const int volatile a;", Style);

  Style.QualifierAlignment = FormatStyle::QAS_Right;
  Style.QualifierOrder = {"type", "const", "volatile"};

  verifyFormat("int const volatile a;", "const volatile int a;", Style);
  verifyFormat("int const volatile a;", "volatile const int a;", Style);
  verifyFormat("int const volatile a;", "int const volatile a;", Style);
  verifyFormat("int const volatile a;", "int volatile const a;", Style);
  verifyFormat("int const volatile a;", "const int volatile a;", Style);

  Style.QualifierAlignment = FormatStyle::QAS_Left;
  Style.QualifierOrder = {"volatile", "const", "type"};

  verifyFormat("volatile const int a;", "const volatile int a;", Style);
  verifyFormat("volatile const int a;", "volatile const int a;", Style);
  verifyFormat("volatile const int a;", "int const volatile a;", Style);
  verifyFormat("volatile const int a;", "int volatile const a;", Style);
  verifyFormat("volatile const int a;", "const int volatile a;", Style);

  Style.QualifierAlignment = FormatStyle::QAS_Right;
  Style.QualifierOrder = {"type", "volatile", "const"};

  verifyFormat("int volatile const a;", "const volatile int a;", Style);
  verifyFormat("int volatile const a;", "volatile const int a;", Style);
  verifyFormat("int volatile const a;", "int const volatile a;", Style);
  verifyFormat("int volatile const a;", "int volatile const a;", Style);
  verifyFormat("int volatile const a;", "const int volatile a;", Style);

  Style.QualifierAlignment = FormatStyle::QAS_Custom;
  Style.QualifierOrder = {"type", "volatile", "const"};

  verifyFormat("int volatile const a;", "const volatile int a;", Style);
  verifyFormat("int volatile const a;", "volatile const int a;", Style);
  verifyFormat("int volatile const a;", "int const volatile a;", Style);
  verifyFormat("int volatile const a;", "int volatile const a;", Style);
  verifyFormat("int volatile const a;", "const int volatile a;", Style);
}

TEST_F(QualifierFixerTest, InlineStatics) {
  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Left;
  Style.QualifierOrder = {"inline", "static", "const", "volatile", "type"};
  EXPECT_EQ(Style.QualifierOrder.size(), (size_t)5);

  verifyFormat("inline static const volatile int a;",
               "const inline static volatile int a;", Style);
  verifyFormat("inline static const volatile int a;",
               "volatile inline static const int a;", Style);
  verifyFormat("inline static const volatile int a;",
               "int const inline static  volatile a;", Style);
  verifyFormat("inline static const volatile int a;",
               "int volatile inline static  const a;", Style);
  verifyFormat("inline static const volatile int a;",
               "const int inline static  volatile a;", Style);
}

TEST_F(QualifierFixerTest, AmpEqual) {
  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Custom;
  Style.QualifierOrder = {"static", "type", "const"};
  EXPECT_EQ(Style.QualifierOrder.size(), (size_t)3);

  verifyFormat("foo(std::string const & = std::string()) const",
               "foo(const std::string & = std::string()) const", Style);
  verifyFormat("foo(std::string const & = std::string())",
               "foo(const std::string & = std::string())", Style);
}

TEST_F(QualifierFixerTest, MoveConstBeyondTypeSmall) {

  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Custom;
  Style.QualifierOrder = {"type", "const"};
  EXPECT_EQ(Style.QualifierOrder.size(), (size_t)2);

  verifyFormat("int const a;", "const int a;", Style);
  verifyFormat("int const *a;", "const int*a;", Style);
  verifyFormat("int const *a;", "const int *a;", Style);
  verifyFormat("int const &a;", "const int &a;", Style);
  verifyFormat("int const &&a;", "const int &&a;", Style);
}

TEST_F(QualifierFixerTest, MoveConstBeforeTypeSmall) {

  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Custom;
  Style.QualifierOrder = {"const", "type"};
  EXPECT_EQ(Style.QualifierOrder.size(), (size_t)2);

  verifyFormat("const int a;", "int const a;", Style);
  verifyFormat("const int *a;", "int const *a;", Style);
  verifyFormat("const int *a const;", "int const *a const;", Style);

  verifyFormat("const int a = foo();", "int const a = foo();", Style);
  verifyFormat("const int *a = foo();", "int const *a = foo();", Style);
  verifyFormat("const int *a const = foo();", "int const *a const = foo();",
               Style);

  verifyFormat("const auto a = foo();", "auto const a = foo();", Style);
  verifyFormat("const auto *a = foo();", "auto const *a = foo();", Style);
  verifyFormat("const auto *a const = foo();", "auto const *a const = foo();",
               Style);
}

TEST_F(QualifierFixerTest, MoveConstBeyondType) {

  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Custom;
  Style.QualifierOrder = {"static", "inline", "type", "const", "volatile"};
  EXPECT_EQ(Style.QualifierOrder.size(), (size_t)5);

  verifyFormat("static inline int const volatile a;",
               "const inline static volatile int a;", Style);
  verifyFormat("static inline int const volatile a;",
               "volatile inline static const int a;", Style);
  verifyFormat("static inline int const volatile a;",
               "int const inline static  volatile a;", Style);
  verifyFormat("static inline int const volatile a;",
               "int volatile inline static  const a;", Style);
  verifyFormat("static inline int const volatile a;",
               "const int inline static  volatile a;", Style);

  verifyFormat("static inline int const volatile *a const;",
               "const int inline static  volatile *a const;", Style);
}

TEST_F(QualifierFixerTest, PrepareLeftRightOrdering) {
  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Custom;
  Style.QualifierOrder = {"static", "inline", "type", "const", "volatile"};

  std::vector<std::string> Left;
  std::vector<std::string> Right;
  std::vector<tok::TokenKind> ConfiguredTokens;
  QualifierAlignmentFixer::PrepareLeftRightOrdering(Style.QualifierOrder, Left,
                                                    Right, ConfiguredTokens);

  EXPECT_EQ(Left.size(), (size_t)2);
  EXPECT_EQ(Right.size(), (size_t)2);

  std::vector<std::string> LeftResult = {"inline", "static"};
  std::vector<std::string> RightResult = {"const", "volatile"};
  EXPECT_EQ(Left, LeftResult);
  EXPECT_EQ(Right, RightResult);
}

TEST_F(QualifierFixerTest, IsQualifierType) {

  std::vector<tok::TokenKind> ConfiguredTokens;
  ConfiguredTokens.push_back(tok::kw_const);
  ConfiguredTokens.push_back(tok::kw_static);
  ConfiguredTokens.push_back(tok::kw_inline);
  ConfiguredTokens.push_back(tok::kw_restrict);
  ConfiguredTokens.push_back(tok::kw_constexpr);

  auto Tokens =
      annotate("const static inline auto restrict int double long constexpr");

  EXPECT_TRUE(LeftRightQualifierAlignmentFixer::isQualifierOrType(
      Tokens[0], ConfiguredTokens));
  EXPECT_TRUE(LeftRightQualifierAlignmentFixer::isQualifierOrType(
      Tokens[1], ConfiguredTokens));
  EXPECT_TRUE(LeftRightQualifierAlignmentFixer::isQualifierOrType(
      Tokens[2], ConfiguredTokens));
  EXPECT_TRUE(LeftRightQualifierAlignmentFixer::isQualifierOrType(
      Tokens[3], ConfiguredTokens));
  EXPECT_TRUE(LeftRightQualifierAlignmentFixer::isQualifierOrType(
      Tokens[4], ConfiguredTokens));
  EXPECT_TRUE(LeftRightQualifierAlignmentFixer::isQualifierOrType(
      Tokens[5], ConfiguredTokens));
  EXPECT_TRUE(LeftRightQualifierAlignmentFixer::isQualifierOrType(
      Tokens[6], ConfiguredTokens));
  EXPECT_TRUE(LeftRightQualifierAlignmentFixer::isQualifierOrType(
      Tokens[7], ConfiguredTokens));
  EXPECT_TRUE(LeftRightQualifierAlignmentFixer::isQualifierOrType(
      Tokens[8], ConfiguredTokens));

  auto NotTokens = annotate("for while do Foo Bar ");

  EXPECT_FALSE(LeftRightQualifierAlignmentFixer::isQualifierOrType(
      NotTokens[0], ConfiguredTokens));
  EXPECT_FALSE(LeftRightQualifierAlignmentFixer::isQualifierOrType(
      NotTokens[1], ConfiguredTokens));
  EXPECT_FALSE(LeftRightQualifierAlignmentFixer::isQualifierOrType(
      NotTokens[2], ConfiguredTokens));
  EXPECT_FALSE(LeftRightQualifierAlignmentFixer::isQualifierOrType(
      NotTokens[3], ConfiguredTokens));
  EXPECT_FALSE(LeftRightQualifierAlignmentFixer::isQualifierOrType(
      NotTokens[4], ConfiguredTokens));
  EXPECT_FALSE(LeftRightQualifierAlignmentFixer::isQualifierOrType(
      NotTokens[5], ConfiguredTokens));
}

TEST_F(QualifierFixerTest, IsMacro) {

  auto Tokens = annotate("INT INTPR Foo int");

  EXPECT_TRUE(LeftRightQualifierAlignmentFixer::isPossibleMacro(Tokens[0]));
  EXPECT_TRUE(LeftRightQualifierAlignmentFixer::isPossibleMacro(Tokens[1]));
  EXPECT_FALSE(LeftRightQualifierAlignmentFixer::isPossibleMacro(Tokens[2]));
  EXPECT_FALSE(LeftRightQualifierAlignmentFixer::isPossibleMacro(Tokens[3]));
}

TEST_F(QualifierFixerTest, OverlappingQualifier) {

  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Left;
  Style.QualifierOrder = {"const", "type"};

  verifyFormat("Foo(const Bar &name);", "Foo(Bar const &name);", Style);
}

TEST_F(QualifierFixerTest, DontPushQualifierThroughNonSpecifiedTypes) {

  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Left;
  Style.QualifierOrder = {"const", "type"};

  verifyFormat("inline static const int a;", Style);

  Style.QualifierOrder = {"static", "const", "type"};

  verifyFormat("inline static const int a;", Style);
  verifyFormat("static inline const int a;", "static inline const int a;",
               Style);

  verifyFormat("static const int a;", "const static int a;", Style);
}

TEST_F(QualifierFixerTest, UnsignedQualifier) {

  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Left;
  Style.QualifierOrder = {"const", "type"};

  verifyFormat("Foo(const unsigned char *bytes)",
               "Foo(unsigned const char *bytes)", Style);

  Style.QualifierAlignment = FormatStyle::QAS_Right;
  Style.QualifierOrder = {"type", "const"};

  verifyFormat("Foo(unsigned char const *bytes)",
               "Foo(unsigned const char *bytes)", Style);
}

TEST_F(QualifierFixerTest, NoOpQualifierReplacements) {

  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Custom;
  Style.QualifierOrder = {"static", "const", "type"};

  ReplacementCount = 0;
  EXPECT_EQ(ReplacementCount, 0);
  verifyFormat("static const uint32 foo[] = {0, 31};", Style);
  verifyFormat("#define MACRO static const", Style);
  verifyFormat("using sc = static const", Style);
  EXPECT_EQ(ReplacementCount, 0);
}

TEST_F(QualifierFixerTest, QualifierTemplates) {
  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Custom;
  Style.QualifierOrder = {"static", "const", "type"};

  ReplacementCount = 0;
  EXPECT_EQ(ReplacementCount, 0);
  verifyFormat("using A = B<>;", Style);
  verifyFormat("using A = B /**/<>;", Style);
  verifyFormat("template <class C> using A = B<Foo<C>, 1>;", Style);
  verifyFormat("template <class C> using A = B /**/<Foo<C>, 1>;", Style);
  verifyFormat("template <class C> using A = B /* */<Foo<C>, 1>;", Style);
  verifyFormat("template <class C> using A = B /*foo*/<Foo<C>, 1>;", Style);
  verifyFormat("template <class C> using A = B /**/ /**/<Foo<C>, 1>;", Style);
  verifyFormat("template <class C> using A = B<Foo</**/ C>, 1>;", Style);
  verifyFormat("template <class C> using A = /**/ B<Foo<C>, 1>;", Style);
  EXPECT_EQ(ReplacementCount, 0);
  verifyFormat("template <class C>\n"
               "using A = B // foo\n"
               "    <Foo<C>, 1>;",
               Style);

  ReplacementCount = 0;
  Style.QualifierOrder = {"type", "static", "const"};
  verifyFormat("using A = B<>;", Style);
  verifyFormat("using A = B /**/<>;", Style);
  verifyFormat("template <class C> using A = B<Foo<C>, 1>;", Style);
  verifyFormat("template <class C> using A = B /**/<Foo<C>, 1>;", Style);
  verifyFormat("template <class C> using A = B /* */<Foo<C>, 1>;", Style);
  verifyFormat("template <class C> using A = B /*foo*/<Foo<C>, 1>;", Style);
  verifyFormat("template <class C> using A = B /**/ /**/<Foo<C>, 1>;", Style);
  verifyFormat("template <class C> using A = B<Foo</**/ C>, 1>;", Style);
  verifyFormat("template <class C> using A = /**/ B<Foo<C>, 1>;", Style);
  EXPECT_EQ(ReplacementCount, 0);
  verifyFormat("template <class C>\n"
               "using A = B // foo\n"
               "    <Foo<C>, 1>;",
               Style);
}

TEST_F(QualifierFixerTest, WithConstraints) {
  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Custom;
  Style.QualifierOrder = {"constexpr", "type"};

  verifyFormat("template <typename T>\n"
               "  requires Concept<F>\n"
               "constexpr constructor();",
               Style);
  verifyFormat("template <typename T>\n"
               "  requires Concept1<F> && Concept2<F>\n"
               "constexpr constructor();",
               Style);
}

TEST_F(QualifierFixerTest, DisableRegions) {
  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Custom;
  Style.QualifierOrder = {"inline", "static", "const", "type"};

  ReplacementCount = 0;
  verifyFormat("// clang-format off\n"
               "int const inline static a = 0;\n"
               "// clang-format on\n",
               Style);
  EXPECT_EQ(ReplacementCount, 0);
  verifyFormat("// clang-format off\n"
               "int const inline static a = 0;\n"
               "// clang-format on\n"
               "inline static const int a = 0;\n",
               "// clang-format off\n"
               "int const inline static a = 0;\n"
               "// clang-format on\n"
               "int const inline static a = 0;\n",
               Style);
}

TEST_F(QualifierFixerTest, TemplatesRight) {
  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Custom;
  Style.QualifierOrder = {"type", "const"};

  verifyFormat("template <typename T>\n"
               "  requires Concept<T const>\n"
               "void f();",
               "template <typename T>\n"
               "  requires Concept<const T>\n"
               "void f();",
               Style);
  verifyFormat("TemplateType<T const> t;", "TemplateType<const T> t;", Style);
  verifyFormat("TemplateType<Container const> t;",
               "TemplateType<const Container> t;", Style);
}

TEST_F(QualifierFixerTest, TemplatesLeft) {
  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Custom;
  Style.QualifierOrder = {"const", "type"};

  verifyFormat("template <const T> t;", "template <T const> t;", Style);
  verifyFormat("template <typename T>\n"
               "  requires Concept<const T>\n"
               "void f();",
               "template <typename T>\n"
               "  requires Concept<T const>\n"
               "void f();",
               Style);
  verifyFormat("TemplateType<const T> t;", "TemplateType<T const> t;", Style);
  verifyFormat("TemplateType<const Container> t;",
               "TemplateType<Container const> t;", Style);
}

} // namespace format
} // namespace clang
