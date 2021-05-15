//===- unittest/Format/FormatTestObjC.cpp - Formatting unit tests----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Format/Format.h"

#include "../Tooling/ReplacementTest.h"
#include "FormatTestUtils.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"

#define DEBUG_TYPE "format-test"

using testing::ScopedTrace;

namespace clang {
namespace format {
namespace {

class FormatTestObjC : public ::testing::Test {
protected:
  FormatTestObjC() {
    Style = getLLVMStyle();
    Style.Language = FormatStyle::LK_ObjC;
  }

  enum StatusCheck { SC_ExpectComplete, SC_ExpectIncomplete, SC_DoNotCheck };

  std::string format(llvm::StringRef Code,
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
    auto Result = applyAllReplacements(Code, Replaces);
    EXPECT_TRUE(static_cast<bool>(Result));
    LLVM_DEBUG(llvm::errs() << "\n" << *Result << "\n\n");
    return *Result;
  }

  void _verifyFormat(const char *File, int Line, StringRef Code) {
    ScopedTrace t(File, Line, ::testing::Message() << Code.str());
    EXPECT_EQ(Code.str(), format(Code)) << "Expected code is not stable";
    EXPECT_EQ(Code.str(), format(test::messUp(Code)));
  }

  void _verifyIncompleteFormat(const char *File, int Line, StringRef Code) {
    ScopedTrace t(File, Line, ::testing::Message() << Code.str());
    EXPECT_EQ(Code.str(), format(test::messUp(Code), SC_ExpectIncomplete));
  }

  FormatStyle Style;
};

#define verifyIncompleteFormat(...)                                            \
  _verifyIncompleteFormat(__FILE__, __LINE__, __VA_ARGS__)
#define verifyFormat(...) _verifyFormat(__FILE__, __LINE__, __VA_ARGS__)

TEST(FormatTestObjCStyle, DetectsObjCInHeaders) {
  auto Style = getStyle("LLVM", "a.h", "none",
                        "@interface\n"
                        "- (id)init;");
  ASSERT_TRUE((bool)Style);
  EXPECT_EQ(FormatStyle::LK_ObjC, Style->Language);

  Style = getStyle("LLVM", "a.h", "none",
                   "@interface\n"
                   "+ (id)init;");
  ASSERT_TRUE((bool)Style);
  EXPECT_EQ(FormatStyle::LK_ObjC, Style->Language);

  Style = getStyle("LLVM", "a.h", "none",
                   "@interface\n"
                   "@end\n"
                   "//comment");
  ASSERT_TRUE((bool)Style);
  EXPECT_EQ(FormatStyle::LK_ObjC, Style->Language);

  Style = getStyle("LLVM", "a.h", "none",
                   "@interface\n"
                   "@end //comment");
  ASSERT_TRUE((bool)Style);
  EXPECT_EQ(FormatStyle::LK_ObjC, Style->Language);

  // No recognizable ObjC.
  Style = getStyle("LLVM", "a.h", "none", "void f() {}");
  ASSERT_TRUE((bool)Style);
  EXPECT_EQ(FormatStyle::LK_Cpp, Style->Language);

  Style = getStyle("{}", "a.h", "none", "@interface Foo\n@end\n");
  ASSERT_TRUE((bool)Style);
  EXPECT_EQ(FormatStyle::LK_ObjC, Style->Language);

  Style = getStyle("{}", "a.h", "none",
                   "const int interface = 1;\nconst int end = 2;\n");
  ASSERT_TRUE((bool)Style);
  EXPECT_EQ(FormatStyle::LK_Cpp, Style->Language);

  Style = getStyle("{}", "a.h", "none", "@protocol Foo\n@end\n");
  ASSERT_TRUE((bool)Style);
  EXPECT_EQ(FormatStyle::LK_ObjC, Style->Language);

  Style = getStyle("{}", "a.h", "none",
                   "const int protocol = 1;\nconst int end = 2;\n");
  ASSERT_TRUE((bool)Style);
  EXPECT_EQ(FormatStyle::LK_Cpp, Style->Language);

  Style = getStyle("{}", "a.h", "none", "typedef NS_ENUM(int, Foo) {};\n");
  ASSERT_TRUE((bool)Style);
  EXPECT_EQ(FormatStyle::LK_ObjC, Style->Language);

  Style =
      getStyle("{}", "a.h", "none", "typedef NS_CLOSED_ENUM(int, Foo) {};\n");
  ASSERT_TRUE((bool)Style);
  EXPECT_EQ(FormatStyle::LK_ObjC, Style->Language);

  Style = getStyle("{}", "a.h", "none", "enum Foo {};");
  ASSERT_TRUE((bool)Style);
  EXPECT_EQ(FormatStyle::LK_Cpp, Style->Language);

  Style =
      getStyle("{}", "a.h", "none", "inline void Foo() { Log(@\"Foo\"); }\n");
  ASSERT_TRUE((bool)Style);
  EXPECT_EQ(FormatStyle::LK_ObjC, Style->Language);

  Style =
      getStyle("{}", "a.h", "none", "inline void Foo() { Log(\"Foo\"); }\n");
  ASSERT_TRUE((bool)Style);
  EXPECT_EQ(FormatStyle::LK_Cpp, Style->Language);

  Style =
      getStyle("{}", "a.h", "none", "inline void Foo() { id = @[1, 2, 3]; }\n");
  ASSERT_TRUE((bool)Style);
  EXPECT_EQ(FormatStyle::LK_ObjC, Style->Language);

  Style = getStyle("{}", "a.h", "none",
                   "inline void Foo() { id foo = @{1: 2, 3: 4, 5: 6}; }\n");
  ASSERT_TRUE((bool)Style);
  EXPECT_EQ(FormatStyle::LK_ObjC, Style->Language);

  Style = getStyle("{}", "a.h", "none",
                   "inline void Foo() { int foo[] = {1, 2, 3}; }\n");
  ASSERT_TRUE((bool)Style);
  EXPECT_EQ(FormatStyle::LK_Cpp, Style->Language);

  // ObjC characteristic types.
  Style = getStyle("{}", "a.h", "none", "extern NSString *kFoo;\n");
  ASSERT_TRUE((bool)Style);
  EXPECT_EQ(FormatStyle::LK_ObjC, Style->Language);

  Style = getStyle("{}", "a.h", "none", "extern NSInteger Foo();\n");
  ASSERT_TRUE((bool)Style);
  EXPECT_EQ(FormatStyle::LK_ObjC, Style->Language);

  Style = getStyle("{}", "a.h", "none", "NSObject *Foo();\n");
  ASSERT_TRUE((bool)Style);
  EXPECT_EQ(FormatStyle::LK_ObjC, Style->Language);

  Style = getStyle("{}", "a.h", "none", "NSSet *Foo();\n");
  ASSERT_TRUE((bool)Style);
  EXPECT_EQ(FormatStyle::LK_ObjC, Style->Language);
}

TEST(FormatTestObjCStyle, AvoidDetectingDesignatedInitializersAsObjCInHeaders) {
  auto Style = getStyle("LLVM", "a.h", "none",
                        "static const char *names[] = {[0] = \"foo\",\n"
                        "[kBar] = \"bar\"};");
  ASSERT_TRUE((bool)Style);
  EXPECT_EQ(FormatStyle::LK_Cpp, Style->Language);

  Style = getStyle("LLVM", "a.h", "none",
                   "static const char *names[] = {[0] EQ \"foo\",\n"
                   "[kBar] EQ \"bar\"};");
  ASSERT_TRUE((bool)Style);
  EXPECT_EQ(FormatStyle::LK_Cpp, Style->Language);
}

TEST_F(FormatTestObjC, FormatObjCTryCatch) {
  verifyFormat("@try {\n"
               "  f();\n"
               "} @catch (NSException e) {\n"
               "  @throw;\n"
               "} @finally {\n"
               "  exit(42);\n"
               "}");
  verifyFormat("DEBUG({\n"
               "  @try {\n"
               "  } @finally {\n"
               "  }\n"
               "});\n");
}

TEST_F(FormatTestObjC, FormatObjCAutoreleasepool) {
  verifyFormat("@autoreleasepool {\n"
               "  f();\n"
               "}\n"
               "@autoreleasepool {\n"
               "  f();\n"
               "}\n");
  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.AfterControlStatement = FormatStyle::BWACS_Always;
  verifyFormat("@autoreleasepool\n"
               "{\n"
               "  f();\n"
               "}\n"
               "@autoreleasepool\n"
               "{\n"
               "  f();\n"
               "}\n");
}

TEST_F(FormatTestObjC, FormatObjCGenerics) {
  Style.ColumnLimit = 40;
  verifyFormat("int aaaaaaaaaaaaaaaa(\n"
               "    NSArray<aaaaaaaaaaaaaaaaaa *>\n"
               "        aaaaaaaaaaaaaaaaa);\n");
  verifyFormat("int aaaaaaaaaaaaaaaa(\n"
               "    NSArray<aaaaaaaaaaaaaaaaaaa<\n"
               "        aaaaaaaaaaaaaaaa *> *>\n"
               "        aaaaaaaaaaaaaaaaa);\n");
}

TEST_F(FormatTestObjC, FormatObjCSynchronized) {
  verifyFormat("@synchronized(self) {\n"
               "  f();\n"
               "}\n"
               "@synchronized(self) {\n"
               "  f();\n"
               "}\n");
  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.AfterControlStatement = FormatStyle::BWACS_Always;
  verifyFormat("@synchronized(self)\n"
               "{\n"
               "  f();\n"
               "}\n"
               "@synchronized(self)\n"
               "{\n"
               "  f();\n"
               "}\n");
}

TEST_F(FormatTestObjC, FormatObjCInterface) {
  verifyFormat("@interface Foo : NSObject <NSSomeDelegate> {\n"
               "@public\n"
               "  int field1;\n"
               "@protected\n"
               "  int field2;\n"
               "@private\n"
               "  int field3;\n"
               "@package\n"
               "  int field4;\n"
               "}\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("@interface /* wait for it */ Foo\n"
               "+ (id)init;\n"
               "// Look, a comment!\n"
               "- (int)answerWith:(int)i;\n"
               "@end");

  verifyFormat("@interface Foo\n"
               "@end\n"
               "@interface Bar\n"
               "@end");

  verifyFormat("@interface Foo : Bar\n"
               "@property(assign, readwrite) NSInteger bar;\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("FOUNDATION_EXPORT NS_AVAILABLE_IOS(10.0) @interface Foo : Bar\n"
               "@property(assign, readwrite) NSInteger bar;\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("@interface Foo : /**/ Bar /**/ <Baz, /**/ Quux>\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("@interface Foo (HackStuff)\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("@interface Foo ()\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("@interface Foo (HackStuff) <MyProtocol>\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("@interface Foo {\n"
               "  int _i;\n"
               "}\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("@interface Foo : Bar {\n"
               "  int _i;\n"
               "}\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("@interface Foo : Bar <Baz, Quux> {\n"
               "  int _i;\n"
               "}\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("@interface Foo<Baz : Blech> : Bar <Baz, Quux> {\n"
               "  int _i;\n"
               "}\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("@interface Foo<Bar : Baz <Blech>> : Xyzzy <Corge> {\n"
               "  int _i;\n"
               "}\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("@interface Foo<Bar : Baz <Blech>> : Xyzzy <Corge> <Quux> {\n"
               "  int _i;\n"
               "}\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("@interface Foo : Bar <Baz> <Blech>\n"
               "@end");

  verifyFormat("@interface Foo : Bar <Baz> <Blech, Xyzzy, Corge>\n"
               "@end");

  verifyFormat("@interface Foo (HackStuff) {\n"
               "  int _i;\n"
               "}\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("@interface Foo () {\n"
               "  int _i;\n"
               "}\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("@interface Foo (HackStuff) <MyProtocol> {\n"
               "  int _i;\n"
               "}\n"
               "+ (id)init;\n"
               "@end");
  verifyFormat("@interface Foo\n"
               "- (void)foo {\n"
               "}\n"
               "@end\n"
               "@implementation Bar\n"
               "- (void)bar {\n"
               "}\n"
               "@end");
  Style.ColumnLimit = 40;
  verifyFormat("@interface ccccccccccccc () <\n"
               "    ccccccccccccc, ccccccccccccc,\n"
               "    ccccccccccccc, ccccccccccccc> {\n"
               "}");
  verifyFormat("@interface ccccccccccccc (ccccccccccc) <\n"
               "    ccccccccccccc> {\n"
               "}");
  Style.ObjCBinPackProtocolList = FormatStyle::BPS_Never;
  verifyFormat("@interface ddddddddddddd () <\n"
               "    ddddddddddddd,\n"
               "    ddddddddddddd,\n"
               "    ddddddddddddd,\n"
               "    ddddddddddddd> {\n"
               "}");

  Style.BinPackParameters = false;
  Style.ObjCBinPackProtocolList = FormatStyle::BPS_Auto;
  verifyFormat("@interface eeeeeeeeeeeee () <\n"
               "    eeeeeeeeeeeee,\n"
               "    eeeeeeeeeeeee,\n"
               "    eeeeeeeeeeeee,\n"
               "    eeeeeeeeeeeee> {\n"
               "}");
  Style.ObjCBinPackProtocolList = FormatStyle::BPS_Always;
  verifyFormat("@interface fffffffffffff () <\n"
               "    fffffffffffff, fffffffffffff,\n"
               "    fffffffffffff, fffffffffffff> {\n"
               "}");

  Style = getGoogleStyle(FormatStyle::LK_ObjC);
  verifyFormat("@interface Foo : NSObject <NSSomeDelegate> {\n"
               " @public\n"
               "  int field1;\n"
               " @protected\n"
               "  int field2;\n"
               " @private\n"
               "  int field3;\n"
               " @package\n"
               "  int field4;\n"
               "}\n"
               "+ (id)init;\n"
               "@end");
  verifyFormat("@interface Foo : Bar <Baz, Quux>\n"
               "+ (id)init;\n"
               "@end");
  verifyFormat("@interface Foo (HackStuff) <MyProtocol>\n"
               "+ (id)init;\n"
               "@end");
  Style.ColumnLimit = 40;
  // BinPackParameters should be true by default.
  verifyFormat("void eeeeeeee(int eeeee, int eeeee,\n"
               "              int eeeee, int eeeee);\n");
  // ObjCBinPackProtocolList should be BPS_Never by default.
  verifyFormat("@interface fffffffffffff () <\n"
               "    fffffffffffff,\n"
               "    fffffffffffff,\n"
               "    fffffffffffff,\n"
               "    fffffffffffff> {\n"
               "}");
  verifyFormat("@interface ggggggggggggg\n"
               "    : ggggggggggggg <ggggggggggggg>\n"
               "      <ggggggggggggg>\n"
               "@end");
}

TEST_F(FormatTestObjC, FormatObjCImplementation) {
  verifyFormat("@implementation Foo : NSObject {\n"
               "@public\n"
               "  int field1;\n"
               "@protected\n"
               "  int field2;\n"
               "@private\n"
               "  int field3;\n"
               "@package\n"
               "  int field4;\n"
               "}\n"
               "+ (id)init {\n}\n"
               "@end");

  verifyFormat("@implementation Foo\n"
               "+ (id)init {\n"
               "  if (true)\n"
               "    return nil;\n"
               "}\n"
               "// Look, a comment!\n"
               "- (int)answerWith:(int)i {\n"
               "  return i;\n"
               "}\n"
               "+ (int)answerWith:(int)i {\n"
               "  return i;\n"
               "}\n"
               "@end");

  verifyFormat("@implementation Foo\n"
               "@end\n"
               "@implementation Bar\n"
               "@end");

  EXPECT_EQ("@implementation Foo : Bar\n"
            "+ (id)init {\n}\n"
            "- (void)foo {\n}\n"
            "@end",
            format("@implementation Foo : Bar\n"
                   "+(id)init{}\n"
                   "-(void)foo{}\n"
                   "@end"));

  verifyFormat("@implementation Foo {\n"
               "  int _i;\n"
               "}\n"
               "+ (id)init {\n}\n"
               "@end");

  verifyFormat("@implementation Foo : Bar {\n"
               "  int _i;\n"
               "}\n"
               "+ (id)init {\n}\n"
               "@end");

  verifyFormat("@implementation Foo (HackStuff)\n"
               "+ (id)init {\n}\n"
               "@end");
  verifyFormat("@implementation ObjcClass\n"
               "- (void)method;\n"
               "{}\n"
               "@end");

  Style = getGoogleStyle(FormatStyle::LK_ObjC);
  verifyFormat("@implementation Foo : NSObject {\n"
               " @public\n"
               "  int field1;\n"
               " @protected\n"
               "  int field2;\n"
               " @private\n"
               "  int field3;\n"
               " @package\n"
               "  int field4;\n"
               "}\n"
               "+ (id)init {\n}\n"
               "@end");
}

TEST_F(FormatTestObjC, FormatObjCProtocol) {
  verifyFormat("@protocol Foo\n"
               "@property(weak) id delegate;\n"
               "- (NSUInteger)numberOfThings;\n"
               "@end");

  verifyFormat("@protocol MyProtocol <NSObject>\n"
               "- (NSUInteger)numberOfThings;\n"
               "@end");

  verifyFormat("@protocol Foo;\n"
               "@protocol Bar;\n");

  verifyFormat("@protocol Foo\n"
               "@end\n"
               "@protocol Bar\n"
               "@end");

  verifyFormat("FOUNDATION_EXPORT NS_AVAILABLE_IOS(10.0) @protocol Foo\n"
               "@property(assign, readwrite) NSInteger bar;\n"
               "@end");

  verifyFormat("@protocol myProtocol\n"
               "- (void)mandatoryWithInt:(int)i;\n"
               "@optional\n"
               "- (void)optional;\n"
               "@required\n"
               "- (void)required;\n"
               "@optional\n"
               "@property(assign) int madProp;\n"
               "@end\n");

  verifyFormat("@property(nonatomic, assign, readonly)\n"
               "    int *looooooooooooooooooooooooooooongNumber;\n"
               "@property(nonatomic, assign, readonly)\n"
               "    NSString *looooooooooooooooooooooooooooongName;");

  verifyFormat("@implementation PR18406\n"
               "}\n"
               "@end");

  Style = getGoogleStyle(FormatStyle::LK_ObjC);
  verifyFormat("@protocol MyProtocol <NSObject>\n"
               "- (NSUInteger)numberOfThings;\n"
               "@end");
}

TEST_F(FormatTestObjC, FormatObjCMethodDeclarations) {
  verifyFormat("- (void)doSomethingWith:(GTMFoo *)theFoo\n"
               "                   rect:(NSRect)theRect\n"
               "               interval:(float)theInterval {\n"
               "}");
  verifyFormat("- (void)shortf:(GTMFoo *)theFoo\n"
               "      longKeyword:(NSRect)theRect\n"
               "    longerKeyword:(float)theInterval\n"
               "            error:(NSError **)theError {\n"
               "}");
  verifyFormat("- (void)shortf:(GTMFoo *)theFoo\n"
               "          longKeyword:(NSRect)theRect\n"
               "    evenLongerKeyword:(float)theInterval\n"
               "                error:(NSError **)theError {\n"
               "}");
  verifyFormat("+ (instancetype)new;\n");
  Style.ColumnLimit = 60;
  verifyFormat("- (instancetype)initXxxxxx:(id<x>)x\n"
               "                         y:(id<yyyyyyyyyyyyyyyyyyyy>)y\n"
               "    NS_DESIGNATED_INITIALIZER;");
  verifyFormat("- (void)drawRectOn:(id)surface\n"
               "            ofSize:(size_t)height\n"
               "                  :(size_t)width;");
  Style.ColumnLimit = 40;
  // Make sure selectors with 0, 1, or more arguments are indented when wrapped.
  verifyFormat("- (aaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaa;\n");
  verifyFormat("- (aaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaa:(int)a;\n");
  verifyFormat("- (aaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaa:(int)a\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaa:(int)a;\n");
  verifyFormat("- (aaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "     aaaaaaaaaaaaaaaaaaaaaaaaaaa:(int)a\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaa:(int)a;\n");
  verifyFormat("- (aaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaa:(int)a\n"
               "     aaaaaaaaaaaaaaaaaaaaaaaaaaa:(int)a;\n");

  // Continuation indent width should win over aligning colons if the function
  // name is long.
  Style = getGoogleStyle(FormatStyle::LK_ObjC);
  Style.ColumnLimit = 40;
  verifyFormat("- (void)shortf:(GTMFoo *)theFoo\n"
               "    dontAlignNamef:(NSRect)theRect {\n"
               "}");

  // Make sure we don't break aligning for short parameter names.
  verifyFormat("- (void)shortf:(GTMFoo *)theFoo\n"
               "       aShortf:(NSRect)theRect {\n"
               "}");

  // Format pairs correctly.
  Style.ColumnLimit = 80;
  verifyFormat("- (void)drawRectOn:(id)surface\n"
               "            ofSize:(aaaaaaaa)height\n"
               "                  :(size_t)width\n"
               "          atOrigin:(size_t)x\n"
               "                  :(size_t)y\n"
               "             aaaaa:(a)yyy\n"
               "               bbb:(d)cccc;");
  verifyFormat("- (void)drawRectOn:(id)surface ofSize:(aaa)height:(bbb)width;");

  // BraceWrapping AfterFunction is respected for ObjC methods
  Style = getGoogleStyle(FormatStyle::LK_ObjC);
  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.AfterFunction = true;
  verifyFormat("@implementation Foo\n"
               "- (void)foo:(id)bar\n"
               "{\n"
               "}\n"
               "@end\n");
}

TEST_F(FormatTestObjC, FormatObjCMethodExpr) {
  verifyFormat("[foo bar:baz];");
  verifyFormat("[foo bar]->baz;");
  verifyFormat("return [foo bar:baz];");
  verifyFormat("return (a)[foo bar:baz];");
  verifyFormat("f([foo bar:baz]);");
  verifyFormat("f(2, [foo bar:baz]);");
  verifyFormat("f(2, a ? b : c);");
  verifyFormat("[[self initWithInt:4] bar:[baz quux:arrrr]];");

  // Unary operators.
  verifyFormat("int a = +[foo bar:baz];");
  verifyFormat("int a = -[foo bar:baz];");
  verifyFormat("int a = ![foo bar:baz];");
  verifyFormat("int a = ~[foo bar:baz];");
  verifyFormat("int a = ++[foo bar:baz];");
  verifyFormat("int a = --[foo bar:baz];");
  verifyFormat("int a = sizeof [foo bar:baz];");
  verifyFormat("int a = alignof [foo bar:baz];");
  verifyFormat("int a = &[foo bar:baz];");
  verifyFormat("int a = *[foo bar:baz];");
  // FIXME: Make casts work, without breaking f()[4].
  // verifyFormat("int a = (int)[foo bar:baz];");
  // verifyFormat("return (int)[foo bar:baz];");
  // verifyFormat("(void)[foo bar:baz];");
  verifyFormat("return (MyType *)[self.tableView cellForRowAtIndexPath:cell];");

  // Binary operators.
  verifyFormat("[foo bar:baz], [foo bar:baz];");
  verifyFormat("[foo bar:baz] = [foo bar:baz];");
  verifyFormat("[foo bar:baz] *= [foo bar:baz];");
  verifyFormat("[foo bar:baz] /= [foo bar:baz];");
  verifyFormat("[foo bar:baz] %= [foo bar:baz];");
  verifyFormat("[foo bar:baz] += [foo bar:baz];");
  verifyFormat("[foo bar:baz] -= [foo bar:baz];");
  verifyFormat("[foo bar:baz] <<= [foo bar:baz];");
  verifyFormat("[foo bar:baz] >>= [foo bar:baz];");
  verifyFormat("[foo bar:baz] &= [foo bar:baz];");
  verifyFormat("[foo bar:baz] ^= [foo bar:baz];");
  verifyFormat("[foo bar:baz] |= [foo bar:baz];");
  verifyFormat("[foo bar:baz] ? [foo bar:baz] : [foo bar:baz];");
  verifyFormat("[foo bar:baz] || [foo bar:baz];");
  verifyFormat("[foo bar:baz] && [foo bar:baz];");
  verifyFormat("[foo bar:baz] | [foo bar:baz];");
  verifyFormat("[foo bar:baz] ^ [foo bar:baz];");
  verifyFormat("[foo bar:baz] & [foo bar:baz];");
  verifyFormat("[foo bar:baz] == [foo bar:baz];");
  verifyFormat("[foo bar:baz] != [foo bar:baz];");
  verifyFormat("[foo bar:baz] >= [foo bar:baz];");
  verifyFormat("[foo bar:baz] <= [foo bar:baz];");
  verifyFormat("[foo bar:baz] > [foo bar:baz];");
  verifyFormat("[foo bar:baz] < [foo bar:baz];");
  verifyFormat("[foo bar:baz] >> [foo bar:baz];");
  verifyFormat("[foo bar:baz] << [foo bar:baz];");
  verifyFormat("[foo bar:baz] - [foo bar:baz];");
  verifyFormat("[foo bar:baz] + [foo bar:baz];");
  verifyFormat("[foo bar:baz] * [foo bar:baz];");
  verifyFormat("[foo bar:baz] / [foo bar:baz];");
  verifyFormat("[foo bar:baz] % [foo bar:baz];");
  // Whew!

  verifyFormat("return in[42];");
  verifyFormat("for (auto v : in[1]) {\n}");
  verifyFormat("for (int i = 0; i < in[a]; ++i) {\n}");
  verifyFormat("for (int i = 0; in[a] < i; ++i) {\n}");
  verifyFormat("for (int i = 0; i < n; ++i, ++in[a]) {\n}");
  verifyFormat("for (int i = 0; i < n; ++i, in[a]++) {\n}");
  verifyFormat("for (int i = 0; i < f(in[a]); ++i, in[a]++) {\n}");
  verifyFormat("for (id foo in [self getStuffFor:bla]) {\n"
               "}");
  verifyFormat("[self aaaaa:MACRO(a, b:, c:)];");
  verifyFormat("[self aaaaa:MACRO(a, b:c:, d:e:)];");
  verifyFormat("[self aaaaa:MACRO(a, b:c:d:, e:f:g:)];");
  verifyFormat("int XYMyFoo(int a, int b) NS_SWIFT_NAME(foo(self:scale:));");
  verifyFormat("[self aaaaa:(1 + 2) bbbbb:3];");
  verifyFormat("[self aaaaa:(Type)a bbbbb:3];");

  verifyFormat("[self stuffWithInt:(4 + 2) float:4.5];");
  verifyFormat("[self stuffWithInt:a ? b : c float:4.5];");
  verifyFormat("[self stuffWithInt:a ? [self foo:bar] : c];");
  verifyFormat("[self stuffWithInt:a ? (e ? f : g) : c];");
  verifyFormat("[cond ? obj1 : obj2 methodWithParam:param]");
  verifyFormat("[button setAction:@selector(zoomOut:)];");
  verifyFormat("[color getRed:&r green:&g blue:&b alpha:&a];");

  verifyFormat("arr[[self indexForFoo:a]];");
  verifyFormat("throw [self errorFor:a];");
  verifyFormat("@throw [self errorFor:a];");

  verifyFormat("[(id)foo bar:(id)baz quux:(id)snorf];");
  verifyFormat("[(id)foo bar:(id) ? baz : quux];");
  verifyFormat("4 > 4 ? (id)a : (id)baz;");

  unsigned PreviousColumnLimit = Style.ColumnLimit;
  Style.ColumnLimit = 50;
  // Instead of:
  // bool a =
  //     ([object a:42] == 0 || [object a:42
  //                                    b:42] == 0);
  verifyFormat("bool a = ([object a:42] == 0 ||\n"
               "          [object a:42 b:42] == 0);");
  Style.ColumnLimit = PreviousColumnLimit;
  verifyFormat("bool a = ([aaaaaaaa aaaaa] == aaaaaaaaaaaaaaaaa ||\n"
               "          [aaaaaaaa aaaaa] == aaaaaaaaaaaaaaaaaaaa);");

  // This tests that the formatter doesn't break after "backing" but before ":",
  // which would be at 80 columns.
  verifyFormat(
      "void f() {\n"
      "  if ((self = [super initWithContentRect:contentRect\n"
      "                               styleMask:styleMask ?: otherMask\n"
      "                                 backing:NSBackingStoreBuffered\n"
      "                                   defer:YES]))");

  verifyFormat(
      "[foo checkThatBreakingAfterColonWorksOk:\n"
      "         [bar ifItDoes:reduceOverallLineLengthLikeInThisCase]];");

  verifyFormat("[myObj short:arg1 // Force line break\n"
               "          longKeyword:arg2 != nil ? arg2 : @\"longKeyword\"\n"
               "    evenLongerKeyword:arg3 ?: @\"evenLongerKeyword\"\n"
               "                error:arg4];");
  verifyFormat(
      "void f() {\n"
      "  popup_window_.reset([[RenderWidgetPopupWindow alloc]\n"
      "      initWithContentRect:NSMakeRect(origin_global.x, origin_global.y,\n"
      "                                     pos.width(), pos.height())\n"
      "                styleMask:NSBorderlessWindowMask\n"
      "                  backing:NSBackingStoreBuffered\n"
      "                    defer:NO]);\n"
      "}");
  verifyFormat("[contentsContainer replaceSubview:[subviews objectAtIndex:0]\n"
               "                             with:contentsNativeView];");

  verifyFormat(
      "[pboard addTypes:[NSArray arrayWithObject:kBookmarkButtonDragType]\n"
      "           owner:nillllll];");

  verifyFormat(
      "[pboard setData:[NSData dataWithBytes:&button length:sizeof(button)]\n"
      "        forType:kBookmarkButtonDragType];");

  verifyFormat("[defaultCenter addObserver:self\n"
               "                  selector:@selector(willEnterFullscreen)\n"
               "                      name:kWillEnterFullscreenNotification\n"
               "                    object:nil];");
  verifyFormat("[image_rep drawInRect:drawRect\n"
               "             fromRect:NSZeroRect\n"
               "            operation:NSCompositeCopy\n"
               "             fraction:1.0\n"
               "       respectFlipped:NO\n"
               "                hints:nil];");
  verifyFormat("[aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa];");
  verifyFormat("[aaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaa)\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa];");
  verifyFormat("[aaaaaaaaaaaaaaaaaaaaaaa.aaaaaaaa[aaaaaaaaaaaaaaaaaaaaa]\n"
               "    aaaaaaaaaaaaaaaaaaaaaa];");

  verifyFormat(
      "scoped_nsobject<NSTextField> message(\n"
      "    // The frame will be fixed up when |-setMessageText:| is called.\n"
      "    [[NSTextField alloc] initWithFrame:NSMakeRect(0, 0, 0, 0)]);");
  verifyFormat("[self aaaaaa:bbbbbbbbbbbbb\n"
               "    aaaaaaaaaa:bbbbbbbbbbbbbbbbb\n"
               "         aaaaa:bbbbbbbbbbb + bbbbbbbbbbbb\n"
               "          aaaa:bbb];");
  verifyFormat("[self param:function( //\n"
               "                parameter)]");
  verifyFormat(
      "[self aaaaaaaaaa:aaaaaaaaaaaaaaa | aaaaaaaaaaaaaaa | aaaaaaaaaaaaaaa |\n"
      "                 aaaaaaaaaaaaaaa | aaaaaaaaaaaaaaa | aaaaaaaaaaaaaaa |\n"
      "                 aaaaaaaaaaaaaaa | aaaaaaaaaaaaaaa];");

  // Variadic parameters.
  verifyFormat(
      "NSArray *myStrings = [NSArray stringarray:@\"a\", @\"b\", nil];");
  verifyFormat(
      "[self aaaaaaaaaaaaa:aaaaaaaaaaaaaaa, aaaaaaaaaaaaaaa, aaaaaaaaaaaaaaa,\n"
      "                    aaaaaaaaaaaaaaa, aaaaaaaaaaaaaaa, aaaaaaaaaaaaaaa,\n"
      "                    aaaaaaaaaaaaaaa, aaaaaaaaaaaaaaa];");

  verifyFormat("[self // break\n"
               "      a:a\n"
               "    aaa:aaa];");

  // Formats pair-parameters.
  verifyFormat("[I drawRectOn:surface ofSize:aa:bbb atOrigin:cc:dd];");
  verifyFormat("[I drawRectOn:surface //\n"
               "       ofSize:aa:bbb\n"
               "     atOrigin:cc:dd];");

  // Inline block as a first argument.
  verifyFormat("[object justBlock:^{\n"
               "  a = 42;\n"
               "}];");
  verifyFormat("[object\n"
               "    justBlock:^{\n"
               "      a = 42;\n"
               "    }\n"
               "     notBlock:42\n"
               "            a:42];");
  verifyFormat("[object\n"
               "    firstBlock:^{\n"
               "      a = 42;\n"
               "    }\n"
               "    blockWithLongerName:^{\n"
               "      a = 42;\n"
               "    }];");
  verifyFormat("[object\n"
               "    blockWithLongerName:^{\n"
               "      a = 42;\n"
               "    }\n"
               "    secondBlock:^{\n"
               "      a = 42;\n"
               "    }];");
  verifyFormat("[object\n"
               "    firstBlock:^{\n"
               "      a = 42;\n"
               "    }\n"
               "    notBlock:42\n"
               "    secondBlock:^{\n"
               "      a = 42;\n"
               "    }];");

  // Space between cast rparen and selector name component.
  verifyFormat("[((Foo *)foo) bar];");
  verifyFormat("[((Foo *)foo) bar:1 blech:2];");

  Style.ColumnLimit = 20;
  verifyFormat("aaaaa = [a aa:aa\n"
               "           aa:aa];");
  verifyFormat("aaaaaa = [aa aa:aa\n"
               "             aa:aa];");

  // Message receiver taking multiple lines.
  // Non-corner case.
  verifyFormat("[[object block:^{\n"
               "  return 42;\n"
               "}] a:42 b:42];");
  // Arguments just fit into one line.
  verifyFormat("[[object block:^{\n"
               "  return 42;\n"
               "}] aaaaaaa:42 b:42];");
  // Arguments just over a column limit.
  verifyFormat("[[object block:^{\n"
               "  return 42;\n"
               "}] aaaaaaa:42\n"
               "        bb:42];");
  // Arguments just fit into one line.
  Style.ColumnLimit = 23;
  verifyFormat("[[obj a:42\n"
               "      b:42\n"
               "      c:42\n"
               "      d:42] e:42 f:42];");

  // Arguments do not fit into one line with a receiver.
  Style.ColumnLimit = 20;
  verifyFormat("[[obj a:42] a:42\n"
               "            b:42];");
  verifyFormat("[[obj a:42] a:42\n"
               "            b:42\n"
               "            c:42];");
  verifyFormat("[[obj aaaaaa:42\n"
               "           b:42]\n"
               "    cc:42\n"
               "     d:42];");

  // Avoid breaking receiver expression.
  Style.ColumnLimit = 30;
  verifyFormat("fooooooo =\n"
               "    [[obj fooo] aaa:42\n"
               "                aaa:42];");
  verifyFormat("[[[obj foo] bar] aa:42\n"
               "                 bb:42\n"
               "                 cc:42];");

  // Avoid breaking between unary operators and ObjC method expressions.
  Style.ColumnLimit = 45;
  verifyFormat("if (a012345678901234567890123 &&\n"
               "    ![foo bar]) {\n"
               "}");
  verifyFormat("if (a012345678901234567890123 &&\n"
               "    +[foo bar]) {\n"
               "}");
  verifyFormat("if (a012345678901234567890123 &&\n"
               "    -[foo bar]) {\n"
               "}");

  Style.ColumnLimit = 70;
  verifyFormat(
      "void f() {\n"
      "  popup_wdow_.reset([[RenderWidgetPopupWindow alloc]\n"
      "      iniithContentRect:NSMakRet(origin_global.x, origin_global.y,\n"
      "                                 pos.width(), pos.height())\n"
      "                syeMask:NSBorderlessWindowMask\n"
      "                  bking:NSBackingStoreBuffered\n"
      "                    der:NO]);\n"
      "}");

  Style.ColumnLimit = 60;
  verifyFormat("[call aaaaaaaa.aaaaaa.aaaaaaaa.aaaaaaaa.aaaaaaaa.aaaaaaaa\n"
               "        .aaaaaaaa];"); // FIXME: Indentation seems off.
  // FIXME: This violates the column limit.
  verifyFormat(
      "[aaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "    aaaaaaaaaaaaaaaaa:aaaaaaaa\n"
      "                  aaa:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa];");

  Style = getChromiumStyle(FormatStyle::LK_ObjC);
  Style.ColumnLimit = 80;
  verifyFormat(
      "void f() {\n"
      "  popup_window_.reset([[RenderWidgetPopupWindow alloc]\n"
      "      initWithContentRect:NSMakeRect(origin_global.x, origin_global.y,\n"
      "                                     pos.width(), pos.height())\n"
      "                styleMask:NSBorderlessWindowMask\n"
      "                  backing:NSBackingStoreBuffered\n"
      "                    defer:NO]);\n"
      "}");

  // Respect continuation indent and colon alignment (e.g. when object name is
  // short, and first selector is the longest one)
  Style = getLLVMStyle();
  Style.Language = FormatStyle::LK_ObjC;
  Style.ContinuationIndentWidth = 8;
  verifyFormat("[self performSelectorOnMainThread:@selector(loadAccessories)\n"
               "                       withObject:nil\n"
               "                    waitUntilDone:false];");
  verifyFormat("[self performSelector:@selector(loadAccessories)\n"
               "        withObjectOnMainThread:nil\n"
               "                 waitUntilDone:false];");
  verifyFormat(
      "[aaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "        performSelectorOnMainThread:@selector(loadAccessories)\n"
      "                         withObject:nil\n"
      "                      waitUntilDone:false];");
  verifyFormat(
      "[self // force wrapping\n"
      "        performSelectorOnMainThread:@selector(loadAccessories)\n"
      "                         withObject:nil\n"
      "                      waitUntilDone:false];");

  // The appropriate indentation is used after a block statement.
  Style.ContinuationIndentWidth = 4;
  verifyFormat(
      "void aaaaaaaaaaaaaaaaaaaaa(int c) {\n"
      "  if (c) {\n"
      "    f();\n"
      "  }\n"
      "  [dddddddddddddddddddddddddddddddddddddddddddddddddddddddd\n"
      "      eeeeeeeeeeeeeeeeeeeeeeeeeeeee:^(fffffffffffffff gggggggg) {\n"
      "        f(SSSSS, c);\n"
      "      }];\n"
      "}");
}

TEST_F(FormatTestObjC, ObjCAt) {
  verifyFormat("@autoreleasepool");
  verifyFormat("@catch");
  verifyFormat("@class");
  verifyFormat("@compatibility_alias");
  verifyFormat("@defs");
  verifyFormat("@dynamic");
  verifyFormat("@encode");
  verifyFormat("@end");
  verifyFormat("@finally");
  verifyFormat("@implementation");
  verifyFormat("@import");
  verifyFormat("@interface");
  verifyFormat("@optional");
  verifyFormat("@package");
  verifyFormat("@private");
  verifyFormat("@property");
  verifyFormat("@protected");
  verifyFormat("@protocol");
  verifyFormat("@public");
  verifyFormat("@required");
  verifyFormat("@selector");
  verifyFormat("@synchronized");
  verifyFormat("@synthesize");
  verifyFormat("@throw");
  verifyFormat("@try");

  EXPECT_EQ("@interface", format("@ interface"));

  // The precise formatting of this doesn't matter, nobody writes code like
  // this.
  verifyFormat("@ /*foo*/ interface");
}

TEST_F(FormatTestObjC, ObjCBlockTypesAndVariables) {
  verifyFormat("void DoStuffWithBlockType(int (^)(char));");
  verifyFormat("int (^foo)(char, float);");
  verifyFormat("int (^foo[10])(char, float);");
  verifyFormat("int (^foo[kNumEntries])(char, float);");
  verifyFormat("int (^foo[kNumEntries + 10])(char, float);");
  verifyFormat("int (^foo[(kNumEntries + 10)])(char, float);");
}

TEST_F(FormatTestObjC, ObjCSnippets) {
  verifyFormat("@autoreleasepool {\n"
               "  foo();\n"
               "}");
  verifyFormat("@class Foo, Bar;");
  verifyFormat("@compatibility_alias AliasName ExistingClass;");
  verifyFormat("@dynamic textColor;");
  verifyFormat("char *buf1 = @encode(int *);");
  verifyFormat("char *buf1 = @encode(typeof(4 * 5));");
  verifyFormat("char *buf1 = @encode(int **);");
  verifyFormat("Protocol *proto = @protocol(p1);");
  verifyFormat("SEL s = @selector(foo:);");
  verifyFormat("@synchronized(self) {\n"
               "  f();\n"
               "}");

  verifyFormat("@import foo.bar;\n"
               "@import baz;");

  verifyFormat("@synthesize dropArrowPosition = dropArrowPosition_;");

  verifyFormat("@property(assign, nonatomic) CGFloat hoverAlpha;");
  verifyFormat("@property(assign, getter=isEditable) BOOL editable;");

  verifyFormat("extern UIWindow *MainWindow(void) "
               "NS_SWIFT_NAME(getter:MyHelper.mainWindow());");

  verifyFormat("extern UIWindow *MainWindow(void) "
               "CF_SWIFT_NAME(getter:MyHelper.mainWindow());");

  Style.ColumnLimit = 50;
  verifyFormat("@interface Foo\n"
               "- (void)doStuffWithFoo:(id)name\n"
               "                   bar:(id)bar\n"
               "                   baz:(id)baz\n"
               "    NS_SWIFT_NAME(doStuff(withFoo:bar:baz:));\n"
               "@end");

  Style = getMozillaStyle();
  verifyFormat("@property (assign, getter=isEditable) BOOL editable;");
  verifyFormat("@property BOOL editable;");

  Style = getWebKitStyle();
  verifyFormat("@property (assign, getter=isEditable) BOOL editable;");
  verifyFormat("@property BOOL editable;");

  Style = getGoogleStyle(FormatStyle::LK_ObjC);
  verifyFormat("@synthesize dropArrowPosition = dropArrowPosition_;");
  verifyFormat("@property(assign, getter=isEditable) BOOL editable;");
}

TEST_F(FormatTestObjC, ObjCForIn) {
  verifyFormat("- (void)test {\n"
               "  for (NSString *n in arrayOfStrings) {\n"
               "    foo(n);\n"
               "  }\n"
               "}");
  verifyFormat("- (void)test {\n"
               "  for (NSString *n in (__bridge NSArray *)arrayOfStrings) {\n"
               "    foo(n);\n"
               "  }\n"
               "}");
  verifyFormat("for (Foo *x in bar) {\n}");
  verifyFormat("for (Foo *x in [bar baz]) {\n}");
  verifyFormat("for (Foo *x in [bar baz:blech]) {\n}");
  verifyFormat("for (Foo *x in [bar baz:blech, 1, 2, 3, 0]) {\n}");
  verifyFormat("for (Foo *x in [bar baz:^{\n"
               "       [uh oh];\n"
               "     }]) {\n}");
}

TEST_F(FormatTestObjC, ObjCCxxKeywords) {
  verifyFormat("+ (instancetype)new {\n"
               "  return nil;\n"
               "}\n");
  verifyFormat("+ (instancetype)myNew {\n"
               "  return [self new];\n"
               "}\n");
  verifyFormat("SEL NewSelector(void) { return @selector(new); }\n");
  verifyFormat("SEL MacroSelector(void) { return MACRO(new); }\n");
  verifyFormat("+ (instancetype)delete {\n"
               "  return nil;\n"
               "}\n");
  verifyFormat("+ (instancetype)myDelete {\n"
               "  return [self delete];\n"
               "}\n");
  verifyFormat("SEL DeleteSelector(void) { return @selector(delete); }\n");
  verifyFormat("SEL MacroSelector(void) { return MACRO(delete); }\n");
  verifyFormat("MACRO(new:)\n");
  verifyFormat("MACRO(delete:)\n");
  verifyFormat("foo = @{MACRO(new:) : MACRO(delete:)}\n");
  verifyFormat("@implementation Foo\n"
               "// Testing\n"
               "- (Class)class {\n"
               "}\n"
               "- (void)foo {\n"
               "}\n"
               "@end\n");
  verifyFormat("@implementation Foo\n"
               "- (Class)class {\n"
               "}\n"
               "- (void)foo {\n"
               "}\n"
               "@end");
  verifyFormat("@implementation Foo\n"
               "+ (Class)class {\n"
               "}\n"
               "- (void)foo {\n"
               "}\n"
               "@end");
  verifyFormat("@implementation Foo\n"
               "- (Class)class:(Class)klass {\n"
               "}\n"
               "- (void)foo {\n"
               "}\n"
               "@end");
  verifyFormat("@implementation Foo\n"
               "+ (Class)class:(Class)klass {\n"
               "}\n"
               "- (void)foo {\n"
               "}\n"
               "@end");

  verifyFormat("@interface Foo\n"
               "// Testing\n"
               "- (Class)class;\n"
               "- (void)foo;\n"
               "@end\n");
  verifyFormat("@interface Foo\n"
               "- (Class)class;\n"
               "- (void)foo;\n"
               "@end");
  verifyFormat("@interface Foo\n"
               "+ (Class)class;\n"
               "- (void)foo;\n"
               "@end");
  verifyFormat("@interface Foo\n"
               "- (Class)class:(Class)klass;\n"
               "- (void)foo;\n"
               "@end");
  verifyFormat("@interface Foo\n"
               "+ (Class)class:(Class)klass;\n"
               "- (void)foo;\n"
               "@end");
}

TEST_F(FormatTestObjC, ObjCLiterals) {
  verifyFormat("@\"String\"");
  verifyFormat("@1");
  verifyFormat("@+4.8");
  verifyFormat("@-4");
  verifyFormat("@1LL");
  verifyFormat("@.5");
  verifyFormat("@'c'");
  verifyFormat("@true");

  verifyFormat("NSNumber *smallestInt = @(-INT_MAX - 1);");
  verifyFormat("NSNumber *piOverTwo = @(M_PI / 2);");
  verifyFormat("NSNumber *favoriteColor = @(Green);");
  verifyFormat("NSString *path = @(getenv(\"PATH\"));");

  verifyFormat("[dictionary setObject:@(1) forKey:@\"number\"];");
}

TEST_F(FormatTestObjC, ObjCDictLiterals) {
  verifyFormat("@{");
  verifyFormat("@{}");
  verifyFormat("@{@\"one\" : @1}");
  verifyFormat("return @{@\"one\" : @1;");
  verifyFormat("@{@\"one\" : @1}");

  verifyFormat("@{@\"one\" : @{@2 : @1}}");
  verifyFormat("@{\n"
               "  @\"one\" : @{@2 : @1},\n"
               "}");

  verifyFormat("@{1 > 2 ? @\"one\" : @\"two\" : 1 > 2 ? @1 : @2}");
  verifyIncompleteFormat("[self setDict:@{}");
  verifyIncompleteFormat("[self setDict:@{@1 : @2}");
  verifyFormat("NSLog(@\"%@\", @{@1 : @2, @2 : @3}[@1]);");
  verifyFormat(
      "NSDictionary *masses = @{@\"H\" : @1.0078, @\"He\" : @4.0026};");
  verifyFormat(
      "NSDictionary *settings = @{AVEncoderKey : @(AVAudioQualityMax)};");

  verifyFormat("NSDictionary *d = @{\n"
               "  @\"nam\" : NSUserNam(),\n"
               "  @\"dte\" : [NSDate date],\n"
               "  @\"processInfo\" : [NSProcessInfo processInfo]\n"
               "};");
  verifyFormat(
      "@{\n"
      "  NSFontAttributeNameeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee : "
      "regularFont,\n"
      "};");
  verifyFormat(
      "@{\n"
      "  NSFontAttributeNameeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee :\n"
      "      reeeeeeeeeeeeeeeeeeeeeeeegularFont,\n"
      "};");

  // We should try to be robust in case someone forgets the "@".
  verifyFormat("NSDictionary *d = {\n"
               "  @\"nam\" : NSUserNam(),\n"
               "  @\"dte\" : [NSDate date],\n"
               "  @\"processInfo\" : [NSProcessInfo processInfo]\n"
               "};");
  verifyFormat("NSMutableDictionary *dictionary =\n"
               "    [NSMutableDictionary dictionaryWithDictionary:@{\n"
               "      aaaaaaaaaaaaaaaaaaaaa : aaaaaaaaaaaaa,\n"
               "      bbbbbbbbbbbbbbbbbb : bbbbb,\n"
               "      cccccccccccccccc : ccccccccccccccc\n"
               "    }];");

  // Ensure that casts before the key are kept on the same line as the key.
  verifyFormat(
      "NSDictionary *d = @{\n"
      "  (aaaaaaaa id)aaaaaaaaa : (aaaaaaaa id)aaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "  (aaaaaaaa id)aaaaaaaaaaaaaa : (aaaaaaaa id)aaaaaaaaaaaaaa,\n"
      "};");
  Style.ColumnLimit = 40;
  verifyFormat("int Foo() {\n"
               "  a12345 = @{a12345 : a12345};\n"
               "}");
  verifyFormat("int Foo() {\n"
               "  a12345 = @{a12345 : @(a12345)};\n"
               "}");
  verifyFormat("int Foo() {\n"
               "  a12345 = @{(Foo *)a12345 : @(a12345)};\n"
               "}");
  verifyFormat("int Foo() {\n"
               "  a12345 = @{@(a12345) : a12345};\n"
               "}");
  verifyFormat("int Foo() {\n"
               "  a12345 = @{@(a12345) : @YES};\n"
               "}");
  Style.SpacesInContainerLiterals = false;
  verifyFormat("int Foo() {\n"
               "  b12345 = @{b12345: b12345};\n"
               "}");
  verifyFormat("int Foo() {\n"
               "  b12345 = @{(Foo *)b12345: @(b12345)};\n"
               "}");
  Style.SpacesInContainerLiterals = true;

  Style = getGoogleStyle(FormatStyle::LK_ObjC);
  verifyFormat(
      "@{\n"
      "  NSFontAttributeNameeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee : "
      "regularFont,\n"
      "};");
}

TEST_F(FormatTestObjC, ObjCArrayLiterals) {
  verifyIncompleteFormat("@[");
  verifyFormat("@[]");
  verifyFormat(
      "NSArray *array = @[ @\" Hey \", NSApp, [NSNumber numberWithInt:42] ];");
  verifyFormat("return @[ @3, @[], @[ @4, @5 ] ];");
  verifyFormat("NSArray *array = @[ [foo description] ];");

  verifyFormat(
      "NSArray *some_variable = @[\n"
      "  aaaa == bbbbbbbbbbb ? @\"aaaaaaaaaaaa\" : @\"aaaaaaaaaaaaaa\",\n"
      "  @\"aaaaaaaaaaaaaaaaa\",\n"
      "  @\"aaaaaaaaaaaaaaaaa\",\n"
      "  @\"aaaaaaaaaaaaaaaaa\",\n"
      "];");
  verifyFormat(
      "NSArray *some_variable = @[\n"
      "  aaaa == bbbbbbbbbbb ? @\"aaaaaaaaaaaa\" : @\"aaaaaaaaaaaaaa\",\n"
      "  @\"aaaaaaaaaaaaaaaa\", @\"aaaaaaaaaaaaaaaa\", @\"aaaaaaaaaaaaaaaa\"\n"
      "];");
  verifyFormat("NSArray *some_variable = @[\n"
               "  @\"aaaaaaaaaaaaaaaaa\",\n"
               "  @\"aaaaaaaaaaaaaaaaa\",\n"
               "  @\"aaaaaaaaaaaaaaaaa\",\n"
               "  @\"aaaaaaaaaaaaaaaaa\",\n"
               "];");
  verifyFormat("NSArray *array = @[\n"
               "  @\"a\",\n"
               "  @\"a\",\n" // Trailing comma -> one per line.
               "];");

  // We should try to be robust in case someone forgets the "@".
  verifyFormat("NSArray *some_variable = [\n"
               "  @\"aaaaaaaaaaaaaaaaa\",\n"
               "  @\"aaaaaaaaaaaaaaaaa\",\n"
               "  @\"aaaaaaaaaaaaaaaaa\",\n"
               "  @\"aaaaaaaaaaaaaaaaa\",\n"
               "];");
  verifyFormat(
      "- (NSAttributedString *)attributedStringForSegment:(NSUInteger)segment\n"
      "                                             index:(NSUInteger)index\n"
      "                                nonDigitAttributes:\n"
      "                                    (NSDictionary *)noDigitAttributes;");
  verifyFormat("[someFunction someLooooooooooooongParameter:@[\n"
               "  NSBundle.mainBundle.infoDictionary[@\"a\"]\n"
               "]];");
  Style.ColumnLimit = 40;
  verifyFormat("int Foo() {\n"
               "  a12345 = @[ a12345, a12345 ];\n"
               "}");
  verifyFormat("int Foo() {\n"
               "  a123 = @[ (Foo *)a12345, @(a12345) ];\n"
               "}");
  Style.SpacesInContainerLiterals = false;
  verifyFormat("int Foo() {\n"
               "  b12345 = @[b12345, b12345];\n"
               "}");
  verifyFormat("int Foo() {\n"
               "  b12345 = @[(Foo *)b12345, @(b12345)];\n"
               "}");
  Style.SpacesInContainerLiterals = true;
  Style.ColumnLimit = 20;
  // We can't break string literals inside NSArray literals
  // (that raises -Wobjc-string-concatenation).
  verifyFormat("NSArray *foo = @[\n"
               "  @\"aaaaaaaaaaaaaaaaaaaaaaaaaa\"\n"
               "];\n");
}

TEST_F(FormatTestObjC, BreaksCallStatementWhereSemiJustOverTheLimit) {
  Style.ColumnLimit = 60;
  // If the statement starting with 'a = ...' is put on a single line, the ';'
  // is at line 61.
  verifyFormat("int f(int a) {\n"
               "  a = [self aaaaaaaaaa:bbbbbbbbb\n"
               "             ccccccccc:dddddddd\n"
               "                    ee:fddd];\n"
               "}");
}

TEST_F(FormatTestObjC, AlwaysBreakBeforeMultilineStrings) {
  Style = getGoogleStyle(FormatStyle::LK_ObjC);
  Style.ColumnLimit = 40;
  verifyFormat("aaaa = @\"bbbb\"\n"
               "       @\"cccc\";");
  verifyFormat("aaaa(@\"bbbb\"\n"
               "     @\"cccc\");");
  verifyFormat("aaaa(qqq, @\"bbbb\"\n"
               "          @\"cccc\");");
  verifyFormat("[aaaa qqqq:@\"bbbb\"\n"
               "           @\"cccc\"];");
  verifyFormat("aaaa = [aaaa qqqq:@\"bbbb\"\n"
               "                  @\"cccc\"];");
  verifyFormat("[aaaa qqqq:@\"bbbb\"\n"
               "           @\"cccc\"\n"
               "        rr:42\n"
               "    ssssss:@\"ee\"\n"
               "           @\"fffff\"];");
}

TEST_F(FormatTestObjC, DisambiguatesCallsFromCppLambdas) {
  verifyFormat("x = ([a foo:bar] && b->c == 'd');");
  verifyFormat("x = ([a foo:bar] + b->c == 'd');");
  verifyFormat("x = ([a foo:bar] + !b->c == 'd');");
  verifyFormat("x = ([a foo:bar] + ~b->c == 'd');");
  verifyFormat("x = ([a foo:bar] - b->c == 'd');");
  verifyFormat("x = ([a foo:bar] / b->c == 'd');");
  verifyFormat("x = ([a foo:bar] % b->c == 'd');");
  verifyFormat("x = ([a foo:bar] | b->c == 'd');");
  verifyFormat("x = ([a foo:bar] || b->c == 'd');");
  verifyFormat("x = ([a foo:bar] && b->c == 'd');");
  verifyFormat("x = ([a foo:bar] == b->c == 'd');");
  verifyFormat("x = ([a foo:bar] != b->c == 'd');");
  verifyFormat("x = ([a foo:bar] <= b->c == 'd');");
  verifyFormat("x = ([a foo:bar] >= b->c == 'd');");
  verifyFormat("x = ([a foo:bar] << b->c == 'd');");
  verifyFormat("x = ([a foo:bar] ? b->c == 'd' : 'e');");
  // FIXME: The following are wrongly classified as C++ lambda expressions.
  // For example this code:
  //   x = ([a foo:bar] & b->c == 'd');
  // is formatted as:
  //   x = ([a foo:bar] & b -> c == 'd');
  // verifyFormat("x = ([a foo:bar] & b->c == 'd');");
  // verifyFormat("x = ([a foo:bar] > b->c == 'd');");
  // verifyFormat("x = ([a foo:bar] < b->c == 'd');");
  // verifyFormat("x = ([a foo:bar] >> b->c == 'd');");
}

TEST_F(FormatTestObjC, DisambiguatesCallsFromStructuredBindings) {
  verifyFormat("int f() {\n"
               "  if (a && [f arg])\n"
               "    return 0;\n"
               "}");
  verifyFormat("int f() {\n"
               "  if (a & [f arg])\n"
               "    return 0;\n"
               "}");
  verifyFormat("int f() {\n"
               "  for (auto &[elem] : list)\n"
               "    return 0;\n"
               "}");
  verifyFormat("int f() {\n"
               "  for (auto &&[elem] : list)\n"
               "    return 0;\n"
               "}");
  verifyFormat(
      "int f() {\n"
      "  for (auto /**/ const /**/ volatile /**/ && /**/ [elem] : list)\n"
      "    return 0;\n"
      "}");
}

TEST_F(FormatTestObjC, BreakLineBeforeNestedBlockParam) {
  Style = getGoogleStyle(FormatStyle::LK_ObjC);
  Style.ObjCBreakBeforeNestedBlockParam = false;
  Style.ColumnLimit = 0;

  verifyFormat("[self.test1 t:self callback:^(typeof(self) self, NSNumber *u, "
               "NSNumber *v) {\n"
               "  u = v;\n"
               "}]");

  verifyFormat("[self.test1 t:self w:self callback:^(typeof(self) self, "
               "NSNumber *u, NSNumber *v) {\n"
               "  u = v;\n"
               "}]");

  verifyFormat("[self.test1 t:self w:self callback:^(typeof(self) self, "
               "NSNumber *u, NSNumber *v) {\n"
               "  u = c;\n"
               "} w:self callback2:^(typeof(self) self, NSNumber *a, NSNumber "
               "*b, NSNumber *c) {\n"
               "  b = c;\n"
               "}]");
  verifyFormat("[self.test1 t:self w:self callback:^(typeof(self) self, "
               "NSNumber *u, NSNumber *v) {\n"
               "  u = v;\n"
               "} z:self]");

  Style.ColumnLimit = 80;
  verifyFormat(
      "[self.test_method a:self b:self\n"
      "           callback:^(typeof(self) self, NSNumber *u, NSNumber *v) {\n"
      "             u = v;\n"
      "           }]");

  verifyFormat("[self block:^(void) {\n"
               "  doStuff();\n"
               "} completionHandler:^(void) {\n"
               "  doStuff();\n"
               "  [self block:^(void) {\n"
               "    doStuff();\n"
               "  } completionHandler:^(void) {\n"
               "    doStuff();\n"
               "  }];\n"
               "}];");

  Style.ColumnLimit = 0;
  verifyFormat("[[SessionService sharedService] "
               "loadWindowWithCompletionBlock:^(SessionWindow *window) {\n"
               "  if (window) {\n"
               "    [self windowDidLoad:window];\n"
               "  } else {\n"
               "    [self errorLoadingWindow];\n"
               "  }\n"
               "}];");
  verifyFormat("[controller test:^{\n"
               "  doStuff();\n"
               "} withTimeout:5 completionHandler:^{\n"
               "  doStuff();\n"
               "}];");
  verifyFormat(
      "[self setupTextFieldSignals:@[\n"
      "  self.documentWidthField,\n"
      "  self.documentHeightField,\n"
      "] solver:^(NSTextField *textField) {\n"
      "  return [self.representedObject solveEquationForTextField:textField];\n"
      "}];");
}

TEST_F(FormatTestObjC, IfNotUnlikely) {
  Style = getGoogleStyle(FormatStyle::LK_ObjC);

  verifyFormat("if (argc < 5) [obj func:arg];");
  verifyFormat("if (argc < 5) [[obj1 method1:arg1] method2:arg2];");
  verifyFormat("if (argc < 5) [[foo bar] baz:i[0]];");
  verifyFormat("if (argc < 5) [[foo bar] baz:i[0]][1];");

  verifyFormat("if (argc < 5)\n"
               "  [obj func:arg];\n"
               "else\n"
               "  [obj func:arg2];");

  verifyFormat("if (argc < 5) [[unlikely]]\n"
               "  [obj func:arg];\n"
               "else [[likely]]\n"
               "  [obj func:arg2];");
}

} // end namespace
} // end namespace format
} // end namespace clang
