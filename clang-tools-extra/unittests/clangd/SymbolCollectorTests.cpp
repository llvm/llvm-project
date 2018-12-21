//===-- SymbolCollectorTests.cpp  -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "TestFS.h"
#include "TestTU.h"
#include "index/SymbolCollector.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Index/IndexingAction.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <memory>
#include <string>

using namespace llvm;
namespace clang {
namespace clangd {
namespace {

using testing::_;
using testing::AllOf;
using testing::Contains;
using testing::Eq;
using testing::Field;
using testing::IsEmpty;
using testing::Not;
using testing::Pair;
using testing::UnorderedElementsAre;
using testing::UnorderedElementsAreArray;

// GMock helpers for matching Symbol.
MATCHER_P(Labeled, Label, "") {
  return (arg.Name + arg.Signature).str() == Label;
}
MATCHER_P(ReturnType, D, "") { return arg.ReturnType == D; }
MATCHER_P(Doc, D, "") { return arg.Documentation == D; }
MATCHER_P(Snippet, S, "") {
  return (arg.Name + arg.CompletionSnippetSuffix).str() == S;
}
MATCHER_P(QName, Name, "") { return (arg.Scope + arg.Name).str() == Name; }
MATCHER_P(DeclURI, P, "") {
  return StringRef(arg.CanonicalDeclaration.FileURI) == P;
}
MATCHER_P(DefURI, P, "") { return StringRef(arg.Definition.FileURI) == P; }
MATCHER_P(IncludeHeader, P, "") {
  return (arg.IncludeHeaders.size() == 1) &&
         (arg.IncludeHeaders.begin()->IncludeHeader == P);
}
MATCHER_P2(IncludeHeaderWithRef, IncludeHeader, References,  "") {
  return (arg.IncludeHeader == IncludeHeader) && (arg.References == References);
}
MATCHER_P(DeclRange, Pos, "") {
  return std::make_tuple(arg.CanonicalDeclaration.Start.line(),
                         arg.CanonicalDeclaration.Start.column(),
                         arg.CanonicalDeclaration.End.line(),
                         arg.CanonicalDeclaration.End.column()) ==
         std::make_tuple(Pos.start.line, Pos.start.character, Pos.end.line,
                         Pos.end.character);
}
MATCHER_P(DefRange, Pos, "") {
  return std::make_tuple(
             arg.Definition.Start.line(), arg.Definition.Start.column(),
             arg.Definition.End.line(), arg.Definition.End.column()) ==
         std::make_tuple(Pos.start.line, Pos.start.character, Pos.end.line,
                         Pos.end.character);
}
MATCHER_P(RefCount, R, "") { return int(arg.References) == R; }
MATCHER_P(ForCodeCompletion, IsIndexedForCodeCompletion, "") {
  return static_cast<bool>(arg.Flags & Symbol::IndexedForCodeCompletion) ==
         IsIndexedForCodeCompletion;
}
MATCHER(Deprecated, "") { return arg.Flags & Symbol::Deprecated; }
MATCHER(ImplementationDetail, "") {
  return arg.Flags & Symbol::ImplementationDetail;
}
MATCHER(RefRange, "") {
  const Ref &Pos = testing::get<0>(arg);
  const Range &Range = testing::get<1>(arg);
  return std::make_tuple(Pos.Location.Start.line(), Pos.Location.Start.column(),
                         Pos.Location.End.line(), Pos.Location.End.column()) ==
         std::make_tuple(Range.start.line, Range.start.character,
                         Range.end.line, Range.end.character);
}
testing::Matcher<const std::vector<Ref> &>
HaveRanges(const std::vector<Range> Ranges) {
  return testing::UnorderedPointwise(RefRange(), Ranges);
}

class ShouldCollectSymbolTest : public ::testing::Test {
public:
  void build(StringRef HeaderCode, StringRef Code = "") {
    File.HeaderFilename = HeaderName;
    File.Filename = FileName;
    File.HeaderCode = HeaderCode;
    File.Code = Code;
    AST = File.build();
  }

  // build() must have been called.
  bool shouldCollect(StringRef Name, bool Qualified = true) {
    assert(AST.hasValue());
    return SymbolCollector::shouldCollectSymbol(
        Qualified ? findDecl(*AST, Name) : findUnqualifiedDecl(*AST, Name),
        AST->getASTContext(), SymbolCollector::Options());
  }

protected:
  std::string HeaderName = "f.h";
  std::string FileName = "f.cpp";
  TestTU File;
  Optional<ParsedAST> AST;  // Initialized after build.
};

TEST_F(ShouldCollectSymbolTest, ShouldCollectSymbol) {
  build(R"(
    namespace nx {
    class X{};
    auto f() { int Local; } // auto ensures function body is parsed.
    struct { int x; } var;
    namespace { class InAnonymous {}; }
    }
  )",
        "class InMain {};");
  auto AST = File.build();
  EXPECT_TRUE(shouldCollect("nx"));
  EXPECT_TRUE(shouldCollect("nx::X"));
  EXPECT_TRUE(shouldCollect("nx::f"));

  EXPECT_FALSE(shouldCollect("InMain"));
  EXPECT_FALSE(shouldCollect("Local", /*Qualified=*/false));
  EXPECT_FALSE(shouldCollect("InAnonymous", /*Qualified=*/false));
}

TEST_F(ShouldCollectSymbolTest, NoPrivateProtoSymbol) {
  HeaderName = "f.proto.h";
  build(
      R"(// Generated by the protocol buffer compiler.  DO NOT EDIT!
         namespace nx {
           class Top_Level {};
           class TopLevel {};
           enum Kind {
             KIND_OK,
             Kind_Not_Ok,
           };
         })");
  EXPECT_TRUE(shouldCollect("nx::TopLevel"));
  EXPECT_TRUE(shouldCollect("nx::Kind::KIND_OK"));
  EXPECT_TRUE(shouldCollect("nx::Kind"));

  EXPECT_FALSE(shouldCollect("nx::Top_Level"));
  EXPECT_FALSE(shouldCollect("nx::Kind::Kind_Not_Ok"));
}

TEST_F(ShouldCollectSymbolTest, DoubleCheckProtoHeaderComment) {
  HeaderName = "f.proto.h";
  build(R"(
    namespace nx {
      class Top_Level {};
      enum Kind {
        Kind_Fine
      };
    }
  )");
  EXPECT_TRUE(shouldCollect("nx::Top_Level"));
  EXPECT_TRUE(shouldCollect("nx::Kind_Fine"));
}

class SymbolIndexActionFactory : public tooling::FrontendActionFactory {
public:
  SymbolIndexActionFactory(SymbolCollector::Options COpts,
                           CommentHandler *PragmaHandler)
      : COpts(std::move(COpts)), PragmaHandler(PragmaHandler) {}

  clang::FrontendAction *create() override {
    class WrappedIndexAction : public WrapperFrontendAction {
    public:
      WrappedIndexAction(std::shared_ptr<SymbolCollector> C,
                         const index::IndexingOptions &Opts,
                         CommentHandler *PragmaHandler)
          : WrapperFrontendAction(
                index::createIndexingAction(C, Opts, nullptr)),
            PragmaHandler(PragmaHandler) {}

      std::unique_ptr<ASTConsumer>
      CreateASTConsumer(CompilerInstance &CI, StringRef InFile) override {
        if (PragmaHandler)
          CI.getPreprocessor().addCommentHandler(PragmaHandler);
        return WrapperFrontendAction::CreateASTConsumer(CI, InFile);
      }

    private:
      index::IndexingOptions IndexOpts;
      CommentHandler *PragmaHandler;
    };
    index::IndexingOptions IndexOpts;
    IndexOpts.SystemSymbolFilter =
        index::IndexingOptions::SystemSymbolFilterKind::All;
    IndexOpts.IndexFunctionLocals = false;
    Collector = std::make_shared<SymbolCollector>(COpts);
    return new WrappedIndexAction(Collector, std::move(IndexOpts),
                                  PragmaHandler);
  }

  std::shared_ptr<SymbolCollector> Collector;
  SymbolCollector::Options COpts;
  CommentHandler *PragmaHandler;
};

class SymbolCollectorTest : public ::testing::Test {
public:
  SymbolCollectorTest()
      : InMemoryFileSystem(new vfs::InMemoryFileSystem),
        TestHeaderName(testPath("symbol.h")),
        TestFileName(testPath("symbol.cc")) {
    TestHeaderURI = URI::create(TestHeaderName).toString();
    TestFileURI = URI::create(TestFileName).toString();
  }

  bool runSymbolCollector(StringRef HeaderCode, StringRef MainCode,
                          const std::vector<std::string> &ExtraArgs = {}) {
    IntrusiveRefCntPtr<FileManager> Files(
        new FileManager(FileSystemOptions(), InMemoryFileSystem));

    auto Factory = llvm::make_unique<SymbolIndexActionFactory>(
        CollectorOpts, PragmaHandler.get());

    std::vector<std::string> Args = {
        "symbol_collector", "-fsyntax-only", "-xc++",
        "-std=c++11",       "-include",      TestHeaderName};
    Args.insert(Args.end(), ExtraArgs.begin(), ExtraArgs.end());
    // This allows to override the "-xc++" with something else, i.e.
    // -xobjective-c++.
    Args.push_back(TestFileName);

    tooling::ToolInvocation Invocation(
        Args,
        Factory->create(), Files.get(),
        std::make_shared<PCHContainerOperations>());

    InMemoryFileSystem->addFile(TestHeaderName, 0,
                                MemoryBuffer::getMemBuffer(HeaderCode));
    InMemoryFileSystem->addFile(TestFileName, 0,
                                MemoryBuffer::getMemBuffer(MainCode));
    Invocation.run();
    Symbols = Factory->Collector->takeSymbols();
    Refs = Factory->Collector->takeRefs();
    return true;
  }

protected:
  IntrusiveRefCntPtr<vfs::InMemoryFileSystem> InMemoryFileSystem;
  std::string TestHeaderName;
  std::string TestHeaderURI;
  std::string TestFileName;
  std::string TestFileURI;
  SymbolSlab Symbols;
  RefSlab Refs;
  SymbolCollector::Options CollectorOpts;
  std::unique_ptr<CommentHandler> PragmaHandler;
};

TEST_F(SymbolCollectorTest, CollectSymbols) {
  const std::string Header = R"(
    class Foo {
      Foo() {}
      Foo(int a) {}
      void f();
      friend void f1();
      friend class Friend;
      Foo& operator=(const Foo&);
      ~Foo();
      class Nested {
      void f();
      };
    };
    class Friend {
    };

    void f1();
    inline void f2() {}
    static const int KInt = 2;
    const char* kStr = "123";

    namespace {
    void ff() {} // ignore
    }

    void f1() {}

    namespace foo {
    // Type alias
    typedef int int32;
    using int32_t = int32;

    // Variable
    int v1;

    // Namespace
    namespace bar {
    int v2;
    }
    // Namespace alias
    namespace baz = bar;

    // FIXME: using declaration is not supported as the IndexAction will ignore
    // implicit declarations (the implicit using shadow declaration) by default,
    // and there is no way to customize this behavior at the moment.
    using bar::v2;
    } // namespace foo
  )";
  runSymbolCollector(Header, /*Main=*/"");
  EXPECT_THAT(Symbols,
              UnorderedElementsAreArray(
                  {AllOf(QName("Foo"), ForCodeCompletion(true)),
                   AllOf(QName("Foo::Foo"), ForCodeCompletion(false)),
                   AllOf(QName("Foo::Foo"), ForCodeCompletion(false)),
                   AllOf(QName("Foo::f"), ForCodeCompletion(false)),
                   AllOf(QName("Foo::~Foo"), ForCodeCompletion(false)),
                   AllOf(QName("Foo::operator="), ForCodeCompletion(false)),
                   AllOf(QName("Foo::Nested"), ForCodeCompletion(false)),
                   AllOf(QName("Foo::Nested::f"), ForCodeCompletion(false)),

                   AllOf(QName("Friend"), ForCodeCompletion(true)),
                   AllOf(QName("f1"), ForCodeCompletion(true)),
                   AllOf(QName("f2"), ForCodeCompletion(true)),
                   AllOf(QName("KInt"), ForCodeCompletion(true)),
                   AllOf(QName("kStr"), ForCodeCompletion(true)),
                   AllOf(QName("foo"), ForCodeCompletion(true)),
                   AllOf(QName("foo::bar"), ForCodeCompletion(true)),
                   AllOf(QName("foo::int32"), ForCodeCompletion(true)),
                   AllOf(QName("foo::int32_t"), ForCodeCompletion(true)),
                   AllOf(QName("foo::v1"), ForCodeCompletion(true)),
                   AllOf(QName("foo::bar::v2"), ForCodeCompletion(true)),
                   AllOf(QName("foo::baz"), ForCodeCompletion(true))}));
}

TEST_F(SymbolCollectorTest, Template) {
  Annotations Header(R"(
    // Template is indexed, specialization and instantiation is not.
    template <class T> struct [[Tmpl]] {T $xdecl[[x]] = 0;};
    template <> struct Tmpl<int> {};
    extern template struct Tmpl<float>;
    template struct Tmpl<double>;
  )");
  runSymbolCollector(Header.code(), /*Main=*/"");
  EXPECT_THAT(Symbols,
              UnorderedElementsAreArray(
                  {AllOf(QName("Tmpl"), DeclRange(Header.range())),
                   AllOf(QName("Tmpl::x"), DeclRange(Header.range("xdecl")))}));
}

TEST_F(SymbolCollectorTest, ObjCSymbols) {
  const std::string Header = R"(
    @interface Person
    - (void)someMethodName:(void*)name1 lastName:(void*)lName;
    @end

    @implementation Person
    - (void)someMethodName:(void*)name1 lastName:(void*)lName{
      int foo;
      ^(int param){ int bar; };
    }
    @end

    @interface Person (MyCategory)
    - (void)someMethodName2:(void*)name2;
    @end

    @implementation Person (MyCategory)
    - (void)someMethodName2:(void*)name2 {
      int foo2;
    }
    @end

    @protocol MyProtocol
    - (void)someMethodName3:(void*)name3;
    @end
  )";
  TestFileName = testPath("test.m");
  runSymbolCollector(Header, /*Main=*/"", {"-fblocks", "-xobjective-c++"});
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(
                  QName("Person"), QName("Person::someMethodName:lastName:"),
                  QName("MyCategory"), QName("Person::someMethodName2:"),
                  QName("MyProtocol"), QName("MyProtocol::someMethodName3:")));
}

TEST_F(SymbolCollectorTest, Locations) {
  Annotations Header(R"cpp(
    // Declared in header, defined in main.
    extern int $xdecl[[X]];
    class $clsdecl[[Cls]];
    void $printdecl[[print]]();

    // Declared in header, defined nowhere.
    extern int $zdecl[[Z]];

    void $foodecl[[fo\
o]]();
  )cpp");
  Annotations Main(R"cpp(
    int $xdef[[X]] = 42;
    class $clsdef[[Cls]] {};
    void $printdef[[print]]() {}

    // Declared/defined in main only.
    int Y;
  )cpp");
  runSymbolCollector(Header.code(), Main.code());
  EXPECT_THAT(
      Symbols,
      UnorderedElementsAre(
          AllOf(QName("X"), DeclRange(Header.range("xdecl")),
                DefRange(Main.range("xdef"))),
          AllOf(QName("Cls"), DeclRange(Header.range("clsdecl")),
                DefRange(Main.range("clsdef"))),
          AllOf(QName("print"), DeclRange(Header.range("printdecl")),
                DefRange(Main.range("printdef"))),
          AllOf(QName("Z"), DeclRange(Header.range("zdecl"))),
          AllOf(QName("foo"), DeclRange(Header.range("foodecl")))
          ));
}

TEST_F(SymbolCollectorTest, Refs) {
  Annotations Header(R"(
  class $foo[[Foo]] {
  public:
    $foo[[Foo]]() {}
    $foo[[Foo]](int);
  };
  class $bar[[Bar]];
  void $func[[func]]();

  namespace $ns[[NS]] {} // namespace ref is ignored
  )");
  Annotations Main(R"(
  class $bar[[Bar]] {};

  void $func[[func]]();

  void fff() {
    $foo[[Foo]] foo;
    $bar[[Bar]] bar;
    $func[[func]]();
    int abc = 0;
    $foo[[Foo]] foo2 = abc;
  }
  )");
  Annotations SymbolsOnlyInMainCode(R"(
  int a;
  void b() {}
  static const int c = 0;
  class d {};
  )");
  CollectorOpts.RefFilter = RefKind::All;
  runSymbolCollector(Header.code(),
                     (Main.code() + SymbolsOnlyInMainCode.code()).str());
  auto HeaderSymbols = TestTU::withHeaderCode(Header.code()).headerSymbols();

  EXPECT_THAT(Refs, Contains(Pair(findSymbol(Symbols, "Foo").ID,
                                  HaveRanges(Main.ranges("foo")))));
  EXPECT_THAT(Refs, Contains(Pair(findSymbol(Symbols, "Bar").ID,
                                  HaveRanges(Main.ranges("bar")))));
  EXPECT_THAT(Refs, Contains(Pair(findSymbol(Symbols, "func").ID,
                                  HaveRanges(Main.ranges("func")))));
  EXPECT_THAT(Refs, Not(Contains(Pair(findSymbol(Symbols, "NS").ID, _))));
  // Symbols *only* in the main file (a, b, c) had no refs collected.
  auto MainSymbols =
      TestTU::withHeaderCode(SymbolsOnlyInMainCode.code()).headerSymbols();
  EXPECT_THAT(Refs, Not(Contains(Pair(findSymbol(MainSymbols, "a").ID, _))));
  EXPECT_THAT(Refs, Not(Contains(Pair(findSymbol(MainSymbols, "b").ID, _))));
  EXPECT_THAT(Refs, Not(Contains(Pair(findSymbol(MainSymbols, "c").ID, _))));
}

TEST_F(SymbolCollectorTest, RefsInHeaders) {
  CollectorOpts.RefFilter = RefKind::All;
  CollectorOpts.RefsInHeaders = true;
  Annotations Header(R"(
  class [[Foo]] {};
  )");
  runSymbolCollector(Header.code(), "");
  EXPECT_THAT(Refs, Contains(Pair(findSymbol(Symbols, "Foo").ID,
                                  HaveRanges(Header.ranges()))));
}

TEST_F(SymbolCollectorTest, References) {
  const std::string Header = R"(
    class W;
    class X {};
    class Y;
    class Z {}; // not used anywhere
    Y* y = nullptr;  // used in header doesn't count
    #define GLOBAL_Z(name) Z name;
  )";
  const std::string Main = R"(
    W* w = nullptr;
    W* w2 = nullptr; // only one usage counts
    X x();
    class V;
    V* v = nullptr; // Used, but not eligible for indexing.
    class Y{}; // definition doesn't count as a reference
    GLOBAL_Z(z); // Not a reference to Z, we don't spell the type.
  )";
  CollectorOpts.CountReferences = true;
  runSymbolCollector(Header, Main);
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(AllOf(QName("W"), RefCount(1)),
                                   AllOf(QName("X"), RefCount(1)),
                                   AllOf(QName("Y"), RefCount(0)),
                                   AllOf(QName("Z"), RefCount(0)), QName("y")));
}

TEST_F(SymbolCollectorTest, SymbolRelativeNoFallback) {
  runSymbolCollector("class Foo {};", /*Main=*/"");
  EXPECT_THAT(Symbols, UnorderedElementsAre(
                           AllOf(QName("Foo"), DeclURI(TestHeaderURI))));
}

TEST_F(SymbolCollectorTest, SymbolRelativeWithFallback) {
  TestHeaderName = "x.h";
  TestFileName = "x.cpp";
  TestHeaderURI = URI::create(testPath(TestHeaderName)).toString();
  CollectorOpts.FallbackDir = testRoot();
  runSymbolCollector("class Foo {};", /*Main=*/"");
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(AllOf(QName("Foo"), DeclURI(TestHeaderURI))));
}

TEST_F(SymbolCollectorTest, UnittestURIScheme) {
  // Use test URI scheme from URITests.cpp
  TestHeaderName = testPath("x.h");
  TestFileName = testPath("x.cpp");
  runSymbolCollector("class Foo {};", /*Main=*/"");
  EXPECT_THAT(Symbols, UnorderedElementsAre(
                           AllOf(QName("Foo"), DeclURI("unittest:///x.h"))));
}

TEST_F(SymbolCollectorTest, IncludeEnums) {
  const std::string Header = R"(
    enum {
      Red
    };
    enum Color {
      Green
    };
    enum class Color2 {
      Yellow
    };
    namespace ns {
    enum {
      Black
    };
    }
  )";
  runSymbolCollector(Header, /*Main=*/"");
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(
                  AllOf(QName("Red"), ForCodeCompletion(true)),
                  AllOf(QName("Color"), ForCodeCompletion(true)),
                  AllOf(QName("Green"), ForCodeCompletion(true)),
                  AllOf(QName("Color2"), ForCodeCompletion(true)),
                  AllOf(QName("Color2::Yellow"), ForCodeCompletion(false)),
                  AllOf(QName("ns"), ForCodeCompletion(true)),
                  AllOf(QName("ns::Black"), ForCodeCompletion(true))));
}

TEST_F(SymbolCollectorTest, NamelessSymbols) {
  const std::string Header = R"(
    struct {
      int a;
    } Foo;
  )";
  runSymbolCollector(Header, /*Main=*/"");
  EXPECT_THAT(Symbols, UnorderedElementsAre(QName("Foo"),
                                            QName("(anonymous struct)::a")));
}

TEST_F(SymbolCollectorTest, SymbolFormedFromRegisteredSchemeFromMacro) {

  Annotations Header(R"(
    #define FF(name) \
      class name##_Test {};

    $expansion[[FF]](abc);

    #define FF2() \
      class $spelling[[Test]] {};

    FF2();
  )");

  runSymbolCollector(Header.code(), /*Main=*/"");
  EXPECT_THAT(
      Symbols,
      UnorderedElementsAre(
          AllOf(QName("abc_Test"), DeclRange(Header.range("expansion")),
                DeclURI(TestHeaderURI)),
          AllOf(QName("Test"), DeclRange(Header.range("spelling")),
                DeclURI(TestHeaderURI))));
}

TEST_F(SymbolCollectorTest, SymbolFormedByCLI) {
  Annotations Header(R"(
    #ifdef NAME
    class $expansion[[NAME]] {};
    #endif
  )");
  runSymbolCollector(Header.code(), /*Main=*/"", /*ExtraArgs=*/{"-DNAME=name"});
  EXPECT_THAT(Symbols, UnorderedElementsAre(AllOf(
                           QName("name"), DeclRange(Header.range("expansion")),
                           DeclURI(TestHeaderURI))));
}

TEST_F(SymbolCollectorTest, IgnoreSymbolsInMainFile) {
  const std::string Header = R"(
    class Foo {};
    void f1();
    inline void f2() {}
  )";
  const std::string Main = R"(
    namespace {
    void ff() {} // ignore
    }
    void main_f() {} // ignore
    void f1() {}
  )";
  runSymbolCollector(Header, Main);
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(QName("Foo"), QName("f1"), QName("f2")));
}

TEST_F(SymbolCollectorTest, ClassMembers) {
  const std::string Header = R"(
    class Foo {
      void f() {}
      void g();
      static void sf() {}
      static void ssf();
      static int x;
    };
  )";
  const std::string Main = R"(
    void Foo::g() {}
    void Foo::ssf() {}
  )";
  runSymbolCollector(Header, Main);
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(QName("Foo"), QName("Foo::f"),
                                   QName("Foo::g"), QName("Foo::sf"),
                                   QName("Foo::ssf"), QName("Foo::x")));
}

TEST_F(SymbolCollectorTest, Scopes) {
  const std::string Header = R"(
    namespace na {
    class Foo {};
    namespace nb {
    class Bar {};
    }
    }
  )";
  runSymbolCollector(Header, /*Main=*/"");
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(QName("na"), QName("na::nb"),
                                   QName("na::Foo"), QName("na::nb::Bar")));
}

TEST_F(SymbolCollectorTest, ExternC) {
  const std::string Header = R"(
    extern "C" { class Foo {}; }
    namespace na {
    extern "C" { class Bar {}; }
    }
  )";
  runSymbolCollector(Header, /*Main=*/"");
  EXPECT_THAT(Symbols, UnorderedElementsAre(QName("na"), QName("Foo"),
                                            QName("na::Bar")));
}

TEST_F(SymbolCollectorTest, SkipInlineNamespace) {
  const std::string Header = R"(
    namespace na {
    inline namespace nb {
    class Foo {};
    }
    }
    namespace na {
    // This is still inlined.
    namespace nb {
    class Bar {};
    }
    }
  )";
  runSymbolCollector(Header, /*Main=*/"");
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(QName("na"), QName("na::nb"),
                                   QName("na::Foo"), QName("na::Bar")));
}

TEST_F(SymbolCollectorTest, SymbolWithDocumentation) {
  const std::string Header = R"(
    namespace nx {
    /// Foo comment.
    int ff(int x, double y) { return 0; }
    }
  )";
  runSymbolCollector(Header, /*Main=*/"");
  EXPECT_THAT(
      Symbols,
      UnorderedElementsAre(
          QName("nx"), AllOf(QName("nx::ff"), Labeled("ff(int x, double y)"),
                             ReturnType("int"), Doc("Foo comment."))));
}

TEST_F(SymbolCollectorTest, Snippet) {
  const std::string Header = R"(
    namespace nx {
    void f() {}
    int ff(int x, double y) { return 0; }
    }
  )";
  runSymbolCollector(Header, /*Main=*/"");
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(
                  QName("nx"),
                  AllOf(QName("nx::f"), Labeled("f()"), Snippet("f()")),
                  AllOf(QName("nx::ff"), Labeled("ff(int x, double y)"),
                        Snippet("ff(${1:int x}, ${2:double y})"))));
}

TEST_F(SymbolCollectorTest, IncludeHeaderSameAsFileURI) {
  CollectorOpts.CollectIncludePath = true;
  runSymbolCollector("class Foo {};", /*Main=*/"");
  EXPECT_THAT(Symbols, UnorderedElementsAre(
                           AllOf(QName("Foo"), DeclURI(TestHeaderURI))));
  EXPECT_THAT(Symbols.begin()->IncludeHeaders,
              UnorderedElementsAre(IncludeHeaderWithRef(TestHeaderURI, 1u)));
}

#ifndef _WIN32
TEST_F(SymbolCollectorTest, CanonicalSTLHeader) {
  CollectorOpts.CollectIncludePath = true;
  CanonicalIncludes Includes;
  addSystemHeadersMapping(&Includes);
  CollectorOpts.Includes = &Includes;
  // bits/basic_string.h$ should be mapped to <string>
  TestHeaderName = "/nasty/bits/basic_string.h";
  TestFileName = "/nasty/bits/basic_string.cpp";
  TestHeaderURI = URI::create(TestHeaderName).toString();
  runSymbolCollector("class string {};", /*Main=*/"");
  EXPECT_THAT(Symbols, UnorderedElementsAre(AllOf(QName("string"),
                                                  DeclURI(TestHeaderURI),
                                                  IncludeHeader("<string>"))));
}
#endif

TEST_F(SymbolCollectorTest, STLiosfwd) {
  CollectorOpts.CollectIncludePath = true;
  CanonicalIncludes Includes;
  addSystemHeadersMapping(&Includes);
  CollectorOpts.Includes = &Includes;
  // Symbols from <iosfwd> should be mapped individually.
  TestHeaderName = testPath("iosfwd");
  TestFileName = testPath("iosfwd.cpp");
  std::string Header = R"(
    namespace std {
      class no_map {};
      class ios {};
      class ostream {};
      class filebuf {};
    } // namespace std
  )";
  runSymbolCollector(Header, /*Main=*/"");
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(
                  QName("std"),
                  AllOf(QName("std::no_map"), IncludeHeader("<iosfwd>")),
                  AllOf(QName("std::ios"), IncludeHeader("<ios>")),
                  AllOf(QName("std::ostream"), IncludeHeader("<ostream>")),
                  AllOf(QName("std::filebuf"), IncludeHeader("<fstream>"))));
}

TEST_F(SymbolCollectorTest, IWYUPragma) {
  CollectorOpts.CollectIncludePath = true;
  CanonicalIncludes Includes;
  PragmaHandler = collectIWYUHeaderMaps(&Includes);
  CollectorOpts.Includes = &Includes;
  const std::string Header = R"(
    // IWYU pragma: private, include the/good/header.h
    class Foo {};
  )";
  runSymbolCollector(Header, /*Main=*/"");
  EXPECT_THAT(Symbols, UnorderedElementsAre(
                           AllOf(QName("Foo"), DeclURI(TestHeaderURI),
                                 IncludeHeader("\"the/good/header.h\""))));
}

TEST_F(SymbolCollectorTest, IWYUPragmaWithDoubleQuotes) {
  CollectorOpts.CollectIncludePath = true;
  CanonicalIncludes Includes;
  PragmaHandler = collectIWYUHeaderMaps(&Includes);
  CollectorOpts.Includes = &Includes;
  const std::string Header = R"(
    // IWYU pragma: private, include "the/good/header.h"
    class Foo {};
  )";
  runSymbolCollector(Header, /*Main=*/"");
  EXPECT_THAT(Symbols, UnorderedElementsAre(
                           AllOf(QName("Foo"), DeclURI(TestHeaderURI),
                                 IncludeHeader("\"the/good/header.h\""))));
}

TEST_F(SymbolCollectorTest, SkipIncFileWhenCanonicalizeHeaders) {
  CollectorOpts.CollectIncludePath = true;
  CanonicalIncludes Includes;
  Includes.addMapping(TestHeaderName, "<canonical>");
  CollectorOpts.Includes = &Includes;
  auto IncFile = testPath("test.inc");
  auto IncURI = URI::create(IncFile).toString();
  InMemoryFileSystem->addFile(IncFile, 0,
                              MemoryBuffer::getMemBuffer("class X {};"));
  runSymbolCollector("#include \"test.inc\"\nclass Y {};", /*Main=*/"",
                     /*ExtraArgs=*/{"-I", testRoot()});
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(AllOf(QName("X"), DeclURI(IncURI),
                                         IncludeHeader("<canonical>")),
                                   AllOf(QName("Y"), DeclURI(TestHeaderURI),
                                         IncludeHeader("<canonical>"))));
}

TEST_F(SymbolCollectorTest, MainFileIsHeaderWhenSkipIncFile) {
  CollectorOpts.CollectIncludePath = true;
  CanonicalIncludes Includes;
  CollectorOpts.Includes = &Includes;
  TestFileName = testPath("main.h");
  TestFileURI = URI::create(TestFileName).toString();
  auto IncFile = testPath("test.inc");
  auto IncURI = URI::create(IncFile).toString();
  InMemoryFileSystem->addFile(IncFile, 0,
                              MemoryBuffer::getMemBuffer("class X {};"));
  runSymbolCollector("", /*Main=*/"#include \"test.inc\"",
                     /*ExtraArgs=*/{"-I", testRoot()});
  EXPECT_THAT(Symbols, UnorderedElementsAre(AllOf(QName("X"), DeclURI(IncURI),
                                                  IncludeHeader(TestFileURI))));
}

TEST_F(SymbolCollectorTest, MainFileIsHeaderWithoutExtensionWhenSkipIncFile) {
  CollectorOpts.CollectIncludePath = true;
  CanonicalIncludes Includes;
  CollectorOpts.Includes = &Includes;
  TestFileName = testPath("no_ext_main");
  TestFileURI = URI::create(TestFileName).toString();
  auto IncFile = testPath("test.inc");
  auto IncURI = URI::create(IncFile).toString();
  InMemoryFileSystem->addFile(IncFile, 0,
                              MemoryBuffer::getMemBuffer("class X {};"));
  runSymbolCollector("", /*Main=*/"#include \"test.inc\"",
                     /*ExtraArgs=*/{"-I", testRoot()});
  EXPECT_THAT(Symbols, UnorderedElementsAre(AllOf(QName("X"), DeclURI(IncURI),
                                                  IncludeHeader(TestFileURI))));
}

TEST_F(SymbolCollectorTest, FallbackToIncFileWhenIncludingFileIsCC) {
  CollectorOpts.CollectIncludePath = true;
  CanonicalIncludes Includes;
  CollectorOpts.Includes = &Includes;
  auto IncFile = testPath("test.inc");
  auto IncURI = URI::create(IncFile).toString();
  InMemoryFileSystem->addFile(IncFile, 0,
                              MemoryBuffer::getMemBuffer("class X {};"));
  runSymbolCollector("", /*Main=*/"#include \"test.inc\"",
                     /*ExtraArgs=*/{"-I", testRoot()});
  EXPECT_THAT(Symbols, UnorderedElementsAre(AllOf(QName("X"), DeclURI(IncURI),
                                                  IncludeHeader(IncURI))));
}

TEST_F(SymbolCollectorTest, AvoidUsingFwdDeclsAsCanonicalDecls) {
  CollectorOpts.CollectIncludePath = true;
  Annotations Header(R"(
    // Forward declarations of TagDecls.
    class C;
    struct S;
    union U;

    // Canonical declarations.
    class $cdecl[[C]] {};
    struct $sdecl[[S]] {};
    union $udecl[[U]] {int $xdecl[[x]]; bool $ydecl[[y]];};
  )");
  runSymbolCollector(Header.code(), /*Main=*/"");
  EXPECT_THAT(
      Symbols,
      UnorderedElementsAre(
          AllOf(QName("C"), DeclURI(TestHeaderURI),
                DeclRange(Header.range("cdecl")), IncludeHeader(TestHeaderURI),
                DefURI(TestHeaderURI), DefRange(Header.range("cdecl"))),
          AllOf(QName("S"), DeclURI(TestHeaderURI),
                DeclRange(Header.range("sdecl")), IncludeHeader(TestHeaderURI),
                DefURI(TestHeaderURI), DefRange(Header.range("sdecl"))),
          AllOf(QName("U"), DeclURI(TestHeaderURI),
                DeclRange(Header.range("udecl")), IncludeHeader(TestHeaderURI),
                DefURI(TestHeaderURI), DefRange(Header.range("udecl"))),
          AllOf(QName("U::x"), DeclURI(TestHeaderURI),
                DeclRange(Header.range("xdecl")), DefURI(TestHeaderURI),
                DefRange(Header.range("xdecl"))),
          AllOf(QName("U::y"), DeclURI(TestHeaderURI),
                DeclRange(Header.range("ydecl")), DefURI(TestHeaderURI),
                DefRange(Header.range("ydecl")))));
}

TEST_F(SymbolCollectorTest, ClassForwardDeclarationIsCanonical) {
  CollectorOpts.CollectIncludePath = true;
  runSymbolCollector(/*Header=*/"class X;", /*Main=*/"class X {};");
  EXPECT_THAT(Symbols, UnorderedElementsAre(AllOf(
                           QName("X"), DeclURI(TestHeaderURI),
                           IncludeHeader(TestHeaderURI), DefURI(TestFileURI))));
}

TEST_F(SymbolCollectorTest, UTF16Character) {
  // ö is 2-bytes.
  Annotations Header(/*Header=*/"class [[pörk]] {};");
  runSymbolCollector(Header.code(), /*Main=*/"");
  EXPECT_THAT(Symbols, UnorderedElementsAre(
                           AllOf(QName("pörk"), DeclRange(Header.range()))));
}

TEST_F(SymbolCollectorTest, DoNotIndexSymbolsInFriendDecl) {
  Annotations Header(R"(
    namespace nx {
      class $z[[Z]] {};
      class X {
        friend class Y;
        friend class Z;
        friend void foo();
        friend void $bar[[bar]]() {}
      };
      class $y[[Y]] {};
      void $foo[[foo]]();
    }
  )");
  runSymbolCollector(Header.code(), /*Main=*/"");

  EXPECT_THAT(Symbols,
              UnorderedElementsAre(
                  QName("nx"), QName("nx::X"),
                  AllOf(QName("nx::Y"), DeclRange(Header.range("y"))),
                  AllOf(QName("nx::Z"), DeclRange(Header.range("z"))),
                  AllOf(QName("nx::foo"), DeclRange(Header.range("foo"))),
                  AllOf(QName("nx::bar"), DeclRange(Header.range("bar")))));
}

TEST_F(SymbolCollectorTest, ReferencesInFriendDecl) {
  const std::string Header = R"(
    class X;
    class Y;
  )";
  const std::string Main = R"(
    class C {
      friend ::X;
      friend class Y;
    };
  )";
  CollectorOpts.CountReferences = true;
  runSymbolCollector(Header, Main);
  EXPECT_THAT(Symbols, UnorderedElementsAre(AllOf(QName("X"), RefCount(1)),
                                            AllOf(QName("Y"), RefCount(1))));
}

TEST_F(SymbolCollectorTest, Origin) {
  CollectorOpts.Origin = SymbolOrigin::Static;
  runSymbolCollector("class Foo {};", /*Main=*/"");
  EXPECT_THAT(Symbols, UnorderedElementsAre(
                           Field(&Symbol::Origin, SymbolOrigin::Static)));
}

TEST_F(SymbolCollectorTest, CollectMacros) {
  CollectorOpts.CollectIncludePath = true;
  Annotations Header(R"(
    #define X 1
    #define $mac[[MAC]](x) int x
    #define $used[[USED]](y) float y;

    MAC(p);
  )");
  const std::string Main = R"(
    #define MAIN 1  // not indexed
    USED(t);
  )";
  CollectorOpts.CountReferences = true;
  CollectorOpts.CollectMacro = true;
  runSymbolCollector(Header.code(), Main);
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(QName("p"),
                                   AllOf(QName("X"), DeclURI(TestHeaderURI),
                                         IncludeHeader(TestHeaderURI)),
                                   AllOf(Labeled("MAC(x)"), RefCount(0),
                                         DeclRange(Header.range("mac"))),
                                   AllOf(Labeled("USED(y)"), RefCount(1),
                                         DeclRange(Header.range("used")))));
}

TEST_F(SymbolCollectorTest, DeprecatedSymbols) {
  const std::string Header = R"(
    void TestClangc() __attribute__((deprecated("", "")));
    void TestClangd();
  )";
  runSymbolCollector(Header, /**/ "");
  EXPECT_THAT(Symbols, UnorderedElementsAre(
                           AllOf(QName("TestClangc"), Deprecated()),
                           AllOf(QName("TestClangd"), Not(Deprecated()))));
}

TEST_F(SymbolCollectorTest, ImplementationDetail) {
  const std::string Header = R"(
    #define DECL_NAME(x, y) x##_##y##_Decl
    #define DECL(x, y) class DECL_NAME(x, y) {};
    DECL(X, Y); // X_Y_Decl

    class Public {};
  )";
  runSymbolCollector(Header, /**/ "");
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(
                  AllOf(QName("X_Y_Decl"), ImplementationDetail()),
                  AllOf(QName("Public"), Not(ImplementationDetail()))));
}

} // namespace
} // namespace clangd
} // namespace clang
