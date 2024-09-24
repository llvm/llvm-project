//===-- SymbolCollectorTests.cpp  -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "TestFS.h"
#include "TestTU.h"
#include "URI.h"
#include "clang-include-cleaner/Record.h"
#include "index/SymbolCollector.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Index/IndexingAction.h"
#include "clang/Index/IndexingOptions.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "gmock/gmock-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace clang {
namespace clangd {
namespace {

using ::testing::_;
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;
using ::testing::UnorderedElementsAreArray;

// GMock helpers for matching Symbol.
MATCHER_P(labeled, Label, "") {
  return (arg.Name + arg.Signature).str() == Label;
}
MATCHER_P(returnType, D, "") { return arg.ReturnType == D; }
MATCHER_P(doc, D, "") { return arg.Documentation == D; }
MATCHER_P(snippet, S, "") {
  return (arg.Name + arg.CompletionSnippetSuffix).str() == S;
}
MATCHER_P(qName, Name, "") { return (arg.Scope + arg.Name).str() == Name; }
MATCHER_P(hasName, Name, "") { return arg.Name == Name; }
MATCHER_P(templateArgs, TemplArgs, "") {
  return arg.TemplateSpecializationArgs == TemplArgs;
}
MATCHER_P(hasKind, Kind, "") { return arg.SymInfo.Kind == Kind; }
MATCHER_P(declURI, P, "") {
  return StringRef(arg.CanonicalDeclaration.FileURI) == P;
}
MATCHER_P(defURI, P, "") { return StringRef(arg.Definition.FileURI) == P; }
MATCHER(includeHeader, "") { return !arg.IncludeHeaders.empty(); }
MATCHER_P(includeHeader, P, "") {
  return (arg.IncludeHeaders.size() == 1) &&
         (arg.IncludeHeaders.begin()->IncludeHeader == P);
}
MATCHER_P2(IncludeHeaderWithRef, includeHeader, References, "") {
  return (arg.IncludeHeader == includeHeader) && (arg.References == References);
}
bool rangesMatch(const SymbolLocation &Loc, const Range &R) {
  return std::make_tuple(Loc.Start.line(), Loc.Start.column(), Loc.End.line(),
                         Loc.End.column()) ==
         std::make_tuple(R.start.line, R.start.character, R.end.line,
                         R.end.character);
}
MATCHER_P(declRange, Pos, "") {
  return rangesMatch(arg.CanonicalDeclaration, Pos);
}
MATCHER_P(defRange, Pos, "") { return rangesMatch(arg.Definition, Pos); }
MATCHER_P(refCount, R, "") { return int(arg.References) == R; }
MATCHER_P(forCodeCompletion, IsIndexedForCodeCompletion, "") {
  return static_cast<bool>(arg.Flags & Symbol::IndexedForCodeCompletion) ==
         IsIndexedForCodeCompletion;
}
MATCHER(deprecated, "") { return arg.Flags & Symbol::Deprecated; }
MATCHER(implementationDetail, "") {
  return arg.Flags & Symbol::ImplementationDetail;
}
MATCHER(visibleOutsideFile, "") {
  return static_cast<bool>(arg.Flags & Symbol::VisibleOutsideFile);
}
MATCHER(refRange, "") {
  const Ref &Pos = ::testing::get<0>(arg);
  const Range &Range = ::testing::get<1>(arg);
  return rangesMatch(Pos.Location, Range);
}
MATCHER_P2(OverriddenBy, Subject, Object, "") {
  return arg == Relation{Subject.ID, RelationKind::OverriddenBy, Object.ID};
}
MATCHER(isSpelled, "") {
  return static_cast<bool>(arg.Kind & RefKind::Spelled);
}
::testing::Matcher<const std::vector<Ref> &>
haveRanges(const std::vector<Range> Ranges) {
  return ::testing::UnorderedPointwise(refRange(), Ranges);
}

class ShouldCollectSymbolTest : public ::testing::Test {
public:
  void build(llvm::StringRef HeaderCode, llvm::StringRef Code = "") {
    File.HeaderFilename = HeaderName;
    File.Filename = FileName;
    File.HeaderCode = std::string(HeaderCode);
    File.Code = std::string(Code);
    AST = File.build();
  }

  // build() must have been called.
  bool shouldCollect(llvm::StringRef Name, bool Qualified = true) {
    assert(AST);
    const NamedDecl &ND =
        Qualified ? findDecl(*AST, Name) : findUnqualifiedDecl(*AST, Name);
    const SourceManager &SM = AST->getSourceManager();
    bool MainFile = isInsideMainFile(ND.getBeginLoc(), SM);
    return SymbolCollector::shouldCollectSymbol(
        ND, AST->getASTContext(), SymbolCollector::Options(), MainFile);
  }

protected:
  std::string HeaderName = "f.h";
  std::string FileName = "f.cpp";
  TestTU File;
  std::optional<ParsedAST> AST; // Initialized after build.
};

TEST_F(ShouldCollectSymbolTest, ShouldCollectSymbol) {
  build(R"(
    namespace nx {
    class X{};
    auto f() { int Local; } // auto ensures function body is parsed.
    struct { int x; } var;
    }
  )",
        R"(
    class InMain {};
    namespace { class InAnonymous {}; }
    static void g();
  )");
  auto AST = File.build();
  EXPECT_TRUE(shouldCollect("nx"));
  EXPECT_TRUE(shouldCollect("nx::X"));
  EXPECT_TRUE(shouldCollect("nx::f"));
  EXPECT_TRUE(shouldCollect("InMain"));
  EXPECT_TRUE(shouldCollect("InAnonymous", /*Qualified=*/false));
  EXPECT_TRUE(shouldCollect("g"));

  EXPECT_FALSE(shouldCollect("Local", /*Qualified=*/false));
}

TEST_F(ShouldCollectSymbolTest, CollectLocalClassesAndVirtualMethods) {
  build(R"(
    namespace nx {
    auto f() {
      int Local;
      auto LocalLambda = [&](){
        Local++;
        class ClassInLambda{};
        return Local;
      };
    } // auto ensures function body is parsed.
    auto foo() {
      class LocalBase {
        virtual void LocalVirtual();
        void LocalConcrete();
        int BaseMember;
      };
    }
    } // namespace nx
  )",
        "");
  auto AST = File.build();
  EXPECT_FALSE(shouldCollect("Local", /*Qualified=*/false));
  EXPECT_TRUE(shouldCollect("ClassInLambda", /*Qualified=*/false));
  EXPECT_TRUE(shouldCollect("LocalBase", /*Qualified=*/false));
  EXPECT_TRUE(shouldCollect("LocalVirtual", /*Qualified=*/false));
  EXPECT_TRUE(shouldCollect("LocalConcrete", /*Qualified=*/false));
  EXPECT_FALSE(shouldCollect("BaseMember", /*Qualified=*/false));
  EXPECT_FALSE(shouldCollect("Local", /*Qualified=*/false));
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
  SymbolIndexActionFactory(SymbolCollector::Options COpts)
      : COpts(std::move(COpts)) {}

  std::unique_ptr<FrontendAction> create() override {
    class IndexAction : public ASTFrontendAction {
    public:
      IndexAction(std::shared_ptr<index::IndexDataConsumer> DataConsumer,
                  const index::IndexingOptions &Opts,
                  std::shared_ptr<include_cleaner::PragmaIncludes> PI)
          : DataConsumer(std::move(DataConsumer)), Opts(Opts),
            PI(std::move(PI)) {}

      std::unique_ptr<ASTConsumer>
      CreateASTConsumer(CompilerInstance &CI, llvm::StringRef InFile) override {
        PI->record(CI);
        return createIndexingASTConsumer(DataConsumer, Opts,
                                         CI.getPreprocessorPtr());
      }

      bool BeginInvocation(CompilerInstance &CI) override {
        // Make the compiler parse all comments.
        CI.getLangOpts().CommentOpts.ParseAllComments = true;
        return true;
      }

    private:
      std::shared_ptr<index::IndexDataConsumer> DataConsumer;
      index::IndexingOptions Opts;
      std::shared_ptr<include_cleaner::PragmaIncludes> PI;
    };
    index::IndexingOptions IndexOpts;
    IndexOpts.SystemSymbolFilter =
        index::IndexingOptions::SystemSymbolFilterKind::All;
    IndexOpts.IndexFunctionLocals = true;
    std::shared_ptr<include_cleaner::PragmaIncludes> PI =
        std::make_shared<include_cleaner::PragmaIncludes>();
    COpts.PragmaIncludes = PI.get();
    Collector = std::make_shared<SymbolCollector>(COpts);
    return std::make_unique<IndexAction>(Collector, std::move(IndexOpts),
                                         std::move(PI));
  }

  std::shared_ptr<SymbolCollector> Collector;
  SymbolCollector::Options COpts;
};

class SymbolCollectorTest : public ::testing::Test {
public:
  SymbolCollectorTest()
      : InMemoryFileSystem(new llvm::vfs::InMemoryFileSystem),
        TestHeaderName(testPath("symbol.h")),
        TestFileName(testPath("symbol.cc")) {
    TestHeaderURI = URI::create(TestHeaderName).toString();
    TestFileURI = URI::create(TestFileName).toString();
  }

  // Note that unlike TestTU, no automatic header guard is added.
  // HeaderCode should start with #pragma once to be treated as modular.
  bool runSymbolCollector(llvm::StringRef HeaderCode, llvm::StringRef MainCode,
                          const std::vector<std::string> &ExtraArgs = {}) {
    llvm::IntrusiveRefCntPtr<FileManager> Files(
        new FileManager(FileSystemOptions(), InMemoryFileSystem));

    auto Factory = std::make_unique<SymbolIndexActionFactory>(CollectorOpts);

    std::vector<std::string> Args = {"symbol_collector", "-fsyntax-only",
                                     "-xc++", "-include", TestHeaderName};
    Args.insert(Args.end(), ExtraArgs.begin(), ExtraArgs.end());
    // This allows to override the "-xc++" with something else, i.e.
    // -xobjective-c++.
    Args.push_back(TestFileName);

    tooling::ToolInvocation Invocation(
        Args, Factory->create(), Files.get(),
        std::make_shared<PCHContainerOperations>());

    // Multiple calls to runSymbolCollector with different contents will fail
    // to update the filesystem! Why are we sharing one across tests, anyway?
    EXPECT_TRUE(InMemoryFileSystem->addFile(
        TestHeaderName, 0, llvm::MemoryBuffer::getMemBuffer(HeaderCode)));
    EXPECT_TRUE(InMemoryFileSystem->addFile(
        TestFileName, 0, llvm::MemoryBuffer::getMemBuffer(MainCode)));
    Invocation.run();
    Symbols = Factory->Collector->takeSymbols();
    Refs = Factory->Collector->takeRefs();
    Relations = Factory->Collector->takeRelations();
    return true;
  }

protected:
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem;
  std::string TestHeaderName;
  std::string TestHeaderURI;
  std::string TestFileName;
  std::string TestFileURI;
  SymbolSlab Symbols;
  RefSlab Refs;
  RelationSlab Relations;
  SymbolCollector::Options CollectorOpts;
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

    void f1() {
      auto LocalLambda = [&](){
        class ClassInLambda{};
      };
    }

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

    using bar::v2;
    } // namespace foo
  )";
  runSymbolCollector(Header, /*Main=*/"");
  EXPECT_THAT(Symbols,
              UnorderedElementsAreArray(
                  {AllOf(qName("Foo"), forCodeCompletion(true)),
                   AllOf(qName("Foo::Foo"), forCodeCompletion(false)),
                   AllOf(qName("Foo::Foo"), forCodeCompletion(false)),
                   AllOf(qName("Foo::f"), forCodeCompletion(false)),
                   AllOf(qName("Foo::~Foo"), forCodeCompletion(false)),
                   AllOf(qName("Foo::operator="), forCodeCompletion(false)),
                   AllOf(qName("Foo::Nested"), forCodeCompletion(false)),
                   AllOf(qName("Foo::Nested::f"), forCodeCompletion(false)),
                   AllOf(qName("ClassInLambda"), forCodeCompletion(false)),
                   AllOf(qName("Friend"), forCodeCompletion(true)),
                   AllOf(qName("f1"), forCodeCompletion(true)),
                   AllOf(qName("f2"), forCodeCompletion(true)),
                   AllOf(qName("KInt"), forCodeCompletion(true)),
                   AllOf(qName("kStr"), forCodeCompletion(true)),
                   AllOf(qName("foo"), forCodeCompletion(true)),
                   AllOf(qName("foo::bar"), forCodeCompletion(true)),
                   AllOf(qName("foo::int32"), forCodeCompletion(true)),
                   AllOf(qName("foo::int32_t"), forCodeCompletion(true)),
                   AllOf(qName("foo::v1"), forCodeCompletion(true)),
                   AllOf(qName("foo::bar::v2"), forCodeCompletion(true)),
                   AllOf(qName("foo::v2"), forCodeCompletion(true)),
                   AllOf(qName("foo::baz"), forCodeCompletion(true))}));
}

TEST_F(SymbolCollectorTest, FileLocal) {
  const std::string Header = R"(
    class Foo {};
    namespace {
      class Ignored {};
    }
    void bar();
  )";
  const std::string Main = R"(
    class ForwardDecl;
    void bar() {}
    static void a();
    class B {};
    namespace {
      void c();
    }
  )";
  runSymbolCollector(Header, Main);
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(
                  AllOf(qName("Foo"), visibleOutsideFile()),
                  AllOf(qName("bar"), visibleOutsideFile()),
                  AllOf(qName("a"), Not(visibleOutsideFile())),
                  AllOf(qName("B"), Not(visibleOutsideFile())),
                  AllOf(qName("c"), Not(visibleOutsideFile())),
                  // FIXME: ForwardDecl likely *is* visible outside.
                  AllOf(qName("ForwardDecl"), Not(visibleOutsideFile()))));
}

TEST_F(SymbolCollectorTest, Template) {
  Annotations Header(R"(
    // Primary template and explicit specialization are indexed, instantiation
    // is not.
    template <class T, class U> struct [[Tmpl]] {T $xdecl[[x]] = 0;};
    template <> struct $specdecl[[Tmpl]]<int, bool> {};
    template <class U> struct $partspecdecl[[Tmpl]]<bool, U> {};
    extern template struct Tmpl<float, bool>;
    template struct Tmpl<double, bool>;
  )");
  runSymbolCollector(Header.code(), /*Main=*/"");
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(
                  AllOf(qName("Tmpl"), declRange(Header.range()),
                        forCodeCompletion(true)),
                  AllOf(qName("Tmpl"), declRange(Header.range("specdecl")),
                        forCodeCompletion(false)),
                  AllOf(qName("Tmpl"), declRange(Header.range("partspecdecl")),
                        forCodeCompletion(false)),
                  AllOf(qName("Tmpl::x"), declRange(Header.range("xdecl")),
                        forCodeCompletion(false))));
}

TEST_F(SymbolCollectorTest, templateArgs) {
  Annotations Header(R"(
    template <class X> class $barclasstemp[[Bar]] {};
    template <class T, class U, template<typename> class Z, int Q>
    struct [[Tmpl]] { T $xdecl[[x]] = 0; };

    // template-template, non-type and type full spec
    template <> struct $specdecl[[Tmpl]]<int, bool, Bar, 3> {};

    // template-template, non-type and type partial spec
    template <class U, int T> struct $partspecdecl[[Tmpl]]<bool, U, Bar, T> {};
    // instantiation
    extern template struct Tmpl<float, bool, Bar, 8>;
    // instantiation
    template struct Tmpl<double, bool, Bar, 2>;

    template <typename ...> class $fooclasstemp[[Foo]] {};
    // parameter-packs full spec
    template<> class $parampack[[Foo]]<Bar<int>, int, double> {};
    // parameter-packs partial spec
    template<class T> class $parampackpartial[[Foo]]<T, T> {};

    template <int ...> class $bazclasstemp[[Baz]] {};
    // non-type parameter-packs full spec
    template<> class $parampacknontype[[Baz]]<3, 5, 8> {};
    // non-type parameter-packs partial spec
    template<int T> class $parampacknontypepartial[[Baz]]<T, T> {};

    template <template <class> class ...> class $fozclasstemp[[Foz]] {};
    // template-template parameter-packs full spec
    template<> class $parampacktempltempl[[Foz]]<Bar, Bar> {};
    // template-template parameter-packs partial spec
    template<template <class> class T>
    class $parampacktempltemplpartial[[Foz]]<T, T> {};
  )");
  runSymbolCollector(Header.code(), /*Main=*/"");
  EXPECT_THAT(
      Symbols,
      AllOf(
          Contains(AllOf(qName("Tmpl"), templateArgs("<int, bool, Bar, 3>"),
                         declRange(Header.range("specdecl")),
                         forCodeCompletion(false))),
          Contains(AllOf(qName("Tmpl"), templateArgs("<bool, U, Bar, T>"),
                         declRange(Header.range("partspecdecl")),
                         forCodeCompletion(false))),
          Contains(AllOf(qName("Foo"), templateArgs("<Bar<int>, int, double>"),
                         declRange(Header.range("parampack")),
                         forCodeCompletion(false))),
          Contains(AllOf(qName("Foo"), templateArgs("<T, T>"),
                         declRange(Header.range("parampackpartial")),
                         forCodeCompletion(false))),
          Contains(AllOf(qName("Baz"), templateArgs("<3, 5, 8>"),
                         declRange(Header.range("parampacknontype")),
                         forCodeCompletion(false))),
          Contains(AllOf(qName("Baz"), templateArgs("<T, T>"),
                         declRange(Header.range("parampacknontypepartial")),
                         forCodeCompletion(false))),
          Contains(AllOf(qName("Foz"), templateArgs("<Bar, Bar>"),
                         declRange(Header.range("parampacktempltempl")),
                         forCodeCompletion(false))),
          Contains(AllOf(qName("Foz"), templateArgs("<T, T>"),
                         declRange(Header.range("parampacktempltemplpartial")),
                         forCodeCompletion(false)))));
}

TEST_F(SymbolCollectorTest, ObjCRefs) {
  Annotations Header(R"(
  @interface Person
  - (void)$talk[[talk]];
  - (void)$say[[say]]:(id)something;
  @end
  @interface Person (Category)
  - (void)categoryMethod;
  - (void)multiArg:(id)a method:(id)b;
  @end
  )");
  Annotations Main(R"(
  @implementation Person
  - (void)$talk[[talk]] {}
  - (void)$say[[say]]:(id)something {}
  @end

  void fff(Person *p) {
    [p $talk[[talk]]];
    [p $say[[say]]:0];
    [p categoryMethod];
    [p multiArg:0 method:0];
  }
  )");
  CollectorOpts.RefFilter = RefKind::All;
  CollectorOpts.CollectMainFileRefs = true;
  TestFileName = testPath("test.m");
  runSymbolCollector(Header.code(), Main.code(),
                     {"-fblocks", "-xobjective-c++", "-Wno-objc-root-class"});
  EXPECT_THAT(Refs, Contains(Pair(findSymbol(Symbols, "Person::talk").ID,
                                  haveRanges(Main.ranges("talk")))));
  EXPECT_THAT(Refs, Contains(Pair(findSymbol(Symbols, "Person::say:").ID,
                                  haveRanges(Main.ranges("say")))));
  EXPECT_THAT(Refs,
              Contains(Pair(findSymbol(Symbols, "Person::categoryMethod").ID,
                            ElementsAre(isSpelled()))));
  EXPECT_THAT(Refs,
              Contains(Pair(findSymbol(Symbols, "Person::multiArg:method:").ID,
                            ElementsAre(isSpelled()))));
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
                  qName("Person"), qName("Person::someMethodName:lastName:"),
                  AllOf(qName("MyCategory"), forCodeCompletion(false)),
                  qName("Person::someMethodName2:"), qName("MyProtocol"),
                  qName("MyProtocol::someMethodName3:")));
}

TEST_F(SymbolCollectorTest, ObjCPropertyImpl) {
  const std::string Header = R"(
    @interface Container
    @property(nonatomic) int magic;
    @end

    @implementation Container
    @end
  )";
  TestFileName = testPath("test.m");
  runSymbolCollector(Header, /*Main=*/"", {"-xobjective-c++"});
  EXPECT_THAT(Symbols, Contains(qName("Container")));
  EXPECT_THAT(Symbols, Contains(qName("Container::magic")));
  // FIXME: Results also contain Container::_magic on some platforms.
  //        Figure out why it's platform-dependent.
}

TEST_F(SymbolCollectorTest, ObjCLocations) {
  Annotations Header(R"(
    // Declared in header, defined in main.
    @interface $dogdecl[[Dog]]
    @end
    @interface $fluffydecl[[Dog]] (Fluffy)
    @end
  )");
  Annotations Main(R"(
    @interface Dog ()
    @end
    @implementation $dogdef[[Dog]]
    @end
    @implementation $fluffydef[[Dog]] (Fluffy)
    @end
    // Category with no declaration (only implementation).
    @implementation $ruff[[Dog]] (Ruff)
    @end
    // Implicitly defined interface.
    @implementation $catdog[[CatDog]]
    @end
  )");
  runSymbolCollector(Header.code(), Main.code(),
                     {"-xobjective-c++", "-Wno-objc-root-class"});
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(
                  AllOf(qName("Dog"), declRange(Header.range("dogdecl")),
                        defRange(Main.range("dogdef"))),
                  AllOf(qName("Fluffy"), declRange(Header.range("fluffydecl")),
                        defRange(Main.range("fluffydef"))),
                  AllOf(qName("CatDog"), declRange(Main.range("catdog")),
                        defRange(Main.range("catdog"))),
                  AllOf(qName("Ruff"), declRange(Main.range("ruff")),
                        defRange(Main.range("ruff")))));
}

TEST_F(SymbolCollectorTest, ObjCForwardDecls) {
  Annotations Header(R"(
    // Forward declared in header, declared and defined in main.
    @protocol Barker;
    @class Dog;
    // Never fully declared so Clang latches onto this decl.
    @class $catdogdecl[[CatDog]];
  )");
  Annotations Main(R"(
    @protocol $barkerdecl[[Barker]]
    - (void)woof;
    @end
    @interface $dogdecl[[Dog]]<Barker>
    - (void)woof;
    @end
    @implementation $dogdef[[Dog]]
    - (void)woof {}
    @end
    @implementation $catdogdef[[CatDog]]
    @end
  )");
  runSymbolCollector(Header.code(), Main.code(),
                     {"-xobjective-c++", "-Wno-objc-root-class"});
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(
                  AllOf(qName("CatDog"), declRange(Header.range("catdogdecl")),
                        defRange(Main.range("catdogdef"))),
                  AllOf(qName("Dog"), declRange(Main.range("dogdecl")),
                        defRange(Main.range("dogdef"))),
                  AllOf(qName("Barker"), declRange(Main.range("barkerdecl"))),
                  qName("Barker::woof"), qName("Dog::woof")));
}

TEST_F(SymbolCollectorTest, ObjCClassExtensions) {
  Annotations Header(R"(
    @interface $catdecl[[Cat]]
    @end
  )");
  Annotations Main(R"(
    @interface Cat ()
    - (void)meow;
    @end
    @interface Cat ()
    - (void)pur;
    @end
  )");
  runSymbolCollector(Header.code(), Main.code(),
                     {"-xobjective-c++", "-Wno-objc-root-class"});
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(
                  AllOf(qName("Cat"), declRange(Header.range("catdecl"))),
                  qName("Cat::meow"), qName("Cat::pur")));
}

TEST_F(SymbolCollectorTest, ObjCFrameworkIncludeHeader) {
  CollectorOpts.CollectIncludePath = true;
  auto FrameworksPath = testPath("Frameworks/");
  std::string FrameworkHeader = R"(
    __attribute((objc_root_class))
    @interface NSObject
    @end
  )";
  InMemoryFileSystem->addFile(
      testPath("Frameworks/Foundation.framework/Headers/NSObject.h"), 0,
      llvm::MemoryBuffer::getMemBuffer(FrameworkHeader));
  std::string PrivateFrameworkHeader = R"(
    #import <Foundation/NSObject.h>

    @interface PrivateClass : NSObject
    @end
  )";
  InMemoryFileSystem->addFile(
      testPath(
          "Frameworks/Foundation.framework/PrivateHeaders/NSObject+Private.h"),
      0, llvm::MemoryBuffer::getMemBuffer(PrivateFrameworkHeader));

  std::string Header = R"(
    #import <Foundation/NSObject+Private.h>
    #import <Foundation/NSObject.h>

    @interface Container : NSObject
    @end
  )";
  std::string Main = "";
  TestFileName = testPath("test.m");
  runSymbolCollector(Header, Main, {"-F", FrameworksPath, "-xobjective-c++"});
  EXPECT_THAT(
      Symbols,
      UnorderedElementsAre(
          AllOf(qName("NSObject"), includeHeader("<Foundation/NSObject.h>")),
          AllOf(qName("PrivateClass"),
                includeHeader("<Foundation/NSObject+Private.h>")),
          AllOf(qName("Container"))));

  // After adding the umbrella headers, we should use that spelling instead.
  std::string UmbrellaHeader = R"(
    #import <Foundation/NSObject.h>
  )";
  InMemoryFileSystem->addFile(
      testPath("Frameworks/Foundation.framework/Headers/Foundation.h"), 0,
      llvm::MemoryBuffer::getMemBuffer(UmbrellaHeader));
  std::string PrivateUmbrellaHeader = R"(
    #import <Foundation/NSObject+Private.h>
  )";
  InMemoryFileSystem->addFile(
      testPath("Frameworks/Foundation.framework/PrivateHeaders/"
               "Foundation_Private.h"),
      0, llvm::MemoryBuffer::getMemBuffer(PrivateUmbrellaHeader));
  runSymbolCollector(Header, Main, {"-F", FrameworksPath, "-xobjective-c++"});
  EXPECT_THAT(
      Symbols,
      UnorderedElementsAre(
          AllOf(qName("NSObject"), includeHeader("<Foundation/Foundation.h>")),
          AllOf(qName("PrivateClass"),
                includeHeader("<Foundation/Foundation_Private.h>")),
          AllOf(qName("Container"))));

  runSymbolCollector(Header, Main,
                     {"-iframework", FrameworksPath, "-xobjective-c++"});
  EXPECT_THAT(
      Symbols,
      UnorderedElementsAre(
          AllOf(qName("NSObject"), includeHeader("<Foundation/Foundation.h>")),
          AllOf(qName("PrivateClass"),
                includeHeader("<Foundation/Foundation_Private.h>")),
          AllOf(qName("Container"))));
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
    int $ydecl[[Y]];
  )cpp");
  runSymbolCollector(Header.code(), Main.code());
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(
                  AllOf(qName("X"), declRange(Header.range("xdecl")),
                        defRange(Main.range("xdef"))),
                  AllOf(qName("Cls"), declRange(Header.range("clsdecl")),
                        defRange(Main.range("clsdef"))),
                  AllOf(qName("print"), declRange(Header.range("printdecl")),
                        defRange(Main.range("printdef"))),
                  AllOf(qName("Z"), declRange(Header.range("zdecl"))),
                  AllOf(qName("foo"), declRange(Header.range("foodecl"))),
                  AllOf(qName("Y"), declRange(Main.range("ydecl")))));
}

TEST_F(SymbolCollectorTest, Refs) {
  Annotations Header(R"(
  #define MACRO(X) (X + 1)
  class Foo {
  public:
    Foo() {}
    Foo(int);
  };
  class Bar;
  void func();

  namespace NS {} // namespace ref is ignored
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
    abc = $macro[[MACRO]](1);
  }
  )");
  Annotations SymbolsOnlyInMainCode(R"(
  #define FUNC(X) (X+1)
  int a;
  void b() {}
  static const int c = FUNC(1);
  class d {};
  )");
  CollectorOpts.RefFilter = RefKind::All;
  CollectorOpts.CollectMacro = true;
  runSymbolCollector(Header.code(),
                     (Main.code() + SymbolsOnlyInMainCode.code()).str());
  EXPECT_THAT(Refs, Contains(Pair(findSymbol(Symbols, "Foo").ID,
                                  haveRanges(Main.ranges("foo")))));
  EXPECT_THAT(Refs, Contains(Pair(findSymbol(Symbols, "Bar").ID,
                                  haveRanges(Main.ranges("bar")))));
  EXPECT_THAT(Refs, Contains(Pair(findSymbol(Symbols, "func").ID,
                                  haveRanges(Main.ranges("func")))));
  EXPECT_THAT(Refs, Not(Contains(Pair(findSymbol(Symbols, "NS").ID, _))));
  EXPECT_THAT(Refs, Contains(Pair(findSymbol(Symbols, "MACRO").ID,
                                  haveRanges(Main.ranges("macro")))));
  // - (a, b) externally visible and should have refs.
  // - (c, FUNC) externally invisible and had no refs collected.
  auto MainSymbols =
      TestTU::withHeaderCode(SymbolsOnlyInMainCode.code()).headerSymbols();
  EXPECT_THAT(Refs, Contains(Pair(findSymbol(MainSymbols, "a").ID, _)));
  EXPECT_THAT(Refs, Contains(Pair(findSymbol(MainSymbols, "b").ID, _)));
  EXPECT_THAT(Refs, Not(Contains(Pair(findSymbol(MainSymbols, "c").ID, _))));
  EXPECT_THAT(Refs, Not(Contains(Pair(findSymbol(MainSymbols, "FUNC").ID, _))));

  // Run the collector again with CollectMainFileRefs = true.
  // We need to recreate InMemoryFileSystem because runSymbolCollector()
  // calls MemoryBuffer::getMemBuffer(), which makes the buffers unusable
  // after runSymbolCollector() exits.
  InMemoryFileSystem = new llvm::vfs::InMemoryFileSystem();
  CollectorOpts.CollectMainFileRefs = true;
  runSymbolCollector(Header.code(),
                     (Main.code() + SymbolsOnlyInMainCode.code()).str());
  EXPECT_THAT(Refs, Contains(Pair(findSymbol(Symbols, "a").ID, _)));
  EXPECT_THAT(Refs, Contains(Pair(findSymbol(Symbols, "b").ID, _)));
  EXPECT_THAT(Refs, Contains(Pair(findSymbol(Symbols, "c").ID, _)));
  // However, references to main-file macros are not collected.
  EXPECT_THAT(Refs, Not(Contains(Pair(findSymbol(Symbols, "FUNC").ID, _))));
}

TEST_F(SymbolCollectorTest, RefContainers) {
  Annotations Code(R"cpp(
    int $toplevel1[[f1]](int);
    void f2() {
      (void) $ref1a[[f1]](1);
      auto fptr = &$ref1b[[f1]];
    }
    int $toplevel2[[v1]] = $ref2[[f1]](2);
    void f3(int arg = $ref3[[f1]](3));
    struct S1 {
      int $classscope1[[member1]] = $ref4[[f1]](4);
      int $classscope2[[member2]] = 42;
    };
    constexpr int f4(int x) { return x + 1; }
    template <int I = $ref5[[f4]](0)> struct S2 {};
    S2<$ref6[[f4]](0)> v2;
    S2<$ref7a[[f4]](0)> f5(S2<$ref7b[[f4]](0)>);
    namespace N {
      void $namespacescope1[[f6]]();
      int $namespacescope2[[v3]];
    }
  )cpp");
  CollectorOpts.RefFilter = RefKind::All;
  CollectorOpts.CollectMainFileRefs = true;
  runSymbolCollector("", Code.code());
  auto FindRefWithRange = [&](Range R) -> std::optional<Ref> {
    for (auto &Entry : Refs) {
      for (auto &Ref : Entry.second) {
        if (rangesMatch(Ref.Location, R))
          return Ref;
      }
    }
    return std::nullopt;
  };
  auto Container = [&](llvm::StringRef RangeName) {
    auto Ref = FindRefWithRange(Code.range(RangeName));
    EXPECT_TRUE(bool(Ref));
    return Ref->Container;
  };
  EXPECT_EQ(Container("ref1a"),
            findSymbol(Symbols, "f2").ID); // function body (call)
  EXPECT_EQ(Container("ref1b"),
            findSymbol(Symbols, "f2").ID); // function body (address-of)
  EXPECT_EQ(Container("ref2"),
            findSymbol(Symbols, "v1").ID); // variable initializer
  EXPECT_EQ(Container("ref3"),
            findSymbol(Symbols, "f3").ID); // function parameter default value
  EXPECT_EQ(Container("ref4"),
            findSymbol(Symbols, "S1::member1").ID); // member initializer
  EXPECT_EQ(Container("ref5"),
            findSymbol(Symbols, "S2").ID); // template parameter default value
  EXPECT_EQ(Container("ref6"),
            findSymbol(Symbols, "v2").ID); // type of variable
  EXPECT_EQ(Container("ref7a"),
            findSymbol(Symbols, "f5").ID); // return type of function
  EXPECT_EQ(Container("ref7b"),
            findSymbol(Symbols, "f5").ID); // parameter type of function

  EXPECT_FALSE(Container("classscope1").isNull());
  EXPECT_FALSE(Container("namespacescope1").isNull());

  EXPECT_EQ(Container("toplevel1"), Container("toplevel2"));
  EXPECT_EQ(Container("classscope1"), Container("classscope2"));
  EXPECT_EQ(Container("namespacescope1"), Container("namespacescope2"));

  EXPECT_NE(Container("toplevel1"), Container("namespacescope1"));
  EXPECT_NE(Container("toplevel1"), Container("classscope1"));
  EXPECT_NE(Container("classscope1"), Container("namespacescope1"));
}

TEST_F(SymbolCollectorTest, MacroRefInHeader) {
  Annotations Header(R"(
  #define $foo[[FOO]](X) (X + 1)
  #define $bar[[BAR]](X) (X + 2)

  // Macro defined multiple times.
  #define $ud1[[UD]] 1
  int ud_1 = $ud1[[UD]];
  #undef UD

  #define $ud2[[UD]] 2
  int ud_2 = $ud2[[UD]];
  #undef UD

  // Macros from token concatenations not included.
  #define $concat[[CONCAT]](X) X##A()
  #define $prepend[[PREPEND]](X) MACRO##X()
  #define $macroa[[MACROA]]() 123
  int B = $concat[[CONCAT]](MACRO);
  int D = $prepend[[PREPEND]](A);

  void fff() {
    int abc = $foo[[FOO]](1) + $bar[[BAR]]($foo[[FOO]](1));
  }
  )");
  CollectorOpts.RefFilter = RefKind::All;
  CollectorOpts.RefsInHeaders = true;
  // Need this to get the SymbolID for macros for tests.
  CollectorOpts.CollectMacro = true;

  runSymbolCollector(Header.code(), "");

  EXPECT_THAT(Refs, Contains(Pair(findSymbol(Symbols, "FOO").ID,
                                  haveRanges(Header.ranges("foo")))));
  EXPECT_THAT(Refs, Contains(Pair(findSymbol(Symbols, "BAR").ID,
                                  haveRanges(Header.ranges("bar")))));
  // No unique ID for multiple symbols named UD. Check for ranges only.
  EXPECT_THAT(Refs, Contains(Pair(_, haveRanges(Header.ranges("ud1")))));
  EXPECT_THAT(Refs, Contains(Pair(_, haveRanges(Header.ranges("ud2")))));
  EXPECT_THAT(Refs, Contains(Pair(findSymbol(Symbols, "CONCAT").ID,
                                  haveRanges(Header.ranges("concat")))));
  EXPECT_THAT(Refs, Contains(Pair(findSymbol(Symbols, "PREPEND").ID,
                                  haveRanges(Header.ranges("prepend")))));
  EXPECT_THAT(Refs, Contains(Pair(findSymbol(Symbols, "MACROA").ID,
                                  haveRanges(Header.ranges("macroa")))));
}

TEST_F(SymbolCollectorTest, MacroRefWithoutCollectingSymbol) {
  Annotations Header(R"(
  #define $foo[[FOO]](X) (X + 1)
  int abc = $foo[[FOO]](1);
  )");
  CollectorOpts.RefFilter = RefKind::All;
  CollectorOpts.RefsInHeaders = true;
  CollectorOpts.CollectMacro = false;
  runSymbolCollector(Header.code(), "");
  EXPECT_THAT(Refs, Contains(Pair(_, haveRanges(Header.ranges("foo")))));
}

TEST_F(SymbolCollectorTest, MacrosWithRefFilter) {
  Annotations Header("#define $macro[[MACRO]](X) (X + 1)");
  Annotations Main("void foo() { int x = $macro[[MACRO]](1); }");
  CollectorOpts.RefFilter = RefKind::Unknown;
  runSymbolCollector(Header.code(), Main.code());
  EXPECT_THAT(Refs, IsEmpty());
}

TEST_F(SymbolCollectorTest, SpelledReferences) {
  struct {
    llvm::StringRef Header;
    llvm::StringRef Main;
    llvm::StringRef TargetSymbolName;
  } TestCases[] = {
    {
      R"cpp(
        struct Foo;
        #define MACRO Foo
      )cpp",
      R"cpp(
        struct $spelled[[Foo]] {
          $spelled[[Foo]]();
          ~$spelled[[Foo]]();
        };
        $spelled[[Foo]] Variable1;
        $implicit[[MACRO]] Variable2;
      )cpp",
      "Foo",
    },
    {
      R"cpp(
        class Foo {
        public:
          Foo() = default;
        };
      )cpp",
      R"cpp(
        void f() { Foo $implicit[[f]]; f = $spelled[[Foo]]();}
      )cpp",
      "Foo::Foo" /// constructor.
    },
    { // Unclean identifiers
      R"cpp(
        struct Foo {};
      )cpp",
      R"cpp(
        $spelled[[Fo\
o]] f{};
      )cpp",
      "Foo",
    },
  };
  CollectorOpts.RefFilter = RefKind::All;
  CollectorOpts.RefsInHeaders = false;
  for (const auto& T : TestCases) {
    SCOPED_TRACE(T.Header + "\n---\n" + T.Main);
    Annotations Header(T.Header);
    Annotations Main(T.Main);
    // Reset the file system.
    InMemoryFileSystem = new llvm::vfs::InMemoryFileSystem;
    runSymbolCollector(Header.code(), Main.code());

    const auto SpelledRanges = Main.ranges("spelled");
    const auto ImplicitRanges = Main.ranges("implicit");
    RefSlab::Builder SpelledSlabBuilder, ImplicitSlabBuilder;
    const auto TargetID = findSymbol(Symbols, T.TargetSymbolName).ID;
    for (const auto &SymbolAndRefs : Refs) {
      const auto ID = SymbolAndRefs.first;
      if (ID != TargetID)
        continue;
      for (const auto &Ref : SymbolAndRefs.second)
        if ((Ref.Kind & RefKind::Spelled) != RefKind::Unknown)
          SpelledSlabBuilder.insert(ID, Ref);
        else
          ImplicitSlabBuilder.insert(ID, Ref);
    }
    const auto SpelledRefs = std::move(SpelledSlabBuilder).build(),
               ImplicitRefs = std::move(ImplicitSlabBuilder).build();
    EXPECT_EQ(SpelledRanges.empty(), SpelledRefs.empty());
    EXPECT_EQ(ImplicitRanges.empty(), ImplicitRefs.empty());
    if (!SpelledRanges.empty())
      EXPECT_THAT(SpelledRefs,
                  Contains(Pair(TargetID, haveRanges(SpelledRanges))));
    if (!ImplicitRanges.empty())
      EXPECT_THAT(ImplicitRefs,
                  Contains(Pair(TargetID, haveRanges(ImplicitRanges))));
  }
}

TEST_F(SymbolCollectorTest, NameReferences) {
  CollectorOpts.RefFilter = RefKind::All;
  CollectorOpts.RefsInHeaders = true;
  Annotations Header(R"(
    class [[Foo]] {
    public:
      [[Foo]]() {}
      ~[[Foo]]() {}
    };
  )");
  CollectorOpts.RefFilter = RefKind::All;
  runSymbolCollector(Header.code(), "");
  // When we find references for class Foo, we expect to see all
  // constructor/destructor references.
  EXPECT_THAT(Refs, Contains(Pair(findSymbol(Symbols, "Foo").ID,
                                  haveRanges(Header.ranges()))));
}

TEST_F(SymbolCollectorTest, RefsOnMacros) {
  // Refs collected from SymbolCollector behave in the same way as
  // AST-based xrefs.
  CollectorOpts.RefFilter = RefKind::All;
  CollectorOpts.RefsInHeaders = true;
  Annotations Header(R"(
    #define TYPE(X) X
    #define FOO Foo
    #define CAT(X, Y) X##Y
    class [[Foo]] {};
    void test() {
      TYPE([[Foo]]) foo;
      [[FOO]] foo2;
      TYPE(TYPE([[Foo]])) foo3;
      [[CAT]](Fo, o) foo4;
    }
  )");
  CollectorOpts.RefFilter = RefKind::All;
  runSymbolCollector(Header.code(), "");
  EXPECT_THAT(Refs, Contains(Pair(findSymbol(Symbols, "Foo").ID,
                                  haveRanges(Header.ranges()))));
}

TEST_F(SymbolCollectorTest, HeaderAsMainFile) {
  CollectorOpts.RefFilter = RefKind::All;
  Annotations Header(R"(
  class $Foo[[Foo]] {};

  void $Func[[Func]]() {
    $Foo[[Foo]] fo;
  }
  )");
  // We should collect refs to main-file symbols in all cases:

  // 1. The main file is normal .cpp file.
  TestFileName = testPath("foo.cpp");
  runSymbolCollector("", Header.code());
  EXPECT_THAT(Refs,
              UnorderedElementsAre(Pair(findSymbol(Symbols, "Foo").ID,
                                        haveRanges(Header.ranges("Foo"))),
                                   Pair(findSymbol(Symbols, "Func").ID,
                                        haveRanges(Header.ranges("Func")))));

  // 2. Run the .h file as main file.
  TestFileName = testPath("foo.h");
  runSymbolCollector("", Header.code(),
                     /*ExtraArgs=*/{"-xobjective-c++-header"});
  EXPECT_THAT(Symbols, UnorderedElementsAre(qName("Foo"), qName("Func")));
  EXPECT_THAT(Refs,
              UnorderedElementsAre(Pair(findSymbol(Symbols, "Foo").ID,
                                        haveRanges(Header.ranges("Foo"))),
                                   Pair(findSymbol(Symbols, "Func").ID,
                                        haveRanges(Header.ranges("Func")))));

  // 3. Run the .hh file as main file (without "-x c++-header").
  TestFileName = testPath("foo.hh");
  runSymbolCollector("", Header.code());
  EXPECT_THAT(Symbols, UnorderedElementsAre(qName("Foo"), qName("Func")));
  EXPECT_THAT(Refs,
              UnorderedElementsAre(Pair(findSymbol(Symbols, "Foo").ID,
                                        haveRanges(Header.ranges("Foo"))),
                                   Pair(findSymbol(Symbols, "Func").ID,
                                        haveRanges(Header.ranges("Func")))));
}

TEST_F(SymbolCollectorTest, RefsInHeaders) {
  CollectorOpts.RefFilter = RefKind::All;
  CollectorOpts.RefsInHeaders = true;
  CollectorOpts.CollectMacro = true;
  Annotations Header(R"(
  #define $macro[[MACRO]](x) (x+1)
  class $foo[[Foo]] {};
  )");
  runSymbolCollector(Header.code(), "");
  EXPECT_THAT(Refs, Contains(Pair(findSymbol(Symbols, "Foo").ID,
                                  haveRanges(Header.ranges("foo")))));
  EXPECT_THAT(Refs, Contains(Pair(findSymbol(Symbols, "MACRO").ID,
                                  haveRanges(Header.ranges("macro")))));
}

TEST_F(SymbolCollectorTest, BaseOfRelations) {
  std::string Header = R"(
  class Base {};
  class Derived : public Base {};
  )";
  runSymbolCollector(Header, /*Main=*/"");
  const Symbol &Base = findSymbol(Symbols, "Base");
  const Symbol &Derived = findSymbol(Symbols, "Derived");
  EXPECT_THAT(Relations,
              Contains(Relation{Base.ID, RelationKind::BaseOf, Derived.ID}));
}

TEST_F(SymbolCollectorTest, OverrideRelationsSimpleInheritance) {
  std::string Header = R"cpp(
    class A {
      virtual void foo();
    };
    class B : public A {
      void foo() override;  // A::foo
      virtual void bar();
    };
    class C : public B {
      void bar() override;  // B::bar
    };
    class D: public C {
      void foo() override;  // B::foo
      void bar() override;  // C::bar
    };
  )cpp";
  runSymbolCollector(Header, /*Main=*/"");
  const Symbol &AFoo = findSymbol(Symbols, "A::foo");
  const Symbol &BFoo = findSymbol(Symbols, "B::foo");
  const Symbol &DFoo = findSymbol(Symbols, "D::foo");

  const Symbol &BBar = findSymbol(Symbols, "B::bar");
  const Symbol &CBar = findSymbol(Symbols, "C::bar");
  const Symbol &DBar = findSymbol(Symbols, "D::bar");

  std::vector<Relation> Result;
  for (const Relation &R : Relations)
    if (R.Predicate == RelationKind::OverriddenBy)
      Result.push_back(R);
  EXPECT_THAT(Result, UnorderedElementsAre(
                          OverriddenBy(AFoo, BFoo), OverriddenBy(BBar, CBar),
                          OverriddenBy(BFoo, DFoo), OverriddenBy(CBar, DBar)));
}

TEST_F(SymbolCollectorTest, OverrideRelationsMultipleInheritance) {
  std::string Header = R"cpp(
    class A {
      virtual void foo();
    };
    class B {
      virtual void bar();
    };
    class C : public B {
      void bar() override;  // B::bar
      virtual void baz();
    }
    class D : public A, C {
      void foo() override;  // A::foo
      void bar() override;  // C::bar
      void baz() override;  // C::baz
    };
  )cpp";
  runSymbolCollector(Header, /*Main=*/"");
  const Symbol &AFoo = findSymbol(Symbols, "A::foo");
  const Symbol &BBar = findSymbol(Symbols, "B::bar");
  const Symbol &CBar = findSymbol(Symbols, "C::bar");
  const Symbol &CBaz = findSymbol(Symbols, "C::baz");
  const Symbol &DFoo = findSymbol(Symbols, "D::foo");
  const Symbol &DBar = findSymbol(Symbols, "D::bar");
  const Symbol &DBaz = findSymbol(Symbols, "D::baz");

  std::vector<Relation> Result;
  for (const Relation &R : Relations)
    if (R.Predicate == RelationKind::OverriddenBy)
      Result.push_back(R);
  EXPECT_THAT(Result, UnorderedElementsAre(
                          OverriddenBy(BBar, CBar), OverriddenBy(AFoo, DFoo),
                          OverriddenBy(CBar, DBar), OverriddenBy(CBaz, DBaz)));
}

TEST_F(SymbolCollectorTest, CountReferences) {
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
    class Y{}; // definition doesn't count as a reference
    V* v = nullptr;
    GLOBAL_Z(z); // Not a reference to Z, we don't spell the type.
  )";
  CollectorOpts.CountReferences = true;
  runSymbolCollector(Header, Main);
  EXPECT_THAT(
      Symbols,
      UnorderedElementsAreArray(
          {AllOf(qName("W"), refCount(1)), AllOf(qName("X"), refCount(1)),
           AllOf(qName("Y"), refCount(0)), AllOf(qName("Z"), refCount(0)),
           AllOf(qName("y"), refCount(0)), AllOf(qName("z"), refCount(0)),
           AllOf(qName("x"), refCount(0)), AllOf(qName("w"), refCount(0)),
           AllOf(qName("w2"), refCount(0)), AllOf(qName("V"), refCount(1)),
           AllOf(qName("v"), refCount(0))}));
}

TEST_F(SymbolCollectorTest, SymbolRelativeNoFallback) {
  runSymbolCollector("class Foo {};", /*Main=*/"");
  EXPECT_THAT(Symbols, UnorderedElementsAre(
                           AllOf(qName("Foo"), declURI(TestHeaderURI))));
}

TEST_F(SymbolCollectorTest, SymbolRelativeWithFallback) {
  TestHeaderName = "x.h";
  TestFileName = "x.cpp";
  TestHeaderURI = URI::create(testPath(TestHeaderName)).toString();
  CollectorOpts.FallbackDir = testRoot();
  runSymbolCollector("class Foo {};", /*Main=*/"");
  EXPECT_THAT(Symbols, UnorderedElementsAre(
                           AllOf(qName("Foo"), declURI(TestHeaderURI))));
}

TEST_F(SymbolCollectorTest, UnittestURIScheme) {
  // Use test URI scheme from URITests.cpp
  TestHeaderName = testPath("x.h");
  TestFileName = testPath("x.cpp");
  runSymbolCollector("class Foo {};", /*Main=*/"");
  EXPECT_THAT(Symbols, UnorderedElementsAre(
                           AllOf(qName("Foo"), declURI("unittest:///x.h"))));
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
    class Color3 {
      enum {
        Blue
      };
    };
  )";
  runSymbolCollector(Header, /*Main=*/"");
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(
                  AllOf(qName("Red"), forCodeCompletion(true)),
                  AllOf(qName("Color"), forCodeCompletion(true)),
                  AllOf(qName("Green"), forCodeCompletion(true)),
                  AllOf(qName("Color2"), forCodeCompletion(true)),
                  AllOf(qName("Color2::Yellow"), forCodeCompletion(true)),
                  AllOf(qName("ns"), forCodeCompletion(true)),
                  AllOf(qName("ns::Black"), forCodeCompletion(true)),
                  AllOf(qName("Color3"), forCodeCompletion(true)),
                  AllOf(qName("Color3::Blue"), forCodeCompletion(true))));
}

TEST_F(SymbolCollectorTest, NamelessSymbols) {
  const std::string Header = R"(
    struct {
      int a;
    } Foo;
  )";
  runSymbolCollector(Header, /*Main=*/"");
  EXPECT_THAT(Symbols, UnorderedElementsAre(qName("Foo"),
                                            qName("(anonymous struct)::a")));
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
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(
                  AllOf(qName("abc_Test"), declRange(Header.range("expansion")),
                        declURI(TestHeaderURI)),
                  AllOf(qName("Test"), declRange(Header.range("spelling")),
                        declURI(TestHeaderURI))));
}

TEST_F(SymbolCollectorTest, SymbolFormedByCLI) {
  Annotations Header(R"(
    #ifdef NAME
    class $expansion[[NAME]] {};
    #endif
  )");
  runSymbolCollector(Header.code(), /*Main=*/"", /*ExtraArgs=*/{"-DNAME=name"});
  EXPECT_THAT(Symbols, UnorderedElementsAre(AllOf(
                           qName("name"), declRange(Header.range("expansion")),
                           declURI(TestHeaderURI))));
}

TEST_F(SymbolCollectorTest, SymbolsInMainFile) {
  const std::string Main = R"(
    class Foo {};
    void f1();
    inline void f2() {}

    namespace {
    void ff() {}
    }
    namespace foo {
    namespace {
    class Bar {};
    }
    }
    void main_f() {}
    void f1() {}
  )";
  runSymbolCollector(/*Header=*/"", Main);
  EXPECT_THAT(Symbols, UnorderedElementsAre(
                           qName("Foo"), qName("f1"), qName("f2"), qName("ff"),
                           qName("foo"), qName("foo::Bar"), qName("main_f")));
}

TEST_F(SymbolCollectorTest, Documentation) {
  const std::string Header = R"(
    // doc Foo
    class Foo {
      // doc f
      int f();
    };
  )";
  CollectorOpts.StoreAllDocumentation = false;
  runSymbolCollector(Header, /* Main */ "");
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(
                  AllOf(qName("Foo"), doc("doc Foo"), forCodeCompletion(true)),
                  AllOf(qName("Foo::f"), doc(""), returnType(""),
                        forCodeCompletion(false))));

  CollectorOpts.StoreAllDocumentation = true;
  runSymbolCollector(Header, /* Main */ "");
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(
                  AllOf(qName("Foo"), doc("doc Foo"), forCodeCompletion(true)),
                  AllOf(qName("Foo::f"), doc("doc f"), returnType(""),
                        forCodeCompletion(false))));
}

TEST_F(SymbolCollectorTest, DocumentationInMain) {
  const std::string Header = R"(
    // doc Foo
    class Foo {
      void f();
    };
  )";
  const std::string Main = R"(
    // doc f
    void Foo::f() {}
  )";
  CollectorOpts.StoreAllDocumentation = true;
  runSymbolCollector(Header, Main);
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(
                  AllOf(qName("Foo"), doc("doc Foo"), forCodeCompletion(true)),
                  AllOf(qName("Foo::f"), doc("doc f"), returnType(""),
                        forCodeCompletion(false))));
}

TEST_F(SymbolCollectorTest, DocumentationAtDeclThenDef) {
  const std::string Header = R"(
    class Foo {
      // doc f decl
      void f();
    };
  )";
  const std::string Main = R"(
    // doc f def
    void Foo::f() {}
  )";
  CollectorOpts.StoreAllDocumentation = true;
  runSymbolCollector(Header, Main);
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(AllOf(qName("Foo")),
                                   AllOf(qName("Foo::f"), doc("doc f decl"))));
}

TEST_F(SymbolCollectorTest, DocumentationAtDefThenDecl) {
  const std::string Header = R"(
    // doc f def
    void f() {}

    // doc f decl
    void f();
  )";
  CollectorOpts.StoreAllDocumentation = true;
  runSymbolCollector(Header, "" /*Main*/);
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(AllOf(qName("f"), doc("doc f def"))));
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
  EXPECT_THAT(
      Symbols,
      UnorderedElementsAre(
          qName("Foo"),
          AllOf(qName("Foo::f"), returnType(""), forCodeCompletion(false)),
          AllOf(qName("Foo::g"), returnType(""), forCodeCompletion(false)),
          AllOf(qName("Foo::sf"), returnType(""), forCodeCompletion(false)),
          AllOf(qName("Foo::ssf"), returnType(""), forCodeCompletion(false)),
          AllOf(qName("Foo::x"), returnType(""), forCodeCompletion(false))));
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
              UnorderedElementsAre(qName("na"), qName("na::nb"),
                                   qName("na::Foo"), qName("na::nb::Bar")));
}

TEST_F(SymbolCollectorTest, ExternC) {
  const std::string Header = R"(
    extern "C" { class Foo {}; }
    namespace na {
    extern "C" { class Bar {}; }
    }
  )";
  runSymbolCollector(Header, /*Main=*/"");
  EXPECT_THAT(Symbols, UnorderedElementsAre(qName("na"), qName("Foo"),
                                            qName("na::Bar")));
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
              UnorderedElementsAre(qName("na"), qName("na::nb"),
                                   qName("na::Foo"), qName("na::Bar")));
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
          qName("nx"), AllOf(qName("nx::ff"), labeled("ff(int x, double y)"),
                             returnType("int"), doc("Foo comment."))));
}

TEST_F(SymbolCollectorTest, snippet) {
  const std::string Header = R"(
    namespace nx {
    void f() {}
    int ff(int x, double y) { return 0; }
    }
  )";
  runSymbolCollector(Header, /*Main=*/"");
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(
                  qName("nx"),
                  AllOf(qName("nx::f"), labeled("f()"), snippet("f()")),
                  AllOf(qName("nx::ff"), labeled("ff(int x, double y)"),
                        snippet("ff(${1:int x}, ${2:double y})"))));
}

TEST_F(SymbolCollectorTest, IncludeHeaderSameAsFileURI) {
  CollectorOpts.CollectIncludePath = true;
  runSymbolCollector("#pragma once\nclass Foo {};", /*Main=*/"");
  EXPECT_THAT(Symbols, UnorderedElementsAre(
                           AllOf(qName("Foo"), declURI(TestHeaderURI))));
  EXPECT_THAT(Symbols.begin()->IncludeHeaders,
              UnorderedElementsAre(IncludeHeaderWithRef(TestHeaderURI, 1u)));
}

TEST_F(SymbolCollectorTest, CanonicalSTLHeader) {
  CollectorOpts.CollectIncludePath = true;
  runSymbolCollector(
      R"cpp(
      namespace std {
        class string {};
        // Move overloads have special handling.
        template <typename _T> T&& move(_T&& __value);
        template <typename _I, typename _O> _O move(_I, _I, _O);
        template <typename _T, typename _O, typename _I> _O move(
          _T&&, _O, _O, _I);
      }
      )cpp",
      /*Main=*/"");
  EXPECT_THAT(
      Symbols,
      UnorderedElementsAre(
          qName("std"),
          AllOf(qName("std::string"), declURI(TestHeaderURI),
                includeHeader("<string>")),
          // Parameter names are demangled.
          AllOf(labeled("move(T &&value)"), includeHeader("<utility>")),
          AllOf(labeled("move(I, I, O)"), includeHeader("<algorithm>")),
          AllOf(labeled("move(T &&, O, O, I)"), includeHeader("<algorithm>"))));
}

TEST_F(SymbolCollectorTest, IWYUPragma) {
  CollectorOpts.CollectIncludePath = true;
  const std::string Header = R"(
    // IWYU pragma: private, include the/good/header.h
    class Foo {};
  )";
  runSymbolCollector(Header, /*Main=*/"");
  EXPECT_THAT(Symbols, UnorderedElementsAre(
                           AllOf(qName("Foo"), declURI(TestHeaderURI),
                                 includeHeader("\"the/good/header.h\""))));
}

TEST_F(SymbolCollectorTest, IWYUPragmaWithDoubleQuotes) {
  CollectorOpts.CollectIncludePath = true;
  const std::string Header = R"(
    // IWYU pragma: private, include "the/good/header.h"
    class Foo {};
  )";
  runSymbolCollector(Header, /*Main=*/"");
  EXPECT_THAT(Symbols, UnorderedElementsAre(
                           AllOf(qName("Foo"), declURI(TestHeaderURI),
                                 includeHeader("\"the/good/header.h\""))));
}

TEST_F(SymbolCollectorTest, IWYUPragmaExport) {
  CollectorOpts.CollectIncludePath = true;
  const std::string Header = R"cpp(#pragma once
    #include "exporter.h"
  )cpp";
  auto ExporterFile = testPath("exporter.h");
  InMemoryFileSystem->addFile(
      ExporterFile, 0, llvm::MemoryBuffer::getMemBuffer(R"cpp(#pragma once
    #include "private.h" // IWYU pragma: export
  )cpp"));
  auto PrivateFile = testPath("private.h");
  InMemoryFileSystem->addFile(
      PrivateFile, 0, llvm::MemoryBuffer::getMemBuffer("class Foo {};"));
  runSymbolCollector(Header, /*Main=*/"",
                     /*ExtraArgs=*/{"-I", testRoot()});
  EXPECT_THAT(Symbols, UnorderedElementsAre(AllOf(
                           qName("Foo"),
                           includeHeader(URI::create(ExporterFile).toString()),
                           declURI(URI::create(PrivateFile).toString()))));
}

TEST_F(SymbolCollectorTest, MainFileIsHeaderWhenSkipIncFile) {
  CollectorOpts.CollectIncludePath = true;
  // To make this case as hard as possible, we won't tell clang main is a
  // header. No extension, no -x c++-header.
  TestFileName = testPath("no_ext_main");
  TestFileURI = URI::create(TestFileName).toString();
  auto IncFile = testPath("test.inc");
  auto IncURI = URI::create(IncFile).toString();
  InMemoryFileSystem->addFile(IncFile, 0,
                              llvm::MemoryBuffer::getMemBuffer("class X {};"));
  runSymbolCollector("", R"cpp(
    // Can't use #pragma once in a main file clang doesn't think is a header.
    #ifndef MAIN_H_
    #define MAIN_H_
    #include "test.inc"
    #endif
  )cpp",
                     /*ExtraArgs=*/{"-I", testRoot()});
  EXPECT_THAT(Symbols, UnorderedElementsAre(AllOf(qName("X"), declURI(IncURI),
                                                  includeHeader(TestFileURI))));
}

TEST_F(SymbolCollectorTest, IncFileInNonHeader) {
  CollectorOpts.CollectIncludePath = true;
  TestFileName = testPath("main.cc");
  TestFileURI = URI::create(TestFileName).toString();
  auto IncFile = testPath("test.inc");
  auto IncURI = URI::create(IncFile).toString();
  InMemoryFileSystem->addFile(IncFile, 0,
                              llvm::MemoryBuffer::getMemBuffer("class X {};"));
  runSymbolCollector("", R"cpp(
    #include "test.inc"
  )cpp",
                     /*ExtraArgs=*/{"-I", testRoot()});
  EXPECT_THAT(Symbols, UnorderedElementsAre(AllOf(qName("X"), declURI(IncURI),
                                                  Not(includeHeader()))));
}

// Features that depend on header-guards are fragile. Header guards are only
// recognized when the file ends, so we have to defer checking for them.
TEST_F(SymbolCollectorTest, HeaderGuardDetected) {
  CollectorOpts.CollectIncludePath = true;
  CollectorOpts.CollectMacro = true;
  runSymbolCollector(R"cpp(
    #ifndef HEADER_GUARD_
    #define HEADER_GUARD_

    // Symbols are seen before the header guard is complete.
    #define MACRO
    int decl();

    #endif // Header guard is recognized here.
  )cpp",
                     "");
  EXPECT_THAT(Symbols, Not(Contains(qName("HEADER_GUARD_"))));
  EXPECT_THAT(Symbols, Each(includeHeader()));
}

TEST_F(SymbolCollectorTest, NonModularHeader) {
  auto TU = TestTU::withHeaderCode("int x();");
  EXPECT_THAT(TU.headerSymbols(), ElementsAre(includeHeader()));

  // Files missing include guards aren't eligible for insertion.
  TU.ImplicitHeaderGuard = false;
  EXPECT_THAT(TU.headerSymbols(), ElementsAre(Not(includeHeader())));

  // We recognize some patterns of trying to prevent insertion.
  TU = TestTU::withHeaderCode(R"cpp(
#ifndef SECRET
#error "This file isn't safe to include directly"
#endif
    int x();
    )cpp");
  TU.ExtraArgs.push_back("-DSECRET"); // *we're* able to include it.
  EXPECT_THAT(TU.headerSymbols(), ElementsAre(Not(includeHeader())));
}

TEST_F(SymbolCollectorTest, AvoidUsingFwdDeclsAsCanonicalDecls) {
  CollectorOpts.CollectIncludePath = true;
  Annotations Header(R"(
    #pragma once
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
          AllOf(qName("C"), declURI(TestHeaderURI),
                declRange(Header.range("cdecl")), includeHeader(TestHeaderURI),
                defURI(TestHeaderURI), defRange(Header.range("cdecl"))),
          AllOf(qName("S"), declURI(TestHeaderURI),
                declRange(Header.range("sdecl")), includeHeader(TestHeaderURI),
                defURI(TestHeaderURI), defRange(Header.range("sdecl"))),
          AllOf(qName("U"), declURI(TestHeaderURI),
                declRange(Header.range("udecl")), includeHeader(TestHeaderURI),
                defURI(TestHeaderURI), defRange(Header.range("udecl"))),
          AllOf(qName("U::x"), declURI(TestHeaderURI),
                declRange(Header.range("xdecl")), defURI(TestHeaderURI),
                defRange(Header.range("xdecl"))),
          AllOf(qName("U::y"), declURI(TestHeaderURI),
                declRange(Header.range("ydecl")), defURI(TestHeaderURI),
                defRange(Header.range("ydecl")))));
}

TEST_F(SymbolCollectorTest, ClassForwardDeclarationIsCanonical) {
  CollectorOpts.CollectIncludePath = true;
  runSymbolCollector(/*Header=*/"#pragma once\nclass X;",
                     /*Main=*/"class X {};");
  EXPECT_THAT(Symbols, UnorderedElementsAre(AllOf(
                           qName("X"), declURI(TestHeaderURI),
                           includeHeader(TestHeaderURI), defURI(TestFileURI))));
}

TEST_F(SymbolCollectorTest, UTF16Character) {
  // ö is 2-bytes.
  Annotations Header(/*Header=*/"class [[pörk]] {};");
  runSymbolCollector(Header.code(), /*Main=*/"");
  EXPECT_THAT(Symbols, UnorderedElementsAre(
                           AllOf(qName("pörk"), declRange(Header.range()))));
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
                  qName("nx"), qName("nx::X"),
                  AllOf(qName("nx::Y"), declRange(Header.range("y"))),
                  AllOf(qName("nx::Z"), declRange(Header.range("z"))),
                  AllOf(qName("nx::foo"), declRange(Header.range("foo"))),
                  AllOf(qName("nx::bar"), declRange(Header.range("bar")))));
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
  EXPECT_THAT(Symbols, UnorderedElementsAre(AllOf(qName("X"), refCount(1)),
                                            AllOf(qName("Y"), refCount(1)),
                                            AllOf(qName("C"), refCount(0))));
}

TEST_F(SymbolCollectorTest, Origin) {
  CollectorOpts.Origin = SymbolOrigin::Static;
  runSymbolCollector("class Foo {};", /*Main=*/"");
  EXPECT_THAT(Symbols, UnorderedElementsAre(
                           Field(&Symbol::Origin, SymbolOrigin::Static)));
  InMemoryFileSystem = new llvm::vfs::InMemoryFileSystem;
  CollectorOpts.CollectMacro = true;
  runSymbolCollector("#define FOO", /*Main=*/"");
  EXPECT_THAT(Symbols, UnorderedElementsAre(
                           Field(&Symbol::Origin, SymbolOrigin::Static)));
}

TEST_F(SymbolCollectorTest, CollectMacros) {
  CollectorOpts.CollectIncludePath = true;
  Annotations Header(R"(
    #pragma once
    #define X 1
    #define $mac[[MAC]](x) int x
    #define $used[[USED]](y) float y;

    MAC(p);
  )");

  Annotations Main(R"(
    #define $main[[MAIN]] 1
     USED(t);
  )");
  CollectorOpts.CountReferences = true;
  CollectorOpts.CollectMacro = true;
  runSymbolCollector(Header.code(), Main.code());
  EXPECT_THAT(
      Symbols,
      UnorderedElementsAre(
          qName("p"), qName("t"),
          AllOf(qName("X"), declURI(TestHeaderURI),
                includeHeader(TestHeaderURI)),
          AllOf(labeled("MAC(x)"), refCount(0),

                declRange(Header.range("mac")), visibleOutsideFile()),
          AllOf(labeled("USED(y)"), refCount(1),
                declRange(Header.range("used")), visibleOutsideFile()),
          AllOf(labeled("MAIN"), refCount(0), declRange(Main.range("main")),
                Not(visibleOutsideFile()))));
}

TEST_F(SymbolCollectorTest, DeprecatedSymbols) {
  const std::string Header = R"(
    void TestClangc() __attribute__((deprecated("", "")));
    void TestClangd();
  )";
  runSymbolCollector(Header, /**/ "");
  EXPECT_THAT(Symbols, UnorderedElementsAre(
                           AllOf(qName("TestClangc"), deprecated()),
                           AllOf(qName("TestClangd"), Not(deprecated()))));
}

TEST_F(SymbolCollectorTest, implementationDetail) {
  const std::string Header = R"(
    #define DECL_NAME(x, y) x##_##y##_Decl
    #define DECL(x, y) class DECL_NAME(x, y) {};
    DECL(X, Y); // X_Y_Decl

    class Public {};
  )";
  runSymbolCollector(Header, /**/ "");
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(
                  AllOf(qName("X_Y_Decl"), implementationDetail()),
                  AllOf(qName("Public"), Not(implementationDetail()))));
}

TEST_F(SymbolCollectorTest, UsingDecl) {
  const char *Header = R"(
  void foo();
  namespace std {
    using ::foo;
  })";
  runSymbolCollector(Header, /**/ "");
  EXPECT_THAT(Symbols, Contains(qName("std::foo")));
}

TEST_F(SymbolCollectorTest, CBuiltins) {
  // In C, printf in stdio.h is a redecl of an implicit builtin.
  const char *Header = R"(
    extern int printf(const char*, ...);
  )";
  runSymbolCollector(Header, /**/ "", {"-xc"});
  EXPECT_THAT(Symbols, Contains(qName("printf")));
}

TEST_F(SymbolCollectorTest, InvalidSourceLoc) {
  const char *Header = R"(
      void operator delete(void*)
        __attribute__((__externally_visible__));)";
  runSymbolCollector(Header, /**/ "");
  EXPECT_THAT(Symbols, Contains(qName("operator delete")));
}

TEST_F(SymbolCollectorTest, BadUTF8) {
  // Extracted from boost/spirit/home/support/char_encoding/iso8859_1.hpp
  // This looks like UTF-8 and fools clang, but has high-ISO-8859-1 comments.
  const char *Header = "int PUNCT = 0;\n"
                       "/* \xa1 */ int types[] = { /* \xa1 */PUNCT };";
  CollectorOpts.RefFilter = RefKind::All;
  CollectorOpts.RefsInHeaders = true;
  runSymbolCollector(Header, "");
  EXPECT_THAT(Symbols, Contains(AllOf(qName("types"), doc("\xef\xbf\xbd "))));
  EXPECT_THAT(Symbols, Contains(qName("PUNCT")));
  // Reference is stored, although offset within line is not reliable.
  EXPECT_THAT(Refs, Contains(Pair(findSymbol(Symbols, "PUNCT").ID, _)));
}

TEST_F(SymbolCollectorTest, MacrosInHeaders) {
  CollectorOpts.CollectMacro = true;
  TestFileName = testPath("test.h");
  runSymbolCollector("", "#define X");
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(AllOf(qName("X"), forCodeCompletion(true))));
}

// Regression test for a crash-bug we used to have.
TEST_F(SymbolCollectorTest, UndefOfModuleMacro) {
  auto TU = TestTU::withCode(R"cpp(#include "bar.h")cpp");
  TU.AdditionalFiles["bar.h"] = R"cpp(
    #include "foo.h"
    #undef X
    )cpp";
  TU.AdditionalFiles["foo.h"] = "#define X 1";
  TU.AdditionalFiles["module.modulemap"] = R"cpp(
    module foo {
     header "foo.h"
     export *
   }
   )cpp";
  TU.ExtraArgs.push_back("-fmodules");
  TU.ExtraArgs.push_back("-fmodule-map-file=" + testPath("module.modulemap"));
  TU.OverlayRealFileSystemForModules = true;

  TU.build();
  // We mostly care about not crashing, but verify that we didn't insert garbage
  // about X too.
  EXPECT_THAT(TU.headerSymbols(), Not(Contains(qName("X"))));
}

TEST_F(SymbolCollectorTest, NoCrashOnObjCMethodCStyleParam) {
  auto TU = TestTU::withCode(R"objc(
    @interface Foo
    - (void)fun:(bool)foo, bool bar;
    @end
  )objc");
  TU.ExtraArgs.push_back("-xobjective-c++");

  TU.build();
  // We mostly care about not crashing.
  EXPECT_THAT(TU.headerSymbols(),
              UnorderedElementsAre(qName("Foo"), qName("Foo::fun:")));
}

TEST_F(SymbolCollectorTest, Reserved) {
  const char *Header = R"cpp(
    #pragma once
    void __foo();
    namespace _X { int secret; }
  )cpp";

  CollectorOpts.CollectReserved = true;
  runSymbolCollector(Header, "");
  EXPECT_THAT(Symbols, UnorderedElementsAre(qName("__foo"), qName("_X"),
                                            qName("_X::secret")));

  CollectorOpts.CollectReserved = false;
  runSymbolCollector(Header, "");
  EXPECT_THAT(Symbols, UnorderedElementsAre(qName("__foo"), qName("_X"),
                                            qName("_X::secret")));

  // Ugly: for some reason we reuse the test filesystem across tests.
  // You can't overwrite the same filename with new content!
  InMemoryFileSystem = new llvm::vfs::InMemoryFileSystem;
  runSymbolCollector("#pragma GCC system_header\n" + std::string(Header), "");
  EXPECT_THAT(Symbols, IsEmpty());
}

TEST_F(SymbolCollectorTest, Concepts) {
  const char *Header = R"cpp(
    template <class T>
    concept A = sizeof(T) <= 8;
  )cpp";
  runSymbolCollector("", Header, {"-std=c++20"});
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(AllOf(
                  qName("A"), hasKind(clang::index::SymbolKind::Concept))));
}

TEST_F(SymbolCollectorTest, IncludeHeaderForwardDecls) {
  CollectorOpts.CollectIncludePath = true;
  const std::string Header = R"cpp(#pragma once 
struct Foo;
#include "full.h"
)cpp";
  auto FullFile = testPath("full.h");
  InMemoryFileSystem->addFile(FullFile, 0,
                              llvm::MemoryBuffer::getMemBuffer(R"cpp(
#pragma once 
struct Foo {};)cpp"));
  runSymbolCollector(Header, /*Main=*/"",
                     /*ExtraArgs=*/{"-I", testRoot()});
  EXPECT_THAT(Symbols, UnorderedElementsAre(AllOf(
                           qName("Foo"),
                           includeHeader(URI::create(FullFile).toString()))))
      << *Symbols.begin();
}
} // namespace
} // namespace clangd
} // namespace clang
