//===-- RenameTests.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "ClangdServer.h"
#include "SyncAPI.h"
#include "TestFS.h"
#include "TestTU.h"
#include "index/Ref.h"
#include "refactor/Rename.h"
#include "support/TestTracer.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <algorithm>

namespace clang {
namespace clangd {
namespace {

using testing::ElementsAre;
using testing::Eq;
using testing::IsEmpty;
using testing::Pair;
using testing::SizeIs;
using testing::UnorderedElementsAre;
using testing::UnorderedElementsAreArray;

// Convert a Range to a Ref.
Ref refWithRange(const clangd::Range &Range, const std::string &URI) {
  Ref Result;
  Result.Kind = RefKind::Reference | RefKind::Spelled;
  Result.Location.Start.setLine(Range.start.line);
  Result.Location.Start.setColumn(Range.start.character);
  Result.Location.End.setLine(Range.end.line);
  Result.Location.End.setColumn(Range.end.character);
  Result.Location.FileURI = URI.c_str();
  return Result;
}

// Build a RefSlab from all marked ranges in the annotation. The ranges are
// assumed to associate with the given SymbolName.
std::unique_ptr<RefSlab> buildRefSlab(const Annotations &Code,
                                      llvm::StringRef SymbolName,
                                      llvm::StringRef Path) {
  RefSlab::Builder Builder;
  TestTU TU;
  TU.HeaderCode = std::string(Code.code());
  auto Symbols = TU.headerSymbols();
  const auto &SymbolID = findSymbol(Symbols, SymbolName).ID;
  std::string PathURI = URI::create(Path).toString();
  for (const auto &Range : Code.ranges())
    Builder.insert(SymbolID, refWithRange(Range, PathURI));

  return std::make_unique<RefSlab>(std::move(Builder).build());
}

std::vector<
    std::pair</*FilePath*/ std::string, /*CodeAfterRename*/ std::string>>
applyEdits(FileEdits FE) {
  std::vector<std::pair<std::string, std::string>> Results;
  for (auto &It : FE)
    Results.emplace_back(
        It.first().str(),
        llvm::cantFail(tooling::applyAllReplacements(
            It.getValue().InitialCode, It.getValue().Replacements)));
  return Results;
}

// Generates an expected rename result by replacing all ranges in the given
// annotation with the NewName.
std::string expectedResult(Annotations Test, llvm::StringRef NewName) {
  std::string Result;
  unsigned NextChar = 0;
  llvm::StringRef Code = Test.code();
  for (const auto &R : Test.llvm::Annotations::ranges()) {
    assert(R.Begin <= R.End && NextChar <= R.Begin);
    Result += Code.substr(NextChar, R.Begin - NextChar);
    Result += NewName;
    NextChar = R.End;
  }
  Result += Code.substr(NextChar);
  return Result;
}

TEST(RenameTest, WithinFileRename) {
  // rename is runnning on all "^" points, and "[[]]" ranges point to the
  // identifier that is being renamed.
  llvm::StringRef Tests[] = {
      // Function.
      R"cpp(
        void [[foo^]]() {
          [[fo^o]]();
        }
      )cpp",

      // Type.
      R"cpp(
        struct [[foo^]] {};
        [[foo]] test() {
           [[f^oo]] x;
           return x;
        }
      )cpp",

      // Local variable.
      R"cpp(
        void bar() {
          if (auto [[^foo]] = 5) {
            [[foo]] = 3;
          }
        }
      )cpp",

      // Rename class, including constructor/destructor.
      R"cpp(
        class [[F^oo]] {
          [[F^oo]]();
          ~[[Foo]]();
          void foo(int x);
        };
        [[Foo]]::[[Fo^o]]() {}
        void [[Foo]]::foo(int x) {}
      )cpp",

      // Rename template class, including constructor/destructor.
      R"cpp(
        template <typename T>
        class [[F^oo]] {
          [[F^oo]]();
          ~[[F^oo]]();
          void f([[Foo]] x);
        };
      )cpp",

      // Rename template class constructor.
      R"cpp(
        class [[F^oo]] {
          template<typename T>
          [[Fo^o]]();

          template<typename T>
          [[F^oo]](T t);
        };
      )cpp",

      // Class in template argument.
      R"cpp(
        class [[F^oo]] {};
        template <typename T> void func();
        template <typename T> class Baz {};
        int main() {
          func<[[F^oo]]>();
          Baz<[[F^oo]]> obj;
          return 0;
        }
      )cpp",

      // Forward class declaration without definition.
      R"cpp(
        class [[F^oo]];
        [[Foo]] *f();
      )cpp",

      // Class methods overrides.
      R"cpp(
        struct A {
         virtual void [[f^oo]]() {}
        };
        struct B : A {
          void [[f^oo]]() override {}
        };
        struct C : B {
          void [[f^oo]]() override {}
        };

        void func() {
          A().[[f^oo]]();
          B().[[f^oo]]();
          C().[[f^oo]]();
        }
      )cpp",

      // Template class (partial) specializations.
      R"cpp(
        template <typename T>
        class [[F^oo]] {};

        template<>
        class [[F^oo]]<bool> {};
        template <typename T>
        class [[F^oo]]<T*> {};

        void test() {
          [[Foo]]<int> x;
          [[Foo]]<bool> y;
          [[Foo]]<int*> z;
        }
      )cpp",

      // Incomplete class specializations
      R"cpp(
        template <typename T>
        class [[Fo^o]] {};
        void func([[Foo]]<int>);
      )cpp",

      // Template class instantiations.
      R"cpp(
        template <typename T>
        class [[F^oo]] {
        public:
          T foo(T arg, T& ref, T* ptr) {
            T value;
            int number = 42;
            value = (T)number;
            value = static_cast<T>(number);
            return value;
          }
          static void foo(T value) {}
          T member;
        };

        template <typename T>
        void func() {
          [[F^oo]]<T> obj;
          obj.member = T();
          [[Foo]]<T>::foo();
        }

        void test() {
          [[F^oo]]<int> i;
          i.member = 0;
          [[F^oo]]<int>::foo(0);

          [[F^oo]]<bool> b;
          b.member = false;
          [[Foo]]<bool>::foo(false);
        }
      )cpp",

      // Template class methods.
      R"cpp(
        template <typename T>
        class A {
        public:
          void [[f^oo]]() {}
        };

        void func() {
          A<int>().[[f^oo]]();
          A<double>().[[f^oo]]();
          A<float>().[[f^oo]]();
        }
      )cpp",

      // Complicated class type.
      R"cpp(
         // Forward declaration.
        class [[Fo^o]];
        class Baz {
          virtual int getValue() const = 0;
        };

        class [[F^oo]] : public Baz  {
        public:
          [[Foo]](int value = 0) : x(value) {}

          [[Foo]] &operator++(int);

          bool operator<([[Foo]] const &rhs);
          int getValue() const;
        private:
          int x;
        };

        void func() {
          [[Foo]] *Pointer = 0;
          [[Foo]] Variable = [[Foo]](10);
          for ([[Foo]] it; it < Variable; it++);
          const [[Foo]] *C = new [[Foo]]();
          const_cast<[[Foo]] *>(C)->getValue();
          [[Foo]] foo;
          const Baz &BazReference = foo;
          const Baz *BazPointer = &foo;
          reinterpret_cast<const [[^Foo]] *>(BazPointer)->getValue();
          static_cast<const [[^Foo]] &>(BazReference).getValue();
          static_cast<const [[^Foo]] *>(BazPointer)->getValue();
        }
      )cpp",

      // Destructor explicit call.
      R"cpp(
        class [[F^oo]] {
        public:
          ~[[^Foo]]();
        };

        [[Foo^]]::~[[^Foo]]() {}

        int main() {
          [[Fo^o]] f;
          f.~/*something*/[[^Foo]]();
          f.~[[^Foo]]();
        }
      )cpp",

      // Derived destructor explicit call.
      R"cpp(
        class [[Bas^e]] {};
        class Derived : public [[Bas^e]] {};

        int main() {
          [[Bas^e]] *foo = new Derived();
          foo->[[^Base]]::~[[^Base]]();
        }
      )cpp",

      // CXXConstructor initializer list.
      R"cpp(
        class Baz {};
        class Qux {
          Baz [[F^oo]];
        public:
          Qux();
        };
        Qux::Qux() : [[F^oo]]() {}
      )cpp",

      // DeclRefExpr.
      R"cpp(
        class C {
        public:
          static int [[F^oo]];
        };

        int foo(int x);
        #define MACRO(a) foo(a)

        void func() {
          C::[[F^oo]] = 1;
          MACRO(C::[[Foo]]);
          int y = C::[[F^oo]];
        }
      )cpp",

      // Macros.
      R"cpp(
        // no rename inside macro body.
        #define M1 foo
        #define M2(x) x
        int [[fo^o]]();
        void boo(int);

        void qoo() {
          [[foo]]();
          boo([[foo]]());
          M1();
          boo(M1());
          M2([[foo]]());
          M2(M1()); // foo is inside the nested macro body.
        }
      )cpp",

      // MemberExpr in macros
      R"cpp(
        class Baz {
        public:
          int [[F^oo]];
        };
        int qux(int x);
        #define MACRO(a) qux(a)

        int main() {
          Baz baz;
          baz.[[Foo]] = 1;
          MACRO(baz.[[Foo]]);
          int y = baz.[[Foo]];
        }
      )cpp",

      // Template parameters.
      R"cpp(
        template <typename [[^T]]>
        class Foo {
          [[T]] foo([[T]] arg, [[T]]& ref, [[^T]]* ptr) {
            [[T]] value;
            int number = 42;
            value = ([[T]])number;
            value = static_cast<[[^T]]>(number);
            return value;
          }
          static void foo([[T]] value) {}
          [[T]] member;
        };
      )cpp",

      // Typedef.
      R"cpp(
        namespace ns {
        class basic_string {};
        typedef basic_string [[s^tring]];
        } // namespace ns

        ns::[[s^tring]] foo();
      )cpp",

      // Variable.
      R"cpp(
        namespace A {
        int [[F^oo]];
        }
        int Foo;
        int Qux = Foo;
        int Baz = A::[[^Foo]];
        void fun() {
          struct {
            int Foo;
          } b = {100};
          int Foo = 100;
          Baz = Foo;
          {
            extern int Foo;
            Baz = Foo;
            Foo = A::[[F^oo]] + Baz;
            A::[[Fo^o]] = b.Foo;
          }
          Foo = b.Foo;
        }
      )cpp",

      // Namespace alias.
      R"cpp(
        namespace a { namespace b { void foo(); } }
        namespace [[^x]] = a::b;
        void bar() {
          [[x]]::foo();
        }
      )cpp",

      // Scope enums.
      R"cpp(
        enum class [[K^ind]] { ABC };
        void ff() {
          [[K^ind]] s;
          s = [[Kind]]::ABC;
        }
      )cpp",

      // template class in template argument list.
      R"cpp(
        template<typename T>
        class [[Fo^o]] {};
        template <template<typename> class Z> struct Bar { };
        template <> struct Bar<[[Foo]]> {};
      )cpp",

      // Designated initializer.
      R"cpp(
        struct Bar {
          int [[Fo^o]];
        };
        Bar bar { .[[^Foo]] = 42 };
      )cpp",

      // Nested designated initializer.
      R"cpp(
        struct Baz {
          int Field;
        };
        struct Bar {
          Baz [[Fo^o]];
        };
        // FIXME:    v selecting here results in renaming Field.
        Bar bar { .[[Foo]].Field = 42 };
      )cpp",
      R"cpp(
        struct Baz {
          int [[Fiel^d]];
        };
        struct Bar {
          Baz Foo;
        };
        Bar bar { .Foo.[[^Field]] = 42 };
      )cpp",
  };
  for (llvm::StringRef T : Tests) {
    SCOPED_TRACE(T);
    Annotations Code(T);
    auto TU = TestTU::withCode(Code.code());
    TU.ExtraArgs.push_back("-fno-delayed-template-parsing");
    auto AST = TU.build();
    llvm::StringRef NewName = "abcde";
    for (const auto &RenamePos : Code.points()) {
      auto RenameResult =
          rename({RenamePos, NewName, AST, testPath(TU.Filename)});
      ASSERT_TRUE(bool(RenameResult)) << RenameResult.takeError();
      ASSERT_EQ(1u, RenameResult->GlobalChanges.size());
      EXPECT_EQ(
          applyEdits(std::move(RenameResult->GlobalChanges)).front().second,
          expectedResult(Code, NewName));
    }
  }
}

TEST(RenameTest, Renameable) {
  struct Case {
    const char *Code;
    const char* ErrorMessage; // null if no error
    bool IsHeaderFile;
    const SymbolIndex *Index;
    llvm::StringRef NewName = "DummyName";
  };
  TestTU OtherFile = TestTU::withCode("Outside s; auto ss = &foo;");
  const char *CommonHeader = R"cpp(
    class Outside {};
    void foo();
  )cpp";
  OtherFile.HeaderCode = CommonHeader;
  OtherFile.Filename = "other.cc";
  // The index has a "Outside" reference and a "foo" reference.
  auto OtherFileIndex = OtherFile.index();
  const SymbolIndex *Index = OtherFileIndex.get();

  const bool HeaderFile = true;
  Case Cases[] = {
      {R"cpp(// allow -- function-local
        void f(int [[Lo^cal]]) {
          [[Local]] = 2;
        }
      )cpp",
       nullptr, HeaderFile, Index},

      {R"cpp(// allow -- symbol is indexable and has no refs in index.
        void [[On^lyInThisFile]]();
      )cpp",
       nullptr, HeaderFile, Index},

      {R"cpp(
        void ^f();
      )cpp",
       "keyword", HeaderFile, Index, "return"},

      {R"cpp(// disallow -- symbol is indexable and has other refs in index.
        void f() {
          Out^side s;
        }
      )cpp",
       "used outside main file", HeaderFile, Index},

      {R"cpp(// disallow -- symbol in anonymous namespace in header is not indexable.
        namespace {
        class Unin^dexable {};
        }
      )cpp",
       "not eligible for indexing", HeaderFile, Index},

      {R"cpp(// allow -- symbol in anonymous namespace in non-header file is indexable.
        namespace {
        class [[F^oo]] {};
        }
      )cpp",
       nullptr, !HeaderFile, Index},

      {R"cpp(// disallow -- namespace symbol isn't supported
        namespace n^s {}
      )cpp",
       "not a supported kind", HeaderFile, Index},

      {
          R"cpp(
         #define MACRO 1
         int s = MAC^RO;
       )cpp",
          "not a supported kind", HeaderFile, Index},

      {
          R"cpp(
        struct X { X operator++(int); };
        void f(X x) {x+^+;})cpp",
          "no symbol", HeaderFile, Index},

      {R"cpp(// foo is declared outside the file.
        void fo^o() {}
      )cpp",
       "used outside main file", !HeaderFile /*cc file*/, Index},

      {R"cpp(
         // We should detect the symbol is used outside the file from the AST.
         void fo^o() {})cpp",
       "used outside main file", !HeaderFile, nullptr /*no index*/},

      {R"cpp(// disallow rename on excluded symbols (e.g. std symbols)
         namespace std {
         class str^ing {};
         }
       )cpp",
       "not a supported kind", !HeaderFile, Index},
      {R"cpp(// disallow rename on excluded symbols (e.g. std symbols)
         namespace std {
         inline namespace __u {
         class str^ing {};
         }
         }
       )cpp",
       "not a supported kind", !HeaderFile, Index},

      {R"cpp(
         void foo(int);
         void foo(char);
         template <typename T> void f(T t) {
           fo^o(t);
         })cpp",
       "multiple symbols", !HeaderFile, nullptr /*no index*/},

      {R"cpp(// disallow rename on unrelated token.
         cl^ass Foo {};
       )cpp",
       "no symbol", !HeaderFile, nullptr},

      {R"cpp(// disallow rename on unrelated token.
         temp^late<typename T>
         class Foo {};
       )cpp",
       "no symbol", !HeaderFile, nullptr},
  };

  for (const auto& Case : Cases) {
    SCOPED_TRACE(Case.Code);
    Annotations T(Case.Code);
    TestTU TU = TestTU::withCode(T.code());
    TU.HeaderCode = CommonHeader;
    TU.ExtraArgs.push_back("-fno-delayed-template-parsing");
    if (Case.IsHeaderFile) {
      // We open the .h file as the main file.
      TU.Filename = "test.h";
      // Parsing the .h file as C++ include.
      TU.ExtraArgs.push_back("-xobjective-c++-header");
    }
    auto AST = TU.build();
    llvm::StringRef NewName = Case.NewName;
    auto Results =
        rename({T.point(), NewName, AST, testPath(TU.Filename), Case.Index});
    bool WantRename = true;
    if (T.ranges().empty())
      WantRename = false;
    if (!WantRename) {
      assert(Case.ErrorMessage && "Error message must be set!");
      EXPECT_FALSE(Results)
          << "expected rename returned an error: " << T.code();
      auto ActualMessage = llvm::toString(Results.takeError());
      EXPECT_THAT(ActualMessage, testing::HasSubstr(Case.ErrorMessage));
    } else {
      EXPECT_TRUE(bool(Results)) << "rename returned an error: "
                                 << llvm::toString(Results.takeError());
      ASSERT_EQ(1u, Results->GlobalChanges.size());
      EXPECT_EQ(applyEdits(std::move(Results->GlobalChanges)).front().second,
                expectedResult(T, NewName));
    }
  }
}

TEST(RenameTest, MainFileReferencesOnly) {
  // filter out references not from main file.
  llvm::StringRef Test =
      R"cpp(
        void test() {
          int [[fo^o]] = 1;
          // rename references not from main file are not included.
          #include "foo.inc"
        })cpp";

  Annotations Code(Test);
  auto TU = TestTU::withCode(Code.code());
  TU.AdditionalFiles["foo.inc"] = R"cpp(
      #define Macro(X) X
      &Macro(foo);
      &foo;
    )cpp";
  auto AST = TU.build();
  llvm::StringRef NewName = "abcde";

  auto RenameResult =
      rename({Code.point(), NewName, AST, testPath(TU.Filename)});
  ASSERT_TRUE(bool(RenameResult)) << RenameResult.takeError() << Code.point();
  ASSERT_EQ(1u, RenameResult->GlobalChanges.size());
  EXPECT_EQ(applyEdits(std::move(RenameResult->GlobalChanges)).front().second,
            expectedResult(Code, NewName));
}

TEST(RenameTest, ProtobufSymbolIsExcluded) {
  Annotations Code("Prot^obuf buf;");
  auto TU = TestTU::withCode(Code.code());
  TU.HeaderCode =
      R"cpp(// Generated by the protocol buffer compiler.  DO NOT EDIT!
      class Protobuf {};
      )cpp";
  TU.HeaderFilename = "protobuf.pb.h";
  auto AST = TU.build();
  auto Results = rename({Code.point(), "newName", AST, testPath(TU.Filename)});
  EXPECT_FALSE(Results);
  EXPECT_THAT(llvm::toString(Results.takeError()),
              testing::HasSubstr("not a supported kind"));
}

TEST(RenameTest, PrepareRename) {
  Annotations FooH("void func();");
  Annotations FooCC(R"cpp(
    #include "foo.h"
    void [[fu^nc]]() {}
  )cpp");
  std::string FooHPath = testPath("foo.h");
  std::string FooCCPath = testPath("foo.cc");
  MockFS FS;
  FS.Files[FooHPath] = std::string(FooH.code());
  FS.Files[FooCCPath] = std::string(FooCC.code());

  auto ServerOpts = ClangdServer::optsForTest();
  ServerOpts.BuildDynamicSymbolIndex = true;

  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, FS, ServerOpts);
  runAddDocument(Server, FooHPath, FooH.code());
  runAddDocument(Server, FooCCPath, FooCC.code());

  auto Results =
      runPrepareRename(Server, FooCCPath, FooCC.point(), {/*CrossFile=*/true});
  // verify that for multi-file rename, we only return main-file occurrences.
  ASSERT_TRUE(bool(Results)) << Results.takeError();
  // We don't know the result is complete in prepareRename (passing a nullptr
  // index internally), so GlobalChanges should be empty.
  EXPECT_TRUE(Results->GlobalChanges.empty());
  EXPECT_THAT(FooCC.ranges(),
              testing::UnorderedElementsAreArray(Results->LocalChanges));

  // single-file rename on global symbols, we should report an error.
  Results =
      runPrepareRename(Server, FooCCPath, FooCC.point(), {/*CrossFile=*/false});
  EXPECT_FALSE(Results);
  EXPECT_THAT(llvm::toString(Results.takeError()),
              testing::HasSubstr("is used outside"));
}

TEST(CrossFileRenameTests, DirtyBuffer) {
  Annotations FooCode("class [[Foo]] {};");
  std::string FooPath = testPath("foo.cc");
  Annotations FooDirtyBuffer("class [[Foo]] {};\n// this is dirty buffer");
  Annotations BarCode("void [[Bar]]() {}");
  std::string BarPath = testPath("bar.cc");
  // Build the index, the index has "Foo" references from foo.cc and "Bar"
  // references from bar.cc.
  FileSymbols FSymbols;
  FSymbols.update(FooPath, nullptr, buildRefSlab(FooCode, "Foo", FooPath),
                  nullptr, false);
  FSymbols.update(BarPath, nullptr, buildRefSlab(BarCode, "Bar", BarPath),
                  nullptr, false);
  auto Index = FSymbols.buildIndex(IndexType::Light);

  Annotations MainCode("class  [[Fo^o]] {};");
  auto MainFilePath = testPath("main.cc");
  // Dirty buffer for foo.cc.
  auto GetDirtyBuffer = [&](PathRef Path) -> llvm::Optional<std::string> {
    if (Path == FooPath)
      return FooDirtyBuffer.code().str();
    return llvm::None;
  };

  // Run rename on Foo, there is a dirty buffer for foo.cc, rename should
  // respect the dirty buffer.
  TestTU TU = TestTU::withCode(MainCode.code());
  auto AST = TU.build();
  llvm::StringRef NewName = "newName";
  auto Results = rename({MainCode.point(),
                         NewName,
                         AST,
                         MainFilePath,
                         Index.get(),
                         {/*CrossFile=*/true},
                         GetDirtyBuffer});
  ASSERT_TRUE(bool(Results)) << Results.takeError();
  EXPECT_THAT(
      applyEdits(std::move(Results->GlobalChanges)),
      UnorderedElementsAre(
          Pair(Eq(FooPath), Eq(expectedResult(FooDirtyBuffer, NewName))),
          Pair(Eq(MainFilePath), Eq(expectedResult(MainCode, NewName)))));

  // Run rename on Bar, there is no dirty buffer for the affected file bar.cc,
  // so we should read file content from VFS.
  MainCode = Annotations("void [[Bar]]() { [[B^ar]](); }");
  TU = TestTU::withCode(MainCode.code());
  // Set a file "bar.cc" on disk.
  TU.AdditionalFiles["bar.cc"] = std::string(BarCode.code());
  AST = TU.build();
  Results = rename({MainCode.point(),
                    NewName,
                    AST,
                    MainFilePath,
                    Index.get(),
                    {/*CrossFile=*/true},
                    GetDirtyBuffer});
  ASSERT_TRUE(bool(Results)) << Results.takeError();
  EXPECT_THAT(
      applyEdits(std::move(Results->GlobalChanges)),
      UnorderedElementsAre(
          Pair(Eq(BarPath), Eq(expectedResult(BarCode, NewName))),
          Pair(Eq(MainFilePath), Eq(expectedResult(MainCode, NewName)))));

  // Run rename on a pagination index which couldn't return all refs in one
  // request, we reject rename on this case.
  class PaginationIndex : public SymbolIndex {
    bool refs(const RefsRequest &Req,
              llvm::function_ref<void(const Ref &)> Callback) const override {
      return true; // has more references
    }

    bool fuzzyFind(
        const FuzzyFindRequest &Req,
        llvm::function_ref<void(const Symbol &)> Callback) const override {
      return false;
    }
    void
    lookup(const LookupRequest &Req,
           llvm::function_ref<void(const Symbol &)> Callback) const override {}

    void relations(const RelationsRequest &Req,
                   llvm::function_ref<void(const SymbolID &, const Symbol &)>
                       Callback) const override {}
    size_t estimateMemoryUsage() const override { return 0; }
  } PIndex;
  Results = rename({MainCode.point(),
                    NewName,
                    AST,
                    MainFilePath,
                    &PIndex,
                    {/*CrossFile=*/true},
                    GetDirtyBuffer});
  EXPECT_FALSE(Results);
  EXPECT_THAT(llvm::toString(Results.takeError()),
              testing::HasSubstr("too many occurrences"));
}

TEST(CrossFileRenameTests, DeduplicateRefsFromIndex) {
  auto MainCode = Annotations("int [[^x]] = 2;");
  auto MainFilePath = testPath("main.cc");
  auto BarCode = Annotations("int [[x]];");
  auto BarPath = testPath("bar.cc");
  auto TU = TestTU::withCode(MainCode.code());
  // Set a file "bar.cc" on disk.
  TU.AdditionalFiles["bar.cc"] = std::string(BarCode.code());
  auto AST = TU.build();
  std::string BarPathURI = URI::create(BarPath).toString();
  Ref XRefInBarCC = refWithRange(BarCode.range(), BarPathURI);
  // The index will return duplicated refs, our code should be robost to handle
  // it.
  class DuplicatedXRefIndex : public SymbolIndex {
  public:
    DuplicatedXRefIndex(const Ref &ReturnedRef) : ReturnedRef(ReturnedRef) {}
    bool refs(const RefsRequest &Req,
              llvm::function_ref<void(const Ref &)> Callback) const override {
      // Return two duplicated refs.
      Callback(ReturnedRef);
      Callback(ReturnedRef);
      return false;
    }

    bool fuzzyFind(const FuzzyFindRequest &,
                   llvm::function_ref<void(const Symbol &)>) const override {
      return false;
    }
    void lookup(const LookupRequest &,
                llvm::function_ref<void(const Symbol &)>) const override {}

    void relations(const RelationsRequest &,
                   llvm::function_ref<void(const SymbolID &, const Symbol &)>)
        const override {}
    size_t estimateMemoryUsage() const override { return 0; }
    Ref ReturnedRef;
  } DIndex(XRefInBarCC);
  llvm::StringRef NewName = "newName";
  auto Results = rename({MainCode.point(),
                         NewName,
                         AST,
                         MainFilePath,
                         &DIndex,
                         {/*CrossFile=*/true}});
  ASSERT_TRUE(bool(Results)) << Results.takeError();
  EXPECT_THAT(
      applyEdits(std::move(Results->GlobalChanges)),
      UnorderedElementsAre(
          Pair(Eq(BarPath), Eq(expectedResult(BarCode, NewName))),
          Pair(Eq(MainFilePath), Eq(expectedResult(MainCode, NewName)))));
}

TEST(CrossFileRenameTests, WithUpToDateIndex) {
  MockCompilationDatabase CDB;
  CDB.ExtraClangFlags = {"-xc++"};
  // rename is runnning on all "^" points in FooH, and "[[]]" ranges are the
  // expected rename occurrences.
  struct Case {
    llvm::StringRef FooH;
    llvm::StringRef FooCC;
  } Cases[] = {
      {
          // classes.
          R"cpp(
        class [[Fo^o]] {
          [[Foo]]();
          ~[[Foo]]();
        };
      )cpp",
          R"cpp(
        #include "foo.h"
        [[Foo]]::[[Foo]]() {}
        [[Foo]]::~[[Foo]]() {}

        void func() {
          [[Foo]] foo;
        }
      )cpp",
      },
      {
          // class templates.
          R"cpp(
        template <typename T>
        class [[Foo]] {};
        // FIXME: explicit template specializations are not supported due the
        // clangd index limitations.
        template <>
        class Foo<double> {};
      )cpp",
          R"cpp(
        #include "foo.h"
        void func() {
          [[F^oo]]<int> foo;
        }
      )cpp",
      },
      {
          // class methods.
          R"cpp(
        class Foo {
          void [[f^oo]]();
        };
      )cpp",
          R"cpp(
        #include "foo.h"
        void Foo::[[foo]]() {}

        void func(Foo* p) {
          p->[[foo]]();
        }
      )cpp",
      },
      {
          // rename on constructor and destructor.
          R"cpp(
        class [[Foo]] {
          [[^Foo]]();
          ~[[Foo^]]();
        };
      )cpp",
          R"cpp(
        #include "foo.h"
        [[Foo]]::[[Foo]]() {}
        [[Foo]]::~[[Foo]]() {}

        void func() {
          [[Foo]] foo;
        }
      )cpp",
      },
      {
          // functions.
          R"cpp(
        void [[f^oo]]();
      )cpp",
          R"cpp(
        #include "foo.h"
        void [[foo]]() {}

        void func() {
          [[foo]]();
        }
      )cpp",
      },
      {
          // typedefs.
          R"cpp(
      typedef int [[IN^T]];
      [[INT]] foo();
      )cpp",
          R"cpp(
        #include "foo.h"
        [[INT]] foo() {}
      )cpp",
      },
      {
          // usings.
          R"cpp(
      using [[I^NT]] = int;
      [[INT]] foo();
      )cpp",
          R"cpp(
        #include "foo.h"
        [[INT]] foo() {}
      )cpp",
      },
      {
          // variables.
          R"cpp(
        static const int [[VA^R]] = 123;
      )cpp",
          R"cpp(
        #include "foo.h"
        int s = [[VAR]];
      )cpp",
      },
      {
          // scope enums.
          R"cpp(
      enum class [[K^ind]] { ABC };
      )cpp",
          R"cpp(
        #include "foo.h"
        [[Kind]] ff() {
          return [[Kind]]::ABC;
        }
      )cpp",
      },
      {
          // enum constants.
          R"cpp(
      enum class Kind { [[A^BC]] };
      )cpp",
          R"cpp(
        #include "foo.h"
        Kind ff() {
          return Kind::[[ABC]];
        }
      )cpp",
      },
      {
          // Implicit references in macro expansions.
          R"cpp(
        class [[Fo^o]] {};
        #define FooFoo Foo
        #define FOO Foo
      )cpp",
          R"cpp(
        #include "foo.h"
        void bar() {
          [[Foo]] x;
          FOO y;
          FooFoo z;
        }
      )cpp",
      },
  };

  trace::TestTracer Tracer;
  for (const auto &T : Cases) {
    SCOPED_TRACE(T.FooH);
    Annotations FooH(T.FooH);
    Annotations FooCC(T.FooCC);
    std::string FooHPath = testPath("foo.h");
    std::string FooCCPath = testPath("foo.cc");

    MockFS FS;
    FS.Files[FooHPath] = std::string(FooH.code());
    FS.Files[FooCCPath] = std::string(FooCC.code());

    auto ServerOpts = ClangdServer::optsForTest();
    ServerOpts.BuildDynamicSymbolIndex = true;
    ClangdServer Server(CDB, FS, ServerOpts);

    // Add all files to clangd server to make sure the dynamic index has been
    // built.
    runAddDocument(Server, FooHPath, FooH.code());
    runAddDocument(Server, FooCCPath, FooCC.code());

    llvm::StringRef NewName = "NewName";
    for (const auto &RenamePos : FooH.points()) {
      EXPECT_THAT(Tracer.takeMetric("rename_files"), SizeIs(0));
      auto FileEditsList = llvm::cantFail(runRename(
          Server, FooHPath, RenamePos, NewName, {/*CrossFile=*/true}));
      EXPECT_THAT(Tracer.takeMetric("rename_files"), ElementsAre(2));
      EXPECT_THAT(
          applyEdits(std::move(FileEditsList.GlobalChanges)),
          UnorderedElementsAre(
              Pair(Eq(FooHPath), Eq(expectedResult(T.FooH, NewName))),
              Pair(Eq(FooCCPath), Eq(expectedResult(T.FooCC, NewName)))));
    }
  }
}

TEST(CrossFileRenameTests, CrossFileOnLocalSymbol) {
  // cross-file rename should work for function-local symbols, even there is no
  // index provided.
  Annotations Code("void f(int [[abc]]) { [[a^bc]] = 3; }");
  auto TU = TestTU::withCode(Code.code());
  auto Path = testPath(TU.Filename);
  auto AST = TU.build();
  llvm::StringRef NewName = "newName";
  auto Results = rename({Code.point(), NewName, AST, Path});
  ASSERT_TRUE(bool(Results)) << Results.takeError();
  EXPECT_THAT(
      applyEdits(std::move(Results->GlobalChanges)),
      UnorderedElementsAre(Pair(Eq(Path), Eq(expectedResult(Code, NewName)))));
}

TEST(CrossFileRenameTests, BuildRenameEdits) {
  Annotations Code("[[😂]]");
  auto LSPRange = Code.range();
  llvm::StringRef FilePath = "/test/TestTU.cpp";
  llvm::StringRef NewName = "abc";
  auto Edit = buildRenameEdit(FilePath, Code.code(), {LSPRange}, NewName);
  ASSERT_TRUE(bool(Edit)) << Edit.takeError();
  ASSERT_EQ(1UL, Edit->Replacements.size());
  EXPECT_EQ(FilePath, Edit->Replacements.begin()->getFilePath());
  EXPECT_EQ(4UL, Edit->Replacements.begin()->getLength());

  // Test invalid range.
  LSPRange.end = {10, 0}; // out of range
  Edit = buildRenameEdit(FilePath, Code.code(), {LSPRange}, NewName);
  EXPECT_FALSE(Edit);
  EXPECT_THAT(llvm::toString(Edit.takeError()),
              testing::HasSubstr("fail to convert"));

  // Normal ascii characters.
  Annotations T(R"cpp(
    [[range]]
              [[range]]
      [[range]]
  )cpp");
  Edit = buildRenameEdit(FilePath, T.code(), T.ranges(), NewName);
  ASSERT_TRUE(bool(Edit)) << Edit.takeError();
  EXPECT_EQ(applyEdits(FileEdits{{T.code(), std::move(*Edit)}}).front().second,
            expectedResult(T, NewName));
}

TEST(CrossFileRenameTests, adjustRenameRanges) {
  // Ranges in IndexedCode indicate the indexed occurrences;
  // ranges in DraftCode indicate the expected mapped result, empty indicates
  // we expect no matched result found.
  struct {
    llvm::StringRef IndexedCode;
    llvm::StringRef DraftCode;
  } Tests[] = {
    {
      // both line and column are changed, not a near miss.
      R"cpp(
        int [[x]] = 0;
      )cpp",
      R"cpp(
        // insert a line.
        double x = 0;
      )cpp",
    },
    {
      // subset.
      R"cpp(
        int [[x]] = 0;
      )cpp",
      R"cpp(
        int [[x]] = 0;
        {int x = 0; }
      )cpp",
    },
    {
      // shift columns.
      R"cpp(int [[x]] = 0; void foo(int x);)cpp",
      R"cpp(double [[x]] = 0; void foo(double x);)cpp",
    },
    {
      // shift lines.
      R"cpp(
        int [[x]] = 0;
        void foo(int x);
      )cpp",
      R"cpp(
        // insert a line.
        int [[x]] = 0;
        void foo(int x);
      )cpp",
    },
  };
  LangOptions LangOpts;
  LangOpts.CPlusPlus = true;
  for (const auto &T : Tests) {
    SCOPED_TRACE(T.DraftCode);
    Annotations Draft(T.DraftCode);
    auto ActualRanges = adjustRenameRanges(
        Draft.code(), "x", Annotations(T.IndexedCode).ranges(), LangOpts);
    if (!ActualRanges)
       EXPECT_THAT(Draft.ranges(), testing::IsEmpty());
    else
      EXPECT_THAT(Draft.ranges(),
                  testing::UnorderedElementsAreArray(*ActualRanges));
  }
}

TEST(RangePatchingHeuristic, GetMappedRanges) {
  // ^ in LexedCode marks the ranges we expect to be mapped; no ^ indicates
  // there are no mapped ranges.
  struct {
    llvm::StringRef IndexedCode;
    llvm::StringRef LexedCode;
  } Tests[] = {
    {
      // no lexed ranges.
      "[[]]",
      "",
    },
    {
      // both line and column are changed, not a near miss.
      R"([[]])",
      R"(
        [[]]
      )",
    },
    {
      // subset.
      "[[]]",
      "^[[]]  [[]]"
    },
    {
      // shift columns.
      "[[]]   [[]]",
      "  ^[[]]   ^[[]]  [[]]"
    },
    {
      R"(
        [[]]

        [[]] [[]]
      )",
      R"(
        // insert a line
        ^[[]]

        ^[[]] ^[[]]
      )",
    },
    {
      R"(
        [[]]

        [[]] [[]]
      )",
      R"(
        // insert a line
        ^[[]]
          ^[[]]  ^[[]] // column is shifted.
      )",
    },
    {
      R"(
        [[]]

        [[]] [[]]
      )",
      R"(
        // insert a line
        [[]]

          [[]]  [[]] // not mapped (both line and column are changed).
      )",
    },
    {
      R"(
        [[]]
                [[]]

                   [[]]
                  [[]]

        }
      )",
      R"(
        // insert a new line
        ^[[]]
                ^[[]]
             [[]] // additional range
                   ^[[]]
                  ^[[]]
            [[]] // additional range
      )",
    },
    {
      // non-distinct result (two best results), not a near miss
      R"(
        [[]]
            [[]]
            [[]]
      )",
      R"(
        [[]]
        [[]]
            [[]]
            [[]]
      )",
    }
  };
  for (const auto &T : Tests) {
    SCOPED_TRACE(T.IndexedCode);
    auto Lexed = Annotations(T.LexedCode);
    auto LexedRanges = Lexed.ranges();
    std::vector<Range> ExpectedMatches;
    for (auto P : Lexed.points()) {
      auto Match = llvm::find_if(LexedRanges, [&P](const Range& R) {
        return R.start == P;
      });
      ASSERT_NE(Match, LexedRanges.end());
      ExpectedMatches.push_back(*Match);
    }

    auto Mapped =
        getMappedRanges(Annotations(T.IndexedCode).ranges(), LexedRanges);
    if (!Mapped)
      EXPECT_THAT(ExpectedMatches, IsEmpty());
    else
      EXPECT_THAT(ExpectedMatches, UnorderedElementsAreArray(*Mapped));
  }
}

TEST(CrossFileRenameTests, adjustmentCost) {
  struct {
    llvm::StringRef RangeCode;
    size_t ExpectedCost;
  } Tests[] = {
    {
      R"(
        $idx[[]]$lex[[]] // diff: 0
      )",
      0,
    },
    {
      R"(
        $idx[[]]
        $lex[[]] // line diff: +1
                       $idx[[]]
                       $lex[[]] // line diff: +1
        $idx[[]]
        $lex[[]] // line diff: +1

          $idx[[]]

          $lex[[]] // line diff: +2
      )",
      1 + 1
    },
    {
       R"(
        $idx[[]]
        $lex[[]] // line diff: +1
                       $idx[[]]

                       $lex[[]] // line diff: +2
        $idx[[]]


        $lex[[]] // line diff: +3
      )",
      1 + 1 + 1
    },
    {
       R"(
        $idx[[]]


        $lex[[]] // line diff: +3
                       $idx[[]]

                       $lex[[]] // line diff: +2
        $idx[[]]
        $lex[[]] // line diff: +1
      )",
      3 + 1 + 1
    },
    {
      R"(
        $idx[[]]
        $lex[[]] // line diff: +1
                       $lex[[]] // line diff: -2

                       $idx[[]]
        $idx[[]]


        $lex[[]] // line diff: +3
      )",
      1 + 3 + 5
    },
    {
      R"(
                       $idx[[]] $lex[[]] // column diff: +1
        $idx[[]]$lex[[]] // diff: 0
      )",
      1
    },
    {
      R"(
        $idx[[]]
        $lex[[]] // diff: +1
                       $idx[[]] $lex[[]] // column diff: +1
        $idx[[]]$lex[[]] // diff: 0
      )",
      1 + 1 + 1
    },
    {
      R"(
        $idx[[]] $lex[[]] // column diff: +1
      )",
      1
    },
    {
      R"(
        // column diffs: +1, +2, +3
        $idx[[]] $lex[[]] $idx[[]]  $lex[[]] $idx[[]]   $lex[[]]
      )",
      1 + 1 + 1,
    },
  };
  for (const auto &T : Tests) {
    SCOPED_TRACE(T.RangeCode);
    Annotations C(T.RangeCode);
    std::vector<size_t> MappedIndex;
    for (size_t I = 0; I < C.ranges("lex").size(); ++I)
      MappedIndex.push_back(I);
    EXPECT_EQ(renameRangeAdjustmentCost(C.ranges("idx"), C.ranges("lex"),
                                        MappedIndex),
              T.ExpectedCost);
  }
}

} // namespace
} // namespace clangd
} // namespace clang
