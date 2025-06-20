//===-- DefineOutline.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestFS.h"
#include "TestIndex.h"
#include "TweakTesting.h"
#include "index/MemIndex.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TWEAK_TEST(DefineOutline);

TEST_F(DefineOutlineTest, TriggersOnFunctionDecl) {
  FileName = "Test.cpp";
  // Not available for free function unless in a header file.
  EXPECT_UNAVAILABLE(R"cpp(
    [[void [[f^o^o]]() [[{
      return;
    }]]]])cpp");

  // Available in soure file.
  EXPECT_AVAILABLE(R"cpp(
    struct Foo {
      void f^oo() {}
    };
  )cpp");

  // Available within named namespace in source file.
  EXPECT_AVAILABLE(R"cpp(
    namespace N {
      struct Foo {
        void f^oo() {}
      };
    } // namespace N
  )cpp");

  // Available within anonymous namespace in source file.
  EXPECT_AVAILABLE(R"cpp(
    namespace {
      struct Foo {
        void f^oo() {}
      };
    } // namespace
  )cpp");

  // Not available for out-of-line method.
  EXPECT_UNAVAILABLE(R"cpp(
    class Bar {
      void baz();
    };

    [[void [[Bar::[[b^a^z]]]]() [[{
      return;
    }]]]])cpp");

  FileName = "Test.hpp";
  // Not available unless function name or fully body is selected.
  EXPECT_UNAVAILABLE(R"cpp(
    // Not a definition
    vo^i[[d^ ^f]]^oo();

    [[vo^id ]]foo[[()]] {[[
      [[(void)(5+3);
      return;]]
    }]])cpp");

  // Available even if there are no implementation files.
  EXPECT_AVAILABLE(R"cpp(
    [[void [[f^o^o]]() [[{
      return;
    }]]]])cpp");

  // Not available for out-of-line methods.
  EXPECT_UNAVAILABLE(R"cpp(
    class Bar {
      void baz();
    };

    [[void [[Bar::[[b^a^z]]]]() [[{
      return;
    }]]]])cpp");

  // Basic check for function body and signature.
  EXPECT_AVAILABLE(R"cpp(
    class Bar {
      [[void [[f^o^o^]]() [[{ return; }]]]]
    };

    void foo();
    [[void [[f^o^o]]() [[{
      return;
    }]]]])cpp");

  // Not available on defaulted/deleted members.
  EXPECT_UNAVAILABLE(R"cpp(
    class Foo {
      Fo^o() = default;
      F^oo(const Foo&) = delete;
    };)cpp");

  // Not available within templated classes with unnamed parameters, as it is
  // hard to spell class name out-of-line in such cases.
  EXPECT_UNAVAILABLE(R"cpp(
    template <typename> struct Foo { void fo^o(){} };
    )cpp");

  // Not available on function template specializations and free function
  // templates.
  EXPECT_UNAVAILABLE(R"cpp(
    template <typename T> void fo^o() {}
    template <> void fo^o<int>() {}
  )cpp");

  // Not available on methods of unnamed classes.
  EXPECT_UNAVAILABLE(R"cpp(
    struct Foo {
      struct { void b^ar() {} } Bar;
    };
  )cpp");

  // Not available on methods of named classes with unnamed parent in parents
  // nesting.
  EXPECT_UNAVAILABLE(R"cpp(
    struct Foo {
      struct {
        struct Bar { void b^ar() {} };
      } Baz;
    };
  )cpp");

  // Not available on definitions in header file within unnamed namespaces
  EXPECT_UNAVAILABLE(R"cpp(
    namespace {
      struct Foo {
        void f^oo() {}
      };
    } // namespace
  )cpp");
}

TEST_F(DefineOutlineTest, FailsWithoutSource) {
  FileName = "Test.hpp";
  llvm::StringRef Test = "void fo^o() { return; }";
  llvm::StringRef Expected =
      "fail: Couldn't find a suitable implementation file.";
  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DefineOutlineTest, ApplyTest) {
  ExtraFiles["Test.cpp"] = "";
  FileName = "Test.hpp";

  struct {
    llvm::StringRef Test;
    llvm::StringRef ExpectedHeader;
    llvm::StringRef ExpectedSource;
  } Cases[] = {
      // Simple check
      {
          "void fo^o() { return; }",
          "void foo() ;",
          "void foo() { return; }",
      },
      // Inline specifier.
      {
          "inline void fo^o() { return; }",
          " void foo() ;",
          " void foo() { return; }",
      },
      // Default args.
      {
          "void fo^o(int x, int y = 5, int = 2, int (*foo)(int) = nullptr) {}",
          "void foo(int x, int y = 5, int = 2, int (*foo)(int) = nullptr) ;",
          "void foo(int x, int y , int , int (*foo)(int) ) {}",
      },
      {
          "struct Bar{Bar();}; void fo^o(Bar x = {}) {}",
          "struct Bar{Bar();}; void foo(Bar x = {}) ;",
          "void foo(Bar x ) {}",
      },
      // Constructors
      {
          R"cpp(
            class Foo {public: Foo(); Foo(int);};
            class Bar {
              Ba^r() {}
              Bar(int x) : f1(x) {}
              Foo f1;
              Foo f2 = 2;
            };)cpp",
          R"cpp(
            class Foo {public: Foo(); Foo(int);};
            class Bar {
              Bar() ;
              Bar(int x) : f1(x) {}
              Foo f1;
              Foo f2 = 2;
            };)cpp",
          "Bar::Bar() {}\n",
      },
      // Ctor with initializer.
      {
          R"cpp(
            class Foo {public: Foo(); Foo(int);};
            class Bar {
              Bar() {}
              B^ar(int x) : f1(x), f2(3) {}
              Foo f1;
              Foo f2 = 2;
            };)cpp",
          R"cpp(
            class Foo {public: Foo(); Foo(int);};
            class Bar {
              Bar() {}
              Bar(int x) ;
              Foo f1;
              Foo f2 = 2;
            };)cpp",
          "Bar::Bar(int x) : f1(x), f2(3) {}\n",
      },
      // Ctor initializer with attribute.
      {
          R"cpp(
              template <typename T> class Foo {
                F^oo(T z) __attribute__((weak)) : bar(2){}
                int bar;
              };)cpp",
          R"cpp(
              template <typename T> class Foo {
                Foo(T z) __attribute__((weak)) ;
                int bar;
              };template <typename T>
inline Foo<T>::Foo(T z) __attribute__((weak)) : bar(2){}
)cpp",
          ""},
      // Virt specifiers.
      {
          R"cpp(
            struct A {
              virtual void f^oo() {}
            };)cpp",
          R"cpp(
            struct A {
              virtual void foo() ;
            };)cpp",
          " void A::foo() {}\n",
      },
      {
          R"cpp(
            struct A {
              virtual virtual void virtual f^oo() {}
            };)cpp",
          R"cpp(
            struct A {
              virtual virtual void virtual foo() ;
            };)cpp",
          "  void  A::foo() {}\n",
      },
      {
          R"cpp(
            struct A {
              virtual void foo() = 0;
            };
            struct B : A {
              void fo^o() override {}
            };)cpp",
          R"cpp(
            struct A {
              virtual void foo() = 0;
            };
            struct B : A {
              void foo() override ;
            };)cpp",
          "void B::foo()  {}\n",
      },
      {
          R"cpp(
            struct A {
              virtual void foo() = 0;
            };
            struct B : A {
              void fo^o() final {}
            };)cpp",
          R"cpp(
            struct A {
              virtual void foo() = 0;
            };
            struct B : A {
              void foo() final ;
            };)cpp",
          "void B::foo()  {}\n",
      },
      {
          R"cpp(
            struct A {
              virtual void foo() = 0;
            };
            struct B : A {
              void fo^o() final override {}
            };)cpp",
          R"cpp(
            struct A {
              virtual void foo() = 0;
            };
            struct B : A {
              void foo() final override ;
            };)cpp",
          "void B::foo()   {}\n",
      },
      {
          R"cpp(
            struct A {
              static void fo^o() {}
            };)cpp",
          R"cpp(
            struct A {
              static void foo() ;
            };)cpp",
          " void A::foo() {}\n",
      },
      {
          R"cpp(
            struct A {
              static static void fo^o() {}
            };)cpp",
          R"cpp(
            struct A {
              static static void foo() ;
            };)cpp",
          "  void A::foo() {}\n",
      },
      {
          R"cpp(
            struct Foo {
              explicit Fo^o(int) {}
            };)cpp",
          R"cpp(
            struct Foo {
              explicit Foo(int) ;
            };)cpp",
          " Foo::Foo(int) {}\n",
      },
      {
          R"cpp(
            struct Foo {
              explicit explicit Fo^o(int) {}
            };)cpp",
          R"cpp(
            struct Foo {
              explicit explicit Foo(int) ;
            };)cpp",
          "  Foo::Foo(int) {}\n",
      },
      {
          R"cpp(
            struct A {
              inline void f^oo(int) {}
            };)cpp",
          R"cpp(
            struct A {
               void foo(int) ;
            };)cpp",
          " void A::foo(int) {}\n",
      },
      // Complex class template
      {
          R"cpp(
            template <typename T, typename ...U> struct O1 {
              template <class V, int A> struct O2 {
                enum E { E1, E2 };
                struct I {
                  E f^oo(T, U..., V, E) { return E1; }
                };
              };
            };)cpp",
          R"cpp(
            template <typename T, typename ...U> struct O1 {
              template <class V, int A> struct O2 {
                enum E { E1, E2 };
                struct I {
                  E foo(T, U..., V, E) ;
                };
              };
            };template <typename T, typename ...U>
template <class V, int A>
inline typename O1<T, U...>::template O2<V, A>::E O1<T, U...>::template O2<V, A>::I::foo(T, U..., V, E) { return E1; }
)cpp",
          ""},
      // Destructors
      {
          "class A { ~A^(){} };",
          "class A { ~A(); };",
          "A::~A(){} ",
      },

      // Member template
      {
          R"cpp(
            struct Foo {
              template <typename T, typename, bool B = true>
              T ^bar() { return {}; }
            };)cpp",
          R"cpp(
            struct Foo {
              template <typename T, typename, bool B = true>
              T bar() ;
            };template <typename T, typename, bool B>
inline T Foo::bar() { return {}; }
)cpp",
          ""},

      // Class template with member template
      {
          R"cpp(
            template <typename T> struct Foo {
              template <typename U, bool> T ^bar(const T& t, const U& u) { return {}; }
            };)cpp",
          R"cpp(
            template <typename T> struct Foo {
              template <typename U, bool> T bar(const T& t, const U& u) ;
            };template <typename T>
template <typename U, bool>
inline T Foo<T>::bar(const T& t, const U& u) { return {}; }
)cpp",
          ""},
  };
  for (const auto &Case : Cases) {
    SCOPED_TRACE(Case.Test);
    llvm::StringMap<std::string> EditedFiles;
    EXPECT_EQ(apply(Case.Test, &EditedFiles), Case.ExpectedHeader);
    if (Case.ExpectedSource.empty()) {
      EXPECT_TRUE(EditedFiles.empty());
    } else {
      EXPECT_THAT(EditedFiles, testing::ElementsAre(FileWithContents(
                                   testPath("Test.cpp"), Case.ExpectedSource)));
    }
  }
}

TEST_F(DefineOutlineTest, InCppFile) {
  FileName = "Test.cpp";

  const auto MakePos = [](uint32_t Line, uint32_t Col) {
    SymbolLocation::Position Pos;
    Pos.setLine(Line);
    Pos.setColumn(Col);
    return Pos;
  };
  struct SymbolSpec {
    StringRef NamespaceName;
    StringRef ClassName;
    StringRef FuncName;
    SymbolLocation::Position DeclStart;
    SymbolLocation::Position DeclEnd;
    SymbolLocation::Position DefStart;
    SymbolLocation::Position DefEnd;
  };
  using SymbolSpecs = std::vector<SymbolSpec>;

  struct {
    llvm::StringRef Test;
    llvm::StringRef ExpectedSource;
    SymbolSpecs Definitions;
  } Cases[] = {
      {
          R"cpp(
            namespace foo {
            namespace {
            struct Foo { void ba^r() {} };
            struct Bar { void foo(); };
            void Bar::foo() {}
            }
            }
        )cpp",
          R"cpp(
            namespace foo {
            namespace {
            struct Foo { void bar() ; };void Foo::bar() {} 
            struct Bar { void foo(); };
            void Bar::foo() {}
            }
            }
        )cpp"},

      // Criterion 1: Distance
      {
          R"cpp(
            struct Foo {
                void ignored1();     // Too far away
                void ignored2();     // No definition
                void ignored3() {}   // Defined inline
                void fo^o() {}
                void neighbor();
            };
            void Foo::ignored1() {}
            void Foo::neighbor() {}
        )cpp",
          R"cpp(
            struct Foo {
                void ignored1();     // Too far away
                void ignored2();     // No definition
                void ignored3() {}   // Defined inline
                void foo() ;
                void neighbor();
            };
            void Foo::ignored1() {}
            void Foo::foo() {}
void Foo::neighbor() {}
        )cpp",
          SymbolSpecs{{"", "Foo", "ignored3", MakePos(4, 21), MakePos(4, 29),
                       MakePos(4, 21), MakePos(4, 29)},
                      {"", "Foo", "ignored1", MakePos(2, 21), MakePos(2, 29),
                       MakePos(8, 22), MakePos(8, 30)},
                      {"", "Foo", "neighbor", MakePos(6, 21), MakePos(6, 29),
                       MakePos(9, 22), MakePos(9, 30)}}},

      // Criterion 2: Prefer preceding
      {
          R"cpp(
            struct Foo {
                void neighbor();
                void fo^o() {}
                void ignored();
            };
            void Foo::neighbor() {}
            void Foo::ignored() {}
        )cpp",
          R"cpp(
            struct Foo {
                void neighbor();
                void foo() ;
                void ignored();
            };
            void Foo::neighbor() {}void Foo::foo() {}

            void Foo::ignored() {}
        )cpp",
          SymbolSpecs{{"", "Foo", "ignored", MakePos(4, 21), MakePos(4, 28),
                       MakePos(7, 22), MakePos(7, 29)},
                      {"", "Foo", "neighbor", MakePos(2, 22), MakePos(2, 29),
                       MakePos(6, 22), MakePos(6, 30)}}},

      // Like above, but with a namespace
      {
          R"cpp(
            namespace NS {
            struct Foo {
                void neighbor();
                void fo^o() {}
                void ignored();
            };
            void Foo::neighbor() {}
            void Foo::ignored() {}
            }
        )cpp",
          R"cpp(
            namespace NS {
            struct Foo {
                void neighbor();
                void foo() ;
                void ignored();
            };
            void Foo::neighbor() {}void Foo::foo() {}

            void Foo::ignored() {}
            }
        )cpp",
          SymbolSpecs{{"NS", "Foo", "ignored", MakePos(5, 21), MakePos(5, 29),
                       MakePos(8, 22), MakePos(8, 30)},
                      {"NS", "Foo", "neighbor", MakePos(3, 21), MakePos(3, 29),
                       MakePos(7, 22), MakePos(7, 30)}}},

      // Like above, but there is no namespace at the definition site
      {
          R"cpp(
            namespace NS {
            struct Foo {
                void neighbor();
                void fo^o() {}
                void ignored();
            };
            }
            void NS::Foo::neighbor() {}
            void NS::Foo::ignored() {}
        )cpp",
          R"cpp(
            namespace NS {
            struct Foo {
                void neighbor();
                void foo() ;
                void ignored();
            };
            }
            void NS::Foo::neighbor() {}void NS::Foo::foo() {}

            void NS::Foo::ignored() {}
        )cpp",
          SymbolSpecs{{"NS", "Foo", "ignored", MakePos(5, 21), MakePos(5, 29),
                       MakePos(9, 26), MakePos(9, 33)},
                      {"NS", "Foo", "neighbor", MakePos(3, 21), MakePos(3, 29),
                       MakePos(8, 26), MakePos(8, 34)}}},
  };

  Path FilePath =
#ifdef _WIN32
      "C:/clangd-test/"
#else
      "/clangd-test/"
#endif
      + FileName.str();
  std::string URI = URI::createFile(FilePath).toString();
  std::string FullNamespace;

  const auto BuildIndex = [&](const SymbolSpecs &Specs) {
    SymbolSlab::Builder Slab;
    for (const auto &S : Specs) {
      std::string USRFormat;
      if (!S.NamespaceName.empty())
        USRFormat += "@N@" + S.NamespaceName.str();
      if (!S.ClassName.empty())
        USRFormat += "@S@" + S.ClassName.str();
      USRFormat += "@F@\\0#";
      Symbol Sym = sym(S.FuncName, index::SymbolKind::Function, USRFormat);
      Sym.CanonicalDeclaration.FileURI = URI.data();
      Sym.CanonicalDeclaration.Start = S.DeclStart;
      Sym.CanonicalDeclaration.End = S.DeclEnd;
      Sym.Definition.FileURI = Sym.CanonicalDeclaration.FileURI;
      Sym.Definition.Start = S.DefStart;
      Sym.Definition.End = S.DefEnd;
      if (!S.NamespaceName.empty()) {
        FullNamespace = S.NamespaceName.str() + "::";
        Sym.Scope = FullNamespace;
      }
      Slab.insert(Sym);
    }
    return MemIndex::build(std::move(Slab).build(), RefSlab(), RelationSlab());
  };

  for (const auto &Case : Cases) {
    SCOPED_TRACE(Case.Test);
    Index = BuildIndex(Case.Definitions);
    EXPECT_EQ(apply(Case.Test, nullptr), Case.ExpectedSource);
  }
}

TEST_F(DefineOutlineTest, HandleMacros) {
  llvm::StringMap<std::string> EditedFiles;
  ExtraFiles["Test.cpp"] = "";
  FileName = "Test.hpp";
  ExtraArgs.push_back("-DVIRTUAL=virtual");
  ExtraArgs.push_back("-DOVER=override");

  struct {
    llvm::StringRef Test;
    llvm::StringRef ExpectedHeader;
    llvm::StringRef ExpectedSource;
  } Cases[] = {
      {R"cpp(
          #define BODY { return; }
          void f^oo()BODY)cpp",
       R"cpp(
          #define BODY { return; }
          void foo();)cpp",
       "void foo()BODY"},

      {R"cpp(
          #define BODY return;
          void f^oo(){BODY})cpp",
       R"cpp(
          #define BODY return;
          void foo();)cpp",
       "void foo(){BODY}"},

      {R"cpp(
          #define TARGET void foo()
          [[TARGET]]{ return; })cpp",
       R"cpp(
          #define TARGET void foo()
          TARGET;)cpp",
       "TARGET{ return; }"},

      {R"cpp(
          #define TARGET foo
          void [[TARGET]](){ return; })cpp",
       R"cpp(
          #define TARGET foo
          void TARGET();)cpp",
       "void TARGET(){ return; }"},
      {R"cpp(#define VIRT virtual
          struct A {
            VIRT void f^oo() {}
          };)cpp",
       R"cpp(#define VIRT virtual
          struct A {
            VIRT void foo() ;
          };)cpp",
       " void A::foo() {}\n"},
      {R"cpp(
          struct A {
            VIRTUAL void f^oo() {}
          };)cpp",
       R"cpp(
          struct A {
            VIRTUAL void foo() ;
          };)cpp",
       " void A::foo() {}\n"},
      {R"cpp(
          struct A {
            virtual void foo() = 0;
          };
          struct B : A {
            void fo^o() OVER {}
          };)cpp",
       R"cpp(
          struct A {
            virtual void foo() = 0;
          };
          struct B : A {
            void foo() OVER ;
          };)cpp",
       "void B::foo()  {}\n"},
      {R"cpp(#define STUPID_MACRO(X) virtual
          struct A {
            STUPID_MACRO(sizeof sizeof int) void f^oo() {}
          };)cpp",
       R"cpp(#define STUPID_MACRO(X) virtual
          struct A {
            STUPID_MACRO(sizeof sizeof int) void foo() ;
          };)cpp",
       " void A::foo() {}\n"},
      {R"cpp(#define STAT static
          struct A {
            STAT void f^oo() {}
          };)cpp",
       R"cpp(#define STAT static
          struct A {
            STAT void foo() ;
          };)cpp",
       " void A::foo() {}\n"},
      {R"cpp(#define STUPID_MACRO(X) static
          struct A {
            STUPID_MACRO(sizeof sizeof int) void f^oo() {}
          };)cpp",
       R"cpp(#define STUPID_MACRO(X) static
          struct A {
            STUPID_MACRO(sizeof sizeof int) void foo() ;
          };)cpp",
       " void A::foo() {}\n"},
  };
  for (const auto &Case : Cases) {
    SCOPED_TRACE(Case.Test);
    EXPECT_EQ(apply(Case.Test, &EditedFiles), Case.ExpectedHeader);
    EXPECT_THAT(EditedFiles, testing::ElementsAre(FileWithContents(
                                 testPath("Test.cpp"), Case.ExpectedSource)));
  }
}

TEST_F(DefineOutlineTest, QualifyReturnValue) {
  FileName = "Test.hpp";
  ExtraFiles["Test.cpp"] = "";

  struct {
    llvm::StringRef Test;
    llvm::StringRef ExpectedHeader;
    llvm::StringRef ExpectedSource;
  } Cases[] = {
      {R"cpp(
        namespace a { class Foo{}; }
        using namespace a;
        Foo fo^o() { return {}; })cpp",
       R"cpp(
        namespace a { class Foo{}; }
        using namespace a;
        Foo foo() ;)cpp",
       "a::Foo foo() { return {}; }"},
      {R"cpp(
        namespace a {
          class Foo {
            class Bar {};
            Bar fo^o() { return {}; }
          };
        })cpp",
       R"cpp(
        namespace a {
          class Foo {
            class Bar {};
            Bar foo() ;
          };
        })cpp",
       "a::Foo::Bar a::Foo::foo() { return {}; }\n"},
      {R"cpp(
        class Foo {};
        Foo fo^o() { return {}; })cpp",
       R"cpp(
        class Foo {};
        Foo foo() ;)cpp",
       "Foo foo() { return {}; }"},
  };
  llvm::StringMap<std::string> EditedFiles;
  for (auto &Case : Cases) {
    apply(Case.Test, &EditedFiles);
    EXPECT_EQ(apply(Case.Test, &EditedFiles), Case.ExpectedHeader);
    EXPECT_THAT(EditedFiles, testing::ElementsAre(FileWithContents(
                                 testPath("Test.cpp"), Case.ExpectedSource)));
  }
}

TEST_F(DefineOutlineTest, QualifyFunctionName) {
  FileName = "Test.hpp";
  struct {
    llvm::StringRef TestHeader;
    llvm::StringRef TestSource;
    llvm::StringRef ExpectedHeader;
    llvm::StringRef ExpectedSource;
  } Cases[] = {
      {
          R"cpp(
            namespace a {
              namespace b {
                class Foo {
                  void fo^o() {}
                };
              }
            })cpp",
          "",
          R"cpp(
            namespace a {
              namespace b {
                class Foo {
                  void foo() ;
                };
              }
            })cpp",
          "void a::b::Foo::foo() {}\n",
      },
      {
          "namespace a { namespace b { void f^oo() {} } }",
          "namespace a{}",
          "namespace a { namespace b { void foo() ; } }",
          "namespace a{void b::foo() {} }",
      },
      {
          "namespace a { namespace b { void f^oo() {} } }",
          "using namespace a;",
          "namespace a { namespace b { void foo() ; } }",
          // FIXME: Take using namespace directives in the source file into
          // account. This can be spelled as b::foo instead.
          "using namespace a;void a::b::foo() {} ",
      },
      {
          "namespace a { class A { ~A^(){} }; }",
          "",
          "namespace a { class A { ~A(); }; }",
          "a::A::~A(){} ",
      },
      {
          "namespace a { class A { ~A^(){} }; }",
          "namespace a{}",
          "namespace a { class A { ~A(); }; }",
          "namespace a{A::~A(){} }",
      },
  };
  llvm::StringMap<std::string> EditedFiles;
  for (auto &Case : Cases) {
    ExtraFiles["Test.cpp"] = std::string(Case.TestSource);
    EXPECT_EQ(apply(Case.TestHeader, &EditedFiles), Case.ExpectedHeader);
    EXPECT_THAT(EditedFiles, testing::ElementsAre(FileWithContents(
                                 testPath("Test.cpp"), Case.ExpectedSource)))
        << Case.TestHeader;
  }
}

TEST_F(DefineOutlineTest, FailsMacroSpecifier) {
  FileName = "Test.hpp";
  ExtraFiles["Test.cpp"] = "";
  ExtraArgs.push_back("-DFINALOVER=final override");

  std::pair<StringRef, StringRef> Cases[] = {
      {
          R"cpp(
          #define VIRT virtual void
          struct A {
            VIRT fo^o() {}
          };)cpp",
          "fail: define outline: couldn't remove `virtual` keyword."},
      {
          R"cpp(
          #define OVERFINAL final override
          struct A {
            virtual void foo() {}
          };
          struct B : A {
            void fo^o() OVERFINAL {}
          };)cpp",
          "fail: define outline: Can't move out of line as function has a "
          "macro `override` specifier.\ndefine outline: Can't move out of line "
          "as function has a macro `final` specifier."},
      {
          R"cpp(
          struct A {
            virtual void foo() {}
          };
          struct B : A {
            void fo^o() FINALOVER {}
          };)cpp",
          "fail: define outline: Can't move out of line as function has a "
          "macro `override` specifier.\ndefine outline: Can't move out of line "
          "as function has a macro `final` specifier."},
  };
  for (const auto &Case : Cases) {
    EXPECT_EQ(apply(Case.first), Case.second);
  }
}

} // namespace
} // namespace clangd
} // namespace clang
