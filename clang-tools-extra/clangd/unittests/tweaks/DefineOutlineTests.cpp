//===-- DefineOutline.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestTU.h"
#include "TweakTesting.h"
#include "gmock/gmock-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::ElementsAre;

namespace clang {
namespace clangd {
namespace {

TWEAK_TEST(DefineOutline);

TEST_F(DefineOutlineTest, TriggersOnFunctionDecl) {
  FileName = "Test.cpp";
  // Not available unless in a header file.
  EXPECT_UNAVAILABLE(R"cpp(
    [[void [[f^o^o]]() [[{
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

  // Not available within templated classes, as it is hard to spell class name
  // out-of-line in such cases.
  EXPECT_UNAVAILABLE(R"cpp(
    template <typename> struct Foo { void fo^o(){} };
    )cpp");

  // Not available on function templates and specializations, as definition must
  // be visible to all translation units.
  EXPECT_UNAVAILABLE(R"cpp(
    template <typename> void fo^o() {};
    template <> void fo^o<int>() {};
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
  llvm::StringMap<std::string> EditedFiles;
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
      // Default args.
      {
          "void fo^o(int x, int y = 5, int = 2, int (*foo)(int) = nullptr) {}",
          "void foo(int x, int y = 5, int = 2, int (*foo)(int) = nullptr) ;",
          "void foo(int x, int y , int , int (*foo)(int) ) {}",
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
              class Foo {
                F^oo(int z) __attribute__((weak)) : bar(2){}
                int bar;
              };)cpp",
          R"cpp(
              class Foo {
                Foo(int z) __attribute__((weak)) ;
                int bar;
              };)cpp",
          "Foo::Foo(int z) __attribute__((weak)) : bar(2){}\n",
      },
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
  };
  for (const auto &Case : Cases) {
    SCOPED_TRACE(Case.Test);
    EXPECT_EQ(apply(Case.Test, &EditedFiles), Case.ExpectedHeader);
    EXPECT_THAT(EditedFiles, testing::ElementsAre(FileWithContents(
                                 testPath("Test.cpp"), Case.ExpectedSource)));
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
