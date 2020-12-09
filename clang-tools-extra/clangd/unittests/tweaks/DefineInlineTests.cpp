//===-- DefineInlineTests.cpp -----------------------------------*- C++ -*-===//
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

TWEAK_TEST(DefineInline);

TEST_F(DefineInlineTest, TriggersOnFunctionDecl) {
  // Basic check for function body and signature.
  EXPECT_AVAILABLE(R"cpp(
  class Bar {
    void baz();
  };

  [[void [[Bar::[[b^a^z]]]]() [[{
    return;
  }]]]]

  void foo();
  [[void [[f^o^o]]() [[{
    return;
  }]]]]
  )cpp");

  EXPECT_UNAVAILABLE(R"cpp(
  // Not a definition
  vo^i[[d^ ^f]]^oo();

  [[vo^id ]]foo[[()]] {[[
    [[(void)(5+3);
    return;]]
  }]]

  // Definition with no body.
  class Bar { Bar() = def^ault; };
  )cpp");
}

TEST_F(DefineInlineTest, NoForwardDecl) {
  Header = "void bar();";
  EXPECT_UNAVAILABLE(R"cpp(
  void bar() {
    return;
  }
  // FIXME: Generate a decl in the header.
  void fo^o() {
    return;
  })cpp");
}

TEST_F(DefineInlineTest, ReferencedDecls) {
  EXPECT_AVAILABLE(R"cpp(
    void bar();
    void foo(int test);

    void fo^o(int baz) {
      int x = 10;
      bar();
    })cpp");

  // Internal symbol usage.
  Header = "void foo(int test);";
  EXPECT_UNAVAILABLE(R"cpp(
    void bar();
    void fo^o(int baz) {
      int x = 10;
      bar();
    })cpp");

  // Becomes available after making symbol visible.
  Header = "void bar();" + Header;
  EXPECT_AVAILABLE(R"cpp(
    void fo^o(int baz) {
      int x = 10;
      bar();
    })cpp");

  // FIXME: Move declaration below bar to make it visible.
  Header.clear();
  EXPECT_UNAVAILABLE(R"cpp(
    void foo();
    void bar();

    void fo^o() {
      bar();
    })cpp");

  // Order doesn't matter within a class.
  EXPECT_AVAILABLE(R"cpp(
    class Bar {
      void foo();
      void bar();
    };

    void Bar::fo^o() {
      bar();
    })cpp");

  // FIXME: Perform include insertion to make symbol visible.
  ExtraFiles["a.h"] = "void bar();";
  Header = "void foo(int test);";
  EXPECT_UNAVAILABLE(R"cpp(
    #include "a.h"
    void fo^o(int baz) {
      int x = 10;
      bar();
    })cpp");
}

TEST_F(DefineInlineTest, TemplateSpec) {
  EXPECT_UNAVAILABLE(R"cpp(
    template <typename T> void foo();
    template<> void foo<char>();

    template<> void f^oo<int>() {
    })cpp");
  EXPECT_UNAVAILABLE(R"cpp(
    template <typename T> void foo();

    template<> void f^oo<int>() {
    })cpp");
  EXPECT_UNAVAILABLE(R"cpp(
    template <typename T> struct Foo { void foo(); };

    template <typename T> void Foo<T>::f^oo() {
    })cpp");
  EXPECT_AVAILABLE(R"cpp(
    template <typename T> void foo();
    void bar();
    template <> void foo<int>();

    template<> void f^oo<int>() {
      bar();
    })cpp");
  EXPECT_UNAVAILABLE(R"cpp(
    namespace bar {
      template <typename T> void f^oo() {}
      template void foo<int>();
    })cpp");
}

TEST_F(DefineInlineTest, CheckForCanonDecl) {
  EXPECT_UNAVAILABLE(R"cpp(
    void foo();

    void bar() {}
    void f^oo() {
      // This bar normally refers to the definition just above, but it is not
      // visible from the forward declaration of foo.
      bar();
    })cpp");
  // Make it available with a forward decl.
  EXPECT_AVAILABLE(R"cpp(
    void bar();
    void foo();

    void bar() {}
    void f^oo() {
      bar();
    })cpp");
}

TEST_F(DefineInlineTest, UsingShadowDecls) {
  // Template body is not parsed until instantiation time on windows, which
  // results in arbitrary failures as function body becomes NULL.
  ExtraArgs.push_back("-fno-delayed-template-parsing");
  EXPECT_UNAVAILABLE(R"cpp(
  namespace ns1 { void foo(int); }
  namespace ns2 { void foo(int*); }
  template <typename T>
  void bar();

  using ns1::foo;
  using ns2::foo;

  template <typename T>
  void b^ar() {
    foo(T());
  })cpp");
}

TEST_F(DefineInlineTest, TransformNestedNamespaces) {
  auto Test = R"cpp(
    namespace a {
      void bar();
      namespace b {
        void baz();
        namespace c {
          void aux();
        }
      }
    }

    void foo();
    using namespace a;
    using namespace b;
    using namespace c;
    void f^oo() {
      bar();
      a::bar();

      baz();
      b::baz();
      a::b::baz();

      aux();
      c::aux();
      b::c::aux();
      a::b::c::aux();
    })cpp";
  auto Expected = R"cpp(
    namespace a {
      void bar();
      namespace b {
        void baz();
        namespace c {
          void aux();
        }
      }
    }

    void foo(){
      a::bar();
      a::bar();

      a::b::baz();
      a::b::baz();
      a::b::baz();

      a::b::c::aux();
      a::b::c::aux();
      a::b::c::aux();
      a::b::c::aux();
    }
    using namespace a;
    using namespace b;
    using namespace c;
    )cpp";
  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DefineInlineTest, TransformUsings) {
  auto Test = R"cpp(
    namespace a { namespace b { namespace c { void aux(); } } }

    void foo();
    void f^oo() {
      using namespace a;
      using namespace b;
      using namespace c;
      using c::aux;
      namespace d = c;
    })cpp";
  auto Expected = R"cpp(
    namespace a { namespace b { namespace c { void aux(); } } }

    void foo(){
      using namespace a;
      using namespace a::b;
      using namespace a::b::c;
      using a::b::c::aux;
      namespace d = a::b::c;
    }
    )cpp";
  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DefineInlineTest, TransformDecls) {
  auto Test = R"cpp(
    void foo();
    void f^oo() {
      class Foo {
      public:
        void foo();
        int x;
      };

      enum En { Zero, One };
      En x = Zero;

      enum class EnClass { Zero, One };
      EnClass y = EnClass::Zero;
    })cpp";
  auto Expected = R"cpp(
    void foo(){
      class Foo {
      public:
        void foo();
        int x;
      };

      enum En { Zero, One };
      En x = Zero;

      enum class EnClass { Zero, One };
      EnClass y = EnClass::Zero;
    }
    )cpp";
  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DefineInlineTest, TransformTemplDecls) {
  auto Test = R"cpp(
    namespace a {
      template <typename T> class Bar {
      public:
        void bar();
      };
      template <typename T> T bar;
      template <typename T> void aux() {}
    }

    void foo();

    using namespace a;
    void f^oo() {
      bar<Bar<int>>.bar();
      aux<Bar<int>>();
    })cpp";
  auto Expected = R"cpp(
    namespace a {
      template <typename T> class Bar {
      public:
        void bar();
      };
      template <typename T> T bar;
      template <typename T> void aux() {}
    }

    void foo(){
      a::bar<a::Bar<int>>.bar();
      a::aux<a::Bar<int>>();
    }

    using namespace a;
    )cpp";
  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DefineInlineTest, TransformMembers) {
  auto Test = R"cpp(
    class Foo {
      void foo();
    };

    void Foo::f^oo() {
      return;
    })cpp";
  auto Expected = R"cpp(
    class Foo {
      void foo(){
      return;
    }
    };

    )cpp";
  EXPECT_EQ(apply(Test), Expected);

  ExtraFiles["a.h"] = R"cpp(
    class Foo {
      void foo();
    };)cpp";

  llvm::StringMap<std::string> EditedFiles;
  Test = R"cpp(
    #include "a.h"
    void Foo::f^oo() {
      return;
    })cpp";
  Expected = R"cpp(
    #include "a.h"
    )cpp";
  EXPECT_EQ(apply(Test, &EditedFiles), Expected);

  Expected = R"cpp(
    class Foo {
      void foo(){
      return;
    }
    };)cpp";
  EXPECT_THAT(EditedFiles,
              ElementsAre(FileWithContents(testPath("a.h"), Expected)));
}

TEST_F(DefineInlineTest, TransformDependentTypes) {
  auto Test = R"cpp(
    namespace a {
      template <typename T> class Bar {};
    }

    template <typename T>
    void foo();

    using namespace a;
    template <typename T>
    void f^oo() {
      Bar<T> B;
      Bar<Bar<T>> q;
    })cpp";
  auto Expected = R"cpp(
    namespace a {
      template <typename T> class Bar {};
    }

    template <typename T>
    void foo(){
      a::Bar<T> B;
      a::Bar<a::Bar<T>> q;
    }

    using namespace a;
    )cpp";

  // Template body is not parsed until instantiation time on windows, which
  // results in arbitrary failures as function body becomes NULL.
  ExtraArgs.push_back("-fno-delayed-template-parsing");
  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DefineInlineTest, TransformFunctionTempls) {
  // Check we select correct specialization decl.
  std::pair<llvm::StringRef, llvm::StringRef> Cases[] = {
      {R"cpp(
          template <typename T>
          void foo(T p);

          template <>
          void foo<int>(int p);

          template <>
          void foo<char>(char p);

          template <>
          void fo^o<int>(int p) {
            return;
          })cpp",
       R"cpp(
          template <typename T>
          void foo(T p);

          template <>
          void foo<int>(int p){
            return;
          }

          template <>
          void foo<char>(char p);

          )cpp"},
      {// Make sure we are not selecting the first specialization all the time.
       R"cpp(
          template <typename T>
          void foo(T p);

          template <>
          void foo<int>(int p);

          template <>
          void foo<char>(char p);

          template <>
          void fo^o<char>(char p) {
            return;
          })cpp",
       R"cpp(
          template <typename T>
          void foo(T p);

          template <>
          void foo<int>(int p);

          template <>
          void foo<char>(char p){
            return;
          }

          )cpp"},
      {R"cpp(
          template <typename T>
          void foo(T p);

          template <>
          void foo<int>(int p);

          template <typename T>
          void fo^o(T p) {
            return;
          })cpp",
       R"cpp(
          template <typename T>
          void foo(T p){
            return;
          }

          template <>
          void foo<int>(int p);

          )cpp"},
  };
  // Template body is not parsed until instantiation time on windows, which
  // results in arbitrary failures as function body becomes NULL.
  ExtraArgs.push_back("-fno-delayed-template-parsing");
  for (const auto &Case : Cases)
    EXPECT_EQ(apply(Case.first), Case.second) << Case.first;
}

TEST_F(DefineInlineTest, TransformTypeLocs) {
  auto Test = R"cpp(
    namespace a {
      template <typename T> class Bar {
      public:
        template <typename Q> class Baz {};
      };
      class Foo{};
    }

    void foo();

    using namespace a;
    void f^oo() {
      Bar<int> B;
      Foo foo;
      a::Bar<Bar<int>>::Baz<Bar<int>> q;
    })cpp";
  auto Expected = R"cpp(
    namespace a {
      template <typename T> class Bar {
      public:
        template <typename Q> class Baz {};
      };
      class Foo{};
    }

    void foo(){
      a::Bar<int> B;
      a::Foo foo;
      a::Bar<a::Bar<int>>::Baz<a::Bar<int>> q;
    }

    using namespace a;
    )cpp";
  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DefineInlineTest, TransformDeclRefs) {
  auto Test = R"cpp(
    namespace a {
      template <typename T> class Bar {
      public:
        void foo();
        static void bar();
        int x;
        static int y;
      };
      void bar();
      void test();
    }

    void foo();
    using namespace a;
    void f^oo() {
      a::Bar<int> B;
      B.foo();
      a::bar();
      Bar<Bar<int>>::bar();
      a::Bar<int>::bar();
      B.x = Bar<int>::y;
      Bar<int>::y = 3;
      bar();
      a::test();
    })cpp";
  auto Expected = R"cpp(
    namespace a {
      template <typename T> class Bar {
      public:
        void foo();
        static void bar();
        int x;
        static int y;
      };
      void bar();
      void test();
    }

    void foo(){
      a::Bar<int> B;
      B.foo();
      a::bar();
      a::Bar<a::Bar<int>>::bar();
      a::Bar<int>::bar();
      B.x = a::Bar<int>::y;
      a::Bar<int>::y = 3;
      a::bar();
      a::test();
    }
    using namespace a;
    )cpp";
  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DefineInlineTest, StaticMembers) {
  auto Test = R"cpp(
    namespace ns { class X { static void foo(); void bar(); }; }
    void ns::X::b^ar() {
      foo();
    })cpp";
  auto Expected = R"cpp(
    namespace ns { class X { static void foo(); void bar(){
      foo();
    } }; }
    )cpp";
  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DefineInlineTest, TransformParamNames) {
  std::pair<llvm::StringRef, llvm::StringRef> Cases[] = {
      {R"cpp(
        void foo(int, bool b, int T\
est);
        void ^foo(int f, bool x, int z) {})cpp",
       R"cpp(
        void foo(int f, bool x, int z){}
        )cpp"},
      {R"cpp(
        #define PARAM int Z
        void foo(PARAM);

        void ^foo(int X) {})cpp",
       "fail: Cant rename parameter inside macro body."},
      {R"cpp(
        #define TYPE int
        #define PARAM TYPE Z
        #define BODY(x) 5 * (x) + 2
        template <int P>
        void foo(PARAM, TYPE Q, TYPE, TYPE W = BODY(P));
        template <int x>
        void ^foo(int Z, int b, int c, int d) {})cpp",
       R"cpp(
        #define TYPE int
        #define PARAM TYPE Z
        #define BODY(x) 5 * (x) + 2
        template <int x>
        void foo(PARAM, TYPE b, TYPE c, TYPE d = BODY(x)){}
        )cpp"},
  };
  ExtraArgs.push_back("-fno-delayed-template-parsing");
  for (const auto &Case : Cases)
    EXPECT_EQ(apply(Case.first), Case.second) << Case.first;
}

TEST_F(DefineInlineTest, TransformTemplParamNames) {
  auto Test = R"cpp(
    struct Foo {
      struct Bar {
        template <class, class X,
                  template<typename> class, template<typename> class Y,
                  int, int Z>
        void foo(X, Y<X>, int W = 5 * Z + 2);
      };
    };

    template <class T, class U,
              template<typename> class V, template<typename> class W,
              int X, int Y>
    void Foo::Bar::f^oo(U, W<U>, int Q) {})cpp";
  auto Expected = R"cpp(
    struct Foo {
      struct Bar {
        template <class T, class U,
                  template<typename> class V, template<typename> class W,
                  int X, int Y>
        void foo(U, W<U>, int Q = 5 * Y + 2){}
      };
    };

    )cpp";
  ExtraArgs.push_back("-fno-delayed-template-parsing");
  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DefineInlineTest, TransformInlineNamespaces) {
  auto Test = R"cpp(
    namespace a { inline namespace b { namespace { struct Foo{}; } } }
    void foo();

    using namespace a;
    void ^foo() {Foo foo;})cpp";
  auto Expected = R"cpp(
    namespace a { inline namespace b { namespace { struct Foo{}; } } }
    void foo(){a::Foo foo;}

    using namespace a;
    )cpp";
  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DefineInlineTest, TokensBeforeSemicolon) {
  std::pair<llvm::StringRef, llvm::StringRef> Cases[] = {
      {R"cpp(
          void foo()    /*Comment -_-*/ /*Com 2*/ ;
          void fo^o() { return ; })cpp",
       R"cpp(
          void foo()    /*Comment -_-*/ /*Com 2*/ { return ; }
          )cpp"},

      {R"cpp(
          void foo();
          void fo^o() { return ; })cpp",
       R"cpp(
          void foo(){ return ; }
          )cpp"},

      {R"cpp(
          #define SEMI ;
          void foo() SEMI
          void fo^o() { return ; })cpp",
       "fail: Couldn't find semicolon for target declaration."},
  };
  for (const auto &Case : Cases)
    EXPECT_EQ(apply(Case.first), Case.second) << Case.first;
}

TEST_F(DefineInlineTest, HandleMacros) {
  EXPECT_UNAVAILABLE(R"cpp(
    #define BODY { return; }
    void foo();
    void f^oo()BODY)cpp");

  EXPECT_UNAVAILABLE(R"cpp(
    #define BODY void foo(){ return; }
    void foo();
    [[BODY]])cpp");

  std::pair<llvm::StringRef, llvm::StringRef> Cases[] = {
      // We don't qualify declarations coming from macros.
      {R"cpp(
          #define BODY Foo
          namespace a { class Foo{}; }
          void foo();
          using namespace a;
          void f^oo(){BODY();})cpp",
       R"cpp(
          #define BODY Foo
          namespace a { class Foo{}; }
          void foo(){BODY();}
          using namespace a;
          )cpp"},

      // Macro is not visible at declaration location, but we proceed.
      {R"cpp(
          void foo();
          #define BODY return;
          void f^oo(){BODY})cpp",
       R"cpp(
          void foo(){BODY}
          #define BODY return;
          )cpp"},

      {R"cpp(
          #define TARGET void foo()
          TARGET;
          void f^oo(){ return; })cpp",
       R"cpp(
          #define TARGET void foo()
          TARGET{ return; }
          )cpp"},

      {R"cpp(
          #define TARGET foo
          void TARGET();
          void f^oo(){ return; })cpp",
       R"cpp(
          #define TARGET foo
          void TARGET(){ return; }
          )cpp"},
  };
  for (const auto &Case : Cases)
    EXPECT_EQ(apply(Case.first), Case.second) << Case.first;
}

TEST_F(DefineInlineTest, DropCommonNameSpecifiers) {
  struct {
    llvm::StringRef Test;
    llvm::StringRef Expected;
  } Cases[] = {
      {R"cpp(
        namespace a { namespace b { void aux(); } }
        namespace ns1 {
          void foo();
          namespace qq { void test(); }
          namespace ns2 {
            void bar();
            namespace ns3 { void baz(); }
          }
        }

        using namespace a;
        using namespace a::b;
        using namespace ns1::qq;
        void ns1::ns2::ns3::b^az() {
          foo();
          bar();
          baz();
          ns1::ns2::ns3::baz();
          aux();
          test();
        })cpp",
       R"cpp(
        namespace a { namespace b { void aux(); } }
        namespace ns1 {
          void foo();
          namespace qq { void test(); }
          namespace ns2 {
            void bar();
            namespace ns3 { void baz(){
          foo();
          bar();
          baz();
          ns1::ns2::ns3::baz();
          a::b::aux();
          qq::test();
        } }
          }
        }

        using namespace a;
        using namespace a::b;
        using namespace ns1::qq;
        )cpp"},
      {R"cpp(
        namespace ns1 {
          namespace qq { struct Foo { struct Bar {}; }; using B = Foo::Bar; }
          namespace ns2 { void baz(); }
        }

        using namespace ns1::qq;
        void ns1::ns2::b^az() { Foo f; B b; })cpp",
       R"cpp(
        namespace ns1 {
          namespace qq { struct Foo { struct Bar {}; }; using B = Foo::Bar; }
          namespace ns2 { void baz(){ qq::Foo f; qq::B b; } }
        }

        using namespace ns1::qq;
        )cpp"},
      {R"cpp(
        namespace ns1 {
          namespace qq {
            template<class T> struct Foo { template <class U> struct Bar {}; };
            template<class T, class U>
            using B = typename Foo<T>::template Bar<U>;
          }
          namespace ns2 { void baz(); }
        }

        using namespace ns1::qq;
        void ns1::ns2::b^az() { B<int, bool> b; })cpp",
       R"cpp(
        namespace ns1 {
          namespace qq {
            template<class T> struct Foo { template <class U> struct Bar {}; };
            template<class T, class U>
            using B = typename Foo<T>::template Bar<U>;
          }
          namespace ns2 { void baz(){ qq::B<int, bool> b; } }
        }

        using namespace ns1::qq;
        )cpp"},
  };
  for (const auto &Case : Cases)
    EXPECT_EQ(apply(Case.Test), Case.Expected) << Case.Test;
}

TEST_F(DefineInlineTest, QualifyWithUsingDirectives) {
  llvm::StringRef Test = R"cpp(
    namespace a {
      void bar();
      namespace b { struct Foo{}; void aux(); }
      namespace c { void cux(); }
    }
    using namespace a;
    using X = b::Foo;
    void foo();

    using namespace b;
    using namespace c;
    void ^foo() {
      cux();
      bar();
      X x;
      aux();
      using namespace c;
      // FIXME: The last reference to cux() in body of foo should not be
      // qualified, since there is a using directive inside the function body.
      cux();
    })cpp";
  llvm::StringRef Expected = R"cpp(
    namespace a {
      void bar();
      namespace b { struct Foo{}; void aux(); }
      namespace c { void cux(); }
    }
    using namespace a;
    using X = b::Foo;
    void foo(){
      c::cux();
      bar();
      X x;
      b::aux();
      using namespace c;
      // FIXME: The last reference to cux() in body of foo should not be
      // qualified, since there is a using directive inside the function body.
      c::cux();
    }

    using namespace b;
    using namespace c;
    )cpp";
  EXPECT_EQ(apply(Test), Expected) << Test;
}

TEST_F(DefineInlineTest, AddInline) {
  ExtraArgs.push_back("-fno-delayed-template-parsing");
  llvm::StringMap<std::string> EditedFiles;
  ExtraFiles["a.h"] = "void foo();";
  apply(R"cpp(#include "a.h"
              void fo^o() {})cpp",
        &EditedFiles);
  EXPECT_THAT(EditedFiles, testing::ElementsAre(FileWithContents(
                               testPath("a.h"), "inline void foo(){}")));

  // Check we put inline before cv-qualifiers.
  ExtraFiles["a.h"] = "const int foo();";
  apply(R"cpp(#include "a.h"
              const int fo^o() {})cpp",
        &EditedFiles);
  EXPECT_THAT(EditedFiles, testing::ElementsAre(FileWithContents(
                               testPath("a.h"), "inline const int foo(){}")));

  // No double inline.
  ExtraFiles["a.h"] = "inline void foo();";
  apply(R"cpp(#include "a.h"
              inline void fo^o() {})cpp",
        &EditedFiles);
  EXPECT_THAT(EditedFiles, testing::ElementsAre(FileWithContents(
                               testPath("a.h"), "inline void foo(){}")));

  // Constexprs don't need "inline".
  ExtraFiles["a.h"] = "constexpr void foo();";
  apply(R"cpp(#include "a.h"
              constexpr void fo^o() {})cpp",
        &EditedFiles);
  EXPECT_THAT(EditedFiles, testing::ElementsAre(FileWithContents(
                               testPath("a.h"), "constexpr void foo(){}")));

  // Class members don't need "inline".
  ExtraFiles["a.h"] = "struct Foo { void foo(); };";
  apply(R"cpp(#include "a.h"
              void Foo::fo^o() {})cpp",
        &EditedFiles);
  EXPECT_THAT(EditedFiles,
              testing::ElementsAre(FileWithContents(
                  testPath("a.h"), "struct Foo { void foo(){} };")));

  // Function template doesn't need to be "inline"d.
  ExtraFiles["a.h"] = "template <typename T> void foo();";
  apply(R"cpp(#include "a.h"
              template <typename T>
              void fo^o() {})cpp",
        &EditedFiles);
  EXPECT_THAT(EditedFiles,
              testing::ElementsAre(FileWithContents(
                  testPath("a.h"), "template <typename T> void foo(){}")));

  // Specializations needs to be marked "inline".
  ExtraFiles["a.h"] = R"cpp(
                            template <typename T> void foo();
                            template <> void foo<int>();)cpp";
  apply(R"cpp(#include "a.h"
              template <>
              void fo^o<int>() {})cpp",
        &EditedFiles);
  EXPECT_THAT(EditedFiles,
              testing::ElementsAre(FileWithContents(testPath("a.h"),
                                                    R"cpp(
                            template <typename T> void foo();
                            template <> inline void foo<int>(){})cpp")));
}

} // namespace
} // namespace clangd
} // namespace clang
