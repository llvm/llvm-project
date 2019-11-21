//===-- HoverTests.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "Hover.h"
#include "TestIndex.h"
#include "TestTU.h"
#include "index/MemIndex.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TEST(Hover, Structured) {
  struct {
    const char *const Code;
    const std::function<void(HoverInfo &)> ExpectedBuilder;
  } Cases[] = {
      // Global scope.
      {R"cpp(
          // Best foo ever.
          void [[fo^o]]() {}
          )cpp",
       [](HoverInfo &HI) {
         HI.NamespaceScope = "";
         HI.Name = "foo";
         HI.Kind = SymbolKind::Function;
         HI.Documentation = "Best foo ever.";
         HI.Definition = "void foo()";
         HI.ReturnType = "void";
         HI.Type = "void ()";
         HI.Parameters.emplace();
       }},
      // Inside namespace
      {R"cpp(
          namespace ns1 { namespace ns2 {
            /// Best foo ever.
            void [[fo^o]]() {}
          }}
          )cpp",
       [](HoverInfo &HI) {
         HI.NamespaceScope = "ns1::ns2::";
         HI.Name = "foo";
         HI.Kind = SymbolKind::Function;
         HI.Documentation = "Best foo ever.";
         HI.Definition = "void foo()";
         HI.ReturnType = "void";
         HI.Type = "void ()";
         HI.Parameters.emplace();
       }},
      // Field
      {R"cpp(
          namespace ns1 { namespace ns2 {
            struct Foo {
              int [[b^ar]];
            };
          }}
          )cpp",
       [](HoverInfo &HI) {
         HI.NamespaceScope = "ns1::ns2::";
         HI.LocalScope = "Foo::";
         HI.Name = "bar";
         HI.Kind = SymbolKind::Field;
         HI.Definition = "int bar";
         HI.Type = "int";
       }},
      // Local to class method.
      {R"cpp(
          namespace ns1 { namespace ns2 {
            struct Foo {
              void foo() {
                int [[b^ar]];
              }
            };
          }}
          )cpp",
       [](HoverInfo &HI) {
         HI.NamespaceScope = "ns1::ns2::";
         HI.LocalScope = "Foo::foo::";
         HI.Name = "bar";
         HI.Kind = SymbolKind::Variable;
         HI.Definition = "int bar";
         HI.Type = "int";
       }},
      // Anon namespace and local scope.
      {R"cpp(
          namespace ns1 { namespace {
            struct {
              int [[b^ar]];
            } T;
          }}
          )cpp",
       [](HoverInfo &HI) {
         HI.NamespaceScope = "ns1::(anonymous)::";
         HI.LocalScope = "(anonymous struct)::";
         HI.Name = "bar";
         HI.Kind = SymbolKind::Field;
         HI.Definition = "int bar";
         HI.Type = "int";
       }},
      // Variable with template type
      {R"cpp(
          template <typename T, class... Ts> class Foo { public: Foo(int); };
          Foo<int, char, bool> [[fo^o]] = Foo<int, char, bool>(5);
          )cpp",
       [](HoverInfo &HI) {
         HI.NamespaceScope = "";
         HI.Name = "foo";
         HI.Kind = SymbolKind::Variable;
         HI.Definition = "Foo<int, char, bool> foo = Foo<int, char, bool>(5)";
         HI.Type = "Foo<int, char, bool>";
       }},
      // Implicit template instantiation
      {R"cpp(
          template <typename T> class vector{};
          [[vec^tor]]<int> foo;
          )cpp",
       [](HoverInfo &HI) {
         HI.NamespaceScope = "";
         HI.Name = "vector";
         HI.Kind = SymbolKind::Class;
         HI.Definition = "template <typename T> class vector {}";
         HI.TemplateParameters = {
             {std::string("typename"), std::string("T"), llvm::None},
         };
       }},
      // Class template
      {R"cpp(
          template <template<typename, bool...> class C,
                    typename = char,
                    int = 0,
                    bool Q = false,
                    class... Ts> class Foo {};
          template <template<typename, bool...> class T>
          [[F^oo]]<T> foo;
          )cpp",
       [](HoverInfo &HI) {
         HI.NamespaceScope = "";
         HI.Name = "Foo";
         HI.Kind = SymbolKind::Class;
         HI.Definition =
             R"cpp(template <template <typename, bool...> class C, typename = char, int = 0,
          bool Q = false, class... Ts>
class Foo {})cpp";
         HI.TemplateParameters = {
             {std::string("template <typename, bool...> class"),
              std::string("C"), llvm::None},
             {std::string("typename"), llvm::None, std::string("char")},
             {std::string("int"), llvm::None, std::string("0")},
             {std::string("bool"), std::string("Q"), std::string("false")},
             {std::string("class..."), std::string("Ts"), llvm::None},
         };
       }},
      // Function template
      {R"cpp(
          template <template<typename, bool...> class C,
                    typename = char,
                    int = 0,
                    bool Q = false,
                    class... Ts> void foo();
          template<typename, bool...> class Foo;

          void bar() {
            [[fo^o]]<Foo>();
          }
          )cpp",
       [](HoverInfo &HI) {
         HI.NamespaceScope = "";
         HI.Name = "foo";
         HI.Kind = SymbolKind::Function;
         HI.Definition =
             R"cpp(template <template <typename, bool...> class C, typename = char, int = 0,
          bool Q = false, class... Ts>
void foo())cpp";
         HI.ReturnType = "void";
         HI.Type = "void ()";
         HI.Parameters.emplace();
         HI.TemplateParameters = {
             {std::string("template <typename, bool...> class"),
              std::string("C"), llvm::None},
             {std::string("typename"), llvm::None, std::string("char")},
             {std::string("int"), llvm::None, std::string("0")},
             {std::string("bool"), std::string("Q"), std::string("false")},
             {std::string("class..."), std::string("Ts"), llvm::None},
         };
       }},
      // Function decl
      {R"cpp(
          template<typename, bool...> class Foo {};
          Foo<bool, true, false> foo(int, bool T = false);

          void bar() {
            [[fo^o]](3);
          }
          )cpp",
       [](HoverInfo &HI) {
         HI.NamespaceScope = "";
         HI.Name = "foo";
         HI.Kind = SymbolKind::Function;
         HI.Definition = "Foo<bool, true, false> foo(int, bool T = false)";
         HI.ReturnType = "Foo<bool, true, false>";
         HI.Type = "Foo<bool, true, false> (int, bool)";
         HI.Parameters = {
             {std::string("int"), llvm::None, llvm::None},
             {std::string("bool"), std::string("T"), std::string("false")},
         };
       }},
      // Pointers to lambdas
      {R"cpp(
        void foo() {
          auto lamb = [](int T, bool B) -> bool { return T && B; };
          auto *b = &lamb;
          auto *[[^c]] = &b;
        }
        )cpp",
       [](HoverInfo &HI) {
         HI.NamespaceScope = "";
         HI.LocalScope = "foo::";
         HI.Name = "c";
         HI.Kind = SymbolKind::Variable;
         HI.Definition = "auto *c = &b";
         HI.Type = "class (lambda) **";
         HI.ReturnType = "bool";
         HI.Parameters = {
             {std::string("int"), std::string("T"), llvm::None},
             {std::string("bool"), std::string("B"), llvm::None},
         };
         return HI;
       }},
      // Lambda parameter with decltype reference
      {R"cpp(
        auto lamb = [](int T, bool B) -> bool { return T && B; };
        void foo(decltype(lamb)& bar) {
          [[ba^r]](0, false);
        }
        )cpp",
       [](HoverInfo &HI) {
         HI.NamespaceScope = "";
         HI.LocalScope = "foo::";
         HI.Name = "bar";
         HI.Kind = SymbolKind::Variable;
         HI.Definition = "decltype(lamb) &bar";
         HI.Type = "decltype(lamb) &";
         HI.ReturnType = "bool";
         HI.Parameters = {
             {std::string("int"), std::string("T"), llvm::None},
             {std::string("bool"), std::string("B"), llvm::None},
         };
         return HI;
       }},
      // Lambda parameter with decltype
      {R"cpp(
        auto lamb = [](int T, bool B) -> bool { return T && B; };
        void foo(decltype(lamb) bar) {
          [[ba^r]](0, false);
        }
        )cpp",
       [](HoverInfo &HI) {
         HI.NamespaceScope = "";
         HI.LocalScope = "foo::";
         HI.Name = "bar";
         HI.Kind = SymbolKind::Variable;
         HI.Definition = "decltype(lamb) bar";
         HI.Type = "class (lambda)";
         HI.ReturnType = "bool";
         HI.Parameters = {
             {std::string("int"), std::string("T"), llvm::None},
             {std::string("bool"), std::string("B"), llvm::None},
         };
         HI.Value = "false";
         return HI;
       }},
      // Lambda variable
      {R"cpp(
        void foo() {
          int bar = 5;
          auto lamb = [&bar](int T, bool B) -> bool { return T && B && bar; };
          bool res = [[lam^b]](bar, false);
        }
        )cpp",
       [](HoverInfo &HI) {
         HI.NamespaceScope = "";
         HI.LocalScope = "foo::";
         HI.Name = "lamb";
         HI.Kind = SymbolKind::Variable;
         HI.Definition = "auto lamb = [&bar](int T, bool B) -> bool {}";
         HI.Type = "class (lambda)";
         HI.ReturnType = "bool";
         HI.Parameters = {
             {std::string("int"), std::string("T"), llvm::None},
             {std::string("bool"), std::string("B"), llvm::None},
         };
         return HI;
       }},
      // Local variable in lambda
      {R"cpp(
        void foo() {
          auto lamb = []{int [[te^st]];};
        }
        )cpp",
       [](HoverInfo &HI) {
         HI.NamespaceScope = "";
         HI.LocalScope = "foo::(anonymous class)::operator()::";
         HI.Name = "test";
         HI.Kind = SymbolKind::Variable;
         HI.Definition = "int test";
         HI.Type = "int";
       }},
      // Partially-specialized class template. (formerly type-parameter-0-0)
      {R"cpp(
        template <typename T> class X;
        template <typename T> class [[^X]]<T*> {};
        )cpp",
       [](HoverInfo &HI) {
         HI.Name = "X<T *>";
         HI.NamespaceScope = "";
         HI.Kind = SymbolKind::Class;
         HI.Definition = "template <typename T> class X<T *> {}";
       }},
      // Constructor of partially-specialized class template
      {R"cpp(
          template<typename, typename=void> struct X;
          template<typename T> struct X<T*>{ [[^X]](); };
          )cpp",
       [](HoverInfo &HI) {
         HI.NamespaceScope = "";
         HI.Name = "X";
         HI.LocalScope = "X<T *>::"; // FIXME: X<T *, void>::
         HI.Kind = SymbolKind::Constructor;
         HI.ReturnType = "X<T *>";
         HI.Definition = "X()";
         HI.Parameters.emplace();
       }},
      {"class X { [[^~]]X(); };", // FIXME: Should be [[~X]]()
       [](HoverInfo &HI) {
         HI.NamespaceScope = "";
         HI.Name = "~X";
         HI.LocalScope = "X::";
         HI.Kind = SymbolKind::Constructor;
         HI.ReturnType = "void";
         HI.Definition = "~X()";
         HI.Parameters.emplace();
       }},

      // auto on lambda
      {R"cpp(
        void foo() {
          [[au^to]] lamb = []{};
        }
        )cpp",
       [](HoverInfo &HI) {
         HI.Name = "class (lambda)";
         HI.Kind = SymbolKind::Class;
       }},
      // auto on template instantiation
      {R"cpp(
        template<typename T> class Foo{};
        void foo() {
          [[au^to]] x = Foo<int>();
        }
        )cpp",
       [](HoverInfo &HI) {
         HI.Name = "class Foo<int>";
         HI.Kind = SymbolKind::Class;
       }},
      // auto on specialized template
      {R"cpp(
        template<typename T> class Foo{};
        template<> class Foo<int>{};
        void foo() {
          [[au^to]] x = Foo<int>();
        }
        )cpp",
       [](HoverInfo &HI) {
         HI.Name = "class Foo<int>";
         HI.Kind = SymbolKind::Class;
       }},

      // macro
      {R"cpp(
        // Best MACRO ever.
        #define MACRO(x,y,z) void foo(x, y, z);
        [[MAC^RO]](int, double d, bool z = false);
        )cpp",
       [](HoverInfo &HI) {
         HI.Name = "MACRO", HI.Kind = SymbolKind::String,
         HI.Definition = "#define MACRO(x, y, z) void foo(x, y, z);";
       }},

      // constexprs
      {R"cpp(
        constexpr int add(int a, int b) { return a + b; }
        int [[b^ar]] = add(1, 2);
        )cpp",
       [](HoverInfo &HI) {
         HI.Name = "bar";
         HI.Definition = "int bar = add(1, 2)";
         HI.Kind = SymbolKind::Variable;
         HI.Type = "int";
         HI.NamespaceScope = "";
         HI.Value = "3";
       }},
      {R"cpp(
        int [[b^ar]] = sizeof(char);
        )cpp",
       [](HoverInfo &HI) {
         HI.Name = "bar";
         HI.Definition = "int bar = sizeof(char)";
         HI.Kind = SymbolKind::Variable;
         HI.Type = "int";
         HI.NamespaceScope = "";
         HI.Value = "1";
       }},
      {R"cpp(
        template<int a, int b> struct Add {
          static constexpr int result = a + b;
        };
        int [[ba^r]] = Add<1, 2>::result;
        )cpp",
       [](HoverInfo &HI) {
         HI.Name = "bar";
         HI.Definition = "int bar = Add<1, 2>::result";
         HI.Kind = SymbolKind::Variable;
         HI.Type = "int";
         HI.NamespaceScope = "";
         HI.Value = "3";
       }},
      {R"cpp(
        enum Color { RED, GREEN, };
        Color x = [[GR^EEN]];
       )cpp",
       [](HoverInfo &HI) {
         HI.Name = "GREEN";
         HI.NamespaceScope = "";
         HI.LocalScope = "Color::";
         HI.Definition = "GREEN";
         HI.Kind = SymbolKind::EnumMember;
         HI.Type = "enum Color";
         HI.Value = "1"; // Numeric when hovering on the enumerator name.
       }},
      {R"cpp(
        enum Color { RED, GREEN, };
        Color x = GREEN;
        Color y = [[^x]];
       )cpp",
       [](HoverInfo &HI) {
         HI.Name = "x";
         HI.NamespaceScope = "";
         HI.Definition = "enum Color x = GREEN";
         HI.Kind = SymbolKind::Variable;
         HI.Type = "enum Color";
         HI.Value = "GREEN (1)"; // Symbolic when hovering on an expression.
       }},
      // FIXME: We should use the Decl referenced, even if from an implicit
      // instantiation. Then the scope would be Add<1, 2>.
      {R"cpp(
        template<int a, int b> struct Add {
          static constexpr int result = a + b;
        };
        int bar = Add<1, 2>::[[resu^lt]];
        )cpp",
       [](HoverInfo &HI) {
         HI.Name = "result";
         HI.Definition = "static constexpr int result = a + b";
         HI.Kind = SymbolKind::Property;
         HI.Type = "const int";
         HI.NamespaceScope = "";
         HI.LocalScope = "Add<a, b>::";
         HI.Value = "3";
       }},
      {R"cpp(
        constexpr int answer() { return 40 + 2; }
        int x = [[ans^wer]]();
        )cpp",
       [](HoverInfo &HI) {
         HI.Name = "answer";
         HI.Definition = "constexpr int answer()";
         HI.Kind = SymbolKind::Function;
         HI.Type = "int ()";
         HI.ReturnType = "int";
         HI.Parameters.emplace();
         HI.NamespaceScope = "";
         HI.Value = "42";
       }},
      {R"cpp(
        const char *[[ba^r]] = "1234";
        )cpp",
       [](HoverInfo &HI) {
         HI.Name = "bar";
         HI.Definition = "const char *bar = \"1234\"";
         HI.Kind = SymbolKind::Variable;
         HI.Type = "const char *";
         HI.NamespaceScope = "";
         HI.Value = "&\"1234\"[0]";
       }},
  };
  for (const auto &Case : Cases) {
    SCOPED_TRACE(Case.Code);

    Annotations T(Case.Code);
    TestTU TU = TestTU::withCode(T.code());
    TU.ExtraArgs.push_back("-std=c++17");
    auto AST = TU.build();
    ASSERT_TRUE(AST.getDiagnostics().empty());

    auto H = getHover(AST, T.point(), format::getLLVMStyle(), nullptr);
    ASSERT_TRUE(H);
    HoverInfo Expected;
    Expected.SymRange = T.range();
    Case.ExpectedBuilder(Expected);

    EXPECT_EQ(H->NamespaceScope, Expected.NamespaceScope);
    EXPECT_EQ(H->LocalScope, Expected.LocalScope);
    EXPECT_EQ(H->Name, Expected.Name);
    EXPECT_EQ(H->Kind, Expected.Kind);
    EXPECT_EQ(H->Documentation, Expected.Documentation);
    EXPECT_EQ(H->Definition, Expected.Definition);
    EXPECT_EQ(H->Type, Expected.Type);
    EXPECT_EQ(H->ReturnType, Expected.ReturnType);
    EXPECT_EQ(H->Parameters, Expected.Parameters);
    EXPECT_EQ(H->TemplateParameters, Expected.TemplateParameters);
    EXPECT_EQ(H->SymRange, Expected.SymRange);
    EXPECT_EQ(H->Value, Expected.Value);
  }
}

TEST(Hover, All) {
  struct OneTest {
    StringRef Input;
    StringRef ExpectedHover;
  };

  OneTest Tests[] = {
      {
          R"cpp(// No hover
            ^int main() {
            }
          )cpp",
          "",
      },
      {
          R"cpp(// Local variable
            int main() {
              int bonjour;
              ^bonjour = 2;
              int test1 = bonjour;
            }
          )cpp",
          "text[Declared in]code[main]\n"
          "codeblock(cpp) [\n"
          "int bonjour\n"
          "]",
      },
      {
          R"cpp(// Local variable in method
            struct s {
              void method() {
                int bonjour;
                ^bonjour = 2;
              }
            };
          )cpp",
          "text[Declared in]code[s::method]\n"
          "codeblock(cpp) [\n"
          "int bonjour\n"
          "]",
      },
      {
          R"cpp(// Struct
            namespace ns1 {
              struct MyClass {};
            } // namespace ns1
            int main() {
              ns1::My^Class* Params;
            }
          )cpp",
          "text[Declared in]code[ns1]\n"
          "codeblock(cpp) [\n"
          "struct MyClass {}\n"
          "]",
      },
      {
          R"cpp(// Class
            namespace ns1 {
              class MyClass {};
            } // namespace ns1
            int main() {
              ns1::My^Class* Params;
            }
          )cpp",
          "text[Declared in]code[ns1]\n"
          "codeblock(cpp) [\n"
          "class MyClass {}\n"
          "]",
      },
      {
          R"cpp(// Union
            namespace ns1 {
              union MyUnion { int x; int y; };
            } // namespace ns1
            int main() {
              ns1::My^Union Params;
            }
          )cpp",
          "text[Declared in]code[ns1]\n"
          "codeblock(cpp) [\n"
          "union MyUnion {}\n"
          "]",
      },
      {
          R"cpp(// Function definition via pointer
            int foo(int) {}
            int main() {
              auto *X = &^foo;
            }
          )cpp",
          "text[Declared in]code[global namespace]\n"
          "codeblock(cpp) [\n"
          "int foo(int)\n"
          "]\n"
          "text[Function definition via pointer]",
      },
      {
          R"cpp(// Function declaration via call
            int foo(int);
            int main() {
              return ^foo(42);
            }
          )cpp",
          "text[Declared in]code[global namespace]\n"
          "codeblock(cpp) [\n"
          "int foo(int)\n"
          "]\n"
          "text[Function declaration via call]",
      },
      {
          R"cpp(// Field
            struct Foo { int x; };
            int main() {
              Foo bar;
              bar.^x;
            }
          )cpp",
          "text[Declared in]code[Foo]\n"
          "codeblock(cpp) [\n"
          "int x\n"
          "]",
      },
      {
          R"cpp(// Field with initialization
            struct Foo { int x = 5; };
            int main() {
              Foo bar;
              bar.^x;
            }
          )cpp",
          "text[Declared in]code[Foo]\n"
          "codeblock(cpp) [\n"
          "int x = 5\n"
          "]",
      },
      {
          R"cpp(// Static field
            struct Foo { static int x; };
            int main() {
              Foo::^x;
            }
          )cpp",
          "text[Declared in]code[Foo]\n"
          "codeblock(cpp) [\n"
          "static int x\n"
          "]",
      },
      {
          R"cpp(// Field, member initializer
            struct Foo {
              int x;
              Foo() : ^x(0) {}
            };
          )cpp",
          "text[Declared in]code[Foo]\n"
          "codeblock(cpp) [\n"
          "int x\n"
          "]",
      },
      {
          R"cpp(// Field, GNU old-style field designator
            struct Foo { int x; };
            int main() {
              Foo bar = { ^x : 1 };
            }
          )cpp",
          "text[Declared in]code[Foo]\n"
          "codeblock(cpp) [\n"
          "int x\n"
          "]",
      },
      {
          R"cpp(// Field, field designator
            struct Foo { int x; };
            int main() {
              Foo bar = { .^x = 2 };
            }
          )cpp",
          "text[Declared in]code[Foo]\n"
          "codeblock(cpp) [\n"
          "int x\n"
          "]",
      },
      {
          R"cpp(// Method call
            struct Foo { int x(); };
            int main() {
              Foo bar;
              bar.^x();
            }
          )cpp",
          "text[Declared in]code[Foo]\n"
          "codeblock(cpp) [\n"
          "int x()\n"
          "]",
      },
      {
          R"cpp(// Static method call
            struct Foo { static int x(); };
            int main() {
              Foo::^x();
            }
          )cpp",
          "text[Declared in]code[Foo]\n"
          "codeblock(cpp) [\n"
          "static int x()\n"
          "]",
      },
      {
          R"cpp(// Typedef
            typedef int Foo;
            int main() {
              ^Foo bar;
            }
          )cpp",
          "text[Declared in]code[global namespace]\n"
          "codeblock(cpp) [\n"
          "typedef int Foo\n"
          "]\n"
          "text[Typedef]",
      },
      {
          R"cpp(// Typedef with embedded definition
            typedef struct Bar {} Foo;
            int main() {
              ^Foo bar;
            }
          )cpp",
          "text[Declared in]code[global namespace]\n"
          "codeblock(cpp) [\n"
          "typedef struct Bar Foo\n"
          "]\n"
          "text[Typedef with embedded definition]",
      },
      {
          R"cpp(// Namespace
            namespace ns {
            struct Foo { static void bar(); }
            } // namespace ns
            int main() { ^ns::Foo::bar(); }
          )cpp",
          "text[Declared in]code[global namespace]\n"
          "codeblock(cpp) [\n"
          "namespace ns {}\n"
          "]",
      },
      {
          R"cpp(// Anonymous namespace
            namespace ns {
              namespace {
                int foo;
              } // anonymous namespace
            } // namespace ns
            int main() { ns::f^oo++; }
          )cpp",
          "text[Declared in]code[ns::(anonymous)]\n"
          "codeblock(cpp) [\n"
          "int foo\n"
          "]",
      },
      {
          R"cpp(// Macro
            #define MACRO 0
            #define MACRO 1
            int main() { return ^MACRO; }
            #define MACRO 2
            #undef macro
          )cpp",
          "codeblock(cpp) [\n"
          "#define MACRO 1\n"
          "]",
      },
      {
          R"cpp(// Macro
            #define MACRO 0
            #define MACRO2 ^MACRO
          )cpp",
          "codeblock(cpp) [\n"
          "#define MACRO 0\n"
          "]",
      },
      {
          R"cpp(// Macro
            #define MACRO {\
              return 0;\
            }
            int main() ^MACRO
          )cpp",
          R"cpp(codeblock(cpp) [
#define MACRO                                                                  \
  { return 0; }
])cpp",
      },
      {
          R"cpp(// Forward class declaration
            class Foo;
            class Foo {};
            F^oo* foo();
          )cpp",
          "text[Declared in]code[global namespace]\n"
          "codeblock(cpp) [\n"
          "class Foo {}\n"
          "]\n"
          "text[Forward class declaration]",
      },
      {
          R"cpp(// Function declaration
            void foo();
            void g() { f^oo(); }
            void foo() {}
          )cpp",
          "text[Declared in]code[global namespace]\n"
          "codeblock(cpp) [\n"
          "void foo()\n"
          "]\n"
          "text[Function declaration]",
      },
      {
          R"cpp(// Enum declaration
            enum Hello {
              ONE, TWO, THREE,
            };
            void foo() {
              Hel^lo hello = ONE;
            }
          )cpp",
          "text[Declared in]code[global namespace]\n"
          "codeblock(cpp) [\n"
          "enum Hello {}\n"
          "]\n"
          "text[Enum declaration]",
      },
      {
          R"cpp(// Enumerator
            enum Hello {
              ONE, TWO, THREE,
            };
            void foo() {
              Hello hello = O^NE;
            }
          )cpp",
          "text[Declared in]code[Hello]\n"
          "codeblock(cpp) [\n"
          "ONE\n"
          "]",
      },
      {
          R"cpp(// Enumerator in anonymous enum
            enum {
              ONE, TWO, THREE,
            };
            void foo() {
              int hello = O^NE;
            }
          )cpp",
          "text[Declared in]code[global namespace]\n"
          "codeblock(cpp) [\n"
          "ONE\n"
          "]",
      },
      {
          R"cpp(// Global variable
            static int hey = 10;
            void foo() {
              he^y++;
            }
          )cpp",
          "text[Declared in]code[global namespace]\n"
          "codeblock(cpp) [\n"
          "static int hey = 10\n"
          "]\n"
          "text[Global variable]",
      },
      {
          R"cpp(// Global variable in namespace
            namespace ns1 {
              static int hey = 10;
            }
            void foo() {
              ns1::he^y++;
            }
          )cpp",
          "text[Declared in]code[ns1]\n"
          "codeblock(cpp) [\n"
          "static int hey = 10\n"
          "]",
      },
      {
          R"cpp(// Field in anonymous struct
            static struct {
              int hello;
            } s;
            void foo() {
              s.he^llo++;
            }
          )cpp",
          "text[Declared in]code[(anonymous struct)]\n"
          "codeblock(cpp) [\n"
          "int hello\n"
          "]",
      },
      {
          R"cpp(// Templated function
            template <typename T>
            T foo() {
              return 17;
            }
            void g() { auto x = f^oo<int>(); }
          )cpp",
          "text[Declared in]code[global namespace]\n"
          "codeblock(cpp) [\n"
          "template <typename T> T foo()\n"
          "]\n"
          "text[Templated function]",
      },
      {
          R"cpp(// Anonymous union
            struct outer {
              union {
                int abc, def;
              } v;
            };
            void g() { struct outer o; o.v.d^ef++; }
          )cpp",
          "text[Declared in]code[outer::(anonymous union)]\n"
          "codeblock(cpp) [\n"
          "int def\n"
          "]",
      },
      {
          R"cpp(// documentation from index
            int nextSymbolIsAForwardDeclFromIndexWithNoLocalDocs;
            void indexSymbol();
            void g() { ind^exSymbol(); }
          )cpp",
          "text[Declared in]code[global namespace]\n"
          "codeblock(cpp) [\n"
          "void indexSymbol()\n"
          "]\n"
          "text[comment from index]",
      },
      {
          R"cpp(// Nothing
            void foo() {
              ^
            }
          )cpp",
          "",
      },
      {
          R"cpp(// Simple initialization with auto
            void foo() {
              ^auto i = 1;
            }
          )cpp",
          "codeblock(cpp) [\n"
          "int\n"
          "]",
      },
      {
          R"cpp(// Simple initialization with const auto
            void foo() {
              const ^auto i = 1;
            }
          )cpp",
          "codeblock(cpp) [\n"
          "int\n"
          "]",
      },
      {
          R"cpp(// Simple initialization with const auto&
            void foo() {
              const ^auto& i = 1;
            }
          )cpp",
          "codeblock(cpp) [\n"
          "int\n"
          "]",
      },
      {
          R"cpp(// Simple initialization with auto&
            void foo() {
              ^auto& i = 1;
            }
          )cpp",
          "codeblock(cpp) [\n"
          "int\n"
          "]",
      },
      {
          R"cpp(// Simple initialization with auto*
            void foo() {
              int a = 1;
              ^auto* i = &a;
            }
          )cpp",
          "codeblock(cpp) [\n"
          "int\n"
          "]",
      },
      {
          R"cpp(// Auto with initializer list.
            namespace std
            {
              template<class _E>
              class initializer_list {};
            }
            void foo() {
              ^auto i = {1,2};
            }
          )cpp",
          "codeblock(cpp) [\n"
          "class std::initializer_list<int>\n"
          "]",
      },
      {
          R"cpp(// User defined conversion to auto
            struct Bar {
              operator ^auto() const { return 10; }
            };
          )cpp",
          "codeblock(cpp) [\n"
          "int\n"
          "]",
      },
      {
          R"cpp(// Simple initialization with decltype(auto)
            void foo() {
              ^decltype(auto) i = 1;
            }
          )cpp",
          "codeblock(cpp) [\n"
          "int\n"
          "]",
      },
      {
          R"cpp(// Simple initialization with const decltype(auto)
            void foo() {
              const int j = 0;
              ^decltype(auto) i = j;
            }
          )cpp",
          "codeblock(cpp) [\n"
          "const int\n"
          "]",
      },
      {
          R"cpp(// Simple initialization with const& decltype(auto)
            void foo() {
              int k = 0;
              const int& j = k;
              ^decltype(auto) i = j;
            }
          )cpp",
          "codeblock(cpp) [\n"
          "const int &\n"
          "]",
      },
      {
          R"cpp(// Simple initialization with & decltype(auto)
            void foo() {
              int k = 0;
              int& j = k;
              ^decltype(auto) i = j;
            }
          )cpp",
          "codeblock(cpp) [\n"
          "int &\n"
          "]",
      },
      {
          R"cpp(// decltype with initializer list: nothing
            namespace std
            {
              template<class _E>
              class initializer_list {};
            }
            void foo() {
              ^decltype(auto) i = {1,2};
            }
          )cpp",
          "",
      },
      {
          R"cpp(// simple trailing return type
            ^auto main() -> int {
              return 0;
            }
          )cpp",
          "codeblock(cpp) [\n"
          "int\n"
          "]",
      },
      {
          R"cpp(// auto function return with trailing type
            struct Bar {};
            ^auto test() -> decltype(Bar()) {
              return Bar();
            }
          )cpp",
          "codeblock(cpp) [\n"
          "struct Bar\n"
          "]",
      },
      {
          R"cpp(// trailing return type
            struct Bar {};
            auto test() -> ^decltype(Bar()) {
              return Bar();
            }
          )cpp",
          "codeblock(cpp) [\n"
          "struct Bar\n"
          "]",
      },
      {
          R"cpp(// auto in function return
            struct Bar {};
            ^auto test() {
              return Bar();
            }
          )cpp",
          "codeblock(cpp) [\n"
          "struct Bar\n"
          "]",
      },
      {
          R"cpp(// auto& in function return
            struct Bar {};
            ^auto& test() {
              return Bar();
            }
          )cpp",
          "codeblock(cpp) [\n"
          "struct Bar\n"
          "]",
      },
      {
          R"cpp(// auto* in function return
            struct Bar {};
            ^auto* test() {
              Bar* bar;
              return bar;
            }
          )cpp",
          "codeblock(cpp) [\n"
          "struct Bar\n"
          "]",
      },
      {
          R"cpp(// const auto& in function return
            struct Bar {};
            const ^auto& test() {
              return Bar();
            }
          )cpp",
          "codeblock(cpp) [\n"
          "struct Bar\n"
          "]",
      },
      {
          R"cpp(// decltype(auto) in function return
            struct Bar {};
            ^decltype(auto) test() {
              return Bar();
            }
          )cpp",
          "codeblock(cpp) [\n"
          "struct Bar\n"
          "]",
      },
      {
          R"cpp(// decltype(auto) reference in function return
            struct Bar {};
            ^decltype(auto) test() {
              int a;
              return (a);
            }
          )cpp",
          "codeblock(cpp) [\n"
          "int &\n"
          "]",
      },
      {
          R"cpp(// decltype lvalue reference
            void foo() {
              int I = 0;
              ^decltype(I) J = I;
            }
          )cpp",
          "codeblock(cpp) [\n"
          "int\n"
          "]",
      },
      {
          R"cpp(// decltype lvalue reference
            void foo() {
              int I= 0;
              int &K = I;
              ^decltype(K) J = I;
            }
          )cpp",
          "codeblock(cpp) [\n"
          "int &\n"
          "]",
      },
      {
          R"cpp(// decltype lvalue reference parenthesis
            void foo() {
              int I = 0;
              ^decltype((I)) J = I;
            }
          )cpp",
          "codeblock(cpp) [\n"
          "int &\n"
          "]",
      },
      {
          R"cpp(// decltype rvalue reference
            void foo() {
              int I = 0;
              ^decltype(static_cast<int&&>(I)) J = static_cast<int&&>(I);
            }
          )cpp",
          "codeblock(cpp) [\n"
          "int &&\n"
          "]",
      },
      {
          R"cpp(// decltype rvalue reference function call
            int && bar();
            void foo() {
              int I = 0;
              ^decltype(bar()) J = bar();
            }
          )cpp",
          "codeblock(cpp) [\n"
          "int &&\n"
          "]",
      },
      {
          R"cpp(// decltype of function with trailing return type.
            struct Bar {};
            auto test() -> decltype(Bar()) {
              return Bar();
            }
            void foo() {
              ^decltype(test()) i = test();
            }
          )cpp",
          "codeblock(cpp) [\n"
          "struct Bar\n"
          "]",
      },
      {
          R"cpp(// decltype of var with decltype.
            void foo() {
              int I = 0;
              decltype(I) J = I;
              ^decltype(J) K = J;
            }
          )cpp",
          "codeblock(cpp) [\n"
          "int\n"
          "]",
      },
      {
          R"cpp(// structured binding. Not supported yet
            struct Bar {};
            void foo() {
              Bar a[2];
              ^auto [x,y] = a;
            }
          )cpp",
          "",
      },
      {
          R"cpp(// Template auto parameter. Nothing (Not useful).
            template<^auto T>
            void func() {
            }
            void foo() {
               func<1>();
            }
          )cpp",
          "",
      },
      {
          R"cpp(// More compilcated structured types.
            int bar();
            ^auto (*foo)() = bar;
          )cpp",
          "codeblock(cpp) [\n"
          "int\n"
          "]",
      },
      {
          R"cpp(// Should not crash when evaluating the initializer.
            struct Test {};
            void test() { Test && te^st = {}; }
          )cpp",
          "text[Declared in]code[test]\n"
          "codeblock(cpp) [\n"
          "struct Test &&test = {}\n"
          "]",
      },
  };

  // Create a tiny index, so tests above can verify documentation is fetched.
  Symbol IndexSym = func("indexSymbol");
  IndexSym.Documentation = "comment from index";
  SymbolSlab::Builder Symbols;
  Symbols.insert(IndexSym);
  auto Index =
      MemIndex::build(std::move(Symbols).build(), RefSlab(), RelationSlab());

  for (const OneTest &Test : Tests) {
    Annotations T(Test.Input);
    TestTU TU = TestTU::withCode(T.code());
    TU.ExtraArgs.push_back("-std=c++17");
    auto AST = TU.build();
    if (auto H =
            getHover(AST, T.point(), format::getLLVMStyle(), Index.get())) {
      EXPECT_NE("", Test.ExpectedHover) << Test.Input;
      EXPECT_EQ(H->present().renderForTests(), Test.ExpectedHover.str())
          << Test.Input;
    } else
      EXPECT_EQ("", Test.ExpectedHover.str()) << Test.Input;
  }
}

} // namespace
} // namespace clangd
} // namespace clang
