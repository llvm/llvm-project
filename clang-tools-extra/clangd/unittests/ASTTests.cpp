//===-- ASTTests.cpp --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AST.h"

#include "Annotations.h"
#include "ParsedAST.h"
#include "TestTU.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/Basic/AttrKinds.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstddef>
#include <string>
#include <vector>

namespace clang {
namespace clangd {
namespace {
using testing::Contains;
using testing::Each;
using testing::IsEmpty;

TEST(GetDeducedType, KwAutoKwDecltypeExpansion) {
  struct Test {
    StringRef AnnotatedCode;
    const char *DeducedType;
  } Tests[] = {
      {"^auto i = 0;", "int"},
      {"^auto f(){ return 1;};", "int"},
      {
          R"cpp( // auto on struct in a namespace
              namespace ns1 { struct S {}; }
              ^auto v = ns1::S{};
          )cpp",
          "ns1::S",
      },
      {
          R"cpp( // decltype on struct
              namespace ns1 { struct S {}; }
              ns1::S i;
              ^decltype(i) j;
          )cpp",
          "ns1::S",
      },
      {
          R"cpp(// decltype(auto) on struct&
            namespace ns1 {
            struct S {};
            } // namespace ns1

            ns1::S i;
            ns1::S& j = i;
            ^decltype(auto) k = j;
          )cpp",
          "ns1::S &",
      },
      {
          R"cpp( // auto on template class
              class X;
              template<typename T> class Foo {};
              ^auto v = Foo<X>();
          )cpp",
          "Foo<X>",
      },
      {
          R"cpp( // auto on initializer list.
              namespace std
              {
                template<class _E>
                class [[initializer_list]] {};
              }

              ^auto i = {1,2};
          )cpp",
          "std::initializer_list<int>",
      },
      {
          R"cpp( // auto in function return type with trailing return type
            struct Foo {};
            ^auto test() -> decltype(Foo()) {
              return Foo();
            }
          )cpp",
          "Foo",
      },
      {
          R"cpp( // decltype in trailing return type
            struct Foo {};
            auto test() -> ^decltype(Foo()) {
              return Foo();
            }
          )cpp",
          "Foo",
      },
      {
          R"cpp( // auto in function return type
            struct Foo {};
            ^auto test() {
              return Foo();
            }
          )cpp",
          "Foo",
      },
      {
          R"cpp( // auto& in function return type
            struct Foo {};
            ^auto& test() {
              static Foo x;
              return x;
            }
          )cpp",
          "Foo",
      },
      {
          R"cpp( // auto* in function return type
            struct Foo {};
            ^auto* test() {
              Foo *x;
              return x;
            }
          )cpp",
          "Foo",
      },
      {
          R"cpp( // const auto& in function return type
            struct Foo {};
            const ^auto& test() {
              static Foo x;
              return x;
            }
          )cpp",
          "Foo",
      },
      {
          R"cpp( // decltype(auto) in function return (value)
            struct Foo {};
            ^decltype(auto) test() {
              return Foo();
            }
          )cpp",
          "Foo",
      },
      {
          R"cpp( // decltype(auto) in function return (ref)
            struct Foo {};
            ^decltype(auto) test() {
              static Foo x;
              return (x);
            }
          )cpp",
          "Foo &",
      },
      {
          R"cpp( // decltype(auto) in function return (const ref)
            struct Foo {};
            ^decltype(auto) test() {
              static const Foo x;
              return (x);
            }
          )cpp",
          "const Foo &",
      },
      {
          R"cpp( // auto on alias
            struct Foo {};
            using Bar = Foo;
            ^auto x = Bar();
          )cpp",
          "Bar",
      },
      {
          R"cpp(
            // Generic lambda param.
            struct Foo{};
            auto Generic = [](^auto x) { return 0; };
            int m = Generic(Foo{});
          )cpp",
          "struct Foo",
      },
      {
          R"cpp(
            // Generic lambda instantiated twice, matching deduction.
            struct Foo{};
            auto Generic = [](^auto x, auto y) { return 0; };
            int m = Generic(Foo{}, "one");
            int n = Generic(Foo{}, 2);
          )cpp",
          // No deduction although both instantiations yield the same result :-(
          nullptr,
      },
      {
          R"cpp(
            // Generic lambda instantiated twice, conflicting deduction.
            struct Foo{};
            auto Generic = [](^auto y) { return 0; };
            int m = Generic("one");
            int n = Generic(2);
          )cpp",
          nullptr,
      },
      {
          R"cpp(
            // Generic function param.
            struct Foo{};
            int generic(^auto x) { return 0; }
            int m = generic(Foo{});
          )cpp",
          "struct Foo",
      },
      {
          R"cpp(
            // More complicated param type involving auto.
            template <class> concept C = true;
            struct Foo{};
            int generic(C ^auto *x) { return 0; }
            const Foo *Ptr = nullptr;
            int m = generic(Ptr);
          )cpp",
          "const struct Foo",
      },
  };
  for (Test T : Tests) {
    Annotations File(T.AnnotatedCode);
    auto TU = TestTU::withCode(File.code());
    TU.ExtraArgs.push_back("-std=c++20");
    auto AST = TU.build();
    SourceManagerForFile SM("foo.cpp", File.code());

    SCOPED_TRACE(T.AnnotatedCode);
    EXPECT_FALSE(File.points().empty());
    for (Position Pos : File.points()) {
      auto Location = sourceLocationInMainFile(SM.get(), Pos);
      ASSERT_TRUE(!!Location) << llvm::toString(Location.takeError());
      auto DeducedType = getDeducedType(AST.getASTContext(), *Location);
      if (T.DeducedType == nullptr) {
        EXPECT_FALSE(DeducedType);
      } else {
        ASSERT_TRUE(DeducedType);
        EXPECT_EQ(DeducedType->getAsString(), T.DeducedType);
      }
    }
  }
}

TEST(ClangdAST, GetOnlyInstantiation) {
  struct {
    const char *Code;
    llvm::StringLiteral NodeType;
    const char *Name;
  } Cases[] = {
      {
          R"cpp(
            template <typename> class X {};
            X<int> x;
          )cpp",
          "CXXRecord",
          "template<> class X<int> {}",
      },
      {
          R"cpp(
            template <typename T> T X = T{};
            int y = X<char>;
          )cpp",
          "Var",
          // VarTemplateSpecializationDecl doesn't print as template<>...
          "char X = char{}",
      },
      {
          R"cpp(
            template <typename T> int X(T) { return 42; }
            int y = X("text");
          )cpp",
          "Function",
          "template<> int X<const char *>(const char *)",
      },
      {
          R"cpp(
            int X(auto *x) { return 42; }
            int y = X("text");
          )cpp",
          "Function",
          "template<> int X<const char>(const char *x)",
      },
  };

  for (const auto &Case : Cases) {
    SCOPED_TRACE(Case.Code);
    auto TU = TestTU::withCode(Case.Code);
    TU.ExtraArgs.push_back("-std=c++20");
    auto AST = TU.build();
    PrintingPolicy PP = AST.getASTContext().getPrintingPolicy();
    PP.TerseOutput = true;
    std::string Name;
    if (auto *Result = getOnlyInstantiation(
            const_cast<NamedDecl *>(&findDecl(AST, [&](const NamedDecl &D) {
              return D.getDescribedTemplate() != nullptr &&
                     D.getDeclKindName() == Case.NodeType;
            })))) {
      llvm::raw_string_ostream OS(Name);
      Result->print(OS, PP);
    }

    if (Case.Name)
      EXPECT_EQ(Case.Name, Name);
    else
      EXPECT_THAT(Name, IsEmpty());
  }
}

TEST(ClangdAST, GetContainedAutoParamType) {
  auto TU = TestTU::withCode(R"cpp(
    int withAuto(
       auto a,
       auto *b,
       const auto *c,
       auto &&d,
       auto *&e,
       auto (*f)(int)
    ){};

    int withoutAuto(
      int a,
      int *b,
      const int *c,
      int &&d,
      int *&e,
      int (*f)(int)
    ){};
  )cpp");
  TU.ExtraArgs.push_back("-std=c++20");
  auto AST = TU.build();

  const auto &WithAuto =
      llvm::cast<FunctionTemplateDecl>(findDecl(AST, "withAuto"));
  auto ParamsWithAuto = WithAuto.getTemplatedDecl()->parameters();
  auto *TemplateParamsWithAuto = WithAuto.getTemplateParameters();
  ASSERT_EQ(ParamsWithAuto.size(), TemplateParamsWithAuto->size());

  for (unsigned I = 0; I < ParamsWithAuto.size(); ++I) {
    SCOPED_TRACE(ParamsWithAuto[I]->getNameAsString());
    auto Loc = getContainedAutoParamType(
        ParamsWithAuto[I]->getTypeSourceInfo()->getTypeLoc());
    ASSERT_FALSE(Loc.isNull());
    EXPECT_EQ(Loc.getTypePtr()->getDecl(), TemplateParamsWithAuto->getParam(I));
  }

  const auto &WithoutAuto =
      llvm::cast<FunctionDecl>(findDecl(AST, "withoutAuto"));
  for (auto *ParamWithoutAuto : WithoutAuto.parameters()) {
    ASSERT_TRUE(getContainedAutoParamType(
                    ParamWithoutAuto->getTypeSourceInfo()->getTypeLoc())
                    .isNull());
  }
}

TEST(ClangdAST, GetQualification) {
  // Tries to insert the decl `Foo` into position of each decl named `insert`.
  // This is done to get an appropriate DeclContext for the insertion location.
  // Qualifications are the required nested name specifier to spell `Foo` at the
  // `insert`ion location.
  // VisibleNamespaces are assumed to be visible at every insertion location.
  const struct {
    llvm::StringRef Test;
    std::vector<llvm::StringRef> Qualifications;
    std::vector<std::string> VisibleNamespaces;
  } Cases[] = {
      {
          R"cpp(
            namespace ns1 { namespace ns2 { class Foo {}; } }
            void insert(); // ns1::ns2::Foo
            namespace ns1 {
              void insert(); // ns2::Foo
              namespace ns2 {
                void insert(); // Foo
              }
              using namespace ns2;
              void insert(); // Foo
            }
            using namespace ns1;
            void insert(); // ns2::Foo
            using namespace ns2;
            void insert(); // Foo
          )cpp",
          {"ns1::ns2::", "ns2::", "", "", "ns2::", ""},
          {},
      },
      {
          R"cpp(
            namespace ns1 { namespace ns2 { class Bar { void Foo(); }; } }
            void insert(); // ns1::ns2::Bar::Foo
            namespace ns1 {
              void insert(); // ns2::Bar::Foo
              namespace ns2 {
                void insert(); // Bar::Foo
              }
              using namespace ns2;
              void insert(); // Bar::Foo
            }
            using namespace ns1;
            void insert(); // ns2::Bar::Foo
            using namespace ns2;
            void insert(); // Bar::Foo
          )cpp",
          {"ns1::ns2::Bar::", "ns2::Bar::", "Bar::", "Bar::", "ns2::Bar::",
           "Bar::"},
          {},
      },
      {
          R"cpp(
            namespace ns1 { namespace ns2 { void Foo(); } }
            void insert(); // ns2::Foo
            namespace ns1 {
              void insert(); // ns2::Foo
              namespace ns2 {
                void insert(); // Foo
              }
            }
          )cpp",
          {"ns2::", "ns2::", ""},
          {"ns1::"},
      },
      {
          R"cpp(
            namespace ns {
            extern "C" {
            typedef int Foo;
            }
            }
            void insert(); // ns::Foo
          )cpp",
          {"ns::"},
          {},
      },
  };
  for (const auto &Case : Cases) {
    Annotations Test(Case.Test);
    TestTU TU = TestTU::withCode(Test.code());
    ParsedAST AST = TU.build();
    std::vector<const Decl *> InsertionPoints;
    const NamedDecl *TargetDecl;
    findDecl(AST, [&](const NamedDecl &ND) {
      if (ND.getNameAsString() == "Foo") {
        TargetDecl = &ND;
        return true;
      }

      if (ND.getNameAsString() == "insert")
        InsertionPoints.push_back(&ND);
      return false;
    });

    ASSERT_EQ(InsertionPoints.size(), Case.Qualifications.size());
    for (size_t I = 0, E = InsertionPoints.size(); I != E; ++I) {
      const Decl *D = InsertionPoints[I];
      if (Case.VisibleNamespaces.empty()) {
        EXPECT_EQ(getQualification(AST.getASTContext(),
                                   D->getLexicalDeclContext(), D->getBeginLoc(),
                                   TargetDecl),
                  Case.Qualifications[I]);
      } else {
        EXPECT_EQ(getQualification(AST.getASTContext(),
                                   D->getLexicalDeclContext(), TargetDecl,
                                   Case.VisibleNamespaces),
                  Case.Qualifications[I]);
      }
    }
  }
}

TEST(ClangdAST, PrintType) {
  const struct {
    llvm::StringRef Test;
    std::vector<llvm::StringRef> Types;
  } Cases[] = {
      {
          R"cpp(
            namespace ns1 { namespace ns2 { class Foo {}; } }
            void insert(); // ns1::ns2::Foo
            namespace ns1 {
              void insert(); // ns2::Foo
              namespace ns2 {
                void insert(); // Foo
              }
            }
          )cpp",
          {"ns1::ns2::Foo", "ns2::Foo", "Foo"},
      },
      {
          R"cpp(
            namespace ns1 {
              typedef int Foo;
            }
            void insert(); // ns1::Foo
            namespace ns1 {
              void insert(); // Foo
            }
          )cpp",
          {"ns1::Foo", "Foo"},
      },
  };
  for (const auto &Case : Cases) {
    Annotations Test(Case.Test);
    TestTU TU = TestTU::withCode(Test.code());
    ParsedAST AST = TU.build();
    std::vector<const DeclContext *> InsertionPoints;
    const TypeDecl *TargetDecl = nullptr;
    findDecl(AST, [&](const NamedDecl &ND) {
      if (ND.getNameAsString() == "Foo") {
        if (const auto *TD = llvm::dyn_cast<TypeDecl>(&ND)) {
          TargetDecl = TD;
          return true;
        }
      } else if (ND.getNameAsString() == "insert")
        InsertionPoints.push_back(ND.getDeclContext());
      return false;
    });

    ASSERT_EQ(InsertionPoints.size(), Case.Types.size());
    for (size_t I = 0, E = InsertionPoints.size(); I != E; ++I) {
      const auto *DC = InsertionPoints[I];
      EXPECT_EQ(printType(AST.getASTContext().getTypeDeclType(TargetDecl), *DC),
                Case.Types[I]);
    }
  }
}

TEST(ClangdAST, IsDeeplyNested) {
  Annotations Test(
      R"cpp(
        namespace ns {
        class Foo {
          void bar() {
            class Bar {};
          }
        };
        })cpp");
  TestTU TU = TestTU::withCode(Test.code());
  ParsedAST AST = TU.build();

  EXPECT_TRUE(isDeeplyNested(&findUnqualifiedDecl(AST, "Foo"), /*MaxDepth=*/1));
  EXPECT_FALSE(
      isDeeplyNested(&findUnqualifiedDecl(AST, "Foo"), /*MaxDepth=*/2));

  EXPECT_TRUE(isDeeplyNested(&findUnqualifiedDecl(AST, "bar"), /*MaxDepth=*/2));
  EXPECT_FALSE(
      isDeeplyNested(&findUnqualifiedDecl(AST, "bar"), /*MaxDepth=*/3));

  EXPECT_TRUE(isDeeplyNested(&findUnqualifiedDecl(AST, "Bar"), /*MaxDepth=*/3));
  EXPECT_FALSE(
      isDeeplyNested(&findUnqualifiedDecl(AST, "Bar"), /*MaxDepth=*/4));
}

MATCHER_P(attrKind, K, "") { return arg->getKind() == K; }

MATCHER(implicitAttr, "") { return arg->isImplicit(); }

TEST(ClangdAST, GetAttributes) {
  const char *Code = R"cpp(
    class X{};
    class [[nodiscard]] Y{};
    void f(int * a, int * __attribute__((nonnull)) b);
    void foo(bool c) {
      if (c)
        [[unlikely]] return;
    }
  )cpp";
  ParsedAST AST = TestTU::withCode(Code).build();
  auto DeclAttrs = [&](llvm::StringRef Name) {
    return getAttributes(DynTypedNode::create(findUnqualifiedDecl(AST, Name)));
  };
  // Implicit attributes may be present (e.g. visibility on windows).
  ASSERT_THAT(DeclAttrs("X"), Each(implicitAttr()));
  ASSERT_THAT(DeclAttrs("Y"), Contains(attrKind(attr::WarnUnusedResult)));
  ASSERT_THAT(DeclAttrs("f"), Each(implicitAttr()));
  ASSERT_THAT(DeclAttrs("a"), Each(implicitAttr()));
  ASSERT_THAT(DeclAttrs("b"), Contains(attrKind(attr::NonNull)));

  Stmt *FooBody = cast<FunctionDecl>(findDecl(AST, "foo")).getBody();
  IfStmt *FooIf = cast<IfStmt>(cast<CompoundStmt>(FooBody)->body_front());
  ASSERT_THAT(getAttributes(DynTypedNode::create(*FooIf)),
              Each(implicitAttr()));
  ASSERT_THAT(getAttributes(DynTypedNode::create(*FooIf->getThen())),
              Contains(attrKind(attr::Unlikely)));
}

TEST(ClangdAST, HasReservedName) {
  ParsedAST AST = TestTU::withCode(R"cpp(
    void __foo();
    namespace std {
      inline namespace __1 { class error_code; }
      namespace __detail { int secret; }
    }
  )cpp")
                      .build();

  EXPECT_TRUE(hasReservedName(findUnqualifiedDecl(AST, "__foo")));
  EXPECT_FALSE(
      hasReservedScope(*findUnqualifiedDecl(AST, "__foo").getDeclContext()));

  EXPECT_FALSE(hasReservedName(findUnqualifiedDecl(AST, "error_code")));
  EXPECT_FALSE(hasReservedScope(
      *findUnqualifiedDecl(AST, "error_code").getDeclContext()));

  EXPECT_FALSE(hasReservedName(findUnqualifiedDecl(AST, "secret")));
  EXPECT_TRUE(
      hasReservedScope(*findUnqualifiedDecl(AST, "secret").getDeclContext()));
}

} // namespace
} // namespace clangd
} // namespace clang
