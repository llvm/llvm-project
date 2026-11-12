//===-- RemoveUsingNamespaceTest.cpp ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TweakTesting.h"
#include "gmock/gmock-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TWEAK_TEST(RemoveUsingNamespace);

TEST_F(RemoveUsingNamespaceTest, All) {
  std::pair<llvm::StringRef /*Input*/, llvm::StringRef /*Expected*/> Cases[] = {
      {// Remove all occurrences of ns. Qualify only unqualified.
       R"cpp(
      namespace ns1 { struct vector {}; }
      namespace ns2 { struct map {}; }
      using namespace n^s1;
      using namespace ns2;
      using namespace ns1;
      int main() {
        ns1::vector v1;
        vector v2;
        map m1;
      }
    )cpp",
       R"cpp(
      namespace ns1 { struct vector {}; }
      namespace ns2 { struct map {}; }
      
      using namespace ns2;
      
      int main() {
        ns1::vector v1;
        ns1::vector v2;
        map m1;
      }
    )cpp"},
      {// Ident to be qualified is a macro arg.
       R"cpp(
      #define DECLARE(x, y) x y
      namespace ns { struct vector {}; }
      using namespace n^s;
      int main() {
        DECLARE(ns::vector, v1);
        DECLARE(vector, v2);
      }
    )cpp",
       R"cpp(
      #define DECLARE(x, y) x y
      namespace ns { struct vector {}; }
      
      int main() {
        DECLARE(ns::vector, v1);
        DECLARE(ns::vector, v2);
      }
    )cpp"},
      {// Nested namespace: Fully qualify ident from inner ns.
       R"cpp(
      namespace aa { namespace bb { struct map {}; }}
      using namespace aa::b^b;
      int main() {
        map m;
      }
    )cpp",
       R"cpp(
      namespace aa { namespace bb { struct map {}; }}
      
      int main() {
        aa::bb::map m;
      }
    )cpp"},
      {// Nested namespace: Fully qualify ident from inner ns.
       R"cpp(
      namespace aa { namespace bb { struct map {}; }}
      using namespace a^a;
      int main() {
        bb::map m;
      }
    )cpp",
       R"cpp(
      namespace aa { namespace bb { struct map {}; }}
      
      int main() {
        aa::bb::map m;
      }
    )cpp"},
      {// Typedef.
       R"cpp(
      namespace aa { namespace bb { struct map {}; }}
      using namespace a^a;
      typedef bb::map map;
      int main() { map M; }
    )cpp",
       R"cpp(
      namespace aa { namespace bb { struct map {}; }}
      
      typedef aa::bb::map map;
      int main() { map M; }
    )cpp"},
      {// FIXME: Nested namespaces: Not aware of using ns decl of outer ns.
       R"cpp(
      namespace aa { namespace bb { struct map {}; }}
      using name[[space aa::b]]b;
      using namespace aa;
      int main() {
        map m;
      }
    )cpp",
       R"cpp(
      namespace aa { namespace bb { struct map {}; }}
      
      using namespace aa;
      int main() {
        aa::bb::map m;
      }
    )cpp"},
      {// Does not qualify ident from inner namespace.
       R"cpp(
      namespace aa { namespace bb { struct map {}; }}
      using namespace aa::bb;
      using namespace a^a;
      int main() {
        map m;
      }
    )cpp",
       R"cpp(
      namespace aa { namespace bb { struct map {}; }}
      using namespace aa::bb;
      
      int main() {
        map m;
      }
    )cpp"},
      {// Available only for top level namespace decl.
       R"cpp(
        namespace aa {
          namespace bb { struct map {}; }
          using namespace b^b;
        }
        int main() { aa::map m; }
    )cpp",
       "unavailable"},
      {// FIXME: Unavailable for namespaces containing using-namespace decl.
       R"cpp(
      namespace aa {
        namespace bb { struct map {}; }
        using namespace bb;
      }
      using namespace a^a;
      int main() {
        map m;
      }
    )cpp",
       "unavailable"},
      {R"cpp(
      namespace a::b { struct Foo {}; }
      using namespace a;
      using namespace a::[[b]];
      using namespace b;
      int main() { Foo F;}
    )cpp",
       R"cpp(
      namespace a::b { struct Foo {}; }
      using namespace a;
      
      
      int main() { a::b::Foo F;}
    )cpp"},
      {R"cpp(
      namespace a::b { struct Foo {}; }
      using namespace a;
      using namespace a::b;
      using namespace [[b]];
      int main() { Foo F;}
    )cpp",
       R"cpp(
      namespace a::b { struct Foo {}; }
      using namespace a;
      
      
      int main() { b::Foo F;}
    )cpp"},
      {// Enumerators.
       R"cpp(
      namespace tokens {
      enum Token {
        comma, identifier, numeric
      };
      }
      using namespace tok^ens;
      int main() {
        auto x = comma;
      }
    )cpp",
       R"cpp(
      namespace tokens {
      enum Token {
        comma, identifier, numeric
      };
      }
      
      int main() {
        auto x = tokens::comma;
      }
    )cpp"},
      {// inline namespaces.
       R"cpp(
      namespace std { inline namespace ns1 { inline namespace ns2 { struct vector {}; }}}
      using namespace st^d;
      int main() {
        vector V;
      }
    )cpp",
       R"cpp(
      namespace std { inline namespace ns1 { inline namespace ns2 { struct vector {}; }}}
      
      int main() {
        std::vector V;
      }
    )cpp"},
      {// Does not qualify operators declared in a non-class context
       R"cpp(
      namespace ns {
      struct Foo {};
      void operator+(const Foo &, int) {}
      }
      using namespace n^s;
      int main() {
        Foo foo;
        foo + 10;
      }
    )cpp",
       R"cpp(
      namespace ns {
      struct Foo {};
      void operator+(const Foo &, int) {}
      }
      
      int main() {
        ns::Foo foo;
        foo + 10;
      }
    )cpp"},
      {// Does not qualify user-defined literals
       R"cpp(
      namespace ns {
      long double operator "" _w(long double);
      }
      using namespace n^s;
      int main() { 1.5_w; }
    )cpp",
       R"cpp(
      namespace ns {
      long double operator "" _w(long double);
      }
      
      int main() { 1.5_w; }
    )cpp"},
      {
          R"cpp(
      namespace a { inline namespace b { void foobar(); } }
      using namespace a::[[b]];
      int main() { foobar(); }
    )cpp",
          R"cpp(
      namespace a { inline namespace b { void foobar(); } }
      
      int main() { a::b::foobar(); }
    )cpp"}};
  for (auto C : Cases)
    EXPECT_EQ(C.second, apply(C.first)) << C.first;
}

} // namespace
} // namespace clangd
} // namespace clang
