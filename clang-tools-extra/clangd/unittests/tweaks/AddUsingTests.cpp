//===-- AddUsingTests.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Config.h"
#include "TweakTesting.h"
#include "support/Context.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"
#include <string>
#include <utility>

namespace clang {
namespace clangd {
namespace {

TWEAK_TEST(AddUsing);

TEST_F(AddUsingTest, Prepare) {
  Config Cfg;
  Cfg.Style.FullyQualifiedNamespaces.push_back("ban");
  WithContextValue WithConfig(Config::Key, std::move(Cfg));

  const std::string Header = R"cpp(
#define NS(name) one::two::name
namespace ban { void foo() {} }
namespace banana { void foo() {} }
namespace one {
void oo() {}
template<typename TT> class tt {};
namespace two {
enum ee { ee_enum_value };
void ff() {}
class cc {
public:
  struct st {};
  static void mm() {}
  cc operator|(const cc& x) const { return x; }
};
}
})cpp";

  EXPECT_AVAILABLE(Header + "void fun() { o^n^e^:^:^t^w^o^:^:^f^f(); }");
  EXPECT_AVAILABLE(Header + "void fun() { o^n^e^::^o^o(); }");
  EXPECT_AVAILABLE(Header + "void fun() { o^n^e^:^:^t^w^o^:^:^e^e E; }");
  EXPECT_AVAILABLE(Header + "void fun() { o^n^e^:^:^t^w^o:^:^c^c C; }");
  EXPECT_UNAVAILABLE(Header +
                     "void fun() { o^n^e^:^:^t^w^o^:^:^c^c^:^:^m^m(); }");
  EXPECT_UNAVAILABLE(Header +
                     "void fun() { o^n^e^:^:^t^w^o^:^:^c^c^:^:^s^t inst; }");
  EXPECT_UNAVAILABLE(Header +
                     "void fun() { o^n^e^:^:^t^w^o^:^:^c^c^:^:^s^t inst; }");
  EXPECT_UNAVAILABLE(Header + "void fun() { N^S(c^c) inst; }");
  // This used to crash. Ideally we would support this case, but for now we just
  // test that we don't crash.
  EXPECT_UNAVAILABLE(Header +
                     "template<typename TT> using foo = one::tt<T^T>;");
  // Test that we don't crash or misbehave on unnamed DeclRefExpr.
  EXPECT_UNAVAILABLE(Header +
                     "void fun() { one::two::cc() ^| one::two::cc(); }");
  // Do not offer code action when operating on a banned namespace.
  EXPECT_UNAVAILABLE(Header + "void fun() { ban::fo^o(); }");
  EXPECT_UNAVAILABLE(Header + "void fun() { ::ban::fo^o(); }");
  EXPECT_AVAILABLE(Header + "void fun() { banana::fo^o(); }");

  // NestedNameSpecifier, but no namespace.
  EXPECT_UNAVAILABLE(Header + "class Foo {}; class F^oo foo;");

  // Nested macro case.
  EXPECT_AVAILABLE(R"cpp(
  #define ID2(X) X
  #define ID(Y, X) Y;ID2(X)
  namespace ns { struct Foo{}; }
  ID(int xyz, ns::F^oo) f;)cpp");

  // Check that we do not trigger in header files.
  FileName = "test.h";
  ExtraArgs.push_back("-xc++-header"); // .h file is treated a C by default.
  EXPECT_UNAVAILABLE(Header + "void fun() { one::two::f^f(); }");
  FileName = "test.hpp";
  EXPECT_UNAVAILABLE(Header + "void fun() { one::two::f^f(); }");
}

TEST_F(AddUsingTest, Crash1072) {
  // Used to crash when traversing catch(...)
  // https://github.com/clangd/clangd/issues/1072
  const char *Code = R"cpp(
  namespace ns { class A; }
  ns::^A *err;
  void catchall() {
    try {} catch(...) {}
  }
  )cpp";
  EXPECT_AVAILABLE(Code);
}

TEST_F(AddUsingTest, Apply) {
  FileName = "test.cpp";
  struct {
    llvm::StringRef TestSource;
    llvm::StringRef ExpectedSource;
  } Cases[]{
      {
          // Function, no other using, namespace.
          R"cpp(
#include "test.hpp"
namespace {
void fun() {
  ^one::two::ff();
}
})cpp",
          R"cpp(
#include "test.hpp"
namespace {using one::two::ff;

void fun() {
  ff();
}
})cpp",
      },
      // Type, no other using, namespace.
      {
          R"cpp(
#include "test.hpp"
namespace {
void fun() {
  ::one::t^wo::cc inst;
}
})cpp",
          R"cpp(
#include "test.hpp"
namespace {using ::one::two::cc;

void fun() {
  cc inst;
}
})cpp",
      },
      // Type, no other using, no namespace.
      {
          R"cpp(
#include "test.hpp"

void fun() {
  one::two::e^e inst;
})cpp",
          R"cpp(
#include "test.hpp"

using one::two::ee;

void fun() {
  ee inst;
})cpp"},
      // Function, other usings.
      {
          R"cpp(
#include "test.hpp"

using one::two::cc;
using one::two::ee;

namespace {
void fun() {
  one::two::f^f();
}
})cpp",
          R"cpp(
#include "test.hpp"

using one::two::cc;
using one::two::ff;using one::two::ee;

namespace {
void fun() {
  ff();
}
})cpp",
      },
      // Function, other usings inside namespace.
      {
          R"cpp(
#include "test.hpp"

using one::two::cc;

namespace {

using one::two::ff;

void fun() {
  o^ne::oo();
}
})cpp",
          R"cpp(
#include "test.hpp"

using one::two::cc;

namespace {

using one::oo;using one::two::ff;

void fun() {
  oo();
}
})cpp"},
      // Using comes after cursor.
      {
          R"cpp(
#include "test.hpp"

namespace {

void fun() {
  one::t^wo::ff();
}

using one::two::cc;

})cpp",
          R"cpp(
#include "test.hpp"

namespace {using one::two::ff;


void fun() {
  ff();
}

using one::two::cc;

})cpp"},
      // Pointer type.
      {R"cpp(
#include "test.hpp"

void fun() {
  one::two::c^c *p;
})cpp",
       R"cpp(
#include "test.hpp"

using one::two::cc;

void fun() {
  cc *p;
})cpp"},
      // Namespace declared via macro.
      {R"cpp(
#include "test.hpp"
#define NS_BEGIN(name) namespace name {

NS_BEGIN(foo)

void fun() {
  one::two::f^f();
}
})cpp",
       R"cpp(
#include "test.hpp"
#define NS_BEGIN(name) namespace name {

using one::two::ff;

NS_BEGIN(foo)

void fun() {
  ff();
}
})cpp"},
      // Inside macro argument.
      {R"cpp(
#include "test.hpp"
#define CALL(name) name()

void fun() {
  CALL(one::t^wo::ff);
})cpp",
       R"cpp(
#include "test.hpp"
#define CALL(name) name()

using one::two::ff;

void fun() {
  CALL(ff);
})cpp"},
      // Parent namespace != lexical parent namespace
      {R"cpp(
#include "test.hpp"
namespace foo { void fun(); }

void foo::fun() {
  one::two::f^f();
})cpp",
       R"cpp(
#include "test.hpp"
using one::two::ff;

namespace foo { void fun(); }

void foo::fun() {
  ff();
})cpp"},
      // Inside a lambda.
      {
          R"cpp(
namespace NS {
void unrelated();
void foo();
}

auto L = [] {
  using NS::unrelated;
  NS::f^oo();
};)cpp",
          R"cpp(
namespace NS {
void unrelated();
void foo();
}

auto L = [] {
  using NS::foo;using NS::unrelated;
  foo();
};)cpp",
      },
      // If all other using are fully qualified, add ::
      {R"cpp(
#include "test.hpp"

using ::one::two::cc;
using ::one::two::ee;

void fun() {
  one::two::f^f();
})cpp",
       R"cpp(
#include "test.hpp"

using ::one::two::cc;
using ::one::two::ff;using ::one::two::ee;

void fun() {
  ff();
})cpp"},
      // Make sure we don't add :: if it's already there
      {R"cpp(
#include "test.hpp"

using ::one::two::cc;
using ::one::two::ee;

void fun() {
  ::one::two::f^f();
})cpp",
       R"cpp(
#include "test.hpp"

using ::one::two::cc;
using ::one::two::ff;using ::one::two::ee;

void fun() {
  ff();
})cpp"},
      // If even one using doesn't start with ::, do not add it
      {R"cpp(
#include "test.hpp"

using ::one::two::cc;
using one::two::ee;

void fun() {
  one::two::f^f();
})cpp",
       R"cpp(
#include "test.hpp"

using ::one::two::cc;
using one::two::ff;using one::two::ee;

void fun() {
  ff();
})cpp"},
      // using alias; insert using for the spelled name.
      {R"cpp(
#include "test.hpp"

void fun() {
  one::u^u u;
})cpp",
       R"cpp(
#include "test.hpp"

using one::uu;

void fun() {
  uu u;
})cpp"},
      // using namespace.
      {R"cpp(
#include "test.hpp"
using namespace one;
namespace {
two::c^c C;
})cpp",
       R"cpp(
#include "test.hpp"
using namespace one;
namespace {using two::cc;

cc C;
})cpp"},
      // Type defined in main file, make sure using is after that.
      {R"cpp(
namespace xx {
  struct yy {};
}

x^x::yy X;
)cpp",
       R"cpp(
namespace xx {
  struct yy {};
}

using xx::yy;

yy X;
)cpp"},
      // Type defined in main file via "using", insert after that.
      {R"cpp(
#include "test.hpp"

namespace xx {
  using yy = one::two::cc;
}

x^x::yy X;
)cpp",
       R"cpp(
#include "test.hpp"

namespace xx {
  using yy = one::two::cc;
}

using xx::yy;

yy X;
)cpp"},
      // Using must come after function definition.
      {R"cpp(
namespace xx {
  void yy();
}

void fun() {
  x^x::yy();
}
)cpp",
       R"cpp(
namespace xx {
  void yy();
}

using xx::yy;

void fun() {
  yy();
}
)cpp"},
      // Existing using with non-namespace part.
      {R"cpp(
#include "test.hpp"
using one::two::ee::ee_one;
one::t^wo::cc c;
)cpp",
       R"cpp(
#include "test.hpp"
using one::two::cc;using one::two::ee::ee_one;
cc c;
)cpp"},
      // Template (like std::vector).
      {R"cpp(
#include "test.hpp"
one::v^ec<int> foo;
)cpp",
       R"cpp(
#include "test.hpp"
using one::vec;

vec<int> foo;
)cpp"},
      // Typo correction.
      {R"cpp(
// error-ok
#include "test.hpp"
c^c C;
)cpp",
       R"cpp(
// error-ok
#include "test.hpp"
using one::two::cc;

cc C;
)cpp"},
      {R"cpp(
// error-ok
#include "test.hpp"
void foo() {
  switch(one::two::ee{}) { case two::ee_^one:break; }
}
)cpp",
       R"cpp(
// error-ok
#include "test.hpp"
using one::two::ee_one;

void foo() {
  switch(one::two::ee{}) { case ee_one:break; }
}
)cpp"},
      {R"cpp(
#include "test.hpp"
void foo() {
  one::f^unc_temp<int>();
})cpp",
       R"cpp(
#include "test.hpp"
using one::func_temp;

void foo() {
  func_temp<int>();
})cpp"},
      {R"cpp(
#include "test.hpp"
void foo() {
  one::va^r_temp<int>;
})cpp",
       R"cpp(
#include "test.hpp"
using one::var_temp;

void foo() {
  var_temp<int>;
})cpp"},
  };
  llvm::StringMap<std::string> EditedFiles;
  for (const auto &Case : Cases) {
    ExtraFiles["test.hpp"] = R"cpp(
namespace one {
void oo() {}
namespace two {
enum ee {ee_one};
void ff() {}
class cc {
public:
  struct st { struct nested {}; };
  static void mm() {}
};
}
using uu = two::cc;
template<typename T> struct vec {};
template <typename T> void func_temp();
template <typename T> T var_temp();
})cpp";
    // Typo correction is disabled in msvc-compatibility mode.
    ExtraArgs.push_back("-fno-ms-compatibility");
    EXPECT_EQ(apply(Case.TestSource, &EditedFiles), Case.ExpectedSource);
  }
}

TEST_F(AddUsingTest, ReplaceAllOccurrencesInScope) {
  FileName = "test.cpp";
  ExtraFiles["test.hpp"] = R"cpp(
namespace one {
namespace two {
void ff();
void ff(int);
struct SomeStruct {
  static void mm();
};
}
template <typename T> struct vec {};
template <typename T> void func_temp();
})cpp";

  // Replaces all occurrences across functions, return types, and parameters.
  EXPECT_EQ(apply(R"cpp(
#include "test.hpp"

one::two::SomeStruct make(one::two::SomeStruct s) {
  one::two::SomeStruct x = s;
  return x;
}

void fun() {
  one::two::f^f();
  one::two::ff(42);
}

void other() {
  one::two::ff();
}
)cpp"),
            R"cpp(
#include "test.hpp"

using one::two::ff;

one::two::SomeStruct make(one::two::SomeStruct s) {
  one::two::SomeStruct x = s;
  return x;
}

void fun() {
  ff();
  ff(42);
}

void other() {
  ff();
}
)cpp");

  EXPECT_EQ(apply(R"cpp(
#include "test.hpp"

one::two::SomeStruct make(one::two::SomeStruct s) {
  one::t^wo::SomeStruct x = s;
  one::two::SomeStruct::mm();
  return x;
}
)cpp"),
            R"cpp(
#include "test.hpp"

using one::two::SomeStruct;

SomeStruct make(SomeStruct s) {
  SomeStruct x = s;
  one::two::SomeStruct::mm();
  return x;
}
)cpp");

  // Templates with varying template arguments.
  EXPECT_EQ(apply(R"cpp(
#include "test.hpp"

void fun() {
  one::v^ec<int> a;
  one::vec<double> b;
}
)cpp"),
            R"cpp(
#include "test.hpp"

using one::vec;

void fun() {
  vec<int> a;
  vec<double> b;
}
)cpp");

  EXPECT_EQ(apply(R"cpp(
#include "test.hpp"

void fun() {
  one::f^unc_temp<int>();
  one::func_temp<double>();
}
)cpp"),
            R"cpp(
#include "test.hpp"

using one::func_temp;

void fun() {
  func_temp<int>();
  func_temp<double>();
}
)cpp");

  // Global qualifier and nested namespace spellings targeting the same symbol.
  EXPECT_EQ(apply(R"cpp(
#include "test.hpp"

void fun() {
  ::one::two::f^f();
  one::two::ff();
}
)cpp"),
            R"cpp(
#include "test.hpp"

using ::one::two::ff;

void fun() {
  ff();
  ff();
}
)cpp");

  // Nested macro arguments are rewritten; macro-body expansions are preserved.
  EXPECT_EQ(apply(R"cpp(
#include "test.hpp"
#define ID(X) X
#define CALL(name) name()
#define CALL_FF one::two::ff()

void fun() {
  CALL(ID(one::two::f^f));
  CALL_FF;
  one::two::ff();
}
)cpp"),
            R"cpp(
#include "test.hpp"
#define ID(X) X
#define CALL(name) name()
#define CALL_FF one::two::ff()

using one::two::ff;

void fun() {
  CALL(ID(ff));
  CALL_FF;
  ff();
}
)cpp");
}

TEST_F(AddUsingTest, RespectsScopeOrderAndShadowing) {
  FileName = "test.cpp";
  ExtraFiles["test.hpp"] = R"cpp(
namespace one {
namespace two {
void ff();
struct cc {};
}
}
)cpp";

  // Only rewrites inside the namespace where `using` is inserted, not in
  // sibling namespaces or before an anchored using-declaration.
  EXPECT_EQ(apply(R"cpp(
#include "test.hpp"

namespace a {
void f1() {
  one::two::f^f();
  one::two::ff();
}
}

namespace b {
using one::two::ff;
void f2() {
  one::two::ff();
}
}
)cpp"),
            R"cpp(
#include "test.hpp"

namespace a {using one::two::ff;

void f1() {
  ff();
  ff();
}
}

namespace b {
using one::two::ff;
void f2() {
  one::two::ff();
}
}
)cpp");

  // Existing using-declaration is preserved while redundant qualifiers after it
  // are removed; references before the insertion point stay qualified.
  EXPECT_EQ(apply(R"cpp(
#include "test.hpp"

void before() {
  one::two::ff();
}

using one::two::cc;

void after() {
  one::two::f^f();
  one::two::ff();
}
)cpp"),
            R"cpp(
#include "test.hpp"

void before() {
  one::two::ff();
}

using one::two::ff;using one::two::cc;

void after() {
  ff();
  ff();
}
)cpp");

  // Block-scoped using only rewrites within that block from the anchor onward.
  EXPECT_EQ(apply(R"cpp(
#include "test.hpp"

void fun() {
  {
    using one::two::cc;
    one::two::f^f();
    one::two::ff();
  }
  one::two::ff();
}
)cpp"),
            R"cpp(
#include "test.hpp"

void fun() {
  {
    using one::two::ff;using one::two::cc;
    ff();
    ff();
  }
  one::two::ff();
}
)cpp");

  // Shadowed occurrences, friend declarations, and out-of-line definitions keep
  // their qualifiers.
  EXPECT_EQ(apply(R"cpp(
namespace xx {
void yy();
}

void trigger() {
  x^x::yy();
}

void shadowedByParam(int yy) {
  xx::yy();
}

void shadowedByLocal() {
  xx::yy();
  int yy = 0;
  (void)yy;
  xx::yy();
}

struct ShadowedByMember {
  void method() { xx::yy(); }
  void yy();
};

class Befriender {
  friend void xx::yy();
};

void xx::yy() {
  xx::yy();
}
)cpp"),
            R"cpp(
namespace xx {
void yy();
}

using xx::yy;

void trigger() {
  yy();
}

void shadowedByParam(int yy) {
  xx::yy();
}

void shadowedByLocal() {
  yy();
  int yy = 0;
  (void)yy;
  xx::yy();
}

struct ShadowedByMember {
  void method() { xx::yy(); }
  void yy();
};

class Befriender {
  friend void xx::yy();
};

void xx::yy() {
  yy();
}
)cpp");
}

} // namespace
} // namespace clangd
} // namespace clang
