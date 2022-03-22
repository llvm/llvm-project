//===-- AddUsingTests.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Config.h"
#include "TweakTesting.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

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
enum ee {};
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

  // Do not offer code action on typo-corrections.
  EXPECT_UNAVAILABLE(Header + "/*error-ok*/c^c C;");

  // NestedNameSpecifier, but no namespace.
  EXPECT_UNAVAILABLE(Header + "class Foo {}; class F^oo foo;");

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
  } Cases[]{{
                // Function, no other using, namespace.
                R"cpp(
#include "test.hpp"
namespace {
void fun() {
  ^o^n^e^:^:^t^w^o^:^:^f^f();
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
  ::on^e::t^wo::c^c inst;
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
  on^e::t^wo::e^e inst;
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
  o^ne::o^o();
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
)cpp"}};
  llvm::StringMap<std::string> EditedFiles;
  for (const auto &Case : Cases) {
    for (const auto &SubCase : expandCases(Case.TestSource)) {
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
})cpp";
      EXPECT_EQ(apply(SubCase, &EditedFiles), Case.ExpectedSource);
    }
  }
}

} // namespace
} // namespace clangd
} // namespace clang
