//===- unittests/StaticAnalyzer/TestReturnValueUnderConstruction.cpp ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Frontend/AnalysisConsumer.h"
#include "clang/StaticAnalyzer/Frontend/CheckerRegistry.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

namespace clang {
namespace ento {
namespace {

class TestReturnValueUnderConstructionChecker
  : public Checker<check::PostCall> {
public:
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const {
    // Only calls with origin expression are checked. These are `returnC()`,
    // `returnD()`, C::C() and D::D().
    if (!Call.getOriginExpr())
      return;

    // Since `returnC` returns an object by value, the invocation results
    // in an object of type `C` constructed into variable `c`. Thus the
    // return value of `CallEvent::getReturnValueUnderConstruction()` must
    // be non-empty and has to be a `MemRegion`.
    Optional<SVal> RetVal = Call.getReturnValueUnderConstruction();
    ASSERT_TRUE(RetVal);
    ASSERT_TRUE(RetVal->getAsRegion());

    const auto *RetReg = cast<TypedValueRegion>(RetVal->getAsRegion());
    const Expr *OrigExpr = Call.getOriginExpr();
    ASSERT_EQ(OrigExpr->getType()->getCanonicalTypeInternal(),
              RetReg->getValueType()->getCanonicalTypeInternal());
  }
};

void addTestReturnValueUnderConstructionChecker(
    AnalysisASTConsumer &AnalysisConsumer, AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages =
    {{"test.TestReturnValueUnderConstruction", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
      Registry.addChecker<TestReturnValueUnderConstructionChecker>(
          "test.TestReturnValueUnderConstruction", "", "");
    });
}

TEST(TestReturnValueUnderConstructionChecker,
     ReturnValueUnderConstructionChecker) {
  EXPECT_TRUE(runCheckerOnCode<addTestReturnValueUnderConstructionChecker>(
      R"(class C {
         public:
           C(int nn): n(nn) {}
           virtual ~C() {}
         private:
           int n;
         };

         C returnC(int m) {
           C c(m);
           return c;
         }

         void foo() {
           C c = returnC(1);
         })"));

  EXPECT_TRUE(runCheckerOnCode<addTestReturnValueUnderConstructionChecker>(
      R"(class C {
         public:
           C(int nn): n(nn) {}
           explicit C(): C(0) {}
           virtual ~C() {}
         private:
           int n;
         };

         C returnC() {
           C c;
           return c;
         }

         void foo() {
           C c = returnC();
         })"));

  EXPECT_TRUE(runCheckerOnCode<addTestReturnValueUnderConstructionChecker>(
      R"(class C {
         public:
           C(int nn): n(nn) {}
           virtual ~C() {}
         private:
           int n;
         };

         class D: public C {
         public:
           D(int nn): C(nn) {}
           virtual ~D() {}
         };

         D returnD(int m) {
           D d(m);
           return d;
         }

         void foo() {
           D d = returnD(1); 
         })"));
}

} // namespace
} // namespace ento
} // namespace clang
