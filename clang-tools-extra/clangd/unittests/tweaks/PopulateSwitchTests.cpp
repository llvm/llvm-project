//===-- PopulateSwitchTest.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TweakTesting.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TWEAK_TEST(PopulateSwitch);

TEST_F(PopulateSwitchTest, Test) {
  struct Case {
    CodeContext Context;
    llvm::StringRef TestSource;
    llvm::StringRef ExpectedSource;
    llvm::StringRef FileName = "TestTU.cpp";
  };

  Case Cases[]{
      {
          // No enumerators
          Function,
          R""(enum Enum {}; ^switch ((Enum)0) {})"",
          "unavailable",
      },
      {
          // All enumerators already in switch (unscoped)
          Function,
          R""(enum Enum {A,B}; ^switch (A) {case A:break;case B:break;})"",
          "unavailable",
      },
      {
          // All enumerators already in switch (scoped)
          Function,
          R""(
            enum class Enum {A,B};
            ^switch (Enum::A) {case Enum::A:break;case Enum::B:break;}
          )"",
          "unavailable",
      },
      {
          // Default case in switch
          Function,
          R""(
            enum class Enum {A,B};
            ^switch (Enum::A) {default:break;}
          )"",
          "unavailable",
      },
      {
          // GNU range in switch
          Function,
          R""(
            enum class Enum {A,B};
            ^switch (Enum::A) {case Enum::A ... Enum::B:break;}
          )"",
          "unavailable",
      },
      {
          // Value dependent case expression
          File,
          R""(
            enum class Enum {A,B};
            template<Enum Value>
            void function() {
                ^switch (Enum::A) {case Value:break;}
            }
          )"",
          "unavailable",
      },
      {
          // Body not CompoundStmt
          Function,
          R""(enum Enum {A}; ^switch (A);)"",
          "unavailable",
      },
      {
          // Selection on switch token
          Function,
          R""(enum Enum {A}; ^switch (A) {})"",
          R""(enum Enum {A}; switch (A) {case A:break;})"",
      },
      {
          // Selection on switch condition
          Function,
          R""(enum Enum {A}; switch (^A) {})"",
          R""(enum Enum {A}; switch (A) {case A:break;})"",
      },
      {
          // Selection of whole switch condition
          Function,
          R""(enum Enum {A}; switch ([[A]]) {})"",
          R""(enum Enum {A}; switch (A) {case A:break;})"",
      },
      {
          // Selection in switch body
          Function,
          R""(enum Enum {A}; switch (A) {^})"",
          R""(enum Enum {A}; switch (A) {case A:break;})"",
      },
      {
          // Scoped enumeration
          Function,
          R""(enum class Enum {A}; ^switch (Enum::A) {})"",
          R""(enum class Enum {A}; switch (Enum::A) {case Enum::A:break;})"",
      },
      {
          // Scoped enumeration with multiple enumerators
          Function,
          R""(
            enum class Enum {A,B};
            ^switch (Enum::A) {}
          )"",
          R""(
            enum class Enum {A,B};
            switch (Enum::A) {case Enum::A:case Enum::B:break;}
          )"",
      },
      {
          // Only filling in missing enumerators (unscoped)
          Function,
          R""(
            enum Enum {A,B,C};
            ^switch (A) {case B:break;}
          )"",
          R""(
            enum Enum {A,B,C};
            switch (A) {case B:break;case A:case C:break;}
          )"",
      },
      {
          // Only filling in missing enumerators,
          // even when using integer literals
          Function,
          R""(
            enum Enum {A,B=1,C};
            ^switch (A) {case 1:break;}
          )"",
          R""(
            enum Enum {A,B=1,C};
            switch (A) {case 1:break;case A:case C:break;}
          )"",
      },
      {
          // Only filling in missing enumerators (scoped)
          Function,
          R""(
            enum class Enum {A,B,C};
            ^switch (Enum::A)
            {case Enum::B:break;}
          )"",
          R""(
            enum class Enum {A,B,C};
            switch (Enum::A)
            {case Enum::B:break;case Enum::A:case Enum::C:break;}
          )"",
      },
      {
          // Scoped enumerations in namespace
          File,
          R""(
            namespace ns { enum class Enum {A}; }
            void function() { ^switch (ns::Enum::A) {} }
          )"",
          R""(
            namespace ns { enum class Enum {A}; }
            void function() { switch (ns::Enum::A) {case ns::Enum::A:break;} }
          )"",
      },
      {
          // Unscoped enumerations in namespace
          File,
          R""(
            namespace ns { enum Enum {A}; }
            void function() { ^switch (ns::A) {} }
          )"",
          R""(
            namespace ns { enum Enum {A}; }
            void function() { switch (ns::A) {case ns::A:break;} }
          )"",
      },
      {
          // Duplicated constant names
          Function,
          R""(enum Enum {A,B,b=B}; ^switch (A) {})"",
          R""(enum Enum {A,B,b=B}; switch (A) {case A:case B:break;})"",
      },
      {
          // Duplicated constant names all in switch
          Function,
          R""(enum Enum {A,B,b=B}; ^switch (A) {case A:case B:break;})"",
          "unavailable",
      },
      {
          // Enum is dependent type
          File,
          R""(template<typename T> void f() {enum Enum {A}; ^switch (A) {}})"",
          "unavailable",
      },
      {// C: Only filling in missing enumerators
       Function,
       R""(
            enum CEnum {A,B,C};
            enum CEnum val = A;
            ^switch (val) {case B:break;}
          )"",
       R""(
            enum CEnum {A,B,C};
            enum CEnum val = A;
            switch (val) {case B:break;case A:case C:break;}
          )"",
       "TestTU.c"},
      {// C: Only filling in missing enumerators w/ typedefs
       Function,
       R""(
            typedef unsigned long UInteger;
            enum ControlState : UInteger;
            typedef enum ControlState ControlState;
            enum ControlState : UInteger {A,B,C};
            ControlState controlState = A;
            switch (^controlState) {case A:break;}
          )"",
       R""(
            typedef unsigned long UInteger;
            enum ControlState : UInteger;
            typedef enum ControlState ControlState;
            enum ControlState : UInteger {A,B,C};
            ControlState controlState = A;
            switch (controlState) {case A:break;case B:case C:break;}
          )"",
       "TestTU.c"},
  };

  for (const auto &Case : Cases) {
    Context = Case.Context;
    FileName = Case.FileName;
    EXPECT_EQ(apply(Case.TestSource), Case.ExpectedSource);
  }
}

} // namespace
} // namespace clangd
} // namespace clang
