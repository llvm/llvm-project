//===- UncheckedStatusOrAccessModelTestFixture.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UncheckedStatusOrAccessModelTestFixture.h"
#include "MockHeaders.h"
#include "llvm/Support/ErrorHandling.h"

#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace clang::dataflow::statusor_model {
namespace {

TEST_P(UncheckedStatusOrAccessModelTest, NoStatusOrMention) {
  ExpectDiagnosticsFor(R"cc(
    void target() { "nop"; }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest,
       UnwrapWithoutCheck_Lvalue_CallToValue) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, NonExplicitInitialization) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"
    STATUSOR_INT target() {
      STATUSOR_INT x = Make<STATUSOR_INT>();
      return x.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest,
       UnwrapWithoutCheck_Lvalue_CallToValue_NewLine) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      sor.  // force newline
      value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest,
       UnwrapWithoutCheck_Rvalue_CallToValue) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      std::move(sor).value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest,
       UnwrapWithoutCheck_Lvalue_CallToValueOrDie) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      sor.ValueOrDie();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest,
       UnwrapWithoutCheck_Rvalue_CallToValueOrDie) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      std::move(sor).ValueOrDie();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest,
       UnwrapWithoutCheck_Lvalue_CallToOperatorStar) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      *sor;  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest,
       UnwrapWithoutCheck_Lvalue_CallToOperatorStarSeparateLine) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      *  // [[unsafe]]
          sor;
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest,
       UnwrapWithoutCheck_Rvalue_CallToOperatorStar) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      *std::move(sor);  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest,
       UnwrapWithoutCheck_Lvalue_CallToOperatorArrow) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    struct Foo {
      void foo();
    };

    void target(absl::StatusOr<Foo> sor) {
      sor->foo();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest,
       UnwrapWithoutCheck_Rvalue_CallToOperatorArrow) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    struct Foo {
      void foo();
    };

    void target(absl::StatusOr<Foo> sor) {
      std::move(sor)->foo();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, UnwrapRvalueWithCheck) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      if (sor.ok()) std::move(sor).value();
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, ParensInDeclInitExpr) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target() {
      auto sor = (Make<STATUSOR_INT>());
      if (sor.ok()) sor.value();
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, ReferenceInDeclInitExpr) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    struct Foo {
      const STATUSOR_INT& GetStatusOrInt() const;
    };

    void target(Foo foo) {
      auto sor = foo.GetStatusOrInt();
      if (sor.ok()) sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    struct Foo {
      STATUSOR_INT& GetStatusOrInt();
    };

    void target(Foo foo) {
      auto sor = foo.GetStatusOrInt();
      if (sor.ok()) sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    struct Foo {
      STATUSOR_INT&& GetStatusOrInt() &&;
    };

    void target(Foo foo) {
      auto sor = std::move(foo).GetStatusOrInt();
      if (sor.ok()) sor.value();
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        void target(STATUSOR_INT sor) {
          if (sor.ok())
            sor.value();
          else
            sor.value();  // [[unsafe]]

          sor.value();  // [[unsafe]]
        }
      )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target() {
      if (auto sor = Make<STATUSOR_INT>(); sor.ok())
        sor.value();
      else
        sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, JoinSafeSafe) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        void target(STATUSOR_INT sor, bool b) {
          if (sor.ok()) {
            if (b)
              sor.value();
            else
              sor.value();
          }
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, JoinUnsafeUnsafe) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor, bool b) {
      if (b)
        sor.value();  // [[unsafe]]
      else
        sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, InversedIfThenElse) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        void target(STATUSOR_INT sor) {
          if (!sor.ok())
            sor.value();  // [[unsafe]]
          else
            sor.value();

          sor.value();  // [[unsafe]]
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, DoubleInversedIfThenElse) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      if (!!sor.ok())
        sor.value();
      else
        sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, TripleInversedIfThenElse) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      if (!!!sor.ok())
        sor.value();  // [[unsafe]]
      else
        sor.value();
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_LhsAndRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      if (x.ok() && y.ok()) {
        x.value();

        y.value();
      } else {
        x.value();  // [[unsafe]]

        y.value();  // [[unsafe]]
      }
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_NotLhsAndRhs) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          if (!x.ok() && y.ok()) {
            y.value();

            x.value();  // [[unsafe]]
          } else {
            x.value();  // [[unsafe]]

            y.value();  // [[unsafe]]
          }
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_LhsAndNotRhs) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          if (x.ok() && !y.ok()) {
            x.value();

            y.value();  // [[unsafe]]
          } else {
            x.value();  // [[unsafe]]

            y.value();  // [[unsafe]]
          }
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_NotLhsAndNotRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      if (!x.ok() && !y.ok()) {
        x.value();  // [[unsafe]]

        y.value();  // [[unsafe]]
      } else {
        x.value();  // [[unsafe]]

        y.value();  // [[unsafe]]
      }
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_Not_LhsAndRhs) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          if (!(x.ok() && y.ok())) {
            x.value();  // [[unsafe]]

            y.value();  // [[unsafe]]
          } else {
            x.value();

            y.value();
          }
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_Not_NotLhsAndRhs) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          if (!(!x.ok() && y.ok())) {
            x.value();  // [[unsafe]]

            y.value();  // [[unsafe]]
          } else {
            y.value();

            x.value();  // [[unsafe]]
          }
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_Not_LhsAndNotRhs) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          if (!(x.ok() && !y.ok())) {
            x.value();  // [[unsafe]]

            y.value();  // [[unsafe]]
          } else {
            x.value();

            y.value();  // [[unsafe]]
          }
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_Not_NotLhsAndNotRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      if (!(!x.ok() && !y.ok())) {
        x.value();  // [[unsafe]]

        y.value();  // [[unsafe]]
      } else {
        x.value();  // [[unsafe]]

        y.value();  // [[unsafe]]
      }
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_LhsOrRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      if (x.ok() || y.ok()) {
        x.value();  // [[unsafe]]

        y.value();  // [[unsafe]]
      } else {
        x.value();  // [[unsafe]]

        y.value();  // [[unsafe]]
      }
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_NotLhsOrRhs) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          if (!x.ok() || y.ok()) {
            x.value();  // [[unsafe]]

            y.value();  // [[unsafe]]
          } else {
            x.value();

            y.value();  // [[unsafe]]
          }
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_LhsOrNotRhs) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          if (x.ok() || !y.ok()) {
            x.value();  // [[unsafe]]

            y.value();  // [[unsafe]]
          } else {
            y.value();

            x.value();  // [[unsafe]]
          }
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_NotLhsOrNotRhs) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          if (!x.ok() || !y.ok()) {
            x.value();  // [[unsafe]]

            y.value();  // [[unsafe]]
          } else {
            x.value();

            y.value();
          }
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_Not_LhsOrRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      if (!(x.ok() || y.ok())) {
        x.value();  // [[unsafe]]

        y.value();  // [[unsafe]]
      } else {
        x.value();  // [[unsafe]]

        y.value();  // [[unsafe]]
      }
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_Not_NotLhsOrRhs) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          if (!(!x.ok() || y.ok())) {
            x.value();

            y.value();  // [[unsafe]]
          } else {
            x.value();  // [[unsafe]]

            y.value();  // [[unsafe]]
          }
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_Not_LhsOrNotRhs) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          if (!(x.ok() || !y.ok())) {
            y.value();

            x.value();  // [[unsafe]]
          } else {
            x.value();  // [[unsafe]]

            y.value();  // [[unsafe]]
          }
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_Not_NotLhsOrNotRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      if (!(!x.ok() || !y.ok())) {
        x.value();

        y.value();
      } else {
        x.value();  // [[unsafe]]

        y.value();  // [[unsafe]]
      }
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, TerminatingIfThenBranch) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      if (!sor.ok()) return;

      sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      if (sor.ok()) return;

      sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          if (!x.ok() || !y.ok()) return;

          x.value();

          y.value();
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, TerminatingIfElseBranch) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      if (sor.ok()) {
      } else {
        return;
      }

      sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      if (!sor.ok()) {
      } else {
        return;
      }

      sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, TerminatingIfThenBranchInLoop) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      while (Make<bool>()) {
        if (!sor.ok()) continue;

        sor.value();
      }

      sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      while (Make<bool>()) {
        if (!sor.ok()) break;

        sor.value();
      }

      sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, TernaryConditionalOperator) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      sor.ok() ? sor.value() : 21;

      sor.ok() ? 21 : sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      !sor.ok() ? 21 : sor.value();

      !sor.ok() ? sor.value() : 21;  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor1, STATUSOR_INT sor2) {
      !((__builtin_expect(false || (!(sor1.ok() && sor2.ok())), false)))
          ? (void)0
          : (void)1;
      do {
        sor1.value();  // [[unsafe]]
        sor2.value();  // [[unsafe]]
      } while (true);
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      while (Make<bool>()) sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      while (sor.ok()) sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      while (!sor.ok()) sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      while (!!sor.ok()) sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      while (!!!sor.ok()) sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_LhsAndRhs) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          while (x.ok() && y.ok()) {
            x.value();

            y.value();
          }
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_NotLhsAndRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!x.ok() && y.ok()) x.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!x.ok() && y.ok()) y.value();
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_LhsAndNotRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (x.ok() && !y.ok()) x.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (x.ok() && !y.ok()) y.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_NotLhsAndNotRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!x.ok() && !y.ok()) x.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!x.ok() && !y.ok()) y.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_Not_LhsAndRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!(x.ok() && y.ok())) x.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!(x.ok() && y.ok())) y.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_Not_NotLhsAndRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!(!x.ok() && y.ok())) x.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!(!x.ok() && y.ok())) y.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_Not_LhsAndNotRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!(x.ok() && !y.ok())) x.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!(x.ok() && !y.ok())) y.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_Not_NotLhsAndNotRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!(!x.ok() && !y.ok())) x.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!(!x.ok() && !y.ok())) y.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_LhsOrRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (x.ok() || y.ok()) x.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (x.ok() || y.ok()) y.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_NotLhsOrRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!x.ok() || y.ok()) x.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!x.ok() || y.ok()) y.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_LhsOrNotRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (x.ok() || !y.ok()) x.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (x.ok() || !y.ok()) y.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_NotLhsOrNotRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!x.ok() || !y.ok()) x.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!x.ok() || !y.ok()) y.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_Not_LhsOrRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!(x.ok() || y.ok())) x.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!(x.ok() || y.ok())) y.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_Not_NotLhsOrRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!(!x.ok() || y.ok())) x.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!(!x.ok() || y.ok())) y.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_Not_LhsOrNotRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!(x.ok() || !y.ok())) x.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!(x.ok() || !y.ok())) y.value();
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_Not_NotLhsOrNotRhs) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          while (!(!x.ok() || !y.ok())) {
            x.value();

            y.value();
          }
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_AccessAfterStmt) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      while (sor.ok()) {
      }

      sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      while (!sor.ok()) {
      }

      sor.value();
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_TerminatingBranch_Return) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      while (!sor.ok()) return;

      sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      while (sor.ok()) return;

      sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_NestedIfWithBinaryCondition) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          while (Make<bool>()) {
            if (x.ok() && y.ok()) {
              x.value();

              y.value();
            }
          }
        }
      )cc");
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          while (Make<bool>()) {
            if (!(!x.ok() || !y.ok())) {
              x.value();

              y.value();
            }
          }
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, BuiltinExpect) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          if (!__builtin_expect(!x.ok() || __builtin_expect(!y.ok(), true), false)) {
            x.value();

            y.value();
          }
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, CopyAssignment) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target() {
      STATUSOR_INT sor = Make<STATUSOR_INT>();
      if (sor.ok()) {
        sor = Make<STATUSOR_INT>();
        sor.value();  // [[unsafe]]
      }
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target() {
      STATUSOR_INT sor = Make<STATUSOR_INT>();
      if (!sor.ok()) return;

      sor = Make<STATUSOR_INT>();
      sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target() {
      STATUSOR_INT x = Make<STATUSOR_INT>();
      if (x.ok()) {
        STATUSOR_INT y = x;
        x = Make<STATUSOR_INT>();

        y.value();

        x.value();  // [[unsafe]]
      }
    }
  )cc");
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        void target() {
          STATUSOR_INT x = Make<STATUSOR_INT>();
          STATUSOR_INT y = x;
          if (!y.ok()) return;

          x.value();

          y = Make<STATUSOR_INT>();
          x.value();
        }
      )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    struct Foo {
      STATUSOR_INT bar;
    };

    void target(Foo foo) {
      foo.bar = Make<STATUSOR_INT>();
      if (foo.bar.ok()) {
        foo.bar.value();

        foo.bar = Make<STATUSOR_INT>();
        foo.bar.value();  // [[unsafe]]
      }
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, ShortCircuitingBinaryOperators) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_BOOL sor) {
      bool b = sor.ok() & sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_BOOL sor) {
      bool b = sor.ok() && sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_BOOL sor) {
      bool b = !sor.ok() && sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_BOOL sor) {
      bool b = sor.ok() || sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_BOOL sor) {
      bool b = !sor.ok() || sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(bool b, STATUSOR_INT sor) {
      if (b || sor.ok()) {
        do {
          sor.value();  // [[unsafe]]
        } while (true);
      }
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(bool b, STATUSOR_INT sor) {
      if (__builtin_expect(b || sor.ok(), false)) {
        do {
          sor.value();  // [[unsafe]]
        } while (false);
      }
    }
  )cc");
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        void target(STATUSOR_INT sor1, STATUSOR_INT sor2) {
          while (sor1.ok() && sor2.ok()) sor1.value();
          while (sor1.ok() && sor2.ok()) sor2.value();
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, References) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target() {
      STATUSOR_INT x = Make<STATUSOR_INT>();
      STATUSOR_INT& y = x;
      if (x.ok()) {
        x.value();

        y.value();
      } else {
        x.value();  // [[unsafe]]

        y.value();  // [[unsafe]]
      }
    }
  )cc");
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        void target() {
          STATUSOR_INT x = Make<STATUSOR_INT>();
          STATUSOR_INT& y = x;
          if (y.ok()) {
            x.value();

            y.value();
          } else {
            x.value();  // [[unsafe]]

            y.value();  // [[unsafe]]
          }
        }
      )cc");
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        void target() {
          STATUSOR_INT x = Make<STATUSOR_INT>();
          STATUSOR_INT& y = x;
          if (!y.ok()) return;

          x.value();

          y = Make<STATUSOR_INT>();
          x.value();  // [[unsafe]]
        }
      )cc");
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        void target() {
          STATUSOR_INT x = Make<STATUSOR_INT>();
          const STATUSOR_INT& y = x;
          if (!y.ok()) return;

          y.value();

          x = Make<STATUSOR_INT>();
          y.value();  // [[unsafe]]
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, NoReturnAttribute) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    __attribute__((noreturn)) void f();

    void target(STATUSOR_INT sor) {
      if (!sor.ok()) f();

      sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void f();

    void target(STATUSOR_INT sor) {
      if (!sor.ok()) f();

      sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    struct Foo {
      __attribute__((noreturn)) ~Foo();
      void Bar();
    };

    void target(STATUSOR_INT sor) {
      if (!sor.ok()) Foo().Bar();

      sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    struct Foo {
      ~Foo();
      void Bar();
    };

    void target(STATUSOR_INT sor) {
      if (!sor.ok()) Foo().Bar();

      sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void f();
    __attribute__((noreturn)) void g();

    void target(STATUSOR_INT sor) {
      sor.ok() ? f() : g();

      sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    __attribute__((noreturn)) void f();
    void g();

    void target(STATUSOR_INT sor) {
      !sor.ok() ? f() : g();

      sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void f();
    void g();

    void target(STATUSOR_INT sor) {
      sor.ok() ? f() : g();

      sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void terminate() __attribute__((noreturn));

    void target(STATUSOR_INT sor) {
      sor.value();  // [[unsafe]]
      terminate();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void terminate() __attribute__((noreturn));

    void target(STATUSOR_INT sor) {
      if (sor.ok()) sor.value();
      terminate();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void terminate() __attribute__((noreturn));

    struct Foo {
      ~Foo() __attribute__((noreturn));
    };

    void target() {
      auto sor = Make<absl::StatusOr<Foo>>();
      !(false || !(sor.ok())) ? (void)0 : terminate();
      sor.value();
      terminate();
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, DeclInLoop) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      while (auto ok = sor.ok()) sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    using BoolAlias = bool;

    void target(STATUSOR_INT sor) {
      while (BoolAlias ok = sor.ok()) sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target() {
      while (Make<bool>()) {
        STATUSOR_INT sor = Make<STATUSOR_INT>();
        sor.value();  // [[unsafe]]
      }
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    using StatusOrInt = STATUSOR_INT;

    void target() {
      while (Make<bool>()) {
        StatusOrInt sor = Make<STATUSOR_INT>();
        sor.value();  // [[unsafe]]
      }
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, NonEvaluatedExprInCondition) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        bool unknown();

        void target(STATUSOR_INT sor) {
          if (unknown() && sor.ok()) sor.value();
          if (sor.ok() && unknown()) sor.value();
        }
      )cc");
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        bool unknown();

        void target(STATUSOR_INT sor) {
          if (!(!unknown() || !sor.ok())) sor.value();
          if (!(!sor.ok() || !unknown())) sor.value();
        }
      )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    bool unknown();

    void target(STATUSOR_INT sor) {
      if (unknown() || sor.ok()) sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    bool unknown();

    void target(STATUSOR_INT sor) {
      if (sor.ok() || unknown()) sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, CorrelatedBranches) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(bool b, STATUSOR_INT sor) {
      if (b || sor.ok()) {
        if (!b) sor.value();
      }
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(bool b, STATUSOR_INT sor) {
      if (b && !sor.ok()) return;
      if (b) sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(bool b, STATUSOR_INT sor) {
      if (sor.ok()) b = true;
      if (b) sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(bool b, STATUSOR_INT sor) {
      if (b) return;
      if (sor.ok()) b = true;
      if (b) sor.value();
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, ConditionWithInitStmt) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target() {
      if (STATUSOR_INT sor = Make<STATUSOR_INT>(); sor.ok()) sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target() {
      if (STATUSOR_INT sor = Make<STATUSOR_INT>(); !sor.ok())
        sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, DeadCode) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      bool b = false;
      if (b) sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      bool b;
      b = false;
      if (b) sor.value();
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, TemporaryDestructors) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      sor.ok() ? sor.value() : Fatal().value();

      sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      !sor.ok() ? Fatal().value() : sor.value();

      sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(bool b, STATUSOR_INT sor) {
      b ? 0 : sor.ok() ? sor.value() : Fatal().value();

      sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(bool b, STATUSOR_INT sor) {
      for (int i = 0; i < 10; i++) {
        (b && sor.ok()) ? sor.value() : Fatal().value();

        if (b) sor.value();
      }
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(bool b, STATUSOR_INT sor) {
      for (int i = 0; i < 10; i++) {
        (b || !sor.ok()) ? Fatal().value() : 0;

        if (!b) sor.value();
      }
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(bool b, STATUSOR_INT sor) {
      for (int i = 0; i < 10; i++) {
        (false || !(b && sor.ok())) ? Fatal().value() : 0;

        do {
          sor.value();
        } while (b);
      }
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, CheckMacro) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      CHECK(sor.ok());
      sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      CHECK(!sor.ok());
      sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, QcheckMacro) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      QCHECK(sor.ok());
      sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      QCHECK(!sor.ok());
      sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, CheckNeMacro) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      CHECK_NE(sor.status(), absl::OkStatus());
      sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, QcheckNeMacro) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      QCHECK_NE(sor.status(), absl::OkStatus());
      sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, GlobalVars) {
  // The following examples are not sound as there could be opaque calls between
  // the ok() and the value() calls that change the StatusOr value.
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    static STATUSOR_INT sor;

    void target() {
      if (sor.ok())
        sor.value();
      else
        sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target() {
      static STATUSOR_INT sor;
      if (sor.ok())
        sor.value();
      else
        sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    struct Foo {
      static STATUSOR_INT sor;
    };

    void target(Foo foo) {
      if (foo.sor.ok())
        foo.sor.value();
      else
        foo.sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    struct Foo {
      static STATUSOR_INT sor;
    };

    void target() {
      if (Foo::sor.ok())
        Foo::sor.value();
      else
        Foo::sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    struct Foo {
      static STATUSOR_INT sor;

      static void target() {
        if (sor.ok())
          sor.value();
        else
          sor.value();  // [[unsafe]]
      }
    };
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    struct Foo {
      static STATUSOR_INT sor;

      void target() {
        if (sor.ok())
          sor.value();
        else
          sor.value();  // [[unsafe]]
      }
    };
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    struct S {
      static const int x = -1;
    };

    int target(S s) { return s.x; }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, ReferenceReceivers) {
  // The following examples are not sound as there could be opaque calls between
  // the ok() and the value() calls that change the StatusOr value. However,
  // this is the behavior that users expect so it is here to stay.
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT& sor) {
      if (sor.ok())
        sor.value();
      else
        sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    struct Foo {
      STATUSOR_INT& sor;
    };

    void target(Foo foo) {
      if (foo.sor.ok())
        foo.sor.value();
      else
        foo.sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    struct Bar {
      STATUSOR_INT sor;
    };

    struct Foo {
      Bar& bar;
    };

    void target(Foo foo) {
      if (foo.bar.sor.ok())
        foo.bar.sor.value();
      else
        foo.bar.sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    struct Foo {
      STATUSOR_INT& sor;
    };

    void target(Foo& foo) {
      if (foo.sor.ok())
        foo.sor.value();
      else
        foo.sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, Lambdas) {
  ExpectDiagnosticsForLambda(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target() {
      [](STATUSOR_INT sor) {
        if (sor.ok())
          sor.value();
        else
          sor.value();  // [[unsafe]]
      }(Make<STATUSOR_INT>());
    }
  )cc");
  ExpectDiagnosticsForLambda(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      [sor]() {
        if (sor.ok())
          sor.value();
        else
          sor.value();  // [[unsafe]]
      }();
    }
  )cc");
  ExpectDiagnosticsForLambda(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      [&sor]() {
        if (sor.ok())
          sor.value();
        else
          sor.value();  // [[unsafe]]
      }();
    }
  )cc");
  ExpectDiagnosticsForLambda(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      [sor2 = sor]() {
        if (sor2.ok())
          sor2.value();
        else
          sor2.value();  // [[unsafe]]
      }();
    }
  )cc");
  ExpectDiagnosticsForLambda(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      [&]() {
        if (sor.ok())
          sor.value();
        else
          sor.value();  // [[unsafe]]
      }();
    }
  )cc");
  ExpectDiagnosticsForLambda(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      [=]() {
        if (sor.ok())
          sor.value();
        else
          sor.value();  // [[unsafe]]
      }();
    }
  )cc");
  ExpectDiagnosticsForLambda(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    struct Foo {
      STATUSOR_INT sor;

      void target() {
        [this]() {
          if (sor.ok())
            sor.value();
          else
            sor.value();  // [[unsafe]]
        }();
      }
    };
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, GoodLambda) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    int target() {
      STATUSOR_INT sor = Make<STATUSOR_INT>();
      if (sor.ok()) return [&s = sor.value()] { return s; }();
      return 0;
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, Status) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void foo();

    void target(STATUS s) {
      if (s.ok()) foo();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void foo();

    void target() {
      STATUS s = Make<STATUSOR_INT>().status();
      if (s.ok()) foo();
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, ExpectThatMacro) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      EXPECT_THAT(sor, testing::status::IsOk());

      sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      EXPECT_THAT(sor.status(), testing::status::IsOk());

      sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target() {
      STATUSOR_INT sor = Make<STATUSOR_INT>();
      EXPECT_THAT(sor, testing::status::IsOk());

      sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, ExpectOkMacro) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      EXPECT_OK(sor);

      sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      EXPECT_OK(sor.status());

      sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target() {
      STATUSOR_INT sor = Make<STATUSOR_INT>();
      EXPECT_OK(sor);

      sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, BreadthFirstBlockTraversalLoop) {
  // Evaluating the CFG blocks of the code below in breadth-first order results
  // in an infinite loop. Each iteration of the while loop below results in a
  // new value being assigned to the storage location of sor1. However,
  // following a bread-first order of evaluation, downstream blocks will join
  // environments of different generations of predecessor blocks having distinct
  // values assigned to the sotrage location of sor1, resulting in not assigning
  // a value to the storage location of sor1 in successors. As iterations of the
  // analysis go, the state of the environment flips between having a value
  // assigned to the storage location of sor1 and not having a value assigned to
  // it. Since the evaluation of the copy constructor expression in bar(sor1)
  // depends on a value being assigned to sor1, the state of the environment
  // also flips between having a storage location assigned to the bar(sor1)
  // expression and not having a storage location assigned to it. This leads to
  // an infinite loop as the environment can't stabilize.
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void foo(int, int);
    STATUSOR_INT bar(STATUSOR_INT);
    void baz(int);

    void target() {
      while (true) {
        STATUSOR_INT sor1 = Make<STATUSOR_INT>();
        if (sor1.ok()) {
          STATUSOR_INT sor2 = Make<STATUSOR_INT>();
          if (sor2.ok()) foo(sor1.value(), sor2.value());
        }

        STATUSOR_INT sor3 = bar(sor1);
        for (int i = 0; i < 5; i++) sor3 = bar(sor1);

        baz(sor3.value());  // [[unsafe]]
      }
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, ReturnValue) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target() {
      STATUSOR_INT sor = Make<STATUSOR_INT>();
      if (sor.ok())
        sor.value();
      else
        sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, Goto) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
    label:
      if (sor.ok())
        sor.value();
      else
        sor.value();  // [[unsafe]]
      goto label;
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
    label:
      if (!sor.ok()) goto label;
      sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      if (!sor.ok()) return;
      goto label;
    label:
      sor.value();
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, JoinDistinctValues) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(bool b) {
      STATUSOR_INT sor;
      if (b)
        sor = Make<STATUSOR_INT>();
      else
        sor = Make<STATUSOR_INT>();

      if (sor.ok())
        sor.value();
      else
        sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(bool b) {
      STATUSOR_INT sor;
      if (b) {
        sor = Make<STATUSOR_INT>();
        if (!sor.ok()) return;
      } else {
        sor = Make<STATUSOR_INT>();
        if (!sor.ok()) return;
      }
      sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(bool b) {
      STATUSOR_INT sor;
      if (b) {
        sor = Make<STATUSOR_INT>();
        if (!sor.ok()) return;
      } else {
        sor = Make<STATUSOR_INT>();
      }
      sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, VarDeclInitExprFromPairAccess) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target() {
      auto sor = Make<std::pair<int, STATUSOR_INT>>().second;
      if (sor.ok())
        sor.value();
      else
        sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target() {
      const auto& sor = Make<std::pair<int, STATUSOR_INT>>().second;
      if (sor.ok())
        sor.value();
      else
        sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, LValueToRValueCastOfChangingValue) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    bool foo();

    void target(bool b1) {
      STATUSOR_INT sor;
      if (b1)
        sor = Make<STATUSOR_INT>();
      else
        sor = Make<STATUSOR_INT>();

      do {
        const auto& b2 = foo();
        if (b2) break;

        sor.value();  // [[unsafe]]
      } while (true);
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, ConstructorInitializer) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    class target {
      target() : foo_(Make<STATUSOR_INT>().value()) {  // [[unsafe]]
      }
      int foo_;
    };
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, AssignStatusToBoolVar) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      bool ok = sor.ok();
      if (ok)
        sor.value();
      else
        sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target(STATUSOR_INT sor) {
      bool not_ok = !sor.ok();
      if (not_ok)
        sor.value();  // [[unsafe]]
      else
        sor.value();
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, StructuredBindings) {
  // Binding to a pair (which is actually a struct in the mock header).
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target() {
      const auto [sor, x] = Make<std::pair<STATUSOR_INT, int>>();
      if (sor.ok()) sor.value();
    }
  )cc");

  // Unsafe case.
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target() {
      const auto [sor, x] = Make<std::pair<STATUSOR_INT, int>>();
      sor.value();  // [[unsafe]]
    }
  )cc");

  // As a reference.
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target() {
      const auto& [sor, x] = Make<std::pair<STATUSOR_INT, int>>();
      if (sor.ok()) sor.value();
    }
  )cc");

  // Binding to a ref in a struct.
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    struct S {
      STATUSOR_INT& sor;
      int i;
    };

    void target() {
      const auto& [sor, i] = Make<S>();
      if (sor.ok()) sor.value();
    }
  )cc");

  // In a loop.
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target() {
      auto vals = Make<std::vector<std::pair<int, STATUSOR_INT>>>();
      for (const auto& [x, sor] : vals)
        if (sor.ok()) sor.value();
    }
  )cc");

  // Similar to the above, but InitExpr already has the storage initialized,
  // and bindings refer to them.
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_access_test_defs.h"

    void target() {
      auto vals = Make<std::vector<std::pair<int, STATUSOR_INT>>>();
      for (const auto& p : vals) {
        const auto& [i, sor] = p;
        if (sor.ok()) sor.value();
      }
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, AssignCompositeLogicExprToVar) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        void target(STATUSOR_INT sor, bool b) {
          bool c = sor.ok() && b;
          if (c) sor.value();
        }
      )cc");

  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        void target(STATUSOR_INT sor, bool b) {
          bool c = !(!sor.ok() || !b);
          if (c) sor.value();
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, Subclass) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        class Foo : public STATUSOR_INT {};

        void target(Foo opt) {
          opt.value();  // [[unsafe]]
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, SubclassStatus) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        class Foo : public STATUS {};

        void target(Foo opt) { opt.ok(); }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, SubclassOk) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        class Foo : public STATUSOR_INT {};

        void target(Foo opt) {
          if (opt.ok()) opt.value();
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, SubclassOperator) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_access_test_defs.h"

        class Foo : public STATUSOR_INT {};

        void target(Foo opt) {
          *opt;  // [[unsafe]]
        }
      )cc");
}

} // namespace

std::string
GetAliasMacros(UncheckedStatusOrAccessModelTestAliasKind AliasKind) {
  switch (AliasKind) {
  case UncheckedStatusOrAccessModelTestAliasKind::kUnaliased:
    return R"cc(
#define STATUSOR_INT ::absl::StatusOr<int>
#define STATUSOR_BOOL ::absl::StatusOr<bool>
#define STATUSOR_VOIDPTR ::absl::StatusOr<void*>
#define STATUS ::absl::Status
      )cc";
  case UncheckedStatusOrAccessModelTestAliasKind::kPartiallyAliased:
    return R"cc(
        template <typename T>
        using StatusOrAlias = ::absl::StatusOr<T>;
#define STATUSOR_INT StatusOrAlias<int>
#define STATUSOR_BOOL StatusOrAlias<bool>
#define STATUSOR_VOIDPTR StatusOrAlias<void*>
#define STATUS ::absl::Status
      )cc";
  case UncheckedStatusOrAccessModelTestAliasKind::kFullyAliased:
    return R"cc(
        using StatusOrIntAlias = ::absl::StatusOr<int>;
#define STATUSOR_INT StatusOrIntAlias
        using StatusOrBoolAlias = ::absl::StatusOr<bool>;
#define STATUSOR_BOOL StatusOrBoolAlias
        using StatusOrVoidPtrAlias = ::absl::StatusOr<void*>;
#define STATUSOR_VOIDPTR StatusOrVoidPtrAlias
        using StatusAlias = ::absl::Status;
#define STATUS StatusAlias
      )cc";
  }
  llvm_unreachable("Unknown alias kind.");
}

std::vector<std::pair<std::string, std::string>>
GetHeaders(UncheckedStatusOrAccessModelTestAliasKind AliasKind) {
  auto Headers = test::getMockHeaders();

  Headers.emplace_back("unchecked_statusor_access_test_defs.h",
                       R"cc(
#include "cstddef.h"
#include "statusor_defs.h"
#include "std_optional.h"
#include "std_vector.h"
#include "std_pair.h"
#include "absl_log.h"
#include "testing_defs.h"

                             template <typename T>
                             T Make();

                             class Fatal {
                              public:
                               ~Fatal() __attribute__((noreturn));
                               int value();
                             };
                       )cc" +
                           GetAliasMacros(AliasKind));
  return Headers;
}
} // namespace clang::dataflow::statusor_model
