// RUN: %clang_analyze_cc1 \
// RUN:  -analyzer-checker=core,apiModeling.llvm.CastValue,debug.ExprInspection\
// RUN:  -verify %s

#include "Inputs/llvm.h"

void clang_analyzer_numTimesReached();
void clang_analyzer_warnIfReached();
void clang_analyzer_eval(bool);

namespace clang {
struct Shape {
  template <typename T>
  const T *castAs() const;

  template <typename T>
  const T *getAs() const;
};
class Triangle : public Shape {};
class Circle : public Shape {};
} // namespace clang

using namespace llvm;
using namespace clang;

void test_regions_dyn_cast(const Shape *A, const Shape *B) {
  if (dyn_cast<Circle>(A) && !dyn_cast<Circle>(B))
    clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
}

void test_regions_isa(const Shape *A, const Shape *B) {
  if (isa<Circle>(A) && !isa<Circle>(B))
    clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
}

namespace test_cast {
void evalLogic(const Shape *S) {
  const Circle *C = cast<Circle>(S);
  clang_analyzer_numTimesReached(); // expected-warning {{1}}

  if (S && C)
    clang_analyzer_eval(C == S); // expected-warning {{TRUE}}

  if (S && !C)
    clang_analyzer_warnIfReached(); // no-warning

  if (!S)
    clang_analyzer_warnIfReached(); // no-warning
}
} // namespace test_cast

namespace test_dyn_cast {
void evalLogic(const Shape *S) {
  const Circle *C = dyn_cast<Circle>(S);
  clang_analyzer_numTimesReached(); // expected-warning {{2}}

  if (S && C)
    clang_analyzer_eval(C == S); // expected-warning {{TRUE}}

  if (S && !C)
    clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}

  if (!S)
    clang_analyzer_warnIfReached(); // no-warning
}
} // namespace test_dyn_cast

namespace test_cast_or_null {
void evalLogic(const Shape *S) {
  const Circle *C = cast_or_null<Circle>(S);
  clang_analyzer_numTimesReached(); // expected-warning {{2}}

  if (S && C)
    clang_analyzer_eval(C == S); // expected-warning {{TRUE}}

  if (S && !C)
    clang_analyzer_warnIfReached(); // no-warning

  if (!S)
    clang_analyzer_eval(!C); // expected-warning {{TRUE}}
}
} // namespace test_cast_or_null

namespace test_dyn_cast_or_null {
void evalLogic(const Shape *S) {
  const Circle *C = dyn_cast_or_null<Circle>(S);
  clang_analyzer_numTimesReached(); // expected-warning {{3}}

  if (S && C)
    clang_analyzer_eval(C == S); // expected-warning {{TRUE}}

  if (S && !C)
    clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}

  if (!S)
    clang_analyzer_eval(!C); // expected-warning {{TRUE}}
}
} // namespace test_dyn_cast_or_null

namespace test_cast_as {
void evalLogic(const Shape *S) {
  const Circle *C = S->castAs<Circle>();
  clang_analyzer_numTimesReached(); // expected-warning {{1}}

  if (S && C)
    clang_analyzer_eval(C == S);
  // expected-warning@-1 {{TRUE}}

  if (S && !C)
    clang_analyzer_warnIfReached(); // no-warning

  if (!S)
    clang_analyzer_warnIfReached(); // no-warning
}
} // namespace test_cast_as

namespace test_get_as {
void evalLogic(const Shape *S) {
  const Circle *C = S->getAs<Circle>();
  clang_analyzer_numTimesReached(); // expected-warning {{2}}

  if (S && C)
    clang_analyzer_eval(C == S);
  // expected-warning@-1 {{TRUE}}

  if (S && !C)
    clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}

  if (!S)
    clang_analyzer_warnIfReached(); // no-warning
}
} // namespace test_get_as

