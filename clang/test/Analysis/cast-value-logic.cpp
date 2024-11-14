// RUN: %clang_analyze_cc1 -std=c++14 \
// RUN:  -analyzer-checker=core,apiModeling.llvm.CastValue,debug.ExprInspection\
// RUN:  -verify %s

#include "Inputs/llvm.h"

void clang_analyzer_numTimesReached();
void clang_analyzer_warnIfReached();
void clang_analyzer_eval(bool);
void clang_analyzer_dump(int);

namespace clang {
struct Shape {
  template <typename T>
  const T *castAs() const;

  template <typename T>
  const T *getAs() const;

  virtual double area();
  virtual void print(int n) const {
    clang_analyzer_dump(n + 1); // expected-warning {{+ 1}} this was analyzed in top-level-context
  }
};
struct Triangle : public Shape {
  void print(int n) const override {
    clang_analyzer_dump(n + 2); // expected-warning {{+ 2}} this was analyzed in top-level-context
  }
};
struct Rectangle : public Shape {
  void print(int n) const override {
    clang_analyzer_dump(n + 3);
    // expected-warning@-1 {{103}} this was inlined from 'test_conflicting_type_information' 'A->print(100)'
    // expected-warning@-2 {{303}} this was inlined from 'test_conflicting_type_information' 'R->print(300)'
  }
};
struct Hexagon : public Shape {
  void print(int n) const override {
    clang_analyzer_dump(n + 4); // expected-warning {{+ 4}}
  }
};
struct Circle : public Shape {
public:
  ~Circle();
};
struct SuspiciouslySpecificCircle : public Circle {};
} // namespace clang

using namespace llvm;
using namespace clang;

void test_conflicting_type_information(const Shape *A) {
  if (auto const *T = dyn_cast<Triangle>(A)) {
    // Now, we associate 'Triangle' with 'A'.
    if (auto const *R = dyn_cast<Rectangle>(A)) {
      // Now, we associate 'Rectangle' with 'A'.
      // This should be unreachable becasue 'A' cannot be a 'Triangle' and a 'Rectangle' at the same time.
      A->print(100); // Splits into 2 paths: (1) conservative eval, (2) inlining `Rectangle::print`
      T->print(200); // On path (2), it will conservative eval, because the dynamic type
      R->print(300); // Splits on path (2) into other 2 paths: (3) conservative eval, (4) inlining `Rectangle::print`
    }
  }
}

void test_regions_dyn_cast(const Shape *A, const Shape *B) {
  if (dyn_cast<Circle>(A) && !dyn_cast<Circle>(B))
    clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
}

void test_regions_isa(const Shape *A, const Shape *B) {
  if (isa<Circle>(A) && !isa<Circle>(B))
    clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
}

void test_regions_isa_variadic(const Shape *A, const Shape *B) {
  if (isa<Triangle, Rectangle, Hexagon>(A) &&
      !isa<Rectangle, Hexagon, Circle>(B))
    clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
}

void test_regions_isa_and_nonnull(const Shape *A, const Shape *B) {
  if (isa_and_nonnull<Circle>(A) && !isa_and_nonnull<Circle>(B))
    clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
}

void test_regions_isa_and_nonnull_variadic(const Shape *A, const Shape *B) {
  if (isa_and_nonnull<Triangle, Rectangle, Hexagon>(A) &&
      !isa_and_nonnull<Rectangle, Hexagon, Circle>(B))
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

namespace crashes {
void test_non_reference_null_region_crash(Shape s) {
  cast<Circle>(s); // no-crash
}

void test_non_reference_temporary_crash() {
  extern std::unique_ptr<Shape> foo();
  auto P = foo();
  auto Q = cast<Circle>(std::move(P)); // no-crash
}

double test_virtual_method_after_call(Shape *S) {
  if (isa<Circle>(S))
    return S->area();
  return S->area() / 2;
}

void test_delete_crash() {
  extern Circle *makeCircle();
  Shape *S = makeCircle();
  delete cast<SuspiciouslySpecificCircle>(S);
}
} // namespace crashes
