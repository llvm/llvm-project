// RUN: %clang_analyze_cc1 -std=c++14 \
// RUN:  -analyzer-checker=core,apiModeling.llvm.CastValue,debug.ExprInspection\
// RUN:  -analyzer-output=text -verify %s

#include "Inputs/llvm.h"

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

void evalReferences(const Shape &S) {
  const auto &C = dyn_cast<Circle>(S);
  // expected-note@-1 {{Assuming 'S' is not a 'Circle'}}
  // expected-note@-2 {{Dereference of null pointer}}
  // expected-warning@-3 {{Dereference of null pointer}}
}

void evalNonNullParamNonNullReturnReference(const Shape &S) {
  // Unmodeled cast from reference to pointer.
  const auto *C = dyn_cast_or_null<Circle>(S);
  // expected-note@-1 {{'C' initialized here}}

  if (!dyn_cast_or_null<Circle>(C)) {
    // expected-note@-1 {{'C' is a 'Circle'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (dyn_cast_or_null<Triangle>(C)) {
    // expected-note@-1 {{Assuming 'C' is not a 'Triangle'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (isa<Triangle>(C)) {
    // expected-note@-1 {{'C' is not a 'Triangle'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (isa<Circle>(C)) {
    // expected-note@-1 {{'C' is a 'Circle'}}
    // expected-note@-2 {{Taking true branch}}

    (void)(1 / !C);
    // expected-note@-1 {{'C' is non-null}}
    // expected-note@-2 {{Division by zero}}
    // expected-warning@-3 {{Division by zero}}
  }
}

void evalNonNullParamNonNullReturn(const Shape *S) {
  const auto *C = cast<Circle>(S);
  // expected-note@-1 {{'S' is a 'Circle'}}
  // expected-note@-2 {{'C' initialized here}}

  if (!isa<Triangle>(C)) {
    // expected-note@-1 {{Assuming 'C' is a 'Triangle'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (!isa<Triangle>(C)) {
    // expected-note@-1 {{'C' is a 'Triangle'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  (void)(1 / !C);
  // expected-note@-1 {{'C' is non-null}}
  // expected-note@-2 {{Division by zero}}
  // expected-warning@-3 {{Division by zero}}
}

void evalNonNullParamNullReturn(const Shape *S) {
  const auto *C = dyn_cast_or_null<Circle>(S);
  // expected-note@-1 {{Assuming 'S' is not a 'Circle'}}

  if (const auto *T = dyn_cast_or_null<Triangle>(S)) {
    // expected-note@-1 {{Assuming 'S' is a 'Triangle'}}
    // expected-note@-2 {{'T' initialized here}}
    // expected-note@-3 {{'T' is non-null}}
    // expected-note@-4 {{Taking true branch}}

    (void)(1 / !T);
    // expected-note@-1 {{'T' is non-null}}
    // expected-note@-2 {{Division by zero}}
    // expected-warning@-3 {{Division by zero}}
  }
}

void evalNullParamNullReturn(const Shape *S) {
  const auto *C = dyn_cast_or_null<Circle>(S);
  // expected-note@-1 {{Assuming null pointer is passed into cast}}
  // expected-note@-2 {{'C' initialized to a null pointer value}}

  (void)(1 / (bool)C);
  // expected-note@-1 {{Division by zero}}
  // expected-warning@-2 {{Division by zero}}
}

void evalZeroParamNonNullReturnPointer(const Shape *S) {
  const auto *C = S->castAs<Circle>();
  // expected-note@-1 {{'S' is a 'Circle'}}
  // expected-note@-2 {{'C' initialized here}}

  (void)(1 / !C);
  // expected-note@-1 {{'C' is non-null}}
  // expected-note@-2 {{Division by zero}}
  // expected-warning@-3 {{Division by zero}}
}

void evalZeroParamNonNullReturn(const Shape &S) {
  const auto *C = S.castAs<Circle>();
  // expected-note@-1 {{'C' initialized here}}

  (void)(1 / !C);
  // expected-note@-1 {{'C' is non-null}}
  // expected-note@-2 {{Division by zero}}
  // expected-warning@-3 {{Division by zero}}
}

void evalZeroParamNullReturn(const Shape *S) {
  const auto &C = S->getAs<Circle>();
  // expected-note@-1 {{Assuming 'S' is not a 'Circle'}}
  // expected-note@-2 {{Storing null pointer value}}
  // expected-note@-3 {{'C' initialized here}}

  if (!dyn_cast_or_null<Triangle>(S)) {
    // expected-note@-1 {{Assuming 'S' is a 'Triangle'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (!dyn_cast_or_null<Triangle>(S)) {
    // expected-note@-1 {{'S' is a 'Triangle'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  (void)(1 / (bool)C);
  // expected-note@-1 {{Division by zero}}
  // expected-warning@-2 {{Division by zero}}
}
