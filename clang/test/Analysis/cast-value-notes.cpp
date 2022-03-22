// RUN: %clang_analyze_cc1 -std=c++14 \
// RUN:  -analyzer-checker=core,apiModeling.llvm.CastValue,debug.ExprInspection\
// RUN:  -analyzer-output=text -verify -DDEFAULT_TRIPLE %s 2>&1 | FileCheck %s -check-prefix=DEFAULT-CHECK
//
// RUN: %clang_analyze_cc1 -std=c++14 -triple amdgcn-unknown-unknown \
// RUN: -analyzer-checker=core,apiModeling.llvm.CastValue,debug.ExprInspection\
// RUN: -analyzer-output=text -verify -DAMDGCN_TRIPLE %s 2>&1 | FileCheck %s -check-prefix=AMDGCN-CHECK

#include "Inputs/llvm.h"

// The amggcn triple case uses an intentionally different address space.
// The core.NullDereference checker intentionally ignores checks
// that use address spaces, so the case is differentiated here.
//
// From https://llvm.org/docs/AMDGPUUsage.html#address-spaces,
// select address space 3 (local), since the pointer size is
// different than Generic.
#define DEVICE __attribute__((address_space(3)))

namespace clang {
struct Shape {
  template <typename T>
  const T *castAs() const;

  template <typename T>
  const T *getAs() const;
};
class Triangle : public Shape {};
class Rectangle : public Shape {};
class Hexagon : public Shape {};
class Circle : public Shape {};
} // namespace clang

using namespace llvm;
using namespace clang;

void clang_analyzer_printState();

#if defined(DEFAULT_TRIPLE)
void evalReferences(const Shape &S) {
  const auto &C = dyn_cast<Circle>(S);
  // expected-note@-1 {{Assuming 'S' is not a 'Circle'}}
  // expected-note@-2 {{Dereference of null pointer}}
  // expected-warning@-3 {{Dereference of null pointer}}
  clang_analyzer_printState();
  // DEFAULT-CHECK: "dynamic_types": [
  // DEFAULT-CHECK-NEXT: { "region": "SymRegion{reg_$0<const struct clang::Shape & S>}", "dyn_type": "const class clang::Circle &", "sub_classable": true }
  (void)C;
}
#elif defined(AMDGCN_TRIPLE)
void evalReferences(const Shape &S) {
  const auto &C = dyn_cast<DEVICE Circle>(S);
  clang_analyzer_printState();
  // AMDGCN-CHECK: "dynamic_types": [
  // AMDGCN-CHECK-NEXT: { "region": "SymRegion{reg_$0<const struct clang::Shape & S>}", "dyn_type": "const __attribute__((address_space(3))) class clang::Circle &", "sub_classable": true }
  (void)C;
}
#else
#error Target must be specified, and must be pinned
#endif

void evalNonNullParamNonNullReturnReference(const Shape &S) {
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

  if (dyn_cast_or_null<Rectangle>(C)) {
    // expected-note@-1 {{Assuming 'C' is not a 'Rectangle'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (dyn_cast_or_null<Hexagon>(C)) {
    // expected-note@-1 {{Assuming 'C' is not a 'Hexagon'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (isa<Triangle>(C)) {
    // expected-note@-1 {{'C' is not a 'Triangle'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (isa<Triangle, Rectangle>(C)) {
    // expected-note@-1 {{'C' is neither a 'Triangle' nor a 'Rectangle'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (isa<Triangle, Rectangle, Hexagon>(C)) {
    // expected-note@-1 {{'C' is neither a 'Triangle' nor a 'Rectangle' nor a 'Hexagon'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (isa<Circle, Rectangle, Hexagon>(C)) {
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

  if (dyn_cast_or_null<Rectangle>(C)) {
    // expected-note@-1 {{Assuming 'C' is not a 'Rectangle'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (dyn_cast_or_null<Hexagon>(C)) {
    // expected-note@-1 {{Assuming 'C' is not a 'Hexagon'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (isa<Triangle>(C)) {
    // expected-note@-1 {{'C' is not a 'Triangle'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (isa<Triangle, Rectangle>(C)) {
    // expected-note@-1 {{'C' is neither a 'Triangle' nor a 'Rectangle'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (isa<Triangle, Rectangle, Hexagon>(C)) {
    // expected-note@-1 {{'C' is neither a 'Triangle' nor a 'Rectangle' nor a 'Hexagon'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (isa<Circle, Rectangle, Hexagon>(C)) {
    // expected-note@-1 {{'C' is a 'Circle'}}
    // expected-note@-2 {{Taking true branch}}

    (void)(1 / !C);
    // expected-note@-1 {{'C' is non-null}}
    // expected-note@-2 {{Division by zero}}
    // expected-warning@-3 {{Division by zero}}
  }
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
