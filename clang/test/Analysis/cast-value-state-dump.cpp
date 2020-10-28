// RUN: %clang_analyze_cc1 -std=c++14 \
// RUN:  -analyzer-checker=core,apiModeling.llvm.CastValue,debug.ExprInspection\
// RUN:  -analyzer-output=text -verify %s 2>&1 | FileCheck %s

#include "Inputs/llvm.h"

void clang_analyzer_printState();

namespace clang {
struct Shape {};
class Triangle : public Shape {};
class Circle : public Shape {};
class Square : public Shape {};
} // namespace clang

using namespace llvm;
using namespace clang;

void evalNonNullParamNonNullReturn(const Shape *S) {
  const auto *C = dyn_cast_or_null<Circle>(S);
  // expected-note@-1 {{Assuming 'S' is a 'Circle'}}
  // expected-note@-2 {{'C' initialized here}}

  // FIXME: We assumed that 'S' is a 'Circle' therefore it is not a 'Square'.
  if (dyn_cast_or_null<Square>(S)) {
    // expected-note@-1 {{Assuming 'S' is not a 'Square'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  clang_analyzer_printState();

  // CHECK:      "dynamic_types": [
  // CHECK-NEXT:   { "region": "SymRegion{reg_$0<const struct clang::Shape * S>}", "dyn_type": "const class clang::Circle", "sub_classable": true }
  // CHECK-NEXT: ],
  // CHECK-NEXT: "dynamic_casts": [
  // CHECK:        { "region": "SymRegion{reg_$0<const struct clang::Shape * S>}", "casts": [
  // CHECK-NEXT:     { "from": "struct clang::Shape", "to": "class clang::Circle", "kind": "success" },
  // CHECK-NEXT:     { "from": "struct clang::Shape", "to": "class clang::Square", "kind": "fail" }
  // CHECK-NEXT:   ] }

  (void)(1 / !C);
  // expected-note@-1 {{'C' is non-null}}
  // expected-note@-2 {{Division by zero}}
  // expected-warning@-3 {{Division by zero}}
}

