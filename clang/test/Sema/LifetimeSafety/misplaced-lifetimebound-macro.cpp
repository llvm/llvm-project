// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety-intra-tu-misplaced-lifetimebound -Wlifetime-safety-annotation-placement -Wno-dangling -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety-intra-tu-misplaced-lifetimebound -Wno-dangling -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

// CHECK-NOT: fix-it:

struct MyObj{};

// We do not emit lifetimebound fix-its on macros.

#define REF_PARAM MyObj &obj

MyObj &macro_param(REF_PARAM); // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers before the definition; add it to the declaration instead}}

MyObj &macro_param(MyObj &obj [[clang::lifetimebound]]) { // expected-note {{'lifetimebound' attribute appears here on the definition}}
  return obj;
}

#define REF_PARAMS MyObj &obj1, MyObj &obj2

MyObj &macro_params(bool condition, REF_PARAMS); // expected-warning 2 {{'lifetimebound' attribute on this definition is not visible to callers before the definition; add it to the declaration instead}}

MyObj &macro_params(bool condition, 
                    MyObj &obj1 [[clang::lifetimebound]],    // expected-note {{'lifetimebound' attribute appears here on the definition}}
                    MyObj &obj2 [[clang::lifetimebound]]) {  // expected-note {{'lifetimebound' attribute appears here on the definition}}
    return condition ? obj1 : obj2;
}
