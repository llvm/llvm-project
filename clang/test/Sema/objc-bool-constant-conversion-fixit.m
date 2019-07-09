// RUN: %clang_cc1 -Werror=constant-conversion %s -fixit-recompile -fixit-to-temporary -E -o - | FileCheck %s

typedef signed char BOOL;

BOOL b;

int main() {
  BOOL b = 2;
  // CHECK: BOOL b = 2 ? YES : NO;

  b = b ? 2 : 1;
  // CHECK: b = b ? 2 ? YES : NO : 1;

  b = b ? 1 : 2;
  // CHECK: b = b ? 1 : 2 ? YES : NO;

  b = b ? 2 : 2;
  // CHECK: b = b ? 2 ? YES : NO : 2 ? YES : NO;

  b = 1 + 1;
  // CHECK: b = (1 + 1) ? YES : NO;

  b = 1 | 2;
  // CHECK: b = (1 | 2) ? YES : NO;

  b = 1 << 1;
  // CHECK: b = (1 << 1) ? YES : NO;
}

@interface BoolProp
@property BOOL b;
@end

void f(BoolProp *bp) {
  bp.b = 43;
  // CHECK: bp.b = 43 ? YES : NO;

  [bp setB:43];
  // CHECK: [bp setB:43 ? YES : NO];
}
