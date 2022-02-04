// RUN: %clang_cc1 -Werror=objc-signed-char-bool %s -fixit-recompile -fixit-to-temporary -E -o - | FileCheck %s

typedef signed char BOOL;

BOOL b;

int main(void) {
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

  int i;

  b = i;
  // CHECK: b = i ? YES : NO;

  b = i * 2;
  // CHECK: b = (i * 2) ? YES : NO;

  b = 1 ? 2 : 3;
  // CHECK: b = 1 ? 2 ? YES : NO : 3 ? YES : NO;
}

@interface BoolProp
@property BOOL b;
@end

void f(BoolProp *bp) {
  bp.b = 43;
  // CHECK: bp.b = 43 ? YES : NO;

  [bp setB:43];
  // CHECK: [bp setB:43 ? YES : NO];

  int i;

  bp.b = i;
  // CHECK: bp.b = i ? YES : NO;

  bp.b = i + 1;
  // CHECK: bp.b = (i + 1) ? YES : NO;
}
