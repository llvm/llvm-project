

// RUN: %clang_cc1  -triple armv7k -target-feature +neon -ast-dump -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1  -triple armv7k -target-feature +neon -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s
#include <arm_neon.h>

// CHECK-LABEL: test
void test() {
  float sign[64];
  float32x4x4_t r;
  __builtin_neon_vld4q_v(&r, sign, 41);
}

// CHECK: CompoundStmt
// CHECK: |-DeclStmt
// CHECK: | `-VarDecl {{.*}} used sign 'float[64]'
// CHECK: |-DeclStmt
// CHECK: | `-VarDecl {{.*}} used r 'float32x4x4_t':'struct float32x4x4_t'
// CHECK: `-CallExpr {{.*}} 'void'
// CHECK:   |-ImplicitCastExpr {{.*}} 'void (*)(void *, const void *, int)' <BuiltinFnToFnPtr>
// CHECK:   | `-DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_neon_vld4q_v' 'void (void *, const void *, int)'
// CHECK:   |-ImplicitCastExpr {{.*}} 'void *' <BoundsSafetyPointerCast>
// CHECK:   | `-ImplicitCastExpr {{.*}} 'void *__bidi_indexable' <BitCast>
// CHECK:   |   `-UnaryOperator {{.*}} 'float32x4x4_t *__bidi_indexable' prefix '&' cannot overflow
// CHECK:   |     `-DeclRefExpr {{.*}} 'float32x4x4_t':'struct float32x4x4_t' lvalue Var {{.*}} 'r' 'float32x4x4_t':'struct float32x4x4_t'
// CHECK:   |-ImplicitCastExpr {{.*}} 'const void *' <BoundsSafetyPointerCast>
// CHECK:   | `-ImplicitCastExpr {{.*}} 'const void *__bidi_indexable' <BitCast>
// CHECK:   |   `-ImplicitCastExpr {{.*}} 'float *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:   |     `-DeclRefExpr {{.*}} 'float[64]' lvalue Var {{.*}} 'sign' 'float[64]'
// CHECK:   `-IntegerLiteral {{.*}} 'int' 41
