

// RUN: %clang_cc1 -ast-dump -fbounds-safety -Wno-incompatible-pointer-types %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -fbounds-safety %s -o /dev/null
// RUN: %clang_cc1 -emit-llvm -fbounds-safety -O2 %s -o /dev/null

// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -Wno-incompatible-pointer-types %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s -o /dev/null
// RUN: %clang_cc1 -emit-llvm -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -O2 %s -o /dev/null

#include <ptrcheck.h>

void implicit(void) {
  int *__indexable i;
  const int *__indexable ci;
  const int *__bidi_indexable cbi;
  const void *__bidi_indexable cbv;

  // CHECK:      BinaryOperator {{.+}} 'int *__indexable' '='
  // CHECK-NEXT: |-DeclRefExpr {{.+}} 'int *__indexable' lvalue Var {{.+}} 'i' 'int *__indexable'
  // CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'int *__indexable' <NoOp>
  // CHECK-NEXT:   `-ImplicitCastExpr {{.+}} 'const int *__indexable' <LValueToRValue>
  // CHECK-NEXT:     `-DeclRefExpr {{.+}} 'const int *__indexable' lvalue Var {{.+}} 'ci' 'const int *__indexable'
  i = ci;

  // CHECK:      BinaryOperator {{.+}} 'const int *__indexable' '='
  // CHECK-NEXT: |-DeclRefExpr {{.+}} 'const int *__indexable' lvalue Var {{.+}} 'ci' 'const int *__indexable'
  // CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'const int *__indexable' <NoOp>
  // CHECK-NEXT:   `-ImplicitCastExpr {{.+}} 'int *__indexable' <LValueToRValue>
  // CHECK-NEXT:     `-DeclRefExpr {{.+}} 'int *__indexable' lvalue Var {{.+}} 'i' 'int *__indexable'
  ci = i;

  // CHECK:      BinaryOperator {{.+}} 'int *__indexable' '='
  // CHECK-NEXT: |-DeclRefExpr {{.+}} 'int *__indexable' lvalue Var {{.+}} 'i' 'int *__indexable'
  // CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'int *__indexable' <BoundsSafetyPointerCast>
  // CHECK-NEXT:   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <NoOp>
  // CHECK-NEXT:     `-ImplicitCastExpr {{.+}} 'const int *__bidi_indexable' <LValueToRValue>
  // CHECK-NEXT:       `-DeclRefExpr {{.+}} 'const int *__bidi_indexable' lvalue Var {{.+}} 'cbi' 'const int *__bidi_indexable'
  i = cbi;

  // CHECK:      BinaryOperator {{.+}} 'const int *__bidi_indexable' '='
  // CHECK-NEXT: |-DeclRefExpr {{.+}} 'const int *__bidi_indexable' lvalue Var {{.+}} 'cbi' 'const int *__bidi_indexable'
  // CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'const int *__bidi_indexable' <BoundsSafetyPointerCast>
  // CHECK-NEXT:   `-ImplicitCastExpr {{.+}} 'const int *__indexable' <NoOp>
  // CHECK-NEXT:     `-ImplicitCastExpr {{.+}} 'int *__indexable' <LValueToRValue>
  // CHECK-NEXT:       `-DeclRefExpr {{.+}} 'int *__indexable' lvalue Var {{.+}} 'i' 'int *__indexable'
  cbi = i;

  // CHECK:      BinaryOperator {{.+}} 'int *__indexable' '='
  // CHECK-NEXT: |-DeclRefExpr {{.+}} 'int *__indexable' lvalue Var {{.+}} 'i' 'int *__indexable'
  // CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'int *__indexable' <BoundsSafetyPointerCast>
  // CHECK-NEXT:   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BitCast>
  // CHECK-NEXT:     `-ImplicitCastExpr {{.+}} 'const void *__bidi_indexable' <LValueToRValue>
  // CHECK-NEXT:       `-DeclRefExpr {{.+}} 'const void *__bidi_indexable' lvalue Var {{.+}} 'cbv' 'const void *__bidi_indexable'
  i = cbv;

  // CHECK:      BinaryOperator {{.+}} 'const void *__bidi_indexable' '='
  // CHECK-NEXT: |-DeclRefExpr {{.+}} 'const void *__bidi_indexable' lvalue Var {{.+}} 'cbv' 'const void *__bidi_indexable'
  // CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'const void *__bidi_indexable' <BoundsSafetyPointerCast>
  // CHECK-NEXT:   `-ImplicitCastExpr {{.+}} 'const void *__indexable' <BitCast>
  // CHECK-NEXT:     `-ImplicitCastExpr {{.+}} 'int *__indexable' <LValueToRValue>
  // CHECK-NEXT:       `-DeclRefExpr {{.+}} 'int *__indexable' lvalue Var {{.+}} 'i' 'int *__indexable'
  cbv = i;
}

void implicit_nested(void) {
  int **__indexable i;
  const int **__indexable ci;
  const int **__bidi_indexable cbi;
  const void **__bidi_indexable cbv;

  // CHECK:      BinaryOperator {{.+}} 'int *__single*__indexable' '='
  // CHECK-NEXT: |-DeclRefExpr {{.+}} 'int *__single*__indexable' lvalue Var {{.+}} 'i' 'int *__single*__indexable'
  // CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'int *__single*__indexable' <NoOp>
  // CHECK-NEXT:   `-ImplicitCastExpr {{.+}} 'const int *__single*__indexable' <LValueToRValue>
  // CHECK-NEXT:     `-DeclRefExpr {{.+}} 'const int *__single*__indexable' lvalue Var {{.+}} 'ci' 'const int *__single*__indexable'
  i = ci;

  // CHECK:      BinaryOperator {{.+}} 'const int *__single*__indexable' '='
  // CHECK-NEXT: |-DeclRefExpr {{.+}} 'const int *__single*__indexable' lvalue Var {{.+}} 'ci' 'const int *__single*__indexable'
  // CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'const int *__single*__indexable' <NoOp>
  // CHECK-NEXT:   `-ImplicitCastExpr {{.+}} 'int *__single*__indexable' <LValueToRValue>
  // CHECK-NEXT:     `-DeclRefExpr {{.+}} 'int *__single*__indexable' lvalue Var {{.+}} 'i' 'int *__single*__indexable'
  ci = i;

  // CHECK:      BinaryOperator {{.+}} 'int *__single*__indexable' '='
  // CHECK-NEXT: |-DeclRefExpr {{.+}} 'int *__single*__indexable' lvalue Var {{.+}} 'i' 'int *__single*__indexable'
  // CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'int *__single*__indexable' <BoundsSafetyPointerCast>
  // CHECK-NEXT:   `-ImplicitCastExpr {{.+}} 'int *__single*__bidi_indexable' <NoOp>
  // CHECK-NEXT:     `-ImplicitCastExpr {{.+}} 'const int *__single*__bidi_indexable' <LValueToRValue>
  // CHECK-NEXT:       `-DeclRefExpr {{.+}} 'const int *__single*__bidi_indexable' lvalue Var {{.+}} 'cbi' 'const int *__single*__bidi_indexable'
  i = cbi;

  // CHECK:      BinaryOperator {{.+}} 'const int *__single*__bidi_indexable' '='
  // CHECK-NEXT: |-DeclRefExpr {{.+}} 'const int *__single*__bidi_indexable' lvalue Var {{.+}} 'cbi' 'const int *__single*__bidi_indexable'
  // CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'const int *__single*__bidi_indexable' <BoundsSafetyPointerCast>
  // CHECK-NEXT:   `-ImplicitCastExpr {{.+}} 'const int *__single*__indexable' <NoOp>
  // CHECK-NEXT:     `-ImplicitCastExpr {{.+}} 'int *__single*__indexable' <LValueToRValue>
  // CHECK-NEXT:       `-DeclRefExpr {{.+}} 'int *__single*__indexable' lvalue Var {{.+}} 'i' 'int *__single*__indexable'
  cbi = i;

  // CHECK:      BinaryOperator {{.+}} 'int *__single*__indexable' '='
  // CHECK-NEXT: |-DeclRefExpr {{.+}} 'int *__single*__indexable' lvalue Var {{.+}} 'i' 'int *__single*__indexable'
  // CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'int *__single*__indexable' <BoundsSafetyPointerCast>
  // CHECK-NEXT:   `-ImplicitCastExpr {{.+}} 'int *__single*__bidi_indexable' <BitCast>
  // CHECK-NEXT:     `-ImplicitCastExpr {{.+}} 'const void *__single*__bidi_indexable' <LValueToRValue>
  // CHECK-NEXT:       `-DeclRefExpr {{.+}} 'const void *__single*__bidi_indexable' lvalue Var {{.+}} 'cbv' 'const void *__single*__bidi_indexable'
  i = cbv;

  // CHECK:      BinaryOperator {{.+}} 'const void *__single*__bidi_indexable' '='
  // CHECK-NEXT: |-DeclRefExpr {{.+}} 'const void *__single*__bidi_indexable' lvalue Var {{.+}} 'cbv' 'const void *__single*__bidi_indexable'
  // CHECK-NEXT: `-ImplicitCastExpr {{.+}} 'const void *__single*__bidi_indexable' <BoundsSafetyPointerCast>
  // CHECK-NEXT:   `-ImplicitCastExpr {{.+}} 'const void *__single*__indexable' <BitCast>
  // CHECK-NEXT:     `-ImplicitCastExpr {{.+}} 'int *__single*__indexable' <LValueToRValue>
  // CHECK-NEXT:       `-DeclRefExpr {{.+}} 'int *__single*__indexable' lvalue Var {{.+}} 'i' 'int *__single*__indexable'
  cbv = i;
}

void explicit(void) {
  int *__indexable i;
  const int *__indexable ci;
  const int *__bidi_indexable cbi;
  const void *__bidi_indexable cbv;

  // CHECK:      BinaryOperator {{.+}} 'int *__indexable' '='
  // CHECK-NEXT: |-DeclRefExpr {{.+}} 'int *__indexable' lvalue Var {{.+}} 'i' 'int *__indexable'
  // CHECK-NEXT: `-CStyleCastExpr {{.+}} 'int *__indexable' <NoOp>
  // CHECK-NEXT:   `-ImplicitCastExpr {{.+}} 'const int *__indexable' <LValueToRValue> part_of_explicit_cast
  // CHECK-NEXT:     `-DeclRefExpr {{.+}} 'const int *__indexable' lvalue Var {{.+}} 'ci' 'const int *__indexable'
  i = (int *__indexable)ci;

  // CHECK:      BinaryOperator {{.+}} 'const int *__indexable' '='
  // CHECK-NEXT: |-DeclRefExpr {{.+}} 'const int *__indexable' lvalue Var {{.+}} 'ci' 'const int *__indexable'
  // CHECK-NEXT: `-CStyleCastExpr {{.+}} 'const int *__indexable' <NoOp>
  // CHECK-NEXT:   `-ImplicitCastExpr {{.+}} 'int *__indexable' <LValueToRValue> part_of_explicit_cast
  // CHECK-NEXT:     `-DeclRefExpr {{.+}} 'int *__indexable' lvalue Var {{.+}} 'i' 'int *__indexable'
  ci = (const int *__indexable)i;

  // CHECK:      BinaryOperator {{.+}} 'int *__indexable' '='
  // CHECK-NEXT: |-DeclRefExpr {{.+}} 'int *__indexable' lvalue Var {{.+}} 'i' 'int *__indexable'
  // CHECK-NEXT: `-CStyleCastExpr {{.+}} 'int *__indexable' <BoundsSafetyPointerCast>
  // CHECK-NEXT:   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <NoOp> part_of_explicit_cast
  // CHECK-NEXT:     `-ImplicitCastExpr {{.+}} 'const int *__bidi_indexable' <LValueToRValue> part_of_explicit_cast
  // CHECK-NEXT:       `-DeclRefExpr {{.+}} 'const int *__bidi_indexable' lvalue Var {{.+}} 'cbi' 'const int *__bidi_indexable'
  i = (int *__indexable)cbi;

  // CHECK:      BinaryOperator {{.+}} 'const int *__bidi_indexable' '='
  // CHECK-NEXT: |-DeclRefExpr {{.+}} 'const int *__bidi_indexable' lvalue Var {{.+}} 'cbi' 'const int *__bidi_indexable'
  // CHECK-NEXT: `-CStyleCastExpr {{.+}} 'const int *__bidi_indexable' <BoundsSafetyPointerCast>
  // CHECK-NEXT:   `-ImplicitCastExpr {{.+}} 'const int *__indexable' <NoOp> part_of_explicit_cast
  // CHECK-NEXT:     `-ImplicitCastExpr {{.+}} 'int *__indexable' <LValueToRValue> part_of_explicit_cast
  // CHECK-NEXT:       `-DeclRefExpr {{.+}} 'int *__indexable' lvalue Var {{.+}} 'i' 'int *__indexable'
  cbi = (const int *__bidi_indexable)i;

  // CHECK:      BinaryOperator {{.+}} 'int *__indexable' '='
  // CHECK-NEXT: |-DeclRefExpr {{.+}} 'int *__indexable' lvalue Var {{.+}} 'i' 'int *__indexable'
  // CHECK-NEXT: `-CStyleCastExpr {{.+}} 'int *__indexable' <BoundsSafetyPointerCast>
  // CHECK-NEXT:   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BitCast> part_of_explicit_cast
  // CHECK-NEXT:     `-ImplicitCastExpr {{.+}} 'const void *__bidi_indexable' <LValueToRValue> part_of_explicit_cast
  // CHECK-NEXT:       `-DeclRefExpr {{.+}} 'const void *__bidi_indexable' lvalue Var {{.+}} 'cbv' 'const void *__bidi_indexable'
  i = (int *__indexable)cbv;

  // CHECK:       BinaryOperator {{.+}} 'const void *__bidi_indexable' '='
  // CHECK-NEXT:  |-DeclRefExpr {{.+}} 'const void *__bidi_indexable' lvalue Var {{.+}} 'cbv' 'const void *__bidi_indexable'
  // CHECK-NEXT:  `-CStyleCastExpr {{.+}} 'const void *__bidi_indexable' <BoundsSafetyPointerCast>
  // CHECK-NEXT:    `-ImplicitCastExpr {{.+}} 'const void *__indexable' <BitCast> part_of_explicit_cast
  // CHECK-NEXT:      `-ImplicitCastExpr {{.+}} 'int *__indexable' <LValueToRValue> part_of_explicit_cast
  // CHECK-NEXT:        `-DeclRefExpr {{.+}} 'int *__indexable' lvalue Var {{.+}} 'i' 'int *__indexable'
  cbv = (const void *__bidi_indexable)i;
}

void explicit_nested(void) {
  int **__indexable i;
  const int **__indexable ci;
  const int **__bidi_indexable cbi;
  const void **__bidi_indexable cbv;

  // CHECK:      BinaryOperator {{.+}} 'int *__single*__indexable' '='
  // CHECK-NEXT: |-DeclRefExpr {{.+}} 'int *__single*__indexable' lvalue Var {{.+}} 'i' 'int *__single*__indexable'
  // CHECK-NEXT: `-CStyleCastExpr {{.+}} 'int *__single*__indexable' <NoOp>
  // CHECK-NEXT:   `-ImplicitCastExpr {{.+}} 'const int *__single*__indexable' <LValueToRValue> part_of_explicit_cast
  // CHECK-NEXT:     `-DeclRefExpr {{.+}} 'const int *__single*__indexable' lvalue Var {{.+}} 'ci' 'const int *__single*__indexable'
  i = (int **__indexable)ci;

  // CHECK:      BinaryOperator {{.+}} 'const int *__single*__indexable' '='
  // CHECK-NEXT: |-DeclRefExpr {{.+}} 'const int *__single*__indexable' lvalue Var {{.+}} 'ci' 'const int *__single*__indexable'
  // CHECK-NEXT: `-CStyleCastExpr {{.+}} 'const int *__single*__indexable' <NoOp>
  // CHECK-NEXT:   `-ImplicitCastExpr {{.+}} 'int *__single*__indexable' <LValueToRValue> part_of_explicit_cast
  // CHECK-NEXT:     `-DeclRefExpr {{.+}} 'int *__single*__indexable' lvalue Var {{.+}} 'i' 'int *__single*__indexable'
  ci = (const int **__indexable)i;

  // CHECK:      BinaryOperator {{.+}} 'int *__single*__indexable' '='
  // CHECK-NEXT: |-DeclRefExpr {{.+}} 'int *__single*__indexable' lvalue Var {{.+}} 'i' 'int *__single*__indexable'
  // CHECK-NEXT: `-CStyleCastExpr {{.+}} 'int *__single*__indexable' <BoundsSafetyPointerCast>
  // CHECK-NEXT:   `-ImplicitCastExpr {{.+}} 'int *__single*__bidi_indexable' <NoOp> part_of_explicit_cast
  // CHECK-NEXT:     `-ImplicitCastExpr {{.+}} 'const int *__single*__bidi_indexable' <LValueToRValue> part_of_explicit_cast
  // CHECK-NEXT:       `-DeclRefExpr {{.+}} 'const int *__single*__bidi_indexable' lvalue Var {{.+}} 'cbi' 'const int *__single*__bidi_indexable'
  i = (int **__indexable)cbi;

  // CHECK:      BinaryOperator {{.+}} 'const int *__single*__bidi_indexable' '='
  // CHECK-NEXT: |-DeclRefExpr {{.+}} 'const int *__single*__bidi_indexable' lvalue Var {{.+}} 'cbi' 'const int *__single*__bidi_indexable'
  // CHECK-NEXT: `-CStyleCastExpr {{.+}} 'const int *__single*__bidi_indexable' <BoundsSafetyPointerCast>
  // CHECK-NEXT:   `-ImplicitCastExpr {{.+}} 'const int *__single*__indexable' <NoOp> part_of_explicit_cast
  // CHECK-NEXT:     `-ImplicitCastExpr {{.+}} 'int *__single*__indexable' <LValueToRValue> part_of_explicit_cast
  // CHECK-NEXT:       `-DeclRefExpr {{.+}} 'int *__single*__indexable' lvalue Var {{.+}} 'i' 'int *__single*__indexable'
  cbi = (const int **__bidi_indexable)i;

  // CHECK:      BinaryOperator {{.+}} 'int *__single*__indexable' '='
  // CHECK-NEXT: |-DeclRefExpr {{.+}} 'int *__single*__indexable' lvalue Var {{.+}} 'i' 'int *__single*__indexable'
  // CHECK-NEXT: `-CStyleCastExpr {{.+}} 'int *__single*__indexable' <BoundsSafetyPointerCast>
  // CHECK-NEXT:   `-ImplicitCastExpr {{.+}} 'int *__single*__bidi_indexable' <BitCast> part_of_explicit_cast
  // CHECK-NEXT:     `-ImplicitCastExpr {{.+}} 'const void *__single*__bidi_indexable' <LValueToRValue> part_of_explicit_cast
  // CHECK-NEXT:       `-DeclRefExpr {{.+}} 'const void *__single*__bidi_indexable' lvalue Var {{.+}} 'cbv' 'const void *__single*__bidi_indexable'
  i = (int **__indexable)cbv;

  // CHECK:       BinaryOperator {{.+}} 'const void *__single*__bidi_indexable' '='
  // CHECK-NEXT:  |-DeclRefExpr {{.+}} 'const void *__single*__bidi_indexable' lvalue Var {{.+}} 'cbv' 'const void *__single*__bidi_indexable'
  // CHECK-NEXT:  `-CStyleCastExpr {{.+}} 'const void *__single*__bidi_indexable' <BoundsSafetyPointerCast>
  // CHECK-NEXT:    `-ImplicitCastExpr {{.+}} 'const void *__single*__indexable' <BitCast> part_of_explicit_cast
  // CHECK-NEXT:      `-ImplicitCastExpr {{.+}} 'int *__single*__indexable' <LValueToRValue> part_of_explicit_cast
  // CHECK-NEXT:        `-DeclRefExpr {{.+}} 'int *__single*__indexable' lvalue Var {{.+}} 'i' 'int *__single*__indexable'
  cbv = (const void **__bidi_indexable)i;
}
