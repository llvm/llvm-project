

// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

// TODO: Remove this test when support for disabling return-size checks is removed.

// CHECK: FunctionDecl [[func_cb_in_from_bidi:0x[^ ]+]] {{.+}} cb_in_from_bidi
// CHECK: |-ParmVarDecl [[var_count:0x[^ ]+]]
// CHECK: |-ParmVarDecl [[var_p:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-ReturnStmt
// CHECK:     `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK:       `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK:         `-DeclRefExpr {{.+}} [[var_p]]
int *__counted_by(count) cb_in_from_bidi(int count, int *__bidi_indexable p) {
  return p;
}

// CHECK: FunctionDecl [[func_cb_in_from_indexable:0x[^ ]+]] {{.+}} cb_in_from_indexable
// CHECK: |-ParmVarDecl [[var_count_1:0x[^ ]+]]
// CHECK: |-ParmVarDecl [[var_p_1:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-ReturnStmt
// CHECK:     `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK:       `-ImplicitCastExpr {{.+}} 'int *__indexable' <LValueToRValue>
// CHECK:         `-DeclRefExpr {{.+}} [[var_p_1]]
int *__counted_by(count) cb_in_from_indexable(int count, int *__indexable p) {
  return p;
}

// CHECK: FunctionDecl [[func_cb_in_from_single:0x[^ ]+]] {{.+}} cb_in_from_single
// CHECK: |-ParmVarDecl [[var_count_2:0x[^ ]+]]
// CHECK: |-ParmVarDecl [[var_p_2:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-ReturnStmt
// CHECK:     `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// CHECK:       `-DeclRefExpr {{.+}} [[var_p_2]]
int *__counted_by(count) cb_in_from_single(int count, int *__single p) {
  return p;
}

// CHECK: {{^}}|-FunctionDecl [[func_cb_in_from_int_cast_null:0x[^ ]+]] {{.+}} cb_in_from_int_cast_null
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_count_3:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       `-CStyleCastExpr {{.+}} 'int *' <NullToPointer>
// CHECK-NEXT: {{^}}|         `-IntegerLiteral {{.+}} 0
int *__counted_by(count) cb_in_from_int_cast_null(int count) {
  return (int*)0;
}

// CHECK: {{^}}|-FunctionDecl [[func_cb_in_from_void_cast_null:0x[^ ]+]] {{.+}} cb_in_from_void_cast_null
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_count_4:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <NullToPointer>
// CHECK-NEXT: {{^}}|       `-CStyleCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT: {{^}}|         `-IntegerLiteral {{.+}} 0
int *__counted_by(count) cb_in_from_void_cast_null(int count) {
  return (void*)0;
}

// CHECK: {{^}}|-FunctionDecl [[func_cb_in_from_int_void_int_cast_null:0x[^ ]+]] {{.+}} cb_in_from_int_void_int_cast_null
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_count_5:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       `-CStyleCastExpr {{.+}} 'int *' <BitCast>
// CHECK-NEXT: {{^}}|         `-CStyleCastExpr {{.+}} 'void *' <BitCast>
// CHECK-NEXT: {{^}}|           `-CStyleCastExpr {{.+}} 'int *' <NullToPointer>
// CHECK-NEXT: {{^}}|             `-IntegerLiteral {{.+}} 0
int *__counted_by(count) cb_in_from_int_void_int_cast_null(int count) {
  return (int*)(void*)(int*)0;
}

// CHECK: FunctionDecl [[func_cb_out_from_single:0x[^ ]+]] {{.+}} cb_out_from_single
// CHECK: |-ParmVarDecl [[var_count_3:0x[^ ]+]]
// CHECK: |-ParmVarDecl [[var_p_3:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-ReturnStmt
// CHECK:     `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// CHECK:       `-DeclRefExpr {{.+}} [[var_p_3]]
int *__counted_by(*count) cb_out_from_single(int *__single count, int *__single p) {
  return p;
}

// CHECK: FunctionDecl [[func_sb_from_single:0x[^ ]+]] {{.+}} sb_from_single
// CHECK: |-ParmVarDecl [[var_size:0x[^ ]+]]
// CHECK: |-ParmVarDecl [[var_p_4:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-ReturnStmt
// CHECK:     `-ImplicitCastExpr {{.+}} 'void *__single __sized_by(size)':'void *__single' <BoundsSafetyPointerCast>
// CHECK:       `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <BitCast>
// CHECK:         `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK:           `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// CHECK:             `-DeclRefExpr {{.+}} [[var_p_4]]
void *__sized_by(size) sb_from_single(int size, int *__single p) {
  return p;
}

// CHECK: FunctionDecl [[func_cbn_in_from_single:0x[^ ]+]] {{.+}} cbn_in_from_single
// CHECK: |-ParmVarDecl [[var_count_4:0x[^ ]+]]
// CHECK: |-ParmVarDecl [[var_p_5:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-ReturnStmt
// CHECK:     `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// CHECK:       `-DeclRefExpr {{.+}} [[var_p_5]]
int *__counted_by_or_null(count) cbn_in_from_single(int count, int *__single p) {
  return p;
}

// CHECK: FunctionDecl [[func_eb_in_from_single:0x[^ ]+]] {{.+}} eb_in_from_single
// CHECK: |-ParmVarDecl [[var_end:0x[^ ]+]]
// CHECK: |-ParmVarDecl [[var_p_6:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-ReturnStmt
// CHECK:     `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// CHECK:       `-DeclRefExpr {{.+}} [[var_p_6]]
int *__ended_by(end) eb_in_from_single(int *__single end, int *__single p) {
  return p;
}
