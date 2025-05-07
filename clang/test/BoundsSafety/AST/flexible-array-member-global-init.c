
// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

struct flex_uchar {
	unsigned char len;
	unsigned char data[__counted_by(len - 1)];
};

struct flex_uchar init = { 3, {0, 1} };
// CHECK: `-VarDecl {{.+}} init 'struct flex_uchar' cinit
// CHECK:   `-InitListExpr {{.+}} 'struct flex_uchar'
// CHECK:     |-ImplicitCastExpr {{.+}} 'unsigned char' <IntegralCast>
// CHECK:     | `-IntegerLiteral {{.+}} 'int' 3
// CHECK:     `-InitListExpr {{.+}} 'unsigned char[2]'
// CHECK:       |-ImplicitCastExpr {{.+}} 'unsigned char' <IntegralCast>
// CHECK:       | `-IntegerLiteral {{.+}} 'int' 0
// CHECK:       `-ImplicitCastExpr {{.+}} 'unsigned char' <IntegralCast>
// CHECK:         `-IntegerLiteral {{.+}} 'int' 1
