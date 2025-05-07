
// RUN: %clang_cc1 -ast-dump -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s

#include <ptrcheck.h>

void foo(void) {
    (void)(void *)(char *__single *)0;
    (void)(char *__single *)(void *)0;
}

// CHECK: TranslationUnitDecl
// CHECK: `-FunctionDecl [[func_foo:0x[^ ]+]] {{.+}} foo
// CHECK:   `-CompoundStmt
// CHECK:     |-CStyleCastExpr {{.+}} 'void' <ToVoid>
// CHECK:     | `-CStyleCastExpr {{.+}} 'void *' <BitCast>
// CHECK:     |   `-CStyleCastExpr {{.+}} 'char *__single*' <NullToPointer>
// CHECK:     |     `-IntegerLiteral {{.+}} 0
// CHECK:     `-CStyleCastExpr {{.+}} 'void' <ToVoid>
// CHECK:       `-CStyleCastExpr {{.+}} 'char *__single*' <BitCast>
// CHECK:         `-CStyleCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK:           `-IntegerLiteral {{.+}} 0
