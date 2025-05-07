

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s

int main() {
    int *ip = 0;
    char *cp = (char*)ip;

    return 0;
}

// CHECK: |-DeclStmt
// CHECK: | `-VarDecl {{.+}} cp 'char *__bidi_indexable'{{.*}} cinit
// CHECK: |   `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK: |     `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable'{{.*}} <LValueToRValue> part_of_explicit_cast
// CHECK: |       `-DeclRefExpr {{.+}} 'int *__bidi_indexable'{{.*}} lvalue Var {{.+}} 'ip' 'int *__bidi_indexable'
