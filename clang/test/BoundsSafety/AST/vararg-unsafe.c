

// RUN: %clang_cc1 -fbounds-safety -ast-dump -triple x86_64 -verify %s 2>&1 | FileCheck %s --check-prefix=COMMON --check-prefix=VALIST_ARRAY
// RUN: %clang_cc1 -fbounds-safety -ast-dump -triple i686 %s -verify 2>&1 | FileCheck %s --check-prefix=COMMON --check-prefix=VALIST_CHAR_PTR
// RUN: %clang_cc1 -fbounds-safety -ast-dump -triple arm %s -verify 2>&1 | FileCheck %s --check-prefix=COMMON --check-prefix=VALIST_STRUCT
// RUN: %clang_cc1 -fbounds-safety -ast-dump -triple arm-apple-watchos -verify %s 2>&1 | FileCheck %s --check-prefix=COMMON --check-prefix=VALIST_VOID_PTR
// RUN: %clang_cc1 -fbounds-safety -ast-dump -triple arm64-apple-macosx %s -verify 2>&1 | FileCheck %s --check-prefix=COMMON --check-prefix=VALIST_CHAR_PTR
// RUN: %clang_cc1 -fbounds-safety -ast-dump -triple arm64-apple-ios %s -verify 2>&1 | FileCheck %s --check-prefix=COMMON --check-prefix=VALIST_CHAR_PTR
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump -triple x86_64 -verify %s 2>&1 | FileCheck %s --check-prefix=COMMON --check-prefix=VALIST_ARRAY
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump -triple i686 %s -verify 2>&1 | FileCheck %s --check-prefix=COMMON --check-prefix=VALIST_CHAR_PTR
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump -triple arm %s -verify 2>&1 | FileCheck %s --check-prefix=COMMON --check-prefix=VALIST_STRUCT
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump -triple arm-apple-watchos -verify %s 2>&1 | FileCheck %s --check-prefix=COMMON --check-prefix=VALIST_VOID_PTR
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump -triple arm64-apple-macosx %s -verify 2>&1 | FileCheck %s --check-prefix=COMMON --check-prefix=VALIST_CHAR_PTR
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump -triple arm64-apple-ios %s -verify 2>&1 | FileCheck %s --check-prefix=COMMON --check-prefix=VALIST_CHAR_PTR

// expected-no-diagnostics

#include <stdarg.h>
#include <ptrcheck.h>

void baz(unsigned int n, va_list argp) {
	int *__unsafe_indexable u = 0;
  u = va_arg(argp, int *);
	va_end(argp);
}

void foo(unsigned int n, ...) {
  int *__unsafe_indexable u = 0;
	va_list argp;
	va_start(argp, n);
	baz(n, argp);
  u = va_arg(argp, int *);
	va_end(argp);
}

typedef void (*f_t)(unsigned int n, va_list args);

void test_va_list_on_fp(unsigned int n, va_list args) {
  f_t f = baz;
  f(n, args);
}

void bar(void) {
  int arr[10] = { 0 };
  int *p;
  foo(1, p, arr);
}

// VALIST_ARRAY: |-FunctionDecl {{.*}} used baz 'void (unsigned int, struct __va_list_tag *)'
// VALIST_ARRAY: | |-ParmVarDecl {{.*}} used argp 'struct __va_list_tag *'
// VALIST_ARRAY-LABEL: foo
// VALIST_ARRAY: CompoundStmt
// VALIST_ARRAY: |-DeclStmt
// VALIST_ARRAY: | `-VarDecl {{.*}} used u 'int *__unsafe_indexable' cinit
// VALIST_ARRAY: |   `-ImplicitCastExpr {{.*}} 'int *__unsafe_indexable' <NullToPointer>
// VALIST_ARRAY: |     `-IntegerLiteral {{.*}} 'int' 0
// VALIST_ARRAY: |-DeclStmt
// VALIST_ARRAY: | `-VarDecl {{.*}} used argp 'va_list':'struct __va_list_tag[1]'
// VALIST_ARRAY: |-CallExpr
// VALIST_ARRAY: | |-ImplicitCastExpr {{.*}} 'void (*)(struct __va_list_tag *, ...)' <BuiltinFnToFnPtr>
// VALIST_ARRAY: | | `-DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_va_start' 'void (struct __va_list_tag *, ...)'
// VALIST_ARRAY: | |-ImplicitCastExpr {{.*}} 'struct __va_list_tag *' <ArrayToPointerDecay>
// VALIST_ARRAY: | | `-DeclRefExpr {{.*}} 'va_list':'struct __va_list_tag[1]' lvalue Var {{.*}} 'argp' 'va_list':'struct __va_list_tag[1]'
// VALIST_ARRAY: | `-DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'n' 'unsigned int'
// VALIST_ARRAY: |-BinaryOperator {{.*}} 'int *__unsafe_indexable' '='
// VALIST_ARRAY: | |-DeclRefExpr {{.*}} 'int *__unsafe_indexable' lvalue Var {{.*}} 'u' 'int *__unsafe_indexable'
// VALIST_ARRAY: | `-ImplicitCastExpr {{.*}} 'int *__unsafe_indexable' <BoundsSafetyPointerCast>
// VALIST_ARRAY: |   `-VAArgExpr {{.*}} 'int *'
// VALIST_ARRAY: |     `-ImplicitCastExpr {{.*}} 'struct __va_list_tag *' <ArrayToPointerDecay>
// VALIST_ARRAY: |       `-DeclRefExpr {{.*}} 'va_list':'struct __va_list_tag[1]' lvalue Var {{.*}} 'argp' 'va_list':'struct __va_list_tag[1]'
// VALIST_ARRAY: `-CallExpr
// VALIST_ARRAY:   |-ImplicitCastExpr {{.*}} 'void (*)(struct __va_list_tag *)' <BuiltinFnToFnPtr>
// VALIST_ARRAY:   | `-DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_va_end' 'void (struct __va_list_tag *)'
// VALIST_ARRAY:   `-ImplicitCastExpr {{.*}} 'struct __va_list_tag *' <ArrayToPointerDecay>
// VALIST_ARRAY:     `-DeclRefExpr {{.*}} 'va_list':'struct __va_list_tag[1]' lvalue Var {{.*}} 'argp' 'va_list':'struct __va_list_tag[1]'

// VALIST_ARRAY-LABEL: test_va_list_on_fp
// VALIST_ARRAY: |-ParmVarDecl {{.*}} used n 'unsigned int'
// VALIST_ARRAY: |-ParmVarDecl {{.*}} used args 'struct __va_list_tag *'
// VALIST_ARRAY: `-CompoundStmt
// VALIST_ARRAY:   |-DeclStmt
// VALIST_ARRAY:   | `-VarDecl {{.*}} used f 'void (*__single)(unsigned int, struct __va_list_tag *)' cinit
// VALIST_ARRAY:   |   `-ImplicitCastExpr {{.*}} 'void (*__single)(unsigned int, struct __va_list_tag *)' <FunctionToPointerDecay>
// VALIST_ARRAY:   |     `-DeclRefExpr {{.*}} 'void (unsigned int, struct __va_list_tag *)' Function {{.*}} 'baz' 'void (unsigned int, struct __va_list_tag *)'
// VALIST_ARRAY:   `-CallExpr {{.*}} 'void'
// VALIST_ARRAY:     |-ImplicitCastExpr {{.*}} 'void (*__single)(unsigned int, struct __va_list_tag *)' <LValueToRValue>
// VALIST_ARRAY:     | `-DeclRefExpr {{.*}} 'void (*__single)(unsigned int, struct __va_list_tag *)' lvalue Var {{.*}} 'f' 'void (*__single)(unsigned int, struct __va_list_tag *)'
// VALIST_ARRAY:     |-ImplicitCastExpr {{.*}} 'unsigned int' <LValueToRValue>
// VALIST_ARRAY:     | `-DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'n' 'unsigned int'
// VALIST_ARRAY:     `-ImplicitCastExpr {{.*}} 'struct __va_list_tag *' <LValueToRValue>
// VALIST_ARRAY:       `-DeclRefExpr {{.*}} 'struct __va_list_tag *' lvalue ParmVar {{.*}} 'args' 'struct __va_list_tag *'

// VALIST_CHAR_PTR: |-FunctionDecl {{.*}} used baz 'void (unsigned int, va_list)'
// VALIST_CHAR_PTR: | |-ParmVarDecl {{.*}} used argp 'va_list':'char *'
// VALIST_CHAR_PTR-LABEL: foo
// VALIST_CHAR_PTR: CompoundStmt
// VALIST_CHAR_PTR: |-DeclStmt
// VALIST_CHAR_PTR: | `-VarDecl {{.*}} used u 'int *__unsafe_indexable' cinit
// VALIST_CHAR_PTR: |   `-ImplicitCastExpr {{.*}} 'int *__unsafe_indexable' <NullToPointer>
// VALIST_CHAR_PTR: |     `-IntegerLiteral {{.*}} 'int' 0
// VALIST_CHAR_PTR: |-DeclStmt
// VALIST_CHAR_PTR: | `-VarDecl {{.*}} used argp 'va_list':'char *'
// VALIST_CHAR_PTR: |-CallExpr
// VALIST_CHAR_PTR: | |-ImplicitCastExpr {{.*}} 'void (*)(__builtin_va_list &, ...)' <BuiltinFnToFnPtr>
// VALIST_CHAR_PTR: | | `-DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_va_start' 'void (__builtin_va_list &, ...)'
// VALIST_CHAR_PTR: | |-DeclRefExpr {{.*}} 'va_list':'char *' lvalue Var {{.*}} 'argp' 'va_list':'char *'
// VALIST_CHAR_PTR: | `-DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'n' 'unsigned int'
// VALIST_CHAR_PTR: |-BinaryOperator {{.*}} 'int *__unsafe_indexable' '='
// VALIST_CHAR_PTR: | |-DeclRefExpr {{.*}} 'int *__unsafe_indexable' lvalue Var {{.*}} 'u' 'int *__unsafe_indexable'
// VALIST_CHAR_PTR: | `-ImplicitCastExpr {{.*}} 'int *__unsafe_indexable' <BoundsSafetyPointerCast>
// VALIST_CHAR_PTR: |   `-VAArgExpr {{.*}} 'int *'
// VALIST_CHAR_PTR: |     `-DeclRefExpr {{.*}} 'va_list':'char *' lvalue Var {{.*}} 'argp' 'va_list':'char *'
// VALIST_CHAR_PTR: `-CallExpr
// VALIST_CHAR_PTR:   |-ImplicitCastExpr {{.*}} 'void (*)(__builtin_va_list &)' <BuiltinFnToFnPtr>
// VALIST_CHAR_PTR:   | `-DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_va_end' 'void (__builtin_va_list &)'
// VALIST_CHAR_PTR:   `-DeclRefExpr {{.*}} 'va_list':'char *' lvalue Var {{.*}} 'argp' 'va_list':'char *'
// VALIST_CHAR_PTR-LABEL: test_va_list_on_fp 'void (unsigned int, va_list)'
// VALIST_CHAR_PTR: | |-ParmVarDecl {{.*}} used n 'unsigned int'
// VALIST_CHAR_PTR: | |-ParmVarDecl {{.*}} used args 'va_list':'char *'
// VALIST_CHAR_PTR: | `-CompoundStmt {{.*}}
// VALIST_CHAR_PTR: |   |-DeclStmt
// VALIST_CHAR_PTR: |   | `-VarDecl {{.*}} used f 'void (*__single)(unsigned int, va_list)' cinit
// VALIST_CHAR_PTR: |   |   `-ImplicitCastExpr {{.*}} 'void (*__single)(unsigned int, va_list)' <FunctionToPointerDecay>
// VALIST_CHAR_PTR: |   |     `-DeclRefExpr {{.*}} 'void (unsigned int, va_list)' Function {{.*}} 'baz' 'void (unsigned int, va_list)'
// VALIST_CHAR_PTR: |   `-CallExpr {{.*}} 'void'
// VALIST_CHAR_PTR: |     |-ImplicitCastExpr {{.*}} 'void (*__single)(unsigned int, va_list)' <LValueToRValue>
// VALIST_CHAR_PTR: |     | `-DeclRefExpr {{.*}} 'void (*__single)(unsigned int, va_list)' lvalue Var {{.*}} 'f' 'void (*__single)(unsigned int, va_list)'
// VALIST_CHAR_PTR: |     |-ImplicitCastExpr {{.*}} 'unsigned int' <LValueToRValue>
// VALIST_CHAR_PTR: |     | `-DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'n' 'unsigned int'
// VALIST_CHAR_PTR: |     `-ImplicitCastExpr {{.*}} 'va_list':'char *' <LValueToRValue>
// VALIST_CHAR_PTR: |       `-DeclRefExpr {{.*}} 'va_list':'char *' lvalue ParmVar {{.*}} 'args' 'va_list':'char *'

// VALIST_STRUCT: |-FunctionDecl {{.*}} used baz 'void (unsigned int, va_list)'
// VALIST_STRUCT: | |-ParmVarDecl {{.*}} used argp 'va_list':'struct __va_list'
// VALIST_STRUCT-LABEL: foo
// VALIST_STRUCT: CompoundStmt
// VALIST_STRUCT: |-DeclStmt
// VALIST_STRUCT: | `-VarDecl {{.*}} used u 'int *__unsafe_indexable' cinit
// VALIST_STRUCT: |   `-ImplicitCastExpr {{.*}} 'int *__unsafe_indexable' <NullToPointer>
// VALIST_STRUCT: |     `-IntegerLiteral {{.*}} 'int' 0
// VALIST_STRUCT: |-DeclStmt
// VALIST_STRUCT: | `-VarDecl {{.*}} used argp 'va_list':'struct __va_list'
// VALIST_STRUCT: |-CallExpr {{.*}} 'void'
// VALIST_STRUCT: | |-ImplicitCastExpr {{.*}} 'void (*)(__builtin_va_list &, ...)' <BuiltinFnToFnPtr>
// VALIST_STRUCT: | | `-DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_va_start' 'void (__builtin_va_list &, ...)'
// VALIST_STRUCT: | |-DeclRefExpr {{.*}} 'va_list':'struct __va_list' lvalue Var {{.*}} 'argp' 'va_list':'struct __va_list'
// VALIST_STRUCT: | `-DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'n' 'unsigned int'
// VALIST_STRUCT: |-BinaryOperator {{.*}} 'int *__unsafe_indexable' '='
// VALIST_STRUCT: | |-DeclRefExpr {{.*}} 'int *__unsafe_indexable' lvalue Var {{.*}} 'u' 'int *__unsafe_indexable'
// VALIST_STRUCT: | `-ImplicitCastExpr {{.*}} 'int *__unsafe_indexable' <BoundsSafetyPointerCast>
// VALIST_STRUCT: |   `-VAArgExpr {{.*}} 'int *'
// VALIST_STRUCT: |     `-DeclRefExpr {{.*}} 'va_list':'struct __va_list' lvalue Var {{.*}} 'argp' 'va_list':'struct __va_list'
// VALIST_STRUCT: `-CallExpr {{.*}} 'void'
// VALIST_STRUCT:   |-ImplicitCastExpr {{.*}} 'void (*)(__builtin_va_list &)' <BuiltinFnToFnPtr>
// VALIST_STRUCT:   | `-DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_va_end' 'void (__builtin_va_list &)'
// VALIST_STRUCT:   `-DeclRefExpr {{.*}} 'va_list':'struct __va_list' lvalue Var {{.*}} 'argp' 'va_list':'struct __va_list'

// VALIST_STRUCT-LABEL: test_va_list_on_fp 'void (unsigned int, va_list)'
// VALIST_STRUCT: | |-ParmVarDecl {{.*}} used n 'unsigned int'
// VALIST_STRUCT: | |-ParmVarDecl {{.*}} used args 'va_list':'struct __va_list'
// VALIST_STRUCT: | `-CompoundStmt
// VALIST_STRUCT: |   |-DeclStmt
// VALIST_STRUCT: |   | `-VarDecl {{.*}} used f 'void (*__single)(unsigned int, va_list)' cinit
// VALIST_STRUCT: |   |   `-ImplicitCastExpr {{.*}} 'void (*__single)(unsigned int, va_list)' <FunctionToPointerDecay>
// VALIST_STRUCT: |   |     `-DeclRefExpr {{.*}} 'void (unsigned int, va_list)' Function {{.*}} 'baz' 'void (unsigned int, va_list)'
// VALIST_STRUCT: |   `-CallExpr {{.*}} 'void'
// VALIST_STRUCT: |     |-ImplicitCastExpr {{.*}} 'void (*__single)(unsigned int, va_list)' <LValueToRValue>
// VALIST_STRUCT: |     | `-DeclRefExpr {{.*}} 'void (*__single)(unsigned int, va_list)' lvalue Var {{.*}} 'f' 'void (*__single)(unsigned int, va_list)'
// VALIST_STRUCT: |     |-ImplicitCastExpr {{.*}} 'unsigned int' <LValueToRValue>
// VALIST_STRUCT: |     | `-DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'n' 'unsigned int'
// VALIST_STRUCT: |     `-ImplicitCastExpr {{.*}} 'va_list':'struct __va_list' <LValueToRValue>
// VALIST_STRUCT: |       `-DeclRefExpr {{.*}} 'va_list':'struct __va_list' lvalue ParmVar {{.*}} 'args' 'va_list':'struct __va_list'

// VALIST_VOID_PTR: |-FunctionDecl {{.*}} used baz 'void (unsigned int, va_list)'
// VALIST_VOID_PTR: | |-ParmVarDecl {{.*}} used argp 'va_list':'void *'
// VALIST_VOID_PTR-LABEL: foo
// VALIST_VOID_PTR: CompoundStmt
// VALIST_VOID_PTR: |-DeclStmt
// VALIST_VOID_PTR: | `-VarDecl {{.*}} col:27 used u 'int *__unsafe_indexable' cinit
// VALIST_VOID_PTR: |   `-ImplicitCastExpr {{.*}} 'int *__unsafe_indexable' <NullToPointer>
// VALIST_VOID_PTR: |     `-IntegerLiteral {{.*}} 'int' 0
// VALIST_VOID_PTR: |-DeclStmt
// VALIST_VOID_PTR: | `-VarDecl {{.*}} col:10 used argp 'va_list':'void *'
// VALIST_VOID_PTR: |-CallExpr
// VALIST_VOID_PTR: | |-ImplicitCastExpr {{.*}} 'void (*)(__builtin_va_list &, ...)' <BuiltinFnToFnPtr>
// VALIST_VOID_PTR: | | `-DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_va_start' 'void (__builtin_va_list &, ...)'
// VALIST_VOID_PTR: | |-DeclRefExpr {{.*}} 'va_list':'void *' lvalue Var {{.*}} 'argp' 'va_list':'void *'
// VALIST_VOID_PTR: | `-DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'n' 'unsigned int'
// VALIST_VOID_PTR: |-BinaryOperator {{.*}} 'int *__unsafe_indexable' '='
// VALIST_VOID_PTR: | |-DeclRefExpr {{.*}} 'int *__unsafe_indexable' lvalue Var {{.*}} 'u' 'int *__unsafe_indexable'
// VALIST_VOID_PTR: | `-ImplicitCastExpr {{.*}} 'int *__unsafe_indexable' <BoundsSafetyPointerCast>
// VALIST_VOID_PTR: |   `-VAArgExpr {{.*}} 'int *'
// VALIST_VOID_PTR: |     `-DeclRefExpr {{.*}} 'va_list':'void *' lvalue Var {{.*}} 'argp' 'va_list':'void *'
// VALIST_VOID_PTR: `-CallExpr {{.*}} 'void'
// VALIST_VOID_PTR:   |-ImplicitCastExpr {{.*}} 'void (*)(__builtin_va_list &)' <BuiltinFnToFnPtr>
// VALIST_VOID_PTR:   | `-DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_va_end' 'void (__builtin_va_list &)'
// VALIST_VOID_PTR:   `-DeclRefExpr {{.*}} 'va_list':'void *' lvalue Var {{.*}} 'argp' 'va_list':'void *'

// VALIST_VOID_PTR-LABEL: test_va_list_on_fp 'void (unsigned int, va_list)'
// VALIST_VOID_PTR: | |-ParmVarDecl {{.*}} used n 'unsigned int'
// VALIST_VOID_PTR: | |-ParmVarDecl {{.*}} used args 'va_list':'void *'
// VALIST_VOID_PTR: | `-CompoundStmt
// VALIST_VOID_PTR: |   |-DeclStmt
// VALIST_VOID_PTR: |   | `-VarDecl {{.*}} used f 'void (*__single)(unsigned int, va_list)' cinit
// VALIST_VOID_PTR: |   |   `-ImplicitCastExpr {{.*}} 'void (*__single)(unsigned int, va_list)' <FunctionToPointerDecay>
// VALIST_VOID_PTR: |   |     `-DeclRefExpr {{.*}} 'void (unsigned int, va_list)' Function {{.*}} 'baz' 'void (unsigned int, va_list)'
// VALIST_VOID_PTR: |   `-CallExpr {{.*}} 'void'
// VALIST_VOID_PTR: |     |-ImplicitCastExpr {{.*}} 'void (*__single)(unsigned int, va_list)' <LValueToRValue>
// VALIST_VOID_PTR: |     | `-DeclRefExpr {{.*}} 'void (*__single)(unsigned int, va_list)' lvalue Var {{.*}} 'f' 'void (*__single)(unsigned int, va_list)'
// VALIST_VOID_PTR: |     |-ImplicitCastExpr {{.*}} 'unsigned int' <LValueToRValue>
// VALIST_VOID_PTR: |     | `-DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'n' 'unsigned int'
// VALIST_VOID_PTR: |     `-ImplicitCastExpr {{.*}} 'va_list':'void *' <LValueToRValue>
// VALIST_VOID_PTR: |       `-DeclRefExpr {{.*}} 'va_list':'void *' lvalue ParmVar {{.*}} 'args' 'va_list':'void *'

// COMMON-LABEL: bar
// COMMON: CallExpr
// COMMON: |-ImplicitCastExpr {{.*}} 'void (*__single)(unsigned int, ...)' <FunctionToPointerDecay>
// COMMON: | `-DeclRefExpr {{.*}} 'void (unsigned int, ...)' Function {{.*}} 'foo' 'void (unsigned int, ...)'
// COMMON: |-ImplicitCastExpr {{.*}} 'unsigned int' <IntegralCast>
// COMMON: | `-IntegerLiteral {{.*}} 'int' 1
// COMMON: |-ImplicitCastExpr {{.*}} 'int *__unsafe_indexable' <BoundsSafetyPointerCast>
// COMMON: | `-ImplicitCastExpr {{.*}} 'int *__bidi_indexable' <LValueToRValue>
// COMMON: |   `-DeclRefExpr {{.*}} 'int *__bidi_indexable' lvalue Var {{.*}} 'p' 'int *__bidi_indexable'
// COMMON: `-ImplicitCastExpr {{.*}} 'int *__unsafe_indexable' <BoundsSafetyPointerCast>
// COMMON:   `-ImplicitCastExpr {{.*}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// COMMON:     `-DeclRefExpr {{.*}} 'int[10]' lvalue Var {{.*}} 'arr' 'int[10]'
