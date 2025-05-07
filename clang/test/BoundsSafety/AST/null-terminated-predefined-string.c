
// RUN: %clang_cc1 -fbounds-safety -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s | FileCheck %s

#include <ptrcheck.h>

void foo(const char *);
void bar(void) {
  foo(__func__);
  foo(__FUNCTION__);
  foo(__PRETTY_FUNCTION__);
}
// CHECK:TranslationUnitDecl {{.*}} <<invalid sloc>> <invalid sloc>
// CHECK:|-FunctionDecl {{.*}} used foo 'void (const char *__single __terminated_by(0))'
// CHECK-NEXT:| `-ParmVarDecl {{.*}} 'const char *__single __terminated_by(0)':'const char *__single'
// CHECK-NEXT:`-FunctionDecl {{.*}} bar 'void (void)'
// CHECK-NEXT:  `-CompoundStmt {{.*}}
// CHECK-NEXT:    |-CallExpr {{.*}} 'void'
// CHECK-NEXT:    | |-ImplicitCastExpr {{.*}} 'void (*__single)(const char *__single __terminated_by(0))' <FunctionToPointerDecay>
// CHECK-NEXT:    | | `-DeclRefExpr {{.*}} 'void (const char *__single __terminated_by(0))' Function {{.*}} 'foo' 'void (const char *__single __terminated_by(0))'
// CHECK-NEXT:    | `-ImplicitCastExpr {{.*}} 'const char *__single __terminated_by(0)':'const char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT:    |   `-ImplicitCastExpr {{.*}} 'const char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT:    |     `-PredefinedExpr {{.*}} 'const char[4]' lvalue __func__
// CHECK-NEXT:    |       `-StringLiteral {{.*}} 'const char[4]' lvalue "bar"
// CHECK-NEXT:    |-CallExpr {{.*}} 'void'
// CHECK-NEXT:    | |-ImplicitCastExpr {{.*}} 'void (*__single)(const char *__single __terminated_by(0))' <FunctionToPointerDecay>
// CHECK-NEXT:    | | `-DeclRefExpr {{.*}} 'void (const char *__single __terminated_by(0))' Function {{.*}} 'foo' 'void (const char *__single __terminated_by(0))'
// CHECK-NEXT:    | `-ImplicitCastExpr {{.*}} 'const char *__single __terminated_by(0)':'const char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT:    |   `-ImplicitCastExpr {{.*}} 'const char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT:    |     `-PredefinedExpr {{.*}} 'const char[4]' lvalue __FUNCTION__
// CHECK-NEXT:    |       `-StringLiteral {{.*}} 'const char[4]' lvalue "bar"
// CHECK-NEXT:    `-CallExpr {{.*}} 'void'
// CHECK-NEXT:      |-ImplicitCastExpr {{.*}} 'void (*__single)(const char *__single __terminated_by(0))' <FunctionToPointerDecay>
// CHECK-NEXT:      | `-DeclRefExpr {{.*}} 'void (const char *__single __terminated_by(0))' Function {{.*}} 'foo' 'void (const char *__single __terminated_by(0))'
// CHECK-NEXT:      `-ImplicitCastExpr {{.*}} 'const char *__single __terminated_by(0)':'const char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT:        `-ImplicitCastExpr {{.*}} 'const char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT:          `-PredefinedExpr {{.*}} 'const char[15]' lvalue __PRETTY_FUNCTION__
// CHECK-NEXT:            `-StringLiteral {{.*}} 'const char[15]' lvalue "void bar(void)"
