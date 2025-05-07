

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -fbounds-safety %s -o /dev/null

// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s -o /dev/null

struct s {
	int dummy;
};

struct s *g;

typedef struct s s_t;
s_t *f(void) {
  s_t *l;
  return g ? l : g;
}
// CHECK:TranslationUnitDecl {{.*}} <<invalid sloc>> <invalid sloc>
// CHECK:|-RecordDecl {{.*}} <{{.*}}common-sugared-type.c:9:1, line:11:1> line:9:8 struct s definition
// CHECK-NEXT:| `-FieldDecl {{.*}} <line:10:2, col:6> col:6 dummy 'int'
// CHECK-NEXT:|-VarDecl {{.*}} <line:13:1, col:11> col:11 used g 'struct s *__single'
// CHECK-NEXT:|-TypedefDecl {{.*}} <line:15:1, col:18> col:18 referenced s_t 'struct s'
// CHECK-NEXT:| `-ElaboratedType {{.*}} 'struct s' sugar
// CHECK-NEXT:|   `-RecordType {{.*}} 'struct s'
// CHECK-NEXT:|     `-Record {{.*}} 's'
// CHECK-NEXT:`-FunctionDecl {{.*}} <line:16:1, line:19:1> line:16:6 f 's_t *__single(void)'
// CHECK-NEXT:  `-CompoundStmt {{.*}} <col:14, line:19:1>
// CHECK-NEXT:    |-DeclStmt {{.*}} <line:17:3, col:9>
// CHECK-NEXT:    | `-VarDecl {{.*}} <col:3, col:8> col:8 used l 's_t *__bidi_indexable'
// CHECK-NEXT:    `-ReturnStmt {{.*}} <line:18:3, col:18>
// CHECK-NEXT:      `-ImplicitCastExpr {{.*}} <col:10, col:18> 's_t *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT:        `-ConditionalOperator {{.*}} <col:10, col:18> 'struct s *__bidi_indexable'
// CHECK-NEXT:          |-ImplicitCastExpr {{.*}} <col:10> 'struct s *__single' <LValueToRValue>
// CHECK-NEXT:          | `-DeclRefExpr {{.*}} <col:10> 'struct s *__single' lvalue Var {{.*}} 'g' 'struct s *__single'
// CHECK-NEXT:          |-ImplicitCastExpr {{.*}} <col:14> 's_t *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT:          | `-DeclRefExpr {{.*}} <col:14> 's_t *__bidi_indexable' lvalue Var {{.*}} 'l' 's_t *__bidi_indexable'
// CHECK-NEXT:          `-ImplicitCastExpr {{.*}} <col:18> 'struct s *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK-NEXT:            `-ImplicitCastExpr {{.*}} <col:18> 'struct s *__single' <LValueToRValue>
// CHECK-NEXT:              `-DeclRefExpr {{.*}} <col:18> 'struct s *__single' lvalue Var {{.*}} 'g' 'struct s *__single'
