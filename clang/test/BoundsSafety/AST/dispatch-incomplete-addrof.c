
// RUN: %clang_cc1 -fbounds-safety -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s | FileCheck %s

#define DISPATCH_GLOBAL_OBJECT(type, object) ((type)&(object))
#define DISPATCH_DECL(name) typedef struct name##_s *name##_t
DISPATCH_DECL(dispatch_data);

#define dispatch_data_empty \
			DISPATCH_GLOBAL_OBJECT(dispatch_data_t, _dispatch_data_empty)
extern struct dispatch_data_s _dispatch_data_empty;

void foo(dispatch_data_t _Nonnull arg);

// CHECK-LABEL: test 'void ()'
void test() {
  foo(dispatch_data_empty);
}
// CHECK: `-CompoundStmt
// CHECK:   `-CallExpr {{.*}} 'void'
// CHECK:     |-ImplicitCastExpr {{.*}} 'void (*__single)(struct dispatch_data_s *__single _Nonnull)' <FunctionToPointerDecay>
// CHECK:     | `-DeclRefExpr {{.*}} 'void (struct dispatch_data_s *__single _Nonnull)' Function {{.*}} 'foo' 'void (struct dispatch_data_s *__single _Nonnull)'
// CHECK:     `-ParenExpr {{.*}} 'struct dispatch_data_s *__single'
// CHECK:       `-CStyleCastExpr {{.*}} 'struct dispatch_data_s *__single' <NoOp>
// CHECK:         `-UnaryOperator {{.*}} 'struct dispatch_data_s *__single' prefix '&' cannot overflow
// CHECK:           `-ParenExpr {{.*}} 'struct dispatch_data_s' lvalue
// CHECK:             `-DeclRefExpr {{.*}} 'struct dispatch_data_s' lvalue Var {{.*}} '_dispatch_data_empty' 'struct dispatch_data_s'
