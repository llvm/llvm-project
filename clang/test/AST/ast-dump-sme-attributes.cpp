// Test without serialization:
// RUN: %clang_cc1 -triple aarch64 -target-feature +sme -std=c++2a -ast-dump -ast-dump-filter Foo %s | FileCheck -strict-whitespace %s

// Test with serialization:
// RUN: %clang_cc1 -std=c++20 -triple aarch64 -target-feature +sme -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++20 -triple aarch64 -target-feature +sme -include-pch %t -ast-dump-all -ast-dump-filter Foo /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck --strict-whitespace %s

struct Foo {
// CHECK:      |-CXXRecordDecl {{.*}} implicit struct Foo
// CHECK-NEXT: |-CXXMethodDecl {{.*}} f_streaming 'void () __arm_streaming'
// CHECK-NEXT: |-CXXMethodDecl {{.*}} f_streaming_compatible 'void () __arm_streaming_compatible'
// CHECK-NEXT: |-CXXMethodDecl {{.*}} f_locally_streaming 'void ()'
// CHECK-NEXT: | `-attrDetails: ArmLocallyStreamingAttr
// CHECK-NEXT: |-CXXMethodDecl {{.*}} f_shared_za 'void () __arm_inout("za")'
// CHECK-NEXT: |-CXXMethodDecl {{.*}} f_new_za 'void ()'
// CHECK-NEXT: | `-attrDetails: ArmNewAttr {{.*}} za
// CHECK-NEXT: |-CXXMethodDecl {{.*}} f_preserves_za 'void () __arm_preserves("za")'
  void f_streaming() __arm_streaming;
  void f_streaming_compatible() __arm_streaming_compatible;
  __arm_locally_streaming void f_locally_streaming();
  void f_shared_za() __arm_inout("za");
  __arm_new("za") void f_new_za();
  void f_preserves_za() __arm_preserves("za");



  int test_lambda(int x) {
    auto F = [](int x) __arm_streaming { return x; };
    return F(x);
  }

// CHECK: |-CXXMethodDecl {{.*}} test_lambda 'int (int)' implicit-inline
// CHECK: | |-ParmVarDecl {{.*}} used x 'int'
// CHECK: | | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK: | `-CompoundStmt {{.*}}
// CHECK: |   |-DeclStmt {{.*}} 
// CHECK: |   | `-VarDecl {{.*}} used F '(lambda at {{.*}})' cinit
// CHECK: |   |   |-LambdaExpr {{.*}} <col:14, col:52> '(lambda at {{.*}})'

  typedef void (*s_ptrty) (int, int) __arm_streaming;

// CHECK: |-TypedefDecl {{.*}} referenced s_ptrty 'void (*)(int, int) __arm_streaming'
// CHECK-NEXT: | `-typeDetails: PointerType {{.*}} 'void (*)(int, int) __arm_streaming'
// CHECK-NEXT: |   `-typeDetails: ParenType {{.*}} 'void (int, int) __arm_streaming' sugar
// CHECK-NEXT: |     `-typeDetails: FunctionProtoType {{.*}} 'void (int, int) __arm_streaming' cdecl

  void test_streaming_ptrty(s_ptrty f, int x, int y) { return f(x, y); };

// CHECK: `-CXXMethodDecl {{.*}} test_streaming_ptrty 'void (s_ptrty, int, int)' implicit-inline
// CHECK:   |-ParmVarDecl {{.*}} used f 's_ptrty':'void (*)(int, int) __arm_streaming'
// CHECK:   | `-typeDetails: ElaboratedType {{.*}} 's_ptrty' sugar
// CHECK:   |   `-typeDetails: TypedefType {{.*}} 'Foo::s_ptrty' sugar
// CHECK:   |     |-Typedef {{.*}} 's_ptrty'
// CHECK:   |     `-typeDetails: PointerType {{.*}} 'void (*)(int, int) __arm_streaming'
// CHECK:   |       `-typeDetails: ParenType {{.*}} 'void (int, int) __arm_streaming' sugar
// CHECK:   |         `-typeDetails: FunctionProtoType {{.*}} 'void (int, int) __arm_streaming' cdecl
// CHECK:   |           |-typeDetails: BuiltinType {{.*}} 'void'
// CHECK:   |           |-functionDetails:  cdeclReturnType {{.*}} 'void'
// CHECK:   |           |-typeDetails: BuiltinType {{.*}} 'int'
// CHECK:   |           `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK:   |-ParmVarDecl {{.*}} used x 'int'
// CHECK:   | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK:   |-ParmVarDecl {{.*}} used y 'int'
// CHECK:   | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK:   `-CompoundStmt {{.*}} 
// CHECK:     `-ReturnStmt {{.*}} 
// CHECK:       `-CallExpr {{.*}} 'void'
// CHECK:         |-ImplicitCastExpr {{.*}} 's_ptrty':'void (*)(int, int) __arm_streaming' <LValueToRValue>
// CHECK:         | `-DeclRefExpr {{.*}} 's_ptrty':'void (*)(int, int) __arm_streaming' lvalue ParmVar {{.*}} 'f' 's_ptrty':'void (*)(int, int) __arm_streaming'
// CHECK:         |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK:         | `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK:         `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK:           `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'y' 'int'
};
