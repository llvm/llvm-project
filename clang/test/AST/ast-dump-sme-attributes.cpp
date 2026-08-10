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
// CHECK-NEXT: | `-ArmLocallyStreamingAttr
// CHECK-NEXT: |-CXXMethodDecl {{.*}} f_shared_za 'void () __arm_inout("za")'
// CHECK-NEXT: |-CXXMethodDecl {{.*}} f_new_za 'void ()'
// CHECK-NEXT: | `-ArmNewAttr {{.*}} za
// CHECK-NEXT: |-CXXMethodDecl {{.*}} f_preserves_za 'void () __arm_preserves("za")'
  void f_streaming() __arm_streaming;
  void f_streaming_compatible() __arm_streaming_compatible;
  __arm_locally_streaming void f_locally_streaming();
  void f_shared_za() __arm_inout("za");
  __arm_new("za") void f_new_za();
  void f_preserves_za() __arm_preserves("za");


// CHECK:      |-CXXMethodDecl {{.*}} test_lambda 'int (int)' implicit-inline
// CHECK:         `-CompoundStmt
// CHECK-NEXT:     |-DeclStmt
// CHECK-NEXT:     | `-VarDecl
// CHECK-NEXT:     |   `-LambdaExpr
// CHECK-NEXT:     |     |-CXXRecordDecl
// CHECK:          |     | |-CXXMethodDecl {{.*}} used constexpr operator() 'int (int) __arm_streaming const' inline
// CHECK:          |     | |-CXXConversionDecl {{.*}} implicit constexpr operator int (*)(int) __arm_streaming 'int (*() const noexcept)(int) __arm_streaming' inline
// CHECK:          |     | |-CXXMethodDecl {{.*}} implicit __invoke 'int (int) __arm_streaming' static inline
// CHECK:          `-ReturnStmt
// CHECK:            `-CXXOperatorCallExpr
// CHECK-NEXT:         |-ImplicitCastExpr {{.*}} 'int (*)(int) __arm_streaming const' <FunctionToPointerDecay>
// CHECK-NEXT:         | `-DeclRefExpr {{.*}} 'int (int) __arm_streaming const' lvalue CXXMethod {{.*}} 'operator()' 'int (int) __arm_streaming const'
  int test_lambda(int x) {
    auto F = [](int x) __arm_streaming { return x; };
    return F(x);
  }

// CHECK: |-TypedefDecl {{.*}} referenced s_ptrty 'void (*)(int, int) __arm_streaming'
// CHECK-NEXT: | `-PointerType {{.*}} 'void (*)(int, int) __arm_streaming'
// CHECK-NEXT: |   `-ParenType {{.*}} 'void (int, int) __arm_streaming' sugar
// CHECK-NEXT: |     `-FunctionProtoType {{.*}} 'void (int, int) __arm_streaming' cdecl
  typedef void (*s_ptrty) (int, int) __arm_streaming;

// CHECK:      `-CXXMethodDecl {{.*}} test_streaming_ptrty 'void (s_ptrty, int, int)' implicit-inline
// CHECK-NEXT:   |-ParmVarDecl {{.*}} used f 's_ptrty':'void (*)(int, int) __arm_streaming'
// CHECK-NEXT:   |-ParmVarDecl {{.*}} used x 'int'
// CHECK-NEXT:   |-ParmVarDecl {{.*}} used y 'int'
// CHECK:        `-CompoundStmt
// CHECK-NEXT:     `-ReturnStmt
// CHECK-NEXT:       `-CallExpr
// CHECK-NEXT:         |-ImplicitCastExpr {{.*}} 's_ptrty':'void (*)(int, int) __arm_streaming' <LValueToRValue>
// CHECK-NEXT:         | `-DeclRefExpr {{.*}} 's_ptrty':'void (*)(int, int) __arm_streaming' lvalue ParmVar {{.*}} 'f' 's_ptrty':'void (*)(int, int) __arm_streaming'
// CHECK-NEXT:         |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT:         | `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT:         `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT:           `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'y' 'int'
  void test_streaming_ptrty(s_ptrty f, int x, int y) { return f(x, y); };
};
