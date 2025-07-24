// Test without serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ast-dump %s \
// RUN: | FileCheck --strict-whitespace %s
//
// Test with serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -triple x86_64-unknown-unknown -include-pch %t -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck --strict-whitespace %s

void testArrayInitExpr()
{
    int a[10];
    auto l = [a]{
    };
}

// CHECK: |-FunctionDecl {{.*}} testArrayInitExpr 'void ()'
// CHECK-NEXT: | `-CompoundStmt {{.*}} 
// CHECK-NEXT: |   |-DeclStmt {{.*}} 
// CHECK-NEXT: |   | `-VarDecl {{.*}} used a 'int[10]'
// CHECK-NEXT: |   |   `-typeDetails: ConstantArrayType {{.*}} 'int[10]' 10
// CHECK-NEXT: |   |     `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |   `-DeclStmt {{.*}} 
// CHECK-NEXT: |     `-VarDecl {{.*}} cinit
// CHECK-NEXT: |       |-LambdaExpr {{.*}} 
// CHECK-NEXT: |       | |-CXXRecordDecl {{.*}} implicit class definition
// CHECK-NEXT: |       | | |-DefinitionData lambda pass_in_registers standard_layout trivially_copyable literal can_const_default_init
// CHECK-NEXT: |       | | | |-DefaultConstructor
// CHECK-NEXT: |       | | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: |       | | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT: |       | | | |-CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: |       | | | |-MoveAssignment
// CHECK-NEXT: |       | | | `-Destructor simple irrelevant trivial
// CHECK-NEXT: |       | | |-CXXMethodDecl {{.*}} constexpr operator() 'auto () const -> void' inline
// CHECK-NEXT: |       | | | `-CompoundStmt {{.*}} 
// CHECK-NEXT: |       | | |-FieldDecl {{.*}} implicit 'int[10]'
// CHECK-NEXT: |       | | `-CXXDestructorDecl {{.*}} implicit referenced
// CHECK-NEXT: |       | |-ArrayInitLoopExpr {{.*}} 'int[10]'
// CHECK-NEXT: |       | | |-OpaqueValueExpr {{.*}} 'int[10]' lvalue
// CHECK-NEXT: |       | | | `-DeclRefExpr {{.*}} 'int[10]' lvalue Var {{.*}} 'a' 'int[10]'
// CHECK-NEXT: |       | | `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |       | |   `-ArraySubscriptExpr {{.*}} 'int' lvalue
// CHECK-NEXT: |       | |     |-ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: |       | |     | `-OpaqueValueExpr {{.*}} 'int[10]' lvalue
// CHECK-NEXT: |       | |     |   `-DeclRefExpr {{.*}} 'int[10]' lvalue Var {{.*}} 'a' 'int[10]'
// CHECK-NEXT: |       | |     `-ArrayInitIndexExpr {{.*}} 
// CHECK-NEXT: |       | `-CompoundStmt {{.*}} 
// CHECK-NEXT: |       `-typeDetails: AutoType {{.*}}
// CHECK-NEXT: |         `-typeDetails: RecordType {{.*}}
// CHECK-NEXT: |           `-CXXRecord {{.*}} 

template<typename T, int Size>
class array {
  T data[Size];

  using array_T_size = T[Size];
  using const_array_T_size = const T[Size];
};

// CHECK: |-ClassTemplateDecl {{.*}} array
// CHECK-NEXT: | |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 T
// CHECK-NEXT: | |-NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 1 Size
// CHECK-NEXT: | `-CXXRecordDecl {{.*}} class array definition
// CHECK-NEXT: |   |-DefinitionData standard_layout trivially_copyable trivial
// CHECK-NEXT: |   | |-DefaultConstructor exists trivial needs_implicit
// CHECK-NEXT: |   | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: |   | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT: |   | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: |   | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT: |   | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT: |   |-CXXRecordDecl {{.*}} implicit class array
// CHECK-NEXT: |   |-FieldDecl {{.*}} data 'T[Size]'
// CHECK-NEXT: |   |-TypeAliasDecl {{.*}} array_T_size 'T[Size]'
// CHECK-NEXT: |   | `-typeDetails: DependentSizedArrayType {{.*}} 'T[Size]' dependent
// CHECK-NEXT: |   |   |-typeDetails: TemplateTypeParmType {{.*}} 'T' dependent depth 0 index 0
// CHECK-NEXT: |   |   | `-TemplateTypeParm {{.*}} 'T'
// CHECK-NEXT: |   |   `-DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'Size' 'int'
// CHECK-NEXT: |   `-TypeAliasDecl {{.*}} const_array_T_size 'const T[Size]'
// CHECK-NEXT: |     `-typeDetails: DependentSizedArrayType {{.*}} 'const T[Size]' dependent
// CHECK-NEXT: |       |-qualTypeDetail: QualType {{.*}} 'const T' const
// CHECK-NEXT: |       | `-typeDetails: TemplateTypeParmType {{.*}} 'T' dependent depth 0 index 0
// CHECK-NEXT: |       |   `-TemplateTypeParm {{.*}} 'T'
// CHECK-NEXT: |       `-DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'Size' 'int'

struct V {};
template <typename U, typename Idx, int N>
void testDependentSubscript() {
  U* a;
  U b[5];
  Idx i{};
  enum E { One = 1 };

  // Can types of subscript expressions can be determined?
  // LHS is a type-dependent array, RHS is a known integer type.
  a[1];
  // CHECK: ArraySubscriptExpr {{.*}}line:[[@LINE-1]]{{.*}} 'U'
  b[1];
  // CHECK: ArraySubscriptExpr {{.*}}line:[[@LINE-1]]{{.*}} 'U'

  // Reverse case: RHS is a type-dependent array, LHS is an integer.
  1[a];
  // CHECK: ArraySubscriptExpr {{.*}}line:[[@LINE-1]]{{.*}} 'U'
  1[b];
  // CHECK: ArraySubscriptExpr {{.*}}line:[[@LINE-1]]{{.*}} 'U'

  // LHS is a type-dependent array, RHS is type-dependent.
  a[i];
  // CHECK: ArraySubscriptExpr {{.*}}line:[[@LINE-1]]{{.*}} '<dependent type>'
  b[i];
  // CHECK: ArraySubscriptExpr {{.*}}line:[[@LINE-1]]{{.*}} '<dependent type>'

  V *a2;
  V b2[5];

  // LHS is a known array, RHS is type-dependent.
  a2[i];
  // CHECK: ArraySubscriptExpr {{.*}}line:[[@LINE-1]]{{.*}} '<dependent type>'
  b2[i];
  // CHECK: ArraySubscriptExpr {{.*}}line:[[@LINE-1]]{{.*}} '<dependent type>'

  // LHS is a known array, RHS is a type-dependent index.
  // We know the element type is V, but insist on some dependent type.
  a2[One];
  // CHECK: ArraySubscriptExpr {{.*}}line:[[@LINE-1]]{{.*}} '<dependent type>'
  b2[One];
  // CHECK: ArraySubscriptExpr {{.*}}line:[[@LINE-1]]{{.*}} '<dependent type>'

  V b3[N];
  // LHS is an array with dependent bounds but known elements.
  // We insist on a dependent type.
  b3[0];
  // CHECK: ArraySubscriptExpr {{.*}}line:[[@LINE-1]]{{.*}} '<dependent type>'

  U b4[N];
  // LHS is an array with dependent bounds and dependent elements.
  b4[0];
  // CHECK: ArraySubscriptExpr {{.*}}line:[[@LINE-1]]{{.*}} 'U'
}
