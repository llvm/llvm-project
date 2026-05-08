// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump -disable-llvm-passes -finclude-default-header -o - %s | FileCheck %s

// CHECK: ClassTemplateDecl {{.*}} ConstantBuffer
// CHECK: TemplateTypeParmDecl {{.*}} element_type
// CHECK: CXXRecordDecl {{.*}} ConstantBuffer definition
// CHECK: FinalAttr {{.*}} Implicit final
// CHECK-NEXT: FieldDecl {{.*}} implicit __handle '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(CBuffer)]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]

// CHECK: CXXConstructorDecl {{.*}} ConstantBuffer<element_type> 'void ()' inline
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: BinaryOperator {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(CBuffer)]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// CHECK-SAME: ' '='
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(CBuffer)]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::ConstantBuffer<element_type>' lvalue implicit this
// CHECK-NEXT: CStyleCastExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(CBuffer)]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// CHECK-SAME: ' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'

// CHECK: CXXConstructorDecl {{.*}} ConstantBuffer<element_type> 'void (const hlsl::ConstantBuffer<element_type> &)' inline
// CHECK-NEXT: ParmVarDecl {{.*}} other 'const hlsl::ConstantBuffer<element_type> &'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: BinaryOperator {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(CBuffer)]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// CHECK-SAME: ' '='
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(CBuffer)]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::ConstantBuffer<element_type>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(CBuffer)]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'const hlsl::ConstantBuffer<element_type>' lvalue ParmVar {{.*}} 'other' 'const hlsl::ConstantBuffer<element_type> &'

// CHECK: CXXMethodDecl {{.*}} operator= 'hlsl::ConstantBuffer<element_type> &(const hlsl::ConstantBuffer<element_type> &)' inline
// CHECK-NEXT: ParmVarDecl {{.*}} other 'const hlsl::ConstantBuffer<element_type> &'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: BinaryOperator {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(CBuffer)]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// CHECK-SAME: ' '='
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(CBuffer)]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::ConstantBuffer<element_type>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(CBuffer)]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'const hlsl::ConstantBuffer<element_type>' lvalue ParmVar {{.*}} 'other' 'const hlsl::ConstantBuffer<element_type> &'
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::ConstantBuffer<element_type>' lvalue implicit this

struct S {
  float a;
};
ConstantBuffer<S> cb;

struct Nested {
  S s;
  float b;
};
ConstantBuffer<Nested> cb_nested;

void takes_s(S s) {}
void takes_cb(ConstantBuffer<S> c) {}
void takes_inout_cb(inout ConstantBuffer<S> c) {}

float main() {
  // CHECK: FunctionDecl {{.*}} main
  // CHECK: MemberExpr {{.*}} 'const hlsl_constant float' lvalue .a
  // CHECK-NEXT: CXXMemberCallExpr {{.*}} 'const hlsl_constant S' lvalue
  // CHECK-NEXT: MemberExpr {{.*}} '<bound member function type>' .operator const hlsl_constant S &
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'const hlsl::ConstantBuffer<S>' lvalue <NoOp>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'ConstantBuffer<S>':'hlsl::ConstantBuffer<S>' lvalue Var {{.*}} 'cb' 'ConstantBuffer<S>':'hlsl::ConstantBuffer<S>'
  float f1 = cb.a;

  // CHECK: MemberExpr {{.*}} 'const hlsl_constant float' lvalue .b
  // CHECK-NEXT: CXXMemberCallExpr {{.*}} 'const hlsl_constant Nested' lvalue
  // CHECK-NEXT: MemberExpr {{.*}} '<bound member function type>' .operator const hlsl_constant Nested &
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'const hlsl::ConstantBuffer<Nested>' lvalue <NoOp>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'ConstantBuffer<Nested>':'hlsl::ConstantBuffer<Nested>' lvalue Var {{.*}} 'cb_nested' 'ConstantBuffer<Nested>':'hlsl::ConstantBuffer<Nested>'
  float f2 = cb_nested.b;

  // CHECK: MemberExpr {{.*}} 'const hlsl_constant float' lvalue .a
  // CHECK-NEXT: MemberExpr {{.*}} 'const hlsl_constant S' lvalue .s
  // CHECK-NEXT: CXXMemberCallExpr {{.*}} 'const hlsl_constant Nested' lvalue
  // CHECK-NEXT: MemberExpr {{.*}} '<bound member function type>' .operator const hlsl_constant Nested &
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'const hlsl::ConstantBuffer<Nested>' lvalue <NoOp>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'ConstantBuffer<Nested>':'hlsl::ConstantBuffer<Nested>' lvalue Var {{.*}} 'cb_nested' 'ConstantBuffer<Nested>':'hlsl::ConstantBuffer<Nested>'
  float f3 = cb_nested.s.a;

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(ConstantBuffer<S>)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (ConstantBuffer<S>)' lvalue Function {{.*}} 'takes_cb' 'void (ConstantBuffer<S>)'
  // CHECK-NEXT: CXXConstructExpr {{.*}} 'ConstantBuffer<S>':'hlsl::ConstantBuffer<S>' 'void (const hlsl::ConstantBuffer<S> &)'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'const hlsl::ConstantBuffer<S>' lvalue <NoOp>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'ConstantBuffer<S>':'hlsl::ConstantBuffer<S>' lvalue Var {{.*}} 'cb' 'ConstantBuffer<S>':'hlsl::ConstantBuffer<S>'
  takes_cb(cb);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(inout ConstantBuffer<S>)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (inout ConstantBuffer<S>)' lvalue Function {{.*}} 'takes_inout_cb' 'void (inout ConstantBuffer<S>)'
  // CHECK-NEXT: HLSLOutArgExpr {{.*}} 'ConstantBuffer<S>':'hlsl::ConstantBuffer<S>' lvalue inout
  takes_inout_cb(cb);

  return f1 + f2 + f3;
}
