// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump -disable-llvm-passes -finclude-default-header -o - %s | FileCheck %s

// CHECK: CXXRecordDecl {{.*}} SamplerState definition
// CHECK: FinalAttr {{.*}} Implicit final
// CHECK-NEXT: FieldDecl {{.*}} implicit {{.*}} __handle '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(Sampler)]]

// CHECK: ClassTemplateDecl {{.*}} Texture2D
// CHECK: TemplateTypeParmDecl {{.*}} element_type
// CHECK: CXXRecordDecl {{.*}} Texture2D definition
// CHECK: FinalAttr {{.*}} Implicit final
// CHECK-NEXT: FieldDecl {{.*}} implicit __handle '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(SRV)]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// CHECK-SAME{LITERAL}: [[hlsl::resource_dimension(2D)]]

// CHECK: CXXMethodDecl {{.*}} Sample 'element_type (hlsl::SamplerState, vector<float, 2>)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, 2>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'element_type' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(SRV)]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// CHECK-SAME{LITERAL}: [[hlsl::resource_dimension(2D)]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::Texture2D<element_type>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(Sampler)]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, 2>' lvalue ParmVar {{.*}} 'Location' 'vector<float, 2>'
// CHECK-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} Sample 'element_type (hlsl::SamplerState, vector<float, 2>, vector<int, 2>)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, 2>'
// CHECK-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, 2>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'element_type' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(SRV)]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// CHECK-SAME{LITERAL}: [[hlsl::resource_dimension(2D)]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::Texture2D<element_type>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(Sampler)]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, 2>' lvalue ParmVar {{.*}} 'Location' 'vector<float, 2>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<int, 2>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, 2>'
// CHECK-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} Sample 'element_type (hlsl::SamplerState, vector<float, 2>, vector<int, 2>, float)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, 2>'
// CHECK-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, 2>'
// CHECK-NEXT: ParmVarDecl {{.*}} Clamp 'float'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'element_type' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(SRV)]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// CHECK-SAME{LITERAL}: [[hlsl::resource_dimension(2D)]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::Texture2D<element_type>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(Sampler)]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, 2>' lvalue ParmVar {{.*}} 'Location' 'vector<float, 2>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<int, 2>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, 2>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'Clamp' 'float'
// CHECK-NEXT: AlwaysInlineAttr

Texture2D<float4> t;
SamplerState s;

void main(float2 loc) {
  t.Sample(s, loc);
  t.Sample(s, loc, int2(1, 2));
  t.Sample(s, loc, int2(1, 2), 1.0);
}