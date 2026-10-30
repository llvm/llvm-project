// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump -disable-llvm-passes -finclude-default-header -DTEXTURE=Texture2DMS -o - %s | FileCheck %s -DTEXTURE=Texture2DMS -DINDEX_SIZE=2

// The class template: an `element_type` type parameter followed by an `int`
// `sample_count` non-type parameter that defaults to 0.
// CHECK: ClassTemplateDecl {{.*}} [[TEXTURE]]
// CHECK: TemplateTypeParmDecl {{.*}} element_type
// CHECK: NonTypeTemplateParmDecl {{.*}} 'int' depth 0 index 1 sample_count
// CHECK: TemplateArgument expr '0'

// The resource handle: an SRV whose handle additionally carries [[hlsl::is_ms]].
// CHECK: CXXRecordDecl {{.*}} [[TEXTURE]] definition
// CHECK: FinalAttr {{.*}} Implicit final
// CHECK-NEXT: FieldDecl {{.*}} implicit __handle '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// CHECK-SAME{LITERAL}: [[hlsl::is_ms]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// CHECK-SAME{LITERAL}: [[hlsl::dimension("2D")]]

// Load(location, sampleIndex): the sample index is a separate scalar (there is
// no packed mip level), and the read lowers to __builtin_hlsl_resource_load_ms.
// CHECK: CXXMethodDecl {{.*}} Load 'element_type (vector<int, [[INDEX_SIZE]]>, int)' inline
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<int, [[INDEX_SIZE]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} SampleIndex 'int'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'element_type' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_load_ms' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// CHECK-SAME{LITERAL}: [[hlsl::is_ms]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// CHECK-SAME{LITERAL}: [[hlsl::dimension("2D")]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type, sample_count>' lvalue implicit this
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<int, [[INDEX_SIZE]]>' lvalue ParmVar {{.*}} 'Location' 'vector<int, [[INDEX_SIZE]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'SampleIndex' 'int'
// CHECK-NEXT: AlwaysInlineAttr

// Load(location, sampleIndex, offset): identical, with a trailing 2D offset.
// CHECK: CXXMethodDecl {{.*}} Load 'element_type (vector<int, [[INDEX_SIZE]]>, int, vector<int, 2>)' inline
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<int, [[INDEX_SIZE]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} SampleIndex 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, 2>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'element_type' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_load_ms' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type, sample_count>' lvalue implicit this
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<int, [[INDEX_SIZE]]>' lvalue ParmVar {{.*}} 'Location' 'vector<int, [[INDEX_SIZE]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'SampleIndex' 'int'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<int, 2>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, 2>'
// CHECK-NEXT: AlwaysInlineAttr

// operator[] returns a const reference to sample 0 (same shape as the non-MS
// SRV subscript in Textures-AST.hlsl), via __builtin_hlsl_resource_getpointer.
// CHECK: CXXMethodDecl {{.*}} operator[] 'const hlsl_device element_type &(vector<unsigned int, [[INDEX_SIZE]]>) const' inline
// CHECK-NEXT: ParmVarDecl {{.*}} Index 'vector<unsigned int, [[INDEX_SIZE]]>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: UnaryOperator {{.*}} 'hlsl_device element_type' lvalue prefix '*' cannot overflow
// CHECK-NEXT: CStyleCastExpr {{.*}} 'hlsl_device element_type *' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_getpointer' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'const hlsl::[[TEXTURE]]<element_type, sample_count>' lvalue implicit this
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<unsigned int, [[INDEX_SIZE]]>' lvalue ParmVar {{.*}} 'Index' 'vector<unsigned int, [[INDEX_SIZE]]>'
// CHECK-NEXT: AlwaysInlineAttr

// TODO(MS GetDimensions): when the multisampled GetDimensions overload is
// implemented, dump its CHECK block here (out width/height[/elements]/
// NumberOfSamples, on a samples-based getdimensions builtin), paralleling the
// GetDimensions blocks in Textures-AST.hlsl.

TEXTURE<float> t;

// An explicit, non-default sample count binds the non-type template parameter,
// producing a distinct specialization.
// CHECK: ClassTemplateSpecializationDecl {{.*}} class [[TEXTURE]] definition
// CHECK: TemplateArgument type 'vector<float, 4>'
// CHECK: TemplateArgument integral '4'
TEXTURE<float4, 4> tMS4;

[numthreads(1, 1, 1)]
void main() {
  uint2 i = uint2(0, 0);
  float x = t[i];
  (void)x;
  // TODO: enable once multisampled GetDimensions is implemented, paralleling
  // the t.GetDimensions(w, h) call in Textures-AST.hlsl:
  // uint w, h, n;
  // t.GetDimensions(w, h, n);
}
