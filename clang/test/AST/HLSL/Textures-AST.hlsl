// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -disable-llvm-passes -finclude-default-header -DINDEX_ARG_TYPE=uint3 \
// RUN:   -DINDEX_ARG="uint3(0, 0, 0)" -DTEXTURE=Texture2D -o - %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,SRV -DTEXTURE=Texture2D \
// RUN:   -DINDEX_DIM=2 -DDIM_NAME=2D
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -disable-llvm-passes -finclude-default-header -DINDEX_ARG_TYPE=uint3 \
// RUN:   -DINDEX_ARG="uint3(0, 0, 0)" -DTEXTURE=Texture2DArray -o - %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,SRV,SRV-ARRAY \
// RUN:   -DTEXTURE=Texture2DArray -DINDEX_DIM=3 -DDIM_NAME=2D
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -disable-llvm-passes -finclude-default-header -DINDEX_ARG_TYPE=uint3 \
// RUN:   -DINDEX_ARG="uint3(0, 0, 0)" -DTEXTURE=RWTexture2D -DRW=1 -o - %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,UAV,UAV-STORE,UAV-TRUNC \
// RUN:   -DTEXTURE=RWTexture2D -DINDEX_DIM=2 -DDIM_NAME=2D
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -disable-llvm-passes -finclude-default-header -DINDEX_ARG_TYPE=uint3 \
// RUN:   -DINDEX_ARG="uint3(0, 0, 0)" -DTEXTURE=RWTexture2DArray -DRW=1 -o - %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,UAV,UAV-ARRAY,UAV-STORE,UAV-NOTRUNC \
// RUN:   -DTEXTURE=RWTexture2DArray -DINDEX_DIM=3 -DDIM_NAME=2D

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   INDEX_ARG_TYPE     the declared type of INDEX_ARG
//   INDEX_ARG          a literal operator[] index
//   TEXTURE            resource type name
//   INDEX_DIM          operator[] index components
//   DIM_NAME           hlsl::dimension spelling
//   RW                 dx.Texture UAV operand
//
// Check prefixes:
//   SRV                read-only (SRV) textures
//   SRV-ARRAY          read-only array textures
//   UAV                writable (UAV) textures
//   UAV-STORE          the store through operator[]
//   UAV-TRUNC          types whose index is narrower than INDEX_ARG_TYPE
//   UAV-ARRAY          writable array textures
//   UAV-NOTRUNC        types whose index matches INDEX_ARG_TYPE

// CHECK: ClassTemplateDecl {{.*}} [[TEXTURE]]
// CHECK: TemplateTypeParmDecl {{.*}} element_type
// CHECK: CXXRecordDecl {{.*}} [[TEXTURE]] definition
// CHECK: FinalAttr {{.*}} Implicit final
// CHECK-NEXT: FieldDecl {{.*}} implicit __handle '__hlsl_resource_t
// SRV-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// UAV-SAME{LITERAL}: [[hlsl::resource_class("UAV")]]
// SRV-ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// UAV-ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}

// SRV: CXXMethodDecl {{.*}} operator[] 'const hlsl_device element_type &(vector<unsigned int, [[INDEX_DIM]]>) const' inline
// SRV-NEXT: ParmVarDecl {{.*}} Index 'vector<unsigned int, [[INDEX_DIM]]>'
// SRV-NEXT: CompoundStmt
// SRV-NEXT: ReturnStmt
// SRV-NEXT: UnaryOperator {{.*}} 'hlsl_device element_type' lvalue prefix '*' cannot overflow
// SRV-NEXT: CStyleCastExpr {{.*}} 'hlsl_device element_type *' <Dependent>
// SRV-NEXT: CallExpr {{.*}} '<dependent type>'
// SRV-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_getpointer' 'void (...) noexcept'
// SRV-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// SRV-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// SRV-ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// SRV-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// SRV-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// SRV-SAME: ' lvalue .__handle
// SRV-NEXT: CXXThisExpr {{.*}} 'const hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// SRV-NEXT: DeclRefExpr {{.*}} 'vector<unsigned int, [[INDEX_DIM]]>' lvalue ParmVar {{.*}} 'Index' 'vector<unsigned int, [[INDEX_DIM]]>'
// SRV-NEXT: AlwaysInlineAttr

// UAV: CXXMethodDecl {{.*}} operator[] 'hlsl_device element_type &(vector<unsigned int, [[INDEX_DIM]]>) const' inline
// UAV-NEXT: ParmVarDecl {{.*}} Index 'vector<unsigned int, [[INDEX_DIM]]>'
// UAV-NEXT: CompoundStmt
// UAV-NEXT: ReturnStmt
// UAV-NEXT: UnaryOperator {{.*}} 'hlsl_device element_type' lvalue prefix '*' cannot overflow
// UAV-NEXT: CStyleCastExpr {{.*}} 'hlsl_device element_type *' <Dependent>
// UAV-NEXT: CallExpr {{.*}} '<dependent type>'
// UAV-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_getpointer' 'void (...) noexcept'
// UAV-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// UAV-SAME{LITERAL}: [[hlsl::resource_class("UAV")]]
// UAV-ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// UAV-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// UAV-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// UAV-SAME: ' lvalue .__handle
// UAV-NEXT: CXXThisExpr {{.*}} 'const hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// UAV-NEXT: DeclRefExpr {{.*}} 'vector<unsigned int, [[INDEX_DIM]]>' lvalue ParmVar {{.*}} 'Index' 'vector<unsigned int, [[INDEX_DIM]]>'
// UAV-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} GetDimensions 'void (out unsigned int, out unsigned int)'
// CHECK-NEXT: ParmVarDecl {{.*}} width 'unsigned int &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} out
// CHECK-NEXT: ParmVarDecl {{.*}} height 'unsigned int &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} out
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_getdimensions_xy' 'void (__hlsl_resource_t, unsigned int &, unsigned int &) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// SRV-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// UAV-SAME{LITERAL}: [[hlsl::resource_class("UAV")]]
// SRV-ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// UAV-ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'width' 'unsigned int &__restrict'
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'height' 'unsigned int &__restrict'
// CHECK-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} GetDimensions 'void (unsigned int, out unsigned int, out unsigned int, out unsigned int)'
// CHECK-NEXT: ParmVarDecl {{.*}} mipLevel 'unsigned int'
// CHECK-NEXT: ParmVarDecl {{.*}} width 'unsigned int &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} out
// CHECK-NEXT: ParmVarDecl {{.*}} height 'unsigned int &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} out
// CHECK-NEXT: ParmVarDecl {{.*}} numberOfLevels 'unsigned int &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} out
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_getdimensions_levels_xy' 'void (__hlsl_resource_t, unsigned int, unsigned int &, unsigned int &, unsigned int &) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// SRV-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// UAV-SAME{LITERAL}: [[hlsl::resource_class("UAV")]]
// SRV-ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// UAV-ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'mipLevel' 'unsigned int'
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'width' 'unsigned int &__restrict'
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'height' 'unsigned int &__restrict'
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'numberOfLevels' 'unsigned int &__restrict'
// CHECK-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} GetDimensions 'void (out float, out float)'
// CHECK-NEXT: ParmVarDecl {{.*}} width 'float &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} out
// CHECK-NEXT: ParmVarDecl {{.*}} height 'float &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} out
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_getdimensions_xy_float' 'void (__hlsl_resource_t, float &, float &) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// SRV-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// UAV-SAME{LITERAL}: [[hlsl::resource_class("UAV")]]
// SRV-ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// UAV-ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'width' 'float &__restrict'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'height' 'float &__restrict'
// CHECK-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} GetDimensions 'void (unsigned int, out float, out float, out float)'
// CHECK-NEXT: ParmVarDecl {{.*}} mipLevel 'unsigned int'
// CHECK-NEXT: ParmVarDecl {{.*}} width 'float &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} out
// CHECK-NEXT: ParmVarDecl {{.*}} height 'float &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} out
// CHECK-NEXT: ParmVarDecl {{.*}} numberOfLevels 'float &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} out
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_getdimensions_levels_xy_float' 'void (__hlsl_resource_t, unsigned int, float &, float &, float &) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// SRV-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// UAV-SAME{LITERAL}: [[hlsl::resource_class("UAV")]]
// SRV-ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// UAV-ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'mipLevel' 'unsigned int'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'width' 'float &__restrict'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'height' 'float &__restrict'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'numberOfLevels' 'float &__restrict'
// CHECK-NEXT: AlwaysInlineAttr

// CHECK: ClassTemplatePartialSpecializationDecl {{.*}} class [[TEXTURE]] explicit_specialization
// CHECK: TemplateTypeParmDecl {{.*}} element_type
// CHECK: NonTypeTemplateParmDecl {{.*}} element_count

// SRV-NOT: BinaryOperator {{.*}} 'hlsl_device float' lvalue '='

// UAV-STORE-LABEL: FunctionDecl {{.*}} main 'void ()'
// UAV-STORE: BinaryOperator {{.*}} 'hlsl_device float' lvalue '='
// UAV-STORE-NEXT: CXXOperatorCallExpr {{.*}} 'hlsl_device float' lvalue '[]'
// UAV-STORE-NEXT: ImplicitCastExpr {{.*}} 'hlsl_device float &(*)(vector<unsigned int, [[INDEX_DIM]]>) const' <FunctionToPointerDecay>
// UAV-STORE-NEXT: DeclRefExpr {{.*}} 'hlsl_device float &(vector<unsigned int, [[INDEX_DIM]]>) const' lvalue CXXMethod {{.*}} 'operator[]' 'hlsl_device float &(vector<unsigned int, [[INDEX_DIM]]>) const'
// UAV-STORE-NEXT: ImplicitCastExpr {{.*}} 'const hlsl::[[TEXTURE]]<float>' lvalue <NoOp>
// UAV-STORE-NEXT: DeclRefExpr {{.*}} '[[TEXTURE]]<float>':'hlsl::[[TEXTURE]]<float>' lvalue Var {{.*}} 't' '[[TEXTURE]]<float>':'hlsl::[[TEXTURE]]<float>'
// UAV-TRUNC-NEXT: ImplicitCastExpr {{.*}} 'vector<uint, 2>' <HLSLVectorTruncation>
// UAV-TRUNC-NEXT: ImplicitCastExpr {{.*}} 'uint3':'vector<uint, 3>' <LValueToRValue>
// UAV-TRUNC-NEXT: DeclRefExpr {{.*}} 'uint3':'vector<uint, 3>' lvalue Var {{.*}} 'i' 'uint3':'vector<uint, 3>'
// UAV-NOTRUNC-NEXT: ImplicitCastExpr {{.*}} 'uint3':'vector<uint, 3>' <LValueToRValue>
// UAV-NOTRUNC-NEXT: DeclRefExpr {{.*}} 'uint3':'vector<uint, 3>' lvalue Var {{.*}} 'i' 'uint3':'vector<uint, 3>'
// UAV-STORE-NEXT: FloatingLiteral {{.*}} 'float' {{.*}}

TEXTURE<float> t;

void main() {
  INDEX_ARG_TYPE i = INDEX_ARG;
#if RW
  t[i] = 1.0f;
#endif
  float x = t[i];
  (void)x;
  uint w, h;
  t.GetDimensions(w, h);
}
