// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -disable-llvm-passes -finclude-default-header -DHAS_GETDIM \
// RUN:   -DINDEX_ARG_TYPE=uint3 -DINDEX_ARG="uint3(0, 0, 0)" \
// RUN:   -DTEXTURE=Texture2D -o - %s | FileCheck %s \
// RUN:   --check-prefixes=CHECK,SRV,GETDIM,GETDIM-SRV -DTEXTURE=Texture2D \
// RUN:   -DINDEX_DIM=2 -DDIM_NAME=2D -DINDEX_TYPE="vector<unsigned int, 2>" \
// RUN:   -DLOCATION_TYPE="vector<int, 2>"
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -disable-llvm-passes -finclude-default-header -DINDEX_ARG_TYPE=uint3 \
// RUN:   -DINDEX_ARG="uint3(0, 0, 0)" -DTEXTURE=Texture1D -o - %s | FileCheck \
// RUN:   %s --check-prefixes=CHECK,SRV -DTEXTURE=Texture1D -DINDEX_DIM=1 \
// RUN:   -DDIM_NAME=1D -DINDEX_TYPE="unsigned int" -DLOCATION_TYPE=int
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -disable-llvm-passes -finclude-default-header -DHAS_GETDIM \
// RUN:   -DINDEX_ARG_TYPE=uint3 -DINDEX_ARG="uint3(0, 0, 0)" \
// RUN:   -DTEXTURE=Texture2DArray -o - %s | FileCheck %s \
// RUN:   --check-prefixes=CHECK,SRV,SRV-ARRAY,GETDIM,GETDIM-SRV,GETDIM-SRV-ARRAY \
// RUN:   -DTEXTURE=Texture2DArray -DINDEX_DIM=3 -DDIM_NAME=2D \
// RUN:   -DINDEX_TYPE="vector<unsigned int, 3>" \
// RUN:   -DLOCATION_TYPE="vector<int, 3>"
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -disable-llvm-passes -finclude-default-header -DINDEX_ARG_TYPE=uint3 \
// RUN:   -DINDEX_ARG="uint3(0, 0, 0)" -DTEXTURE=Texture1DArray -o - %s | \
// RUN:   FileCheck %s --check-prefixes=CHECK,SRV,SRV-ARRAY \
// RUN:   -DTEXTURE=Texture1DArray -DINDEX_DIM=2 -DDIM_NAME=1D \
// RUN:   -DINDEX_TYPE="vector<unsigned int, 2>" \
// RUN:   -DLOCATION_TYPE="vector<int, 2>"
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -disable-llvm-passes -finclude-default-header -DHAS_GETDIM \
// RUN:   -DINDEX_ARG_TYPE=uint3 -DINDEX_ARG="uint3(0, 0, 0)" \
// RUN:   -DTEXTURE=RWTexture2D -DRW=1 -o - %s | FileCheck %s \
// RUN:   --check-prefixes=CHECK,UAV,UAV-STORE,UAV-TRUNC,GETDIM,GETDIM-UAV \
// RUN:   -DTEXTURE=RWTexture2D -DINDEX_DIM=2 -DDIM_NAME=2D \
// RUN:   -DINDEX_TYPE="vector<unsigned int, 2>" \
// RUN:   -DLOCATION_TYPE="vector<int, 2>" -DTRUNC_TYPE="vector<uint, 2>"
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -disable-llvm-passes -finclude-default-header -DINDEX_ARG_TYPE=uint3 \
// RUN:   -DINDEX_ARG="uint3(0, 0, 0)" -DTEXTURE=RWTexture1D -DRW=1 -o - %s | \
// RUN:   FileCheck %s --check-prefixes=CHECK,UAV,UAV-STORE,UAV-TRUNC \
// RUN:   -DTEXTURE=RWTexture1D -DINDEX_DIM=1 -DDIM_NAME=1D \
// RUN:   -DINDEX_TYPE="unsigned int" -DLOCATION_TYPE=int \
// RUN:   -DTRUNC_TYPE="uint':'unsigned int"
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -disable-llvm-passes -finclude-default-header -DHAS_GETDIM \
// RUN:   -DINDEX_ARG_TYPE=uint3 -DINDEX_ARG="uint3(0, 0, 0)" \
// RUN:   -DTEXTURE=RWTexture2DArray -DRW=1 -o - %s | FileCheck %s \
// RUN:   --check-prefixes=CHECK,UAV,UAV-ARRAY,UAV-STORE,UAV-NOTRUNC,GETDIM,GETDIM-UAV,GETDIM-UAV-ARRAY \
// RUN:   -DTEXTURE=RWTexture2DArray -DINDEX_DIM=3 -DDIM_NAME=2D \
// RUN:   -DINDEX_TYPE="vector<unsigned int, 3>" \
// RUN:   -DLOCATION_TYPE="vector<int, 3>"
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -disable-llvm-passes -finclude-default-header -DINDEX_ARG_TYPE=uint3 \
// RUN:   -DINDEX_ARG="uint3(0, 0, 0)" -DTEXTURE=RWTexture1DArray -DRW=1 -o - \
// RUN:   %s | FileCheck %s \
// RUN:   --check-prefixes=CHECK,UAV,UAV-ARRAY,UAV-STORE,UAV-TRUNC \
// RUN:   -DTEXTURE=RWTexture1DArray -DINDEX_DIM=2 -DDIM_NAME=1D \
// RUN:   -DINDEX_TYPE="vector<unsigned int, 2>" \
// RUN:   -DLOCATION_TYPE="vector<int, 2>" -DTRUNC_TYPE="vector<uint, 2>"

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   INDEX_ARG_TYPE     the declared type of INDEX_ARG
//   INDEX_ARG          a literal operator[] index
//   TEXTURE            resource type name
//   INDEX_DIM          operator[] index components
//   INDEX_TYPE         operator[] index type
//   LOCATION_TYPE      Load location type
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
//   GETDIM             types that have the GetDimensions overloads
//   GETDIM-SRV         read-only textures that have GetDimensions
//   GETDIM-UAV         writable textures that have GetDimensions
//   GETDIM-SRV-ARRAY   read-only array textures that have GetDimensions
//   GETDIM-UAV-ARRAY   writable array textures that have GetDimensions

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

// UAV: CXXMethodDecl {{.*}} Load 'element_type ([[LOCATION_TYPE]])' inline
// UAV-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
// UAV-NEXT: CompoundStmt
// UAV-NEXT: ReturnStmt
// UAV-NEXT: CStyleCastExpr {{.*}} 'element_type' <Dependent>
// UAV-NEXT: CallExpr {{.*}} '<dependent type>'
// UAV-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_load_level' 'void (...) noexcept'
// UAV-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// UAV-SAME{LITERAL}: [[hlsl::resource_class("UAV")]]
// UAV-ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// UAV-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// UAV-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// UAV-SAME: ' lvalue .__handle
// UAV-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// UAV-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// UAV-NEXT: AlwaysInlineAttr

// SRV: CXXMethodDecl {{.*}} operator[] 'const hlsl_device element_type &([[INDEX_TYPE]]) const' inline
// SRV-NEXT: ParmVarDecl {{.*}} Index '[[INDEX_TYPE]]'
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
// SRV-NEXT: DeclRefExpr {{.*}} '[[INDEX_TYPE]]' lvalue ParmVar {{.*}} 'Index' '[[INDEX_TYPE]]'
// SRV-NEXT: AlwaysInlineAttr

// UAV: CXXMethodDecl {{.*}} operator[] 'hlsl_device element_type &([[INDEX_TYPE]]) const' inline
// UAV-NEXT: ParmVarDecl {{.*}} Index '[[INDEX_TYPE]]'
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
// UAV-NEXT: DeclRefExpr {{.*}} '[[INDEX_TYPE]]' lvalue ParmVar {{.*}} 'Index' '[[INDEX_TYPE]]'
// UAV-NEXT: AlwaysInlineAttr

// GETDIM: CXXMethodDecl {{.*}} GetDimensions 'void (out unsigned int, out unsigned int)'
// GETDIM-NEXT: ParmVarDecl {{.*}} width 'unsigned int &__restrict'
// GETDIM-NEXT: HLSLParamModifierAttr {{.*}} out
// GETDIM-NEXT: ParmVarDecl {{.*}} height 'unsigned int &__restrict'
// GETDIM-NEXT: HLSLParamModifierAttr {{.*}} out
// GETDIM-NEXT: CompoundStmt
// GETDIM-NEXT: CallExpr {{.*}} '<dependent type>'
// GETDIM-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_getdimensions_xy' 'void (__hlsl_resource_t, unsigned int &, unsigned int &) noexcept'
// GETDIM-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// GETDIM-SRV-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// GETDIM-UAV-SAME{LITERAL}: [[hlsl::resource_class("UAV")]]
// GETDIM-SRV-ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// GETDIM-UAV-ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// GETDIM-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// GETDIM-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// GETDIM-SAME: ' lvalue .__handle
// GETDIM-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// GETDIM-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'width' 'unsigned int &__restrict'
// GETDIM-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'height' 'unsigned int &__restrict'
// GETDIM-NEXT: AlwaysInlineAttr

// GETDIM: CXXMethodDecl {{.*}} GetDimensions 'void (unsigned int, out unsigned int, out unsigned int, out unsigned int)'
// GETDIM-NEXT: ParmVarDecl {{.*}} mipLevel 'unsigned int'
// GETDIM-NEXT: ParmVarDecl {{.*}} width 'unsigned int &__restrict'
// GETDIM-NEXT: HLSLParamModifierAttr {{.*}} out
// GETDIM-NEXT: ParmVarDecl {{.*}} height 'unsigned int &__restrict'
// GETDIM-NEXT: HLSLParamModifierAttr {{.*}} out
// GETDIM-NEXT: ParmVarDecl {{.*}} numberOfLevels 'unsigned int &__restrict'
// GETDIM-NEXT: HLSLParamModifierAttr {{.*}} out
// GETDIM-NEXT: CompoundStmt
// GETDIM-NEXT: CallExpr {{.*}} '<dependent type>'
// GETDIM-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_getdimensions_levels_xy' 'void (__hlsl_resource_t, unsigned int, unsigned int &, unsigned int &, unsigned int &) noexcept'
// GETDIM-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// GETDIM-SRV-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// GETDIM-UAV-SAME{LITERAL}: [[hlsl::resource_class("UAV")]]
// GETDIM-SRV-ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// GETDIM-UAV-ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// GETDIM-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// GETDIM-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// GETDIM-SAME: ' lvalue .__handle
// GETDIM-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// GETDIM-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'mipLevel' 'unsigned int'
// GETDIM-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'width' 'unsigned int &__restrict'
// GETDIM-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'height' 'unsigned int &__restrict'
// GETDIM-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'numberOfLevels' 'unsigned int &__restrict'
// GETDIM-NEXT: AlwaysInlineAttr

// GETDIM: CXXMethodDecl {{.*}} GetDimensions 'void (out float, out float)'
// GETDIM-NEXT: ParmVarDecl {{.*}} width 'float &__restrict'
// GETDIM-NEXT: HLSLParamModifierAttr {{.*}} out
// GETDIM-NEXT: ParmVarDecl {{.*}} height 'float &__restrict'
// GETDIM-NEXT: HLSLParamModifierAttr {{.*}} out
// GETDIM-NEXT: CompoundStmt
// GETDIM-NEXT: CallExpr {{.*}} '<dependent type>'
// GETDIM-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_getdimensions_xy_float' 'void (__hlsl_resource_t, float &, float &) noexcept'
// GETDIM-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// GETDIM-SRV-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// GETDIM-UAV-SAME{LITERAL}: [[hlsl::resource_class("UAV")]]
// GETDIM-SRV-ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// GETDIM-UAV-ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// GETDIM-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// GETDIM-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// GETDIM-SAME: ' lvalue .__handle
// GETDIM-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// GETDIM-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'width' 'float &__restrict'
// GETDIM-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'height' 'float &__restrict'
// GETDIM-NEXT: AlwaysInlineAttr

// GETDIM: CXXMethodDecl {{.*}} GetDimensions 'void (unsigned int, out float, out float, out float)'
// GETDIM-NEXT: ParmVarDecl {{.*}} mipLevel 'unsigned int'
// GETDIM-NEXT: ParmVarDecl {{.*}} width 'float &__restrict'
// GETDIM-NEXT: HLSLParamModifierAttr {{.*}} out
// GETDIM-NEXT: ParmVarDecl {{.*}} height 'float &__restrict'
// GETDIM-NEXT: HLSLParamModifierAttr {{.*}} out
// GETDIM-NEXT: ParmVarDecl {{.*}} numberOfLevels 'float &__restrict'
// GETDIM-NEXT: HLSLParamModifierAttr {{.*}} out
// GETDIM-NEXT: CompoundStmt
// GETDIM-NEXT: CallExpr {{.*}} '<dependent type>'
// GETDIM-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_getdimensions_levels_xy_float' 'void (__hlsl_resource_t, unsigned int, float &, float &, float &) noexcept'
// GETDIM-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// GETDIM-SRV-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// GETDIM-UAV-SAME{LITERAL}: [[hlsl::resource_class("UAV")]]
// GETDIM-SRV-ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// GETDIM-UAV-ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// GETDIM-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// GETDIM-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// GETDIM-SAME: ' lvalue .__handle
// GETDIM-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// GETDIM-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'mipLevel' 'unsigned int'
// GETDIM-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'width' 'float &__restrict'
// GETDIM-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'height' 'float &__restrict'
// GETDIM-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'numberOfLevels' 'float &__restrict'
// GETDIM-NEXT: AlwaysInlineAttr

// CHECK: ClassTemplatePartialSpecializationDecl {{.*}} class [[TEXTURE]] explicit_specialization
// CHECK: TemplateTypeParmDecl {{.*}} element_type
// CHECK: NonTypeTemplateParmDecl {{.*}} element_count

// SRV-NOT: BinaryOperator {{.*}} 'hlsl_device float' lvalue '='

// UAV-STORE-LABEL: FunctionDecl {{.*}} main 'void ()'
// UAV-STORE: BinaryOperator {{.*}} 'hlsl_device float' lvalue '='
// UAV-STORE-NEXT: CXXOperatorCallExpr {{.*}} 'hlsl_device float' lvalue '[]'
// UAV-STORE-NEXT: ImplicitCastExpr {{.*}} 'hlsl_device float &(*)([[INDEX_TYPE]]) const' <FunctionToPointerDecay>
// UAV-STORE-NEXT: DeclRefExpr {{.*}} 'hlsl_device float &([[INDEX_TYPE]]) const' lvalue CXXMethod {{.*}} 'operator[]' 'hlsl_device float &([[INDEX_TYPE]]) const'
// UAV-STORE-NEXT: ImplicitCastExpr {{.*}} 'const hlsl::[[TEXTURE]]<float>' lvalue <NoOp>
// UAV-STORE-NEXT: DeclRefExpr {{.*}} '[[TEXTURE]]<float>':'hlsl::[[TEXTURE]]<float>' lvalue Var {{.*}} 't' '[[TEXTURE]]<float>':'hlsl::[[TEXTURE]]<float>'
// UAV-TRUNC-NEXT: ImplicitCastExpr {{.*}} '[[TRUNC_TYPE]]' <HLSLVectorTruncation>
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
#ifdef HAS_GETDIM
  uint w, h;
  t.GetDimensions(w, h);
#endif
}
