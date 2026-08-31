// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -disable-llvm-passes -finclude-default-header -DHAS_OFFSET \
// RUN:   -DHAS_GETDIM_XY -DTEXTURE=Texture2D -DCOORD_TYPE=float2 \
// RUN:   -DGRAD_TYPE=float2 -DLOD_LOCATION=loc -DOFFSET_ARG="int2(1, 2)" -o - \
// RUN:   %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,TEXEL,OFFSET,GETDIM-XY \
// RUN:   -DTEXTURE=Texture2D -DDIM_NAME=2D -DDIM=2 -DCOORD_DIM=2 -DLOAD_DIM=3 \
// RUN:   -DINDEX_TYPE="vector<unsigned int, 2>"
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -disable-llvm-passes -finclude-default-header -DTEXTURE=TextureCube \
// RUN:   -DCOORD_TYPE=float3 -DGRAD_TYPE=float3 -DLOD_LOCATION=loc -o - %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK -DTEXTURE=TextureCube \
// RUN:   -DDIM_NAME=Cube -DDIM=3 -DCOORD_DIM=3
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -disable-llvm-passes -finclude-default-header -DHAS_OFFSET \
// RUN:   -DHAS_GETDIM_XY -DTEXTURE=Texture2DArray -DCOORD_TYPE=float3 \
// RUN:   -DGRAD_TYPE=float2 -DLOD_LOCATION=loc.xy -DOFFSET_ARG="int2(1, 2)" \
// RUN:   -o - %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,ARRAY,TEXEL,OFFSET,GETDIM-XY \
// RUN:   -DTEXTURE=Texture2DArray -DDIM_NAME=2D -DDIM=2 -DCOORD_DIM=3 \
// RUN:   -DLOAD_DIM=4 -DINDEX_TYPE="vector<unsigned int, 3>"

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   HAS_OFFSET         defined for types whose sampling and gathering methods
//                      have overloads taking an offset
//   HAS_GETDIM_XY      defined for types that have the width/height
//                      GetDimensions overloads
//   TEXTURE            resource type name
//   COORD_TYPE         sample location type (DIM components plus the array
//                      slice)
//   GRAD_TYPE          SampleGrad ddx/ddy type, one component per resource
//                      dimension
//   LOD_LOCATION       expression producing a LOD_TYPE location from `loc`
//   OFFSET_ARG         a literal offset argument
//   DIM_NAME           hlsl::dimension spelling
//   DIM                number of resource dimensions (offset, ddx/ddy, LOD
//                      location)
//   COORD_DIM          sample location components (DIM plus the array slice)
//   LOAD_DIM           Load location components (COORD_DIM plus the mip level)
//   INDEX_TYPE         operator[] index type
//
// Check prefixes:
//   TEXEL              the type has integer texel addressing (Load,
//                      operator[], mips), and therefore a `mips` field in its
//                      layout
//   OFFSET             the sampling and gathering methods have offset
//                      overloads
//   GETDIM-XY          the width/height GetDimensions overloads exist
//   ARRAY              the resource has an array slice

// CHECK: CXXRecordDecl {{.*}} SamplerState definition
// CHECK: FinalAttr {{.*}} Implicit final
// CHECK-NEXT: FieldDecl {{.*}} implicit {{.*}} __handle '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]

// CHECK: CXXRecordDecl {{.*}} SamplerComparisonState definition
// CHECK: FinalAttr {{.*}} Implicit final
// CHECK-NEXT: FieldDecl {{.*}} implicit {{.*}} __handle '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]

// CHECK: ClassTemplateDecl {{.*}} [[TEXTURE]]
// CHECK: TemplateTypeParmDecl {{.*}} element_type
// CHECK: CXXRecordDecl {{.*}} [[TEXTURE]] definition
// CHECK: FinalAttr {{.*}} Implicit final
// CHECK-NEXT: FieldDecl {{.*}} implicit __handle '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}

// TEXEL: CXXMethodDecl {{.*}} Load 'element_type (vector<int, [[LOAD_DIM]]>)'
// TEXEL-NEXT: ParmVarDecl {{.*}} Location 'vector<int, [[LOAD_DIM]]>'
// TEXEL-NEXT: CompoundStmt
// TEXEL-NEXT: ReturnStmt
// TEXEL-NEXT: CStyleCastExpr {{.*}} 'element_type' <Dependent>
// TEXEL-NEXT: CallExpr {{.*}} '<dependent type>'
// TEXEL-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_load_level' 'void (...) noexcept'
// TEXEL-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// TEXEL-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// TEXEL-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// TEXEL-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// TEXEL-SAME: ' lvalue .__handle
// TEXEL-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// TEXEL-NEXT: DeclRefExpr {{.*}} 'vector<int, [[LOAD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<int, [[LOAD_DIM]]>'
// TEXEL-NEXT: AlwaysInlineAttr

// TEXEL: CXXMethodDecl {{.*}} Load 'element_type (vector<int, [[LOAD_DIM]]>, vector<int, [[DIM]]>)'
// TEXEL-NEXT: ParmVarDecl {{.*}} Location 'vector<int, [[LOAD_DIM]]>'
// TEXEL-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// TEXEL-NEXT: CompoundStmt
// TEXEL-NEXT: ReturnStmt
// TEXEL-NEXT: CStyleCastExpr {{.*}} 'element_type' <Dependent>
// TEXEL-NEXT: CallExpr {{.*}} '<dependent type>'
// TEXEL-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_load_level' 'void (...) noexcept'
// TEXEL-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// TEXEL-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// TEXEL-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// TEXEL-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// TEXEL-SAME: ' lvalue .__handle
// TEXEL-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// TEXEL-NEXT: DeclRefExpr {{.*}} 'vector<int, [[LOAD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<int, [[LOAD_DIM]]>'
// TEXEL-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// TEXEL-NEXT: AlwaysInlineAttr

// TEXEL: CXXMethodDecl {{.*}} operator[] 'const hlsl_device element_type &([[INDEX_TYPE]]) const' inline
// TEXEL-NEXT: ParmVarDecl {{.*}} Index '[[INDEX_TYPE]]'
// TEXEL-NEXT: CompoundStmt
// TEXEL-NEXT: ReturnStmt
// TEXEL-NEXT: UnaryOperator {{.*}} 'hlsl_device element_type' lvalue prefix '*' cannot overflow
// TEXEL-NEXT: CStyleCastExpr {{.*}} 'hlsl_device element_type *' <Dependent>
// TEXEL-NEXT: CallExpr {{.*}} '<dependent type>'
// TEXEL-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_getpointer' 'void (...) noexcept'
// TEXEL-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// TEXEL-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// TEXEL-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// TEXEL-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// TEXEL-SAME: ' lvalue .__handle
// TEXEL-NEXT: CXXThisExpr {{.*}} 'const hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// TEXEL-NEXT: DeclRefExpr {{.*}} '[[INDEX_TYPE]]' lvalue ParmVar {{.*}} 'Index' '[[INDEX_TYPE]]'
// TEXEL-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} Sample 'element_type (hlsl::SamplerState, vector<float, [[COORD_DIM]]>)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'element_type' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} Sample 'element_type (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, vector<int, [[DIM]]>)'
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// OFFSET-NEXT: CompoundStmt
// OFFSET-NEXT: ReturnStmt
// OFFSET-NEXT: CStyleCastExpr {{.*}} 'element_type' <Dependent>
// OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample' 'void (...) noexcept'
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// OFFSET-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// OFFSET-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// OFFSET-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} Sample 'element_type (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, vector<int, [[DIM]]>, float)'
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} Clamp 'float'
// OFFSET-NEXT: CompoundStmt
// OFFSET-NEXT: ReturnStmt
// OFFSET-NEXT: CStyleCastExpr {{.*}} 'element_type' <Dependent>
// OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample' 'void (...) noexcept'
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// OFFSET-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// OFFSET-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'Clamp' 'float'
// OFFSET-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} SampleBias 'element_type (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, float)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} Bias 'float'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'element_type' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_bias' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'Bias' 'float'
// CHECK-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} SampleBias 'element_type (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, float, vector<int, [[DIM]]>)'
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} Bias 'float'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// OFFSET-NEXT: CompoundStmt
// OFFSET-NEXT: ReturnStmt
// OFFSET-NEXT: CStyleCastExpr {{.*}} 'element_type' <Dependent>
// OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_bias' 'void (...) noexcept'
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// OFFSET-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// OFFSET-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'Bias' 'float'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// OFFSET-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} SampleBias 'element_type (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, float, vector<int, [[DIM]]>, float)'
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} Bias 'float'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} Clamp 'float'
// OFFSET-NEXT: CompoundStmt
// OFFSET-NEXT: ReturnStmt
// OFFSET-NEXT: CStyleCastExpr {{.*}} 'element_type' <Dependent>
// OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_bias' 'void (...) noexcept'
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// OFFSET-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// OFFSET-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'Bias' 'float'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'Clamp' 'float'
// OFFSET-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} SampleGrad 'element_type (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, vector<float, [[DIM]]>, vector<float, [[DIM]]>)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} DDX 'vector<float, [[DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} DDY 'vector<float, [[DIM]]>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'element_type' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_grad' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[DIM]]>' lvalue ParmVar {{.*}} 'DDX' 'vector<float, [[DIM]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[DIM]]>' lvalue ParmVar {{.*}} 'DDY' 'vector<float, [[DIM]]>'
// CHECK-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} SampleGrad 'element_type (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, vector<float, [[DIM]]>, vector<float, [[DIM]]>, vector<int, [[DIM]]>)'
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} DDX 'vector<float, [[DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} DDY 'vector<float, [[DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// OFFSET-NEXT: CompoundStmt
// OFFSET-NEXT: ReturnStmt
// OFFSET-NEXT: CStyleCastExpr {{.*}} 'element_type' <Dependent>
// OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_grad' 'void (...) noexcept'
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// OFFSET-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// OFFSET-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[DIM]]>' lvalue ParmVar {{.*}} 'DDX' 'vector<float, [[DIM]]>'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[DIM]]>' lvalue ParmVar {{.*}} 'DDY' 'vector<float, [[DIM]]>'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// OFFSET-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} SampleGrad 'element_type (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, vector<float, [[DIM]]>, vector<float, [[DIM]]>, vector<int, [[DIM]]>, float)'
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} DDX 'vector<float, [[DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} DDY 'vector<float, [[DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} Clamp 'float'
// OFFSET-NEXT: CompoundStmt
// OFFSET-NEXT: ReturnStmt
// OFFSET-NEXT: CStyleCastExpr {{.*}} 'element_type' <Dependent>
// OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_grad' 'void (...) noexcept'
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// OFFSET-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// OFFSET-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[DIM]]>' lvalue ParmVar {{.*}} 'DDX' 'vector<float, [[DIM]]>'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[DIM]]>' lvalue ParmVar {{.*}} 'DDY' 'vector<float, [[DIM]]>'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'Clamp' 'float'
// OFFSET-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} SampleLevel 'element_type (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, float)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} LOD 'float'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'element_type' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_level' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'LOD' 'float'
// CHECK-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} SampleLevel 'element_type (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, float, vector<int, [[DIM]]>)'
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} LOD 'float'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// OFFSET-NEXT: CompoundStmt
// OFFSET-NEXT: ReturnStmt
// OFFSET-NEXT: CStyleCastExpr {{.*}} 'element_type' <Dependent>
// OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_level' 'void (...) noexcept'
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// OFFSET-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// OFFSET-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'LOD' 'float'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// OFFSET-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} SampleCmp 'float (hlsl::SamplerComparisonState, vector<float, [[COORD_DIM]]>, float)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'float' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_cmp' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// CHECK-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} SampleCmp 'float (hlsl::SamplerComparisonState, vector<float, [[COORD_DIM]]>, float, vector<int, [[DIM]]>)'
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// OFFSET-NEXT: CompoundStmt
// OFFSET-NEXT: ReturnStmt
// OFFSET-NEXT: CStyleCastExpr {{.*}} 'float' <Dependent>
// OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_cmp' 'void (...) noexcept'
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// OFFSET-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// OFFSET-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// OFFSET-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} SampleCmp 'float (hlsl::SamplerComparisonState, vector<float, [[COORD_DIM]]>, float, vector<int, [[DIM]]>, float)'
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} Clamp 'float'
// OFFSET-NEXT: CompoundStmt
// OFFSET-NEXT: ReturnStmt
// OFFSET-NEXT: CStyleCastExpr {{.*}} 'float' <Dependent>
// OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_cmp' 'void (...) noexcept'
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// OFFSET-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// OFFSET-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'Clamp' 'float'
// OFFSET-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} SampleCmpLevelZero 'float (hlsl::SamplerComparisonState, vector<float, [[COORD_DIM]]>, float)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'float' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_cmp_level_zero' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// CHECK-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} SampleCmpLevelZero 'float (hlsl::SamplerComparisonState, vector<float, [[COORD_DIM]]>, float, vector<int, [[DIM]]>)'
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// OFFSET-NEXT: CompoundStmt
// OFFSET-NEXT: ReturnStmt
// OFFSET-NEXT: CStyleCastExpr {{.*}} 'float' <Dependent>
// OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_cmp_level_zero' 'void (...) noexcept'
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// OFFSET-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// OFFSET-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// OFFSET-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} CalculateLevelOfDetail 'float (hlsl::SamplerState, vector<float, [[DIM]]>)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[DIM]]>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'float' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_calculate_lod' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[DIM]]>'
// CHECK-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} CalculateLevelOfDetailUnclamped 'float (hlsl::SamplerState, vector<float, [[DIM]]>)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[DIM]]>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'float' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_calculate_lod_unclamped' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[DIM]]>'
// CHECK-NEXT: AlwaysInlineAttr

// GETDIM-XY: CXXMethodDecl {{.*}} GetDimensions 'void (out unsigned int, out unsigned int)'
// GETDIM-XY-NEXT: ParmVarDecl {{.*}} width 'unsigned int &__restrict'
// GETDIM-XY-NEXT: HLSLParamModifierAttr {{.*}} out
// GETDIM-XY-NEXT: ParmVarDecl {{.*}} height 'unsigned int &__restrict'
// GETDIM-XY-NEXT: HLSLParamModifierAttr {{.*}} out
// GETDIM-XY-NEXT: CompoundStmt
// GETDIM-XY-NEXT: CallExpr {{.*}} '<dependent type>'
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_getdimensions_xy' 'void (__hlsl_resource_t, unsigned int &, unsigned int &) noexcept'
// GETDIM-XY-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// GETDIM-XY-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// GETDIM-XY-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// GETDIM-XY-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// GETDIM-XY-SAME: ' lvalue .__handle
// GETDIM-XY-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'width' 'unsigned int &__restrict'
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'height' 'unsigned int &__restrict'
// GETDIM-XY-NEXT: AlwaysInlineAttr

// GETDIM-XY: CXXMethodDecl {{.*}} GetDimensions 'void (unsigned int, out unsigned int, out unsigned int, out unsigned int)'
// GETDIM-XY-NEXT: ParmVarDecl {{.*}} mipLevel 'unsigned int'
// GETDIM-XY-NEXT: ParmVarDecl {{.*}} width 'unsigned int &__restrict'
// GETDIM-XY-NEXT: HLSLParamModifierAttr {{.*}} out
// GETDIM-XY-NEXT: ParmVarDecl {{.*}} height 'unsigned int &__restrict'
// GETDIM-XY-NEXT: HLSLParamModifierAttr {{.*}} out
// GETDIM-XY-NEXT: ParmVarDecl {{.*}} numberOfLevels 'unsigned int &__restrict'
// GETDIM-XY-NEXT: HLSLParamModifierAttr {{.*}} out
// GETDIM-XY-NEXT: CompoundStmt
// GETDIM-XY-NEXT: CallExpr {{.*}} '<dependent type>'
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_getdimensions_levels_xy' 'void (__hlsl_resource_t, unsigned int, unsigned int &, unsigned int &, unsigned int &) noexcept'
// GETDIM-XY-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// GETDIM-XY-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// GETDIM-XY-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// GETDIM-XY-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// GETDIM-XY-SAME: ' lvalue .__handle
// GETDIM-XY-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'mipLevel' 'unsigned int'
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'width' 'unsigned int &__restrict'
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'height' 'unsigned int &__restrict'
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'numberOfLevels' 'unsigned int &__restrict'
// GETDIM-XY-NEXT: AlwaysInlineAttr

// GETDIM-XY: CXXMethodDecl {{.*}} GetDimensions 'void (out float, out float)'
// GETDIM-XY-NEXT: ParmVarDecl {{.*}} width 'float &__restrict'
// GETDIM-XY-NEXT: HLSLParamModifierAttr {{.*}} out
// GETDIM-XY-NEXT: ParmVarDecl {{.*}} height 'float &__restrict'
// GETDIM-XY-NEXT: HLSLParamModifierAttr {{.*}} out
// GETDIM-XY-NEXT: CompoundStmt
// GETDIM-XY-NEXT: CallExpr {{.*}} '<dependent type>'
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_getdimensions_xy_float' 'void (__hlsl_resource_t, float &, float &) noexcept'
// GETDIM-XY-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// GETDIM-XY-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// GETDIM-XY-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// GETDIM-XY-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// GETDIM-XY-SAME: ' lvalue .__handle
// GETDIM-XY-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'width' 'float &__restrict'
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'height' 'float &__restrict'
// GETDIM-XY-NEXT: AlwaysInlineAttr

// GETDIM-XY: CXXMethodDecl {{.*}} GetDimensions 'void (unsigned int, out float, out float, out float)'
// GETDIM-XY-NEXT: ParmVarDecl {{.*}} mipLevel 'unsigned int'
// GETDIM-XY-NEXT: ParmVarDecl {{.*}} width 'float &__restrict'
// GETDIM-XY-NEXT: HLSLParamModifierAttr {{.*}} out
// GETDIM-XY-NEXT: ParmVarDecl {{.*}} height 'float &__restrict'
// GETDIM-XY-NEXT: HLSLParamModifierAttr {{.*}} out
// GETDIM-XY-NEXT: ParmVarDecl {{.*}} numberOfLevels 'float &__restrict'
// GETDIM-XY-NEXT: HLSLParamModifierAttr {{.*}} out
// GETDIM-XY-NEXT: CompoundStmt
// GETDIM-XY-NEXT: CallExpr {{.*}} '<dependent type>'
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_getdimensions_levels_xy_float' 'void (__hlsl_resource_t, unsigned int, float &, float &, float &) noexcept'
// GETDIM-XY-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// GETDIM-XY-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// GETDIM-XY-SAME{LITERAL}: [[hlsl::contained_type(element_type)]]
// GETDIM-XY-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// GETDIM-XY-SAME: ' lvalue .__handle
// GETDIM-XY-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<element_type>' lvalue implicit this
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'mipLevel' 'unsigned int'
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'width' 'float &__restrict'
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'height' 'float &__restrict'
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'numberOfLevels' 'float &__restrict'
// GETDIM-XY-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} Gather 'vector<element_type, 4> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>)' inline
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, 4>' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned int' 0
// CHECK-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} Gather 'vector<element_type, 4> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, vector<int, [[DIM]]>)' inline
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// OFFSET-NEXT: CompoundStmt
// OFFSET-NEXT: ReturnStmt
// OFFSET-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, 4>' <Dependent>
// OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather' 'void (...) noexcept'
// OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: IntegerLiteral {{.*}} 'unsigned int' 0
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// OFFSET-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} GatherRed 'vector<element_type, 4> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>)' inline
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, 4>' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned int' 0
// CHECK-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} GatherRed 'vector<element_type, 4> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, vector<int, [[DIM]]>)' inline
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// OFFSET-NEXT: CompoundStmt
// OFFSET-NEXT: ReturnStmt
// OFFSET-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, 4>' <Dependent>
// OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather' 'void (...) noexcept'
// OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: IntegerLiteral {{.*}} 'unsigned int' 0
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// OFFSET-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} GatherGreen 'vector<element_type, 4> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>)' inline
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, 4>' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned int' 1
// CHECK-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} GatherGreen 'vector<element_type, 4> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, vector<int, [[DIM]]>)' inline
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// OFFSET-NEXT: CompoundStmt
// OFFSET-NEXT: ReturnStmt
// OFFSET-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, 4>' <Dependent>
// OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather' 'void (...) noexcept'
// OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: IntegerLiteral {{.*}} 'unsigned int' 1
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// OFFSET-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} GatherBlue 'vector<element_type, 4> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>)' inline
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, 4>' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned int' 2
// CHECK-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} GatherBlue 'vector<element_type, 4> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, vector<int, [[DIM]]>)' inline
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// OFFSET-NEXT: CompoundStmt
// OFFSET-NEXT: ReturnStmt
// OFFSET-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, 4>' <Dependent>
// OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather' 'void (...) noexcept'
// OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: IntegerLiteral {{.*}} 'unsigned int' 2
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// OFFSET-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} GatherAlpha 'vector<element_type, 4> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>)' inline
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, 4>' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned int' 3
// CHECK-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} GatherAlpha 'vector<element_type, 4> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, vector<int, [[DIM]]>)' inline
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// OFFSET-NEXT: CompoundStmt
// OFFSET-NEXT: ReturnStmt
// OFFSET-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, 4>' <Dependent>
// OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather' 'void (...) noexcept'
// OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: IntegerLiteral {{.*}} 'unsigned int' 3
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// OFFSET-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} GatherCmp 'vector<float, 4> (hlsl::SamplerComparisonState, vector<float, [[COORD_DIM]]>, float)' inline
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'vector<float, 4>' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather_cmp' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned int' 0
// CHECK-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} GatherCmp 'vector<float, 4> (hlsl::SamplerComparisonState, vector<float, [[COORD_DIM]]>, float, vector<int, [[DIM]]>)' inline
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// OFFSET-NEXT: CompoundStmt
// OFFSET-NEXT: ReturnStmt
// OFFSET-NEXT: CStyleCastExpr {{.*}} 'vector<float, 4>' <Dependent>
// OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather_cmp' 'void (...) noexcept'
// OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// OFFSET-NEXT: IntegerLiteral {{.*}} 'unsigned int' 0
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// OFFSET-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} GatherCmpRed 'vector<float, 4> (hlsl::SamplerComparisonState, vector<float, [[COORD_DIM]]>, float)' inline
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'vector<float, 4>' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather_cmp' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned int' 0
// CHECK-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} GatherCmpGreen 'vector<float, 4> (hlsl::SamplerComparisonState, vector<float, [[COORD_DIM]]>, float)' inline
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'vector<float, 4>' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather_cmp' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned int' 1
// CHECK-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} GatherCmpBlue 'vector<float, 4> (hlsl::SamplerComparisonState, vector<float, [[COORD_DIM]]>, float)' inline
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'vector<float, 4>' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather_cmp' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned int' 2
// CHECK-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} GatherCmpAlpha 'vector<float, 4> (hlsl::SamplerComparisonState, vector<float, [[COORD_DIM]]>, float, vector<int, [[DIM]]>)' inline
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// OFFSET-NEXT: CompoundStmt
// OFFSET-NEXT: ReturnStmt
// OFFSET-NEXT: CStyleCastExpr {{.*}} 'vector<float, 4>' <Dependent>
// OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather_cmp' 'void (...) noexcept'
// OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// OFFSET-NEXT: IntegerLiteral {{.*}} 'unsigned int' 3
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// OFFSET-NEXT: AlwaysInlineAttr

TEXTURE<float> t;
SamplerState s;
SamplerComparisonState scs;

void main(COORD_TYPE loc, float cmp) {
  t.Sample(s, loc);
  t.SampleBias(s, loc, 0.0);
  t.SampleGrad(s, loc, (GRAD_TYPE)0, (GRAD_TYPE)0);
  t.SampleLevel(s, loc, 0.0);
  t.SampleCmp(scs, loc, cmp);
  t.SampleCmpLevelZero(scs, loc, cmp);
  t.CalculateLevelOfDetail(s, LOD_LOCATION);
  t.CalculateLevelOfDetailUnclamped(s, LOD_LOCATION);
  t.Gather(s, loc);

#ifdef HAS_OFFSET
  t.Sample(s, loc, OFFSET_ARG);
  t.Sample(s, loc, OFFSET_ARG, 1.0);
  t.SampleBias(s, loc, 0.0, OFFSET_ARG);
  t.SampleBias(s, loc, 0.0, OFFSET_ARG, 1.0);
  t.SampleGrad(s, loc, (GRAD_TYPE)0, (GRAD_TYPE)0, OFFSET_ARG);
  t.SampleGrad(s, loc, (GRAD_TYPE)0, (GRAD_TYPE)0, OFFSET_ARG, 1.0);
  t.SampleLevel(s, loc, 0.0, OFFSET_ARG);
  t.SampleCmp(scs, loc, cmp, OFFSET_ARG);
  t.SampleCmp(scs, loc, cmp, OFFSET_ARG, 1.0f);
  t.SampleCmpLevelZero(scs, loc, cmp, OFFSET_ARG);
#endif

#ifdef HAS_GETDIM_XY
  uint u_w, u_h, u_l;
  float f_w, f_h, f_l;
  t.GetDimensions(u_w, u_h);
  t.GetDimensions(0, u_w, u_h, u_l);
  t.GetDimensions(f_w, f_h);
  t.GetDimensions(0, f_w, f_h, f_l);
#endif
}
