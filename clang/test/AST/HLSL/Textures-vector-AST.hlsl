// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -disable-llvm-passes -finclude-default-header -DHAS_OFFSET \
// RUN:   -DHAS_GETDIM_XY -DHAS_SAMPLE_CMP -DHAS_GATHER -DTEXTURE=Texture2D \
// RUN:   -DCOORD_TYPE=float2 -DGRAD_TYPE=float2 -DLOD_LOCATION=loc \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -o - %s \
// RUN:   | FileCheck %s \
// RUN:   --check-prefixes=CHECK,TEXEL,OFFSET,GETDIM-XY,SAMPLECMP,SAMPLECMP-OFFSET,GATHER,GATHER-OFFSET \
// RUN:   -DTEXTURE=Texture2D -DDIM_NAME=2D -DDIM=2 -DCOORD_DIM=2 -DLOAD_DIM=3 \
// RUN:   -DINDEX_TYPE="vector<unsigned int, 2>" -DIS_ARRAY=""
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -disable-llvm-passes -finclude-default-header -DHAS_SAMPLE_CMP \
// RUN:   -DHAS_GATHER -DTEXTURE=TextureCube -DCOORD_TYPE=float3 \
// RUN:   -DGRAD_TYPE=float3 -DLOD_LOCATION=loc -o - %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,SAMPLECMP,GATHER \
// RUN:   -DTEXTURE=TextureCube -DDIM_NAME=Cube -DDIM=3 -DCOORD_DIM=3 \
// RUN:   -DIS_ARRAY=""
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -disable-llvm-passes -finclude-default-header -DHAS_SAMPLE_CMP \
// RUN:   -DHAS_GATHER -DTEXTURE=TextureCubeArray -DCOORD_TYPE=float4 \
// RUN:   -DGRAD_TYPE=float3 -DLOD_LOCATION=loc.xyz -o - %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,SAMPLECMP,GATHER \
// RUN:   -DTEXTURE=TextureCubeArray -DDIM_NAME=Cube -DDIM=3 -DCOORD_DIM=4 \
// RUN:   -DIS_ARRAY=" [[hlsl::is_array]]"
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -disable-llvm-passes -finclude-default-header -DHAS_OFFSET \
// RUN:   -DHAS_GETDIM_XY -DHAS_SAMPLE_CMP -DHAS_GATHER \
// RUN:   -DTEXTURE=Texture2DArray -DCOORD_TYPE=float3 -DGRAD_TYPE=float2 \
// RUN:   -DLOD_LOCATION=loc.xy -DOFFSET_ARG="int2(1, 2)" -o - %s \
// RUN:   | FileCheck %s \
// RUN:   --check-prefixes=CHECK,ARRAY,TEXEL,OFFSET,GETDIM-XY,SAMPLECMP,SAMPLECMP-OFFSET,GATHER,GATHER-OFFSET \
// RUN:   -DTEXTURE=Texture2DArray -DDIM_NAME=2D -DDIM=2 -DCOORD_DIM=3 \
// RUN:   -DLOAD_DIM=4 -DINDEX_TYPE="vector<unsigned int, 3>" \
// RUN:   -DIS_ARRAY=" [[hlsl::is_array]]"
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -disable-llvm-passes -finclude-default-header -DHAS_OFFSET \
// RUN:   -DTEXTURE=Texture3D -DCOORD_TYPE=float3 -DGRAD_TYPE=float3 \
// RUN:   -DLOD_LOCATION=loc -DOFFSET_ARG="int3(1, 2, 3)" -o - %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,TEXEL,OFFSET \
// RUN:   -DTEXTURE=Texture3D -DDIM_NAME=3D -DDIM=3 -DCOORD_DIM=3 -DLOAD_DIM=4 \
// RUN:   -DINDEX_TYPE="vector<unsigned int, 3>" -DIS_ARRAY=""

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   HAS_OFFSET         defined for types whose sampling and gathering methods
//                      have overloads taking an offset
//   HAS_GETDIM_XY      defined for types that have the width/height
//                      GetDimensions overloads
//   HAS_SAMPLE_CMP     defined for types that have the comparison sampling
//                      methods
//   HAS_GATHER         defined for types that have the Gather* methods
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
//   SAMPLECMP          the comparison sampling methods exist
//   SAMPLECMP-OFFSET   the comparison sampling methods have offset overloads
//   GATHER             the Gather* methods exist
//   GATHER-OFFSET      the Gather* methods have offset overloads
//   ARRAY              the resource has an array slice
//   IS_ARRAY           the [[hlsl::is_array]] attribute on arrayed resources,
//                      or empty

// CHECK: CXXRecordDecl {{.*}} SamplerState definition
// CHECK: FinalAttr {{.*}} Implicit final
// CHECK-NEXT: FieldDecl {{.*}} implicit {{.*}} __handle '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]

// CHECK: CXXRecordDecl {{.*}} SamplerComparisonState definition
// CHECK: FinalAttr {{.*}} Implicit final
// CHECK-NEXT: FieldDecl {{.*}} implicit {{.*}}__handle '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]

// CHECK: ClassTemplateDecl {{.*}} [[TEXTURE]]
// CHECK: TemplateTypeParmDecl {{.*}} element_type
// CHECK: CXXRecordDecl {{.*}} [[TEXTURE]]
// CHECK: FinalAttr {{.*}} Implicit final
// CHECK: ClassTemplatePartialSpecializationDecl {{.*}} [[TEXTURE]] definition explicit_specialization
// CHECK: TemplateArgument type 'vector<element_type, element_count>':'vector<type-parameter-0-0, element_count>'
// CHECK: TemplateTypeParmDecl {{.*}} element_type
// CHECK: NonTypeTemplateParmDecl {{.*}} element_count
// CHECK-NEXT: FieldDecl {{.*}} implicit __handle '__hlsl_resource_t
// CHECK-SAME: {{\[\[}}hlsl::resource_class("SRV"){{\]\]}}[[IS_ARRAY]] {{\[\[}}hlsl::contained_type(vector<element_type, element_count>){{\]\]}}
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}

// TEXEL: CXXMethodDecl {{.*}} Load 'vector<element_type, element_count> (vector<int, [[LOAD_DIM]]>)'
// TEXEL-NEXT: ParmVarDecl {{.*}} Location 'vector<int, [[LOAD_DIM]]>'
// TEXEL-NEXT: CompoundStmt
// TEXEL-NEXT: ReturnStmt
// TEXEL-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, element_count>' <Dependent>
// TEXEL-NEXT: CallExpr {{.*}} '<dependent type>'
// TEXEL-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_load_level' 'void (...) noexcept'
// TEXEL-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// TEXEL-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// TEXEL-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// TEXEL-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// TEXEL-SAME: ' lvalue .__handle
// TEXEL-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// TEXEL-NEXT: DeclRefExpr {{.*}} 'vector<int, [[LOAD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<int, [[LOAD_DIM]]>'
// TEXEL-NEXT: AlwaysInlineAttr

// TEXEL: CXXMethodDecl {{.*}} Load 'vector<element_type, element_count> (vector<int, [[LOAD_DIM]]>, vector<int, [[DIM]]>)'
// TEXEL-NEXT: ParmVarDecl {{.*}} Location 'vector<int, [[LOAD_DIM]]>'
// TEXEL-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// TEXEL-NEXT: CompoundStmt
// TEXEL-NEXT: ReturnStmt
// TEXEL-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, element_count>' <Dependent>
// TEXEL-NEXT: CallExpr {{.*}} '<dependent type>'
// TEXEL-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_load_level' 'void (...) noexcept'
// TEXEL-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// TEXEL-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// TEXEL-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// TEXEL-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// TEXEL-SAME: ' lvalue .__handle
// TEXEL-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// TEXEL-NEXT: DeclRefExpr {{.*}} 'vector<int, [[LOAD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<int, [[LOAD_DIM]]>'
// TEXEL-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// TEXEL-NEXT: AlwaysInlineAttr

// TEXEL: CXXMethodDecl {{.*}} operator[] 'vector<element_type, element_count> const hlsl_device &([[INDEX_TYPE]]) const' inline
// TEXEL-NEXT: ParmVarDecl {{.*}} Index '[[INDEX_TYPE]]'
// TEXEL-NEXT: CompoundStmt
// TEXEL-NEXT: ReturnStmt
// TEXEL-NEXT: UnaryOperator {{.*}} 'vector<element_type, element_count> hlsl_device' lvalue prefix '*' cannot overflow
// TEXEL-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, element_count> hlsl_device *' <Dependent>
// TEXEL-NEXT: CallExpr {{.*}} '<dependent type>'
// TEXEL-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_getpointer' 'void (...) noexcept'
// TEXEL-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// TEXEL-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// TEXEL-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// TEXEL-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// TEXEL-SAME: ' lvalue .__handle
// TEXEL-NEXT: CXXThisExpr {{.*}} 'const hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// TEXEL-NEXT: DeclRefExpr {{.*}} '[[INDEX_TYPE]]' lvalue ParmVar {{.*}} 'Index' '[[INDEX_TYPE]]'
// TEXEL-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} Sample 'vector<element_type, element_count> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, element_count>' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME: {{\[\[}}hlsl::resource_class("SRV"){{\]\]}}[[IS_ARRAY]] {{\[\[}}hlsl::contained_type(vector<element_type, element_count>){{\]\]}}
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} Sample 'vector<element_type, element_count> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, vector<int, [[DIM]]>)'
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// OFFSET-NEXT: CompoundStmt
// OFFSET-NEXT: ReturnStmt
// OFFSET-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, element_count>' <Dependent>
// OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample' 'void (...) noexcept'
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// OFFSET-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// OFFSET-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// OFFSET-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} Sample 'vector<element_type, element_count> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, vector<int, [[DIM]]>, float)'
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} Clamp 'float'
// OFFSET-NEXT: CompoundStmt
// OFFSET-NEXT: ReturnStmt
// OFFSET-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, element_count>' <Dependent>
// OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample' 'void (...) noexcept'
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// OFFSET-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// OFFSET-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'Clamp' 'float'
// OFFSET-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} SampleBias 'vector<element_type, element_count> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, float)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} Bias 'float'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, element_count>' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_bias' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME: {{\[\[}}hlsl::resource_class("SRV"){{\]\]}}[[IS_ARRAY]] {{\[\[}}hlsl::contained_type(vector<element_type, element_count>){{\]\]}}
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'Bias' 'float'
// CHECK-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} SampleBias 'vector<element_type, element_count> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, float, vector<int, [[DIM]]>)'
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} Bias 'float'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// OFFSET-NEXT: CompoundStmt
// OFFSET-NEXT: ReturnStmt
// OFFSET-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, element_count>' <Dependent>
// OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_bias' 'void (...) noexcept'
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// OFFSET-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// OFFSET-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'Bias' 'float'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// OFFSET-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} SampleBias 'vector<element_type, element_count> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, float, vector<int, [[DIM]]>, float)'
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} Bias 'float'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} Clamp 'float'
// OFFSET-NEXT: CompoundStmt
// OFFSET-NEXT: ReturnStmt
// OFFSET-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, element_count>' <Dependent>
// OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_bias' 'void (...) noexcept'
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// OFFSET-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// OFFSET-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'Bias' 'float'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'Clamp' 'float'
// OFFSET-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} SampleGrad 'vector<element_type, element_count> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, vector<float, [[DIM]]>, vector<float, [[DIM]]>)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} DDX 'vector<float, [[DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} DDY 'vector<float, [[DIM]]>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, element_count>' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_grad' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME: {{\[\[}}hlsl::resource_class("SRV"){{\]\]}}[[IS_ARRAY]] {{\[\[}}hlsl::contained_type(vector<element_type, element_count>){{\]\]}}
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[DIM]]>' lvalue ParmVar {{.*}} 'DDX' 'vector<float, [[DIM]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[DIM]]>' lvalue ParmVar {{.*}} 'DDY' 'vector<float, [[DIM]]>'
// CHECK-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} SampleGrad 'vector<element_type, element_count> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, vector<float, [[DIM]]>, vector<float, [[DIM]]>, vector<int, [[DIM]]>)'
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} DDX 'vector<float, [[DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} DDY 'vector<float, [[DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// OFFSET-NEXT: CompoundStmt
// OFFSET-NEXT: ReturnStmt
// OFFSET-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, element_count>' <Dependent>
// OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_grad' 'void (...) noexcept'
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// OFFSET-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// OFFSET-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[DIM]]>' lvalue ParmVar {{.*}} 'DDX' 'vector<float, [[DIM]]>'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[DIM]]>' lvalue ParmVar {{.*}} 'DDY' 'vector<float, [[DIM]]>'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// OFFSET-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} SampleGrad 'vector<element_type, element_count> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, vector<float, [[DIM]]>, vector<float, [[DIM]]>, vector<int, [[DIM]]>, float)'
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} DDX 'vector<float, [[DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} DDY 'vector<float, [[DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} Clamp 'float'
// OFFSET-NEXT: CompoundStmt
// OFFSET-NEXT: ReturnStmt
// OFFSET-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, element_count>' <Dependent>
// OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_grad' 'void (...) noexcept'
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// OFFSET-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// OFFSET-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
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

// CHECK: CXXMethodDecl {{.*}} SampleLevel 'vector<element_type, element_count> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, float)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} LOD 'float'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, element_count>' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_level' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME: {{\[\[}}hlsl::resource_class("SRV"){{\]\]}}[[IS_ARRAY]] {{\[\[}}hlsl::contained_type(vector<element_type, element_count>){{\]\]}}
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'LOD' 'float'
// CHECK-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} SampleLevel 'vector<element_type, element_count> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, float, vector<int, [[DIM]]>)'
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: ParmVarDecl {{.*}} LOD 'float'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// OFFSET-NEXT: CompoundStmt
// OFFSET-NEXT: ReturnStmt
// OFFSET-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, element_count>' <Dependent>
// OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_level' 'void (...) noexcept'
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// OFFSET-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// OFFSET-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// OFFSET-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// OFFSET-SAME: ' lvalue .__handle
// OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'LOD' 'float'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// OFFSET-NEXT: AlwaysInlineAttr

// SAMPLECMP: CXXMethodDecl {{.*}} SampleCmp 'float (hlsl::SamplerComparisonState, vector<float, [[COORD_DIM]]>, float)'
// SAMPLECMP-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// SAMPLECMP-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// SAMPLECMP-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// SAMPLECMP-NEXT: CompoundStmt
// SAMPLECMP-NEXT: ReturnStmt
// SAMPLECMP-NEXT: CStyleCastExpr {{.*}} 'float' <Dependent>
// SAMPLECMP-NEXT: CallExpr {{.*}} '<dependent type>'
// SAMPLECMP-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_cmp' 'void (...) noexcept'
// SAMPLECMP-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// SAMPLECMP-SAME: {{\[\[}}hlsl::resource_class("SRV"){{\]\]}}[[IS_ARRAY]] {{\[\[}}hlsl::contained_type(vector<element_type, element_count>){{\]\]}}
// SAMPLECMP-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// SAMPLECMP-SAME: ' lvalue .__handle
// SAMPLECMP-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// SAMPLECMP-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// SAMPLECMP-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// SAMPLECMP-SAME: ' lvalue .__handle
// SAMPLECMP-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// SAMPLECMP-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// SAMPLECMP-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// SAMPLECMP-NEXT: AlwaysInlineAttr

// SAMPLECMP-OFFSET: CXXMethodDecl {{.*}} SampleCmp 'float (hlsl::SamplerComparisonState, vector<float, [[COORD_DIM]]>, float, vector<int, [[DIM]]>)'
// SAMPLECMP-OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// SAMPLECMP-OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// SAMPLECMP-OFFSET-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// SAMPLECMP-OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// SAMPLECMP-OFFSET-NEXT: CompoundStmt
// SAMPLECMP-OFFSET-NEXT: ReturnStmt
// SAMPLECMP-OFFSET-NEXT: CStyleCastExpr {{.*}} 'float' <Dependent>
// SAMPLECMP-OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// SAMPLECMP-OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_cmp' 'void (...) noexcept'
// SAMPLECMP-OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// SAMPLECMP-OFFSET-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// SAMPLECMP-OFFSET-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// SAMPLECMP-OFFSET-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// SAMPLECMP-OFFSET-SAME: ' lvalue .__handle
// SAMPLECMP-OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// SAMPLECMP-OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// SAMPLECMP-OFFSET-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// SAMPLECMP-OFFSET-SAME: ' lvalue .__handle
// SAMPLECMP-OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// SAMPLECMP-OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// SAMPLECMP-OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// SAMPLECMP-OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// SAMPLECMP-OFFSET-NEXT: AlwaysInlineAttr

// SAMPLECMP-OFFSET: CXXMethodDecl {{.*}} SampleCmp 'float (hlsl::SamplerComparisonState, vector<float, [[COORD_DIM]]>, float, vector<int, [[DIM]]>, float)'
// SAMPLECMP-OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// SAMPLECMP-OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// SAMPLECMP-OFFSET-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// SAMPLECMP-OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// SAMPLECMP-OFFSET-NEXT: ParmVarDecl {{.*}} Clamp 'float'
// SAMPLECMP-OFFSET-NEXT: CompoundStmt
// SAMPLECMP-OFFSET-NEXT: ReturnStmt
// SAMPLECMP-OFFSET-NEXT: CStyleCastExpr {{.*}} 'float' <Dependent>
// SAMPLECMP-OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// SAMPLECMP-OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_cmp' 'void (...) noexcept'
// SAMPLECMP-OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// SAMPLECMP-OFFSET-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// SAMPLECMP-OFFSET-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// SAMPLECMP-OFFSET-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// SAMPLECMP-OFFSET-SAME: ' lvalue .__handle
// SAMPLECMP-OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// SAMPLECMP-OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// SAMPLECMP-OFFSET-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// SAMPLECMP-OFFSET-SAME: ' lvalue .__handle
// SAMPLECMP-OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// SAMPLECMP-OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// SAMPLECMP-OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// SAMPLECMP-OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// SAMPLECMP-OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'Clamp' 'float'
// SAMPLECMP-OFFSET-NEXT: AlwaysInlineAttr

// SAMPLECMP: CXXMethodDecl {{.*}} SampleCmpLevelZero 'float (hlsl::SamplerComparisonState, vector<float, [[COORD_DIM]]>, float)'
// SAMPLECMP-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// SAMPLECMP-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// SAMPLECMP-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// SAMPLECMP-NEXT: CompoundStmt
// SAMPLECMP-NEXT: ReturnStmt
// SAMPLECMP-NEXT: CStyleCastExpr {{.*}} 'float' <Dependent>
// SAMPLECMP-NEXT: CallExpr {{.*}} '<dependent type>'
// SAMPLECMP-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_cmp_level_zero' 'void (...) noexcept'
// SAMPLECMP-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// SAMPLECMP-SAME: {{\[\[}}hlsl::resource_class("SRV"){{\]\]}}[[IS_ARRAY]] {{\[\[}}hlsl::contained_type(vector<element_type, element_count>){{\]\]}}
// SAMPLECMP-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// SAMPLECMP-SAME: ' lvalue .__handle
// SAMPLECMP-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// SAMPLECMP-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// SAMPLECMP-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// SAMPLECMP-SAME: ' lvalue .__handle
// SAMPLECMP-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// SAMPLECMP-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// SAMPLECMP-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// SAMPLECMP-NEXT: AlwaysInlineAttr

// SAMPLECMP-OFFSET: CXXMethodDecl {{.*}} SampleCmpLevelZero 'float (hlsl::SamplerComparisonState, vector<float, [[COORD_DIM]]>, float, vector<int, [[DIM]]>)'
// SAMPLECMP-OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// SAMPLECMP-OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// SAMPLECMP-OFFSET-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// SAMPLECMP-OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// SAMPLECMP-OFFSET-NEXT: CompoundStmt
// SAMPLECMP-OFFSET-NEXT: ReturnStmt
// SAMPLECMP-OFFSET-NEXT: CStyleCastExpr {{.*}} 'float' <Dependent>
// SAMPLECMP-OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// SAMPLECMP-OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_cmp_level_zero' 'void (...) noexcept'
// SAMPLECMP-OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// SAMPLECMP-OFFSET-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// SAMPLECMP-OFFSET-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// SAMPLECMP-OFFSET-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// SAMPLECMP-OFFSET-SAME: ' lvalue .__handle
// SAMPLECMP-OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// SAMPLECMP-OFFSET-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// SAMPLECMP-OFFSET-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// SAMPLECMP-OFFSET-SAME: ' lvalue .__handle
// SAMPLECMP-OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// SAMPLECMP-OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// SAMPLECMP-OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// SAMPLECMP-OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// SAMPLECMP-OFFSET-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} CalculateLevelOfDetail 'float (hlsl::SamplerState, vector<float, [[DIM]]>)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[DIM]]>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'float' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_calculate_lod' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME: {{\[\[}}hlsl::resource_class("SRV"){{\]\]}}[[IS_ARRAY]] {{\[\[}}hlsl::contained_type(vector<element_type, element_count>){{\]\]}}
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
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
// CHECK-SAME: {{\[\[}}hlsl::resource_class("SRV"){{\]\]}}[[IS_ARRAY]] {{\[\[}}hlsl::contained_type(vector<element_type, element_count>){{\]\]}}
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
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
// GETDIM-XY-NEXT: CallExpr {{.*}}
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} '__builtin_hlsl_resource_getdimensions_xy' 'void (__hlsl_resource_t, unsigned int &, unsigned int &) noexcept'
// GETDIM-XY-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// GETDIM-XY-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// GETDIM-XY-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// GETDIM-XY-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// GETDIM-XY-SAME: ' lvalue .__handle
// GETDIM-XY-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'width' 'unsigned int &__restrict'
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'height' 'unsigned int &__restrict'

// GETDIM-XY: CXXMethodDecl {{.*}} GetDimensions 'void (unsigned int, out unsigned int, out unsigned int, out unsigned int)'
// GETDIM-XY-NEXT: ParmVarDecl {{.*}} mipLevel 'unsigned int'
// GETDIM-XY-NEXT: ParmVarDecl {{.*}} width 'unsigned int &__restrict'
// GETDIM-XY-NEXT: HLSLParamModifierAttr {{.*}} out
// GETDIM-XY-NEXT: ParmVarDecl {{.*}} height 'unsigned int &__restrict'
// GETDIM-XY-NEXT: HLSLParamModifierAttr {{.*}} out
// GETDIM-XY-NEXT: ParmVarDecl {{.*}} numberOfLevels 'unsigned int &__restrict'
// GETDIM-XY-NEXT: HLSLParamModifierAttr {{.*}} out
// GETDIM-XY-NEXT: CompoundStmt
// GETDIM-XY-NEXT: CallExpr {{.*}}
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} '__builtin_hlsl_resource_getdimensions_levels_xy' 'void (__hlsl_resource_t, unsigned int, unsigned int &, unsigned int &, unsigned int &) noexcept'
// GETDIM-XY-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// GETDIM-XY-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// GETDIM-XY-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// GETDIM-XY-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// GETDIM-XY-SAME: ' lvalue .__handle
// GETDIM-XY-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'mipLevel' 'unsigned int'
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'width' 'unsigned int &__restrict'
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'height' 'unsigned int &__restrict'
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'numberOfLevels' 'unsigned int &__restrict'

// GETDIM-XY: CXXMethodDecl {{.*}} GetDimensions 'void (out float, out float)'
// GETDIM-XY-NEXT: ParmVarDecl {{.*}} width 'float &__restrict'
// GETDIM-XY-NEXT: HLSLParamModifierAttr {{.*}} out
// GETDIM-XY-NEXT: ParmVarDecl {{.*}} height 'float &__restrict'
// GETDIM-XY-NEXT: HLSLParamModifierAttr {{.*}} out
// GETDIM-XY-NEXT: CompoundStmt
// GETDIM-XY-NEXT: CallExpr {{.*}}
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} '__builtin_hlsl_resource_getdimensions_xy_float' 'void (__hlsl_resource_t, float &, float &) noexcept'
// GETDIM-XY-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// GETDIM-XY-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// GETDIM-XY-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// GETDIM-XY-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// GETDIM-XY-SAME: ' lvalue .__handle
// GETDIM-XY-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'width' 'float &__restrict'
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'height' 'float &__restrict'

// GETDIM-XY: CXXMethodDecl {{.*}} GetDimensions 'void (unsigned int, out float, out float, out float)'
// GETDIM-XY-NEXT: ParmVarDecl {{.*}} mipLevel 'unsigned int'
// GETDIM-XY-NEXT: ParmVarDecl {{.*}} width 'float &__restrict'
// GETDIM-XY-NEXT: HLSLParamModifierAttr {{.*}} out
// GETDIM-XY-NEXT: ParmVarDecl {{.*}} height 'float &__restrict'
// GETDIM-XY-NEXT: HLSLParamModifierAttr {{.*}} out
// GETDIM-XY-NEXT: ParmVarDecl {{.*}} numberOfLevels 'float &__restrict'
// GETDIM-XY-NEXT: HLSLParamModifierAttr {{.*}} out
// GETDIM-XY-NEXT: CompoundStmt
// GETDIM-XY-NEXT: CallExpr {{.*}}
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} '__builtin_hlsl_resource_getdimensions_levels_xy_float' 'void (__hlsl_resource_t, unsigned int, float &, float &, float &) noexcept'
// GETDIM-XY-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// GETDIM-XY-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// GETDIM-XY-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// GETDIM-XY-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// GETDIM-XY-SAME: ' lvalue .__handle
// GETDIM-XY-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'mipLevel' 'unsigned int'
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'width' 'float &__restrict'
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'height' 'float &__restrict'
// GETDIM-XY-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'numberOfLevels' 'float &__restrict'

// GATHER: CXXMethodDecl {{.*}} Gather 'vector<element_type, 4> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>)' inline
// GATHER-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// GATHER-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// GATHER-NEXT: CompoundStmt
// GATHER-NEXT: ReturnStmt
// GATHER-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, 4>' <Dependent>
// GATHER-NEXT: CallExpr {{.*}} '<dependent type>'
// GATHER-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather' 'void (...) noexcept'
// GATHER-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// GATHER-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// GATHER-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// GATHER-NEXT: IntegerLiteral {{.*}} 'unsigned int' 0
// GATHER-NEXT: AlwaysInlineAttr

// GATHER-OFFSET: CXXMethodDecl {{.*}} Gather 'vector<element_type, 4> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, vector<int, [[DIM]]>)' inline
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// GATHER-OFFSET-NEXT: CompoundStmt
// GATHER-OFFSET-NEXT: ReturnStmt
// GATHER-OFFSET-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, 4>' <Dependent>
// GATHER-OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather' 'void (...) noexcept'
// GATHER-OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// GATHER-OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// GATHER-OFFSET-NEXT: IntegerLiteral {{.*}} 'unsigned int' 0
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// GATHER-OFFSET-NEXT: AlwaysInlineAttr

// GATHER: CXXMethodDecl {{.*}} GatherRed 'vector<element_type, 4> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>)' inline
// GATHER-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// GATHER-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// GATHER-NEXT: CompoundStmt
// GATHER-NEXT: ReturnStmt
// GATHER-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, 4>' <Dependent>
// GATHER-NEXT: CallExpr {{.*}} '<dependent type>'
// GATHER-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather' 'void (...) noexcept'
// GATHER-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// GATHER-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// GATHER-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// GATHER-NEXT: IntegerLiteral {{.*}} 'unsigned int' 0
// GATHER-NEXT: AlwaysInlineAttr

// GATHER-OFFSET: CXXMethodDecl {{.*}} GatherRed 'vector<element_type, 4> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, vector<int, [[DIM]]>)' inline
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// GATHER-OFFSET-NEXT: CompoundStmt
// GATHER-OFFSET-NEXT: ReturnStmt
// GATHER-OFFSET-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, 4>' <Dependent>
// GATHER-OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather' 'void (...) noexcept'
// GATHER-OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// GATHER-OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// GATHER-OFFSET-NEXT: IntegerLiteral {{.*}} 'unsigned int' 0
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// GATHER-OFFSET-NEXT: AlwaysInlineAttr

// GATHER: CXXMethodDecl {{.*}} GatherGreen 'vector<element_type, 4> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>)' inline
// GATHER-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// GATHER-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// GATHER-NEXT: CompoundStmt
// GATHER-NEXT: ReturnStmt
// GATHER-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, 4>' <Dependent>
// GATHER-NEXT: CallExpr {{.*}} '<dependent type>'
// GATHER-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather' 'void (...) noexcept'
// GATHER-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// GATHER-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// GATHER-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// GATHER-NEXT: IntegerLiteral {{.*}} 'unsigned int' 1
// GATHER-NEXT: AlwaysInlineAttr

// GATHER-OFFSET: CXXMethodDecl {{.*}} GatherGreen 'vector<element_type, 4> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, vector<int, [[DIM]]>)' inline
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// GATHER-OFFSET-NEXT: CompoundStmt
// GATHER-OFFSET-NEXT: ReturnStmt
// GATHER-OFFSET-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, 4>' <Dependent>
// GATHER-OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather' 'void (...) noexcept'
// GATHER-OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// GATHER-OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// GATHER-OFFSET-NEXT: IntegerLiteral {{.*}} 'unsigned int' 1
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// GATHER-OFFSET-NEXT: AlwaysInlineAttr

// GATHER: CXXMethodDecl {{.*}} GatherBlue 'vector<element_type, 4> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>)' inline
// GATHER-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// GATHER-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// GATHER-NEXT: CompoundStmt
// GATHER-NEXT: ReturnStmt
// GATHER-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, 4>' <Dependent>
// GATHER-NEXT: CallExpr {{.*}} '<dependent type>'
// GATHER-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather' 'void (...) noexcept'
// GATHER-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// GATHER-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// GATHER-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// GATHER-NEXT: IntegerLiteral {{.*}} 'unsigned int' 2
// GATHER-NEXT: AlwaysInlineAttr

// GATHER-OFFSET: CXXMethodDecl {{.*}} GatherBlue 'vector<element_type, 4> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, vector<int, [[DIM]]>)' inline
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// GATHER-OFFSET-NEXT: CompoundStmt
// GATHER-OFFSET-NEXT: ReturnStmt
// GATHER-OFFSET-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, 4>' <Dependent>
// GATHER-OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather' 'void (...) noexcept'
// GATHER-OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// GATHER-OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// GATHER-OFFSET-NEXT: IntegerLiteral {{.*}} 'unsigned int' 2
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// GATHER-OFFSET-NEXT: AlwaysInlineAttr

// GATHER: CXXMethodDecl {{.*}} GatherAlpha 'vector<element_type, 4> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>)' inline
// GATHER-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// GATHER-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// GATHER-NEXT: CompoundStmt
// GATHER-NEXT: ReturnStmt
// GATHER-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, 4>' <Dependent>
// GATHER-NEXT: CallExpr {{.*}} '<dependent type>'
// GATHER-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather' 'void (...) noexcept'
// GATHER-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// GATHER-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// GATHER-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// GATHER-NEXT: IntegerLiteral {{.*}} 'unsigned int' 3
// GATHER-NEXT: AlwaysInlineAttr

// GATHER-OFFSET: CXXMethodDecl {{.*}} GatherAlpha 'vector<element_type, 4> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, vector<int, [[DIM]]>)' inline
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// GATHER-OFFSET-NEXT: CompoundStmt
// GATHER-OFFSET-NEXT: ReturnStmt
// GATHER-OFFSET-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, 4>' <Dependent>
// GATHER-OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather' 'void (...) noexcept'
// GATHER-OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// GATHER-OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// GATHER-OFFSET-NEXT: IntegerLiteral {{.*}} 'unsigned int' 3
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// GATHER-OFFSET-NEXT: AlwaysInlineAttr

// GATHER: CXXMethodDecl {{.*}} GatherCmp 'vector<float, 4> (hlsl::SamplerComparisonState, vector<float, [[COORD_DIM]]>, float)' inline
// GATHER-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// GATHER-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// GATHER-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// GATHER-NEXT: CompoundStmt
// GATHER-NEXT: ReturnStmt
// GATHER-NEXT: CStyleCastExpr {{.*}} 'vector<float, 4>' <Dependent>
// GATHER-NEXT: CallExpr {{.*}} '<dependent type>'
// GATHER-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather_cmp' 'void (...) noexcept'
// GATHER-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// GATHER-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// GATHER-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// GATHER-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// GATHER-NEXT: IntegerLiteral {{.*}} 'unsigned int' 0
// GATHER-NEXT: AlwaysInlineAttr

// GATHER-OFFSET: CXXMethodDecl {{.*}} GatherCmp 'vector<float, 4> (hlsl::SamplerComparisonState, vector<float, [[COORD_DIM]]>, float, vector<int, [[DIM]]>)' inline
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// GATHER-OFFSET-NEXT: CompoundStmt
// GATHER-OFFSET-NEXT: ReturnStmt
// GATHER-OFFSET-NEXT: CStyleCastExpr {{.*}} 'vector<float, 4>' <Dependent>
// GATHER-OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather_cmp' 'void (...) noexcept'
// GATHER-OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// GATHER-OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// GATHER-OFFSET-NEXT: IntegerLiteral {{.*}} 'unsigned int' 0
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// GATHER-OFFSET-NEXT: AlwaysInlineAttr

// GATHER: CXXMethodDecl {{.*}} GatherCmpRed 'vector<float, 4> (hlsl::SamplerComparisonState, vector<float, [[COORD_DIM]]>, float)' inline
// GATHER-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// GATHER-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// GATHER-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// GATHER-NEXT: CompoundStmt
// GATHER-NEXT: ReturnStmt
// GATHER-NEXT: CStyleCastExpr {{.*}} 'vector<float, 4>' <Dependent>
// GATHER-NEXT: CallExpr {{.*}} '<dependent type>'
// GATHER-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather_cmp' 'void (...) noexcept'
// GATHER-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// GATHER-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// GATHER-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// GATHER-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// GATHER-NEXT: IntegerLiteral {{.*}} 'unsigned int' 0
// GATHER-NEXT: AlwaysInlineAttr

// GATHER: CXXMethodDecl {{.*}} GatherCmpGreen 'vector<float, 4> (hlsl::SamplerComparisonState, vector<float, [[COORD_DIM]]>, float)' inline
// GATHER-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// GATHER-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// GATHER-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// GATHER-NEXT: CompoundStmt
// GATHER-NEXT: ReturnStmt
// GATHER-NEXT: CStyleCastExpr {{.*}} 'vector<float, 4>' <Dependent>
// GATHER-NEXT: CallExpr {{.*}} '<dependent type>'
// GATHER-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather_cmp' 'void (...) noexcept'
// GATHER-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// GATHER-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// GATHER-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// GATHER-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// GATHER-NEXT: IntegerLiteral {{.*}} 'unsigned int' 1
// GATHER-NEXT: AlwaysInlineAttr

// GATHER: CXXMethodDecl {{.*}} GatherCmpBlue 'vector<float, 4> (hlsl::SamplerComparisonState, vector<float, [[COORD_DIM]]>, float)' inline
// GATHER-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// GATHER-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// GATHER-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// GATHER-NEXT: CompoundStmt
// GATHER-NEXT: ReturnStmt
// GATHER-NEXT: CStyleCastExpr {{.*}} 'vector<float, 4>' <Dependent>
// GATHER-NEXT: CallExpr {{.*}} '<dependent type>'
// GATHER-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather_cmp' 'void (...) noexcept'
// GATHER-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// GATHER-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// GATHER-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// GATHER-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// GATHER-NEXT: IntegerLiteral {{.*}} 'unsigned int' 2
// GATHER-NEXT: AlwaysInlineAttr

// GATHER-OFFSET: CXXMethodDecl {{.*}} GatherCmpAlpha 'vector<float, 4> (hlsl::SamplerComparisonState, vector<float, [[COORD_DIM]]>, float, vector<int, [[DIM]]>)' inline
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// GATHER-OFFSET-NEXT: CompoundStmt
// GATHER-OFFSET-NEXT: ReturnStmt
// GATHER-OFFSET-NEXT: CStyleCastExpr {{.*}} 'vector<float, 4>' <Dependent>
// GATHER-OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather_cmp' 'void (...) noexcept'
// GATHER-OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// GATHER-OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// GATHER-OFFSET-NEXT: IntegerLiteral {{.*}} 'unsigned int' 3
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// GATHER-OFFSET-NEXT: AlwaysInlineAttr

TEXTURE<float4> t;
SamplerState s;
SamplerComparisonState scs;

void main(COORD_TYPE loc, float cmp) {
  t.Sample(s, loc);
  t.SampleBias(s, loc, 0.0);
  t.SampleGrad(s, loc, (GRAD_TYPE)0, (GRAD_TYPE)0);
  t.SampleLevel(s, loc, 0.0);
#ifdef HAS_SAMPLE_CMP
  t.SampleCmp(scs, loc, cmp);
  t.SampleCmpLevelZero(scs, loc, cmp);
#endif
  t.CalculateLevelOfDetail(s, LOD_LOCATION);
  t.CalculateLevelOfDetailUnclamped(s, LOD_LOCATION);
#ifdef HAS_GATHER
  t.Gather(s, loc);
#endif

#ifdef HAS_OFFSET
  t.Sample(s, loc, OFFSET_ARG);
  t.Sample(s, loc, OFFSET_ARG, 1.0);
  t.SampleBias(s, loc, 0.0, OFFSET_ARG);
  t.SampleBias(s, loc, 0.0, OFFSET_ARG, 1.0);
  t.SampleGrad(s, loc, (GRAD_TYPE)0, (GRAD_TYPE)0, OFFSET_ARG);
  t.SampleGrad(s, loc, (GRAD_TYPE)0, (GRAD_TYPE)0, OFFSET_ARG, 1.0);
  t.SampleLevel(s, loc, 0.0, OFFSET_ARG);
#ifdef HAS_SAMPLE_CMP
  t.SampleCmp(scs, loc, cmp, OFFSET_ARG);
  t.SampleCmp(scs, loc, cmp, OFFSET_ARG, 1.0f);
  t.SampleCmpLevelZero(scs, loc, cmp, OFFSET_ARG);
#endif
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
