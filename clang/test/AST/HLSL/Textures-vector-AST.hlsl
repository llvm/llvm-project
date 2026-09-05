// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -disable-llvm-passes -finclude-default-header -DHAS_OFFSET \
// RUN:   -DTEXTURE=Texture1D -DCOORD_TYPE=float -DGRAD_TYPE=float \
// RUN:   -DLOD_LOCATION=loc -DOFFSET_ARG="1" -o - %s | FileCheck %s \
// RUN:   --check-prefixes=CHECK,TEXEL,OFFSET -DTEXTURE=Texture1D \
// RUN:   -DDIM_NAME=1D -DLOAD_DIM=2 -DLOCATION_TYPE=float \
// RUN:   -DGRADIENT_TYPE=float -DOFFSET_TYPE=int -DINDEX_TYPE="unsigned int" \
// RUN:   -DIS_ARRAY=""
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -disable-llvm-passes -finclude-default-header -DHAS_OFFSET \
// RUN:   -DTEXTURE=Texture1DArray -DCOORD_TYPE=float2 -DGRAD_TYPE=float \
// RUN:   -DLOD_LOCATION=loc.x -DOFFSET_ARG="1" -o - %s | FileCheck %s \
// RUN:   --check-prefixes=CHECK,TEXEL,OFFSET -DTEXTURE=Texture1DArray \
// RUN:   -DDIM_NAME=1D -DLOAD_DIM=3 -DLOCATION_TYPE="vector<float, 2>" \
// RUN:   -DGRADIENT_TYPE=float -DOFFSET_TYPE=int \
// RUN:   -DINDEX_TYPE="vector<unsigned int, 2>" \
// RUN:   -DIS_ARRAY=" [[hlsl::is_array]]"
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -disable-llvm-passes -finclude-default-header -DHAS_OFFSET \
// RUN:   -DHAS_GETDIM_XY -DHAS_GATHER -DTEXTURE=Texture2D -DCOORD_TYPE=float2 \
// RUN:   -DGRAD_TYPE=float2 -DLOD_LOCATION=loc -DOFFSET_ARG="int2(1, 2)" -o - \
// RUN:   %s | FileCheck %s \
// RUN:   --check-prefixes=CHECK,TEXEL,OFFSET,GATHER,GATHER-OFFSET,GETDIM-XY \
// RUN:   -DTEXTURE=Texture2D -DDIM_NAME=2D -DLOAD_DIM=3 \
// RUN:   -DLOCATION_TYPE="vector<float, 2>" \
// RUN:   -DGRADIENT_TYPE="vector<float, 2>" -DOFFSET_TYPE="vector<int, 2>" \
// RUN:   -DINDEX_TYPE="vector<unsigned int, 2>" -DIS_ARRAY=""
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -disable-llvm-passes -finclude-default-header -DHAS_GATHER \
// RUN:   -DTEXTURE=TextureCube -DCOORD_TYPE=float3 -DGRAD_TYPE=float3 \
// RUN:   -DLOD_LOCATION=loc -o - %s | FileCheck %s \
// RUN:   --check-prefixes=CHECK,GATHER -DTEXTURE=TextureCube -DDIM_NAME=Cube \
// RUN:   -DLOCATION_TYPE="vector<float, 3>" \
// RUN:   -DGRADIENT_TYPE="vector<float, 3>" -DIS_ARRAY=""
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -disable-llvm-passes -finclude-default-header -DHAS_GATHER \
// RUN:   -DTEXTURE=TextureCubeArray -DCOORD_TYPE=float4 -DGRAD_TYPE=float3 \
// RUN:   -DLOD_LOCATION=loc.xyz -o - %s | FileCheck %s \
// RUN:   --check-prefixes=CHECK,GATHER -DTEXTURE=TextureCubeArray \
// RUN:   -DDIM_NAME=Cube -DLOCATION_TYPE="vector<float, 4>" \
// RUN:   -DGRADIENT_TYPE="vector<float, 3>" -DIS_ARRAY=" [[hlsl::is_array]]"
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -disable-llvm-passes -finclude-default-header -DHAS_OFFSET \
// RUN:   -DHAS_GETDIM_XY -DHAS_GATHER -DTEXTURE=Texture2DArray \
// RUN:   -DCOORD_TYPE=float3 -DGRAD_TYPE=float2 -DLOD_LOCATION=loc.xy \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -o - %s | FileCheck %s \
// RUN:   --check-prefixes=CHECK,ARRAY,TEXEL,OFFSET,GATHER,GATHER-OFFSET,GETDIM-XY \
// RUN:   -DTEXTURE=Texture2DArray -DDIM_NAME=2D -DLOAD_DIM=4 \
// RUN:   -DLOCATION_TYPE="vector<float, 3>" \
// RUN:   -DGRADIENT_TYPE="vector<float, 2>" -DOFFSET_TYPE="vector<int, 2>" \
// RUN:   -DINDEX_TYPE="vector<unsigned int, 3>" \
// RUN:   -DIS_ARRAY=" [[hlsl::is_array]]"

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   HAS_OFFSET         defined for types whose sampling and gathering methods
//                      have overloads taking an offset
//   HAS_GETDIM_XY      defined for types that have the width/height
//                      GetDimensions overloads
//   HAS_GATHER         defined for types that have the Gather methods
//   TEXTURE            resource type name
//   COORD_TYPE         sample location type (DIM components plus the array
//                      slice)
//   GRAD_TYPE          SampleGrad ddx/ddy type, one component per resource
//                      dimension
//   LOD_LOCATION       expression producing a LOD_TYPE location from `loc`
//   OFFSET_ARG         a literal offset argument
//   DIM_NAME           hlsl::dimension spelling
//   LOCATION_TYPE      sample location type
//   GRADIENT_TYPE      SampleGrad ddx/ddy and CalculateLevelOfDetail location
//                      type, one component per resource dimension
//   OFFSET_TYPE        offset type, one component per resource dimension
//   LOAD_DIM           Load location components (the location plus the mip
//                      level); always a vector
//   INDEX_TYPE         operator[] index type
//
// Check prefixes:
//   TEXEL              the type has integer texel addressing (Load,
//                      operator[], mips), and therefore a `mips` field in its
//                      layout
//   OFFSET             the sampling and gathering methods have offset
//                      overloads
//   GETDIM-XY          the width/height GetDimensions overloads exist
//   GATHER             the Gather/GatherCmp methods exist
//   GATHER-OFFSET      the Gather methods have offset overloads
//   IS_ARRAY           the [[hlsl::is_array]] attribute on arrayed resources,
//                      or empty

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

// TEXEL: CXXMethodDecl {{.*}} Load 'vector<element_type, element_count> (vector<int, [[LOAD_DIM]]>, [[OFFSET_TYPE]])'
// TEXEL-NEXT: ParmVarDecl {{.*}} Location 'vector<int, [[LOAD_DIM]]>'
// TEXEL-NEXT: ParmVarDecl {{.*}} Offset '[[OFFSET_TYPE]]'
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
// TEXEL-NEXT: DeclRefExpr {{.*}} '[[OFFSET_TYPE]]' lvalue ParmVar {{.*}} 'Offset' '[[OFFSET_TYPE]]'
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

// CHECK: CXXMethodDecl {{.*}} Sample 'vector<element_type, element_count> (hlsl::SamplerState, [[LOCATION_TYPE]])'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
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
// CHECK-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// CHECK-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} Sample 'vector<element_type, element_count> (hlsl::SamplerState, [[LOCATION_TYPE]], [[OFFSET_TYPE]])'
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset '[[OFFSET_TYPE]]'
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
// OFFSET-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// OFFSET-NEXT: DeclRefExpr {{.*}} '[[OFFSET_TYPE]]' lvalue ParmVar {{.*}} 'Offset' '[[OFFSET_TYPE]]'
// OFFSET-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} Sample 'vector<element_type, element_count> (hlsl::SamplerState, [[LOCATION_TYPE]], [[OFFSET_TYPE]], float)'
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset '[[OFFSET_TYPE]]'
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
// OFFSET-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// OFFSET-NEXT: DeclRefExpr {{.*}} '[[OFFSET_TYPE]]' lvalue ParmVar {{.*}} 'Offset' '[[OFFSET_TYPE]]'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'Clamp' 'float'
// OFFSET-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} SampleBias 'vector<element_type, element_count> (hlsl::SamplerState, [[LOCATION_TYPE]], float)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
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
// CHECK-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'Bias' 'float'
// CHECK-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} SampleBias 'vector<element_type, element_count> (hlsl::SamplerState, [[LOCATION_TYPE]], float, [[OFFSET_TYPE]])'
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
// OFFSET-NEXT: ParmVarDecl {{.*}} Bias 'float'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset '[[OFFSET_TYPE]]'
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
// OFFSET-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'Bias' 'float'
// OFFSET-NEXT: DeclRefExpr {{.*}} '[[OFFSET_TYPE]]' lvalue ParmVar {{.*}} 'Offset' '[[OFFSET_TYPE]]'
// OFFSET-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} SampleBias 'vector<element_type, element_count> (hlsl::SamplerState, [[LOCATION_TYPE]], float, [[OFFSET_TYPE]], float)'
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
// OFFSET-NEXT: ParmVarDecl {{.*}} Bias 'float'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset '[[OFFSET_TYPE]]'
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
// OFFSET-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'Bias' 'float'
// OFFSET-NEXT: DeclRefExpr {{.*}} '[[OFFSET_TYPE]]' lvalue ParmVar {{.*}} 'Offset' '[[OFFSET_TYPE]]'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'Clamp' 'float'
// OFFSET-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} SampleGrad 'vector<element_type, element_count> (hlsl::SamplerState, [[LOCATION_TYPE]], [[GRADIENT_TYPE]], [[GRADIENT_TYPE]])'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
// CHECK-NEXT: ParmVarDecl {{.*}} DDX '[[GRADIENT_TYPE]]'
// CHECK-NEXT: ParmVarDecl {{.*}} DDY '[[GRADIENT_TYPE]]'
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
// CHECK-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// CHECK-NEXT: DeclRefExpr {{.*}} '[[GRADIENT_TYPE]]' lvalue ParmVar {{.*}} 'DDX' '[[GRADIENT_TYPE]]'
// CHECK-NEXT: DeclRefExpr {{.*}} '[[GRADIENT_TYPE]]' lvalue ParmVar {{.*}} 'DDY' '[[GRADIENT_TYPE]]'
// CHECK-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} SampleGrad 'vector<element_type, element_count> (hlsl::SamplerState, [[LOCATION_TYPE]], [[GRADIENT_TYPE]], [[GRADIENT_TYPE]], [[OFFSET_TYPE]])'
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
// OFFSET-NEXT: ParmVarDecl {{.*}} DDX '[[GRADIENT_TYPE]]'
// OFFSET-NEXT: ParmVarDecl {{.*}} DDY '[[GRADIENT_TYPE]]'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset '[[OFFSET_TYPE]]'
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
// OFFSET-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// OFFSET-NEXT: DeclRefExpr {{.*}} '[[GRADIENT_TYPE]]' lvalue ParmVar {{.*}} 'DDX' '[[GRADIENT_TYPE]]'
// OFFSET-NEXT: DeclRefExpr {{.*}} '[[GRADIENT_TYPE]]' lvalue ParmVar {{.*}} 'DDY' '[[GRADIENT_TYPE]]'
// OFFSET-NEXT: DeclRefExpr {{.*}} '[[OFFSET_TYPE]]' lvalue ParmVar {{.*}} 'Offset' '[[OFFSET_TYPE]]'
// OFFSET-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} SampleGrad 'vector<element_type, element_count> (hlsl::SamplerState, [[LOCATION_TYPE]], [[GRADIENT_TYPE]], [[GRADIENT_TYPE]], [[OFFSET_TYPE]], float)'
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
// OFFSET-NEXT: ParmVarDecl {{.*}} DDX '[[GRADIENT_TYPE]]'
// OFFSET-NEXT: ParmVarDecl {{.*}} DDY '[[GRADIENT_TYPE]]'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset '[[OFFSET_TYPE]]'
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
// OFFSET-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// OFFSET-NEXT: DeclRefExpr {{.*}} '[[GRADIENT_TYPE]]' lvalue ParmVar {{.*}} 'DDX' '[[GRADIENT_TYPE]]'
// OFFSET-NEXT: DeclRefExpr {{.*}} '[[GRADIENT_TYPE]]' lvalue ParmVar {{.*}} 'DDY' '[[GRADIENT_TYPE]]'
// OFFSET-NEXT: DeclRefExpr {{.*}} '[[OFFSET_TYPE]]' lvalue ParmVar {{.*}} 'Offset' '[[OFFSET_TYPE]]'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'Clamp' 'float'
// OFFSET-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} SampleLevel 'vector<element_type, element_count> (hlsl::SamplerState, [[LOCATION_TYPE]], float)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
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
// CHECK-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'LOD' 'float'
// CHECK-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} SampleLevel 'vector<element_type, element_count> (hlsl::SamplerState, [[LOCATION_TYPE]], float, [[OFFSET_TYPE]])'
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
// OFFSET-NEXT: ParmVarDecl {{.*}} LOD 'float'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset '[[OFFSET_TYPE]]'
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
// OFFSET-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'LOD' 'float'
// OFFSET-NEXT: DeclRefExpr {{.*}} '[[OFFSET_TYPE]]' lvalue ParmVar {{.*}} 'Offset' '[[OFFSET_TYPE]]'
// OFFSET-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} SampleCmp 'float (hlsl::SamplerComparisonState, [[LOCATION_TYPE]], float)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
// CHECK-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'float' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_cmp' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME: {{\[\[}}hlsl::resource_class("SRV"){{\]\]}}[[IS_ARRAY]] {{\[\[}}hlsl::contained_type(vector<element_type, element_count>){{\]\]}}
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// CHECK-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// CHECK-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} SampleCmp 'float (hlsl::SamplerComparisonState, [[LOCATION_TYPE]], float, [[OFFSET_TYPE]])'
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
// OFFSET-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset '[[OFFSET_TYPE]]'
// OFFSET-NEXT: CompoundStmt
// OFFSET-NEXT: ReturnStmt
// OFFSET-NEXT: CStyleCastExpr {{.*}} 'float' <Dependent>
// OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_cmp' 'void (...) noexcept'
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
// OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// OFFSET-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// OFFSET-NEXT: DeclRefExpr {{.*}} '[[OFFSET_TYPE]]' lvalue ParmVar {{.*}} 'Offset' '[[OFFSET_TYPE]]'
// OFFSET-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} SampleCmp 'float (hlsl::SamplerComparisonState, [[LOCATION_TYPE]], float, [[OFFSET_TYPE]], float)'
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
// OFFSET-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset '[[OFFSET_TYPE]]'
// OFFSET-NEXT: ParmVarDecl {{.*}} Clamp 'float'
// OFFSET-NEXT: CompoundStmt
// OFFSET-NEXT: ReturnStmt
// OFFSET-NEXT: CStyleCastExpr {{.*}} 'float' <Dependent>
// OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_cmp' 'void (...) noexcept'
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
// OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// OFFSET-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// OFFSET-NEXT: DeclRefExpr {{.*}} '[[OFFSET_TYPE]]' lvalue ParmVar {{.*}} 'Offset' '[[OFFSET_TYPE]]'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'Clamp' 'float'
// OFFSET-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} SampleCmpLevelZero 'float (hlsl::SamplerComparisonState, [[LOCATION_TYPE]], float)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
// CHECK-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'float' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_cmp_level_zero' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME: {{\[\[}}hlsl::resource_class("SRV"){{\]\]}}[[IS_ARRAY]] {{\[\[}}hlsl::contained_type(vector<element_type, element_count>){{\]\]}}
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// CHECK-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// CHECK-NEXT: AlwaysInlineAttr

// OFFSET: CXXMethodDecl {{.*}} SampleCmpLevelZero 'float (hlsl::SamplerComparisonState, [[LOCATION_TYPE]], float, [[OFFSET_TYPE]])'
// OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// OFFSET-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
// OFFSET-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// OFFSET-NEXT: ParmVarDecl {{.*}} Offset '[[OFFSET_TYPE]]'
// OFFSET-NEXT: CompoundStmt
// OFFSET-NEXT: ReturnStmt
// OFFSET-NEXT: CStyleCastExpr {{.*}} 'float' <Dependent>
// OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_cmp_level_zero' 'void (...) noexcept'
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
// OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// OFFSET-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// OFFSET-NEXT: DeclRefExpr {{.*}} '[[OFFSET_TYPE]]' lvalue ParmVar {{.*}} 'Offset' '[[OFFSET_TYPE]]'
// OFFSET-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} CalculateLevelOfDetail 'float (hlsl::SamplerState, [[GRADIENT_TYPE]])'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location '[[GRADIENT_TYPE]]'
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
// CHECK-NEXT: DeclRefExpr {{.*}} '[[GRADIENT_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[GRADIENT_TYPE]]'
// CHECK-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} CalculateLevelOfDetailUnclamped 'float (hlsl::SamplerState, [[GRADIENT_TYPE]])'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location '[[GRADIENT_TYPE]]'
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
// CHECK-NEXT: DeclRefExpr {{.*}} '[[GRADIENT_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[GRADIENT_TYPE]]'
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

// GATHER: CXXMethodDecl {{.*}} Gather 'vector<element_type, 4> (hlsl::SamplerState, [[LOCATION_TYPE]])' inline
// GATHER-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// GATHER-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
// GATHER-NEXT: CompoundStmt
// GATHER-NEXT: ReturnStmt
// GATHER-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, 4>' <Dependent>
// GATHER-NEXT: CallExpr {{.*}} '<dependent type>'
// GATHER-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather' 'void (...) noexcept'
// GATHER-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// GATHER-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// GATHER-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// GATHER-NEXT: IntegerLiteral {{.*}} 'unsigned int' 0
// GATHER-NEXT: AlwaysInlineAttr

// GATHER-OFFSET: CXXMethodDecl {{.*}} Gather 'vector<element_type, 4> (hlsl::SamplerState, [[LOCATION_TYPE]], [[OFFSET_TYPE]])' inline
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Offset '[[OFFSET_TYPE]]'
// GATHER-OFFSET-NEXT: CompoundStmt
// GATHER-OFFSET-NEXT: ReturnStmt
// GATHER-OFFSET-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, 4>' <Dependent>
// GATHER-OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather' 'void (...) noexcept'
// GATHER-OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// GATHER-OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// GATHER-OFFSET-NEXT: IntegerLiteral {{.*}} 'unsigned int' 0
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} '[[OFFSET_TYPE]]' lvalue ParmVar {{.*}} 'Offset' '[[OFFSET_TYPE]]'
// GATHER-OFFSET-NEXT: AlwaysInlineAttr

// GATHER: CXXMethodDecl {{.*}} GatherRed 'vector<element_type, 4> (hlsl::SamplerState, [[LOCATION_TYPE]])' inline
// GATHER-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// GATHER-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
// GATHER-NEXT: CompoundStmt
// GATHER-NEXT: ReturnStmt
// GATHER-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, 4>' <Dependent>
// GATHER-NEXT: CallExpr {{.*}} '<dependent type>'
// GATHER-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather' 'void (...) noexcept'
// GATHER-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// GATHER-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// GATHER-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// GATHER-NEXT: IntegerLiteral {{.*}} 'unsigned int' 0
// GATHER-NEXT: AlwaysInlineAttr

// GATHER-OFFSET: CXXMethodDecl {{.*}} GatherRed 'vector<element_type, 4> (hlsl::SamplerState, [[LOCATION_TYPE]], [[OFFSET_TYPE]])' inline
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Offset '[[OFFSET_TYPE]]'
// GATHER-OFFSET-NEXT: CompoundStmt
// GATHER-OFFSET-NEXT: ReturnStmt
// GATHER-OFFSET-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, 4>' <Dependent>
// GATHER-OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather' 'void (...) noexcept'
// GATHER-OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// GATHER-OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// GATHER-OFFSET-NEXT: IntegerLiteral {{.*}} 'unsigned int' 0
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} '[[OFFSET_TYPE]]' lvalue ParmVar {{.*}} 'Offset' '[[OFFSET_TYPE]]'
// GATHER-OFFSET-NEXT: AlwaysInlineAttr

// GATHER: CXXMethodDecl {{.*}} GatherGreen 'vector<element_type, 4> (hlsl::SamplerState, [[LOCATION_TYPE]])' inline
// GATHER-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// GATHER-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
// GATHER-NEXT: CompoundStmt
// GATHER-NEXT: ReturnStmt
// GATHER-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, 4>' <Dependent>
// GATHER-NEXT: CallExpr {{.*}} '<dependent type>'
// GATHER-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather' 'void (...) noexcept'
// GATHER-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// GATHER-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// GATHER-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// GATHER-NEXT: IntegerLiteral {{.*}} 'unsigned int' 1
// GATHER-NEXT: AlwaysInlineAttr

// GATHER-OFFSET: CXXMethodDecl {{.*}} GatherGreen 'vector<element_type, 4> (hlsl::SamplerState, [[LOCATION_TYPE]], [[OFFSET_TYPE]])' inline
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Offset '[[OFFSET_TYPE]]'
// GATHER-OFFSET-NEXT: CompoundStmt
// GATHER-OFFSET-NEXT: ReturnStmt
// GATHER-OFFSET-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, 4>' <Dependent>
// GATHER-OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather' 'void (...) noexcept'
// GATHER-OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// GATHER-OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// GATHER-OFFSET-NEXT: IntegerLiteral {{.*}} 'unsigned int' 1
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} '[[OFFSET_TYPE]]' lvalue ParmVar {{.*}} 'Offset' '[[OFFSET_TYPE]]'
// GATHER-OFFSET-NEXT: AlwaysInlineAttr

// GATHER: CXXMethodDecl {{.*}} GatherBlue 'vector<element_type, 4> (hlsl::SamplerState, [[LOCATION_TYPE]])' inline
// GATHER-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// GATHER-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
// GATHER-NEXT: CompoundStmt
// GATHER-NEXT: ReturnStmt
// GATHER-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, 4>' <Dependent>
// GATHER-NEXT: CallExpr {{.*}} '<dependent type>'
// GATHER-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather' 'void (...) noexcept'
// GATHER-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// GATHER-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// GATHER-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// GATHER-NEXT: IntegerLiteral {{.*}} 'unsigned int' 2
// GATHER-NEXT: AlwaysInlineAttr

// GATHER-OFFSET: CXXMethodDecl {{.*}} GatherBlue 'vector<element_type, 4> (hlsl::SamplerState, [[LOCATION_TYPE]], [[OFFSET_TYPE]])' inline
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Offset '[[OFFSET_TYPE]]'
// GATHER-OFFSET-NEXT: CompoundStmt
// GATHER-OFFSET-NEXT: ReturnStmt
// GATHER-OFFSET-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, 4>' <Dependent>
// GATHER-OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather' 'void (...) noexcept'
// GATHER-OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// GATHER-OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// GATHER-OFFSET-NEXT: IntegerLiteral {{.*}} 'unsigned int' 2
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} '[[OFFSET_TYPE]]' lvalue ParmVar {{.*}} 'Offset' '[[OFFSET_TYPE]]'
// GATHER-OFFSET-NEXT: AlwaysInlineAttr

// GATHER: CXXMethodDecl {{.*}} GatherAlpha 'vector<element_type, 4> (hlsl::SamplerState, [[LOCATION_TYPE]])' inline
// GATHER-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// GATHER-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
// GATHER-NEXT: CompoundStmt
// GATHER-NEXT: ReturnStmt
// GATHER-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, 4>' <Dependent>
// GATHER-NEXT: CallExpr {{.*}} '<dependent type>'
// GATHER-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather' 'void (...) noexcept'
// GATHER-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// GATHER-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// GATHER-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// GATHER-NEXT: IntegerLiteral {{.*}} 'unsigned int' 3
// GATHER-NEXT: AlwaysInlineAttr

// GATHER-OFFSET: CXXMethodDecl {{.*}} GatherAlpha 'vector<element_type, 4> (hlsl::SamplerState, [[LOCATION_TYPE]], [[OFFSET_TYPE]])' inline
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Offset '[[OFFSET_TYPE]]'
// GATHER-OFFSET-NEXT: CompoundStmt
// GATHER-OFFSET-NEXT: ReturnStmt
// GATHER-OFFSET-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, 4>' <Dependent>
// GATHER-OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather' 'void (...) noexcept'
// GATHER-OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// GATHER-OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// GATHER-OFFSET-NEXT: IntegerLiteral {{.*}} 'unsigned int' 3
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} '[[OFFSET_TYPE]]' lvalue ParmVar {{.*}} 'Offset' '[[OFFSET_TYPE]]'
// GATHER-OFFSET-NEXT: AlwaysInlineAttr

// GATHER: CXXMethodDecl {{.*}} GatherCmp 'vector<float, 4> (hlsl::SamplerComparisonState, [[LOCATION_TYPE]], float)' inline
// GATHER-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// GATHER-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
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
// GATHER-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// GATHER-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// GATHER-NEXT: IntegerLiteral {{.*}} 'unsigned int' 0
// GATHER-NEXT: AlwaysInlineAttr

// GATHER-OFFSET: CXXMethodDecl {{.*}} GatherCmp 'vector<float, 4> (hlsl::SamplerComparisonState, [[LOCATION_TYPE]], float, [[OFFSET_TYPE]])' inline
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Offset '[[OFFSET_TYPE]]'
// GATHER-OFFSET-NEXT: CompoundStmt
// GATHER-OFFSET-NEXT: ReturnStmt
// GATHER-OFFSET-NEXT: CStyleCastExpr {{.*}} 'vector<float, 4>' <Dependent>
// GATHER-OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather_cmp' 'void (...) noexcept'
// GATHER-OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// GATHER-OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// GATHER-OFFSET-NEXT: IntegerLiteral {{.*}} 'unsigned int' 0
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} '[[OFFSET_TYPE]]' lvalue ParmVar {{.*}} 'Offset' '[[OFFSET_TYPE]]'
// GATHER-OFFSET-NEXT: AlwaysInlineAttr

// GATHER: CXXMethodDecl {{.*}} GatherCmpRed 'vector<float, 4> (hlsl::SamplerComparisonState, [[LOCATION_TYPE]], float)' inline
// GATHER-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// GATHER-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
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
// GATHER-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// GATHER-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// GATHER-NEXT: IntegerLiteral {{.*}} 'unsigned int' 0
// GATHER-NEXT: AlwaysInlineAttr

// GATHER: CXXMethodDecl {{.*}} GatherCmpGreen 'vector<float, 4> (hlsl::SamplerComparisonState, [[LOCATION_TYPE]], float)' inline
// GATHER-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// GATHER-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
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
// GATHER-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// GATHER-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// GATHER-NEXT: IntegerLiteral {{.*}} 'unsigned int' 1
// GATHER-NEXT: AlwaysInlineAttr

// GATHER: CXXMethodDecl {{.*}} GatherCmpBlue 'vector<float, 4> (hlsl::SamplerComparisonState, [[LOCATION_TYPE]], float)' inline
// GATHER-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// GATHER-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
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
// GATHER-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// GATHER-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// GATHER-NEXT: IntegerLiteral {{.*}} 'unsigned int' 2
// GATHER-NEXT: AlwaysInlineAttr

// GATHER-OFFSET: CXXMethodDecl {{.*}} GatherCmpAlpha 'vector<float, 4> (hlsl::SamplerComparisonState, [[LOCATION_TYPE]], float, [[OFFSET_TYPE]])' inline
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Location '[[LOCATION_TYPE]]'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// GATHER-OFFSET-NEXT: ParmVarDecl {{.*}} Offset '[[OFFSET_TYPE]]'
// GATHER-OFFSET-NEXT: CompoundStmt
// GATHER-OFFSET-NEXT: ReturnStmt
// GATHER-OFFSET-NEXT: CStyleCastExpr {{.*}} 'vector<float, 4>' <Dependent>
// GATHER-OFFSET-NEXT: CallExpr {{.*}} '<dependent type>'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_gather_cmp' 'void (...) noexcept'
// GATHER-OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-OFFSET-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<{{.*}}>' lvalue implicit this
// GATHER-OFFSET-NEXT: MemberExpr {{.*}} lvalue .__handle
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} '[[LOCATION_TYPE]]' lvalue ParmVar {{.*}} 'Location' '[[LOCATION_TYPE]]'
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// GATHER-OFFSET-NEXT: IntegerLiteral {{.*}} 'unsigned int' 3
// GATHER-OFFSET-NEXT: DeclRefExpr {{.*}} '[[OFFSET_TYPE]]' lvalue ParmVar {{.*}} 'Offset' '[[OFFSET_TYPE]]'
// GATHER-OFFSET-NEXT: AlwaysInlineAttr

TEXTURE<float4> t;
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
