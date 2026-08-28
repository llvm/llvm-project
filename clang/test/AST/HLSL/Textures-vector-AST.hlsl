// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -disable-llvm-passes -finclude-default-header -DTEXTURE=Texture2D \
// RUN:   -DCOORD_TYPE=float2 -DGRAD_TYPE=float2 -DLOD_LOCATION=loc \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -o - %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK -DTEXTURE=Texture2D \
// RUN:   -DDIM_NAME=2D -DDIM=2 -DCOORD_DIM=2 -DLOAD_DIM=3 \
// RUN:   -DINDEX_TYPE="vector<unsigned int, 2>"
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -disable-llvm-passes -finclude-default-header \
// RUN:   -DTEXTURE=Texture2DArray -DCOORD_TYPE=float3 -DGRAD_TYPE=float2 \
// RUN:   -DLOD_LOCATION=loc.xy -DOFFSET_ARG="int2(1, 2)" -o - %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,ARRAY -DTEXTURE=Texture2DArray \
// RUN:   -DDIM_NAME=2D -DDIM=2 -DCOORD_DIM=3 -DLOAD_DIM=4 \
// RUN:   -DINDEX_TYPE="vector<unsigned int, 3>"

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
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
// CHECK: CXXRecordDecl {{.*}} [[TEXTURE]]
// CHECK: FinalAttr {{.*}} Implicit final
// CHECK: ClassTemplatePartialSpecializationDecl {{.*}} [[TEXTURE]] definition explicit_specialization
// CHECK: TemplateArgument type 'vector<element_type, element_count>':'vector<type-parameter-0-0, element_count>'
// CHECK: TemplateTypeParmDecl {{.*}} element_type
// CHECK: NonTypeTemplateParmDecl {{.*}} element_count
// CHECK-NEXT: FieldDecl {{.*}} implicit __handle '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}

// CHECK: CXXMethodDecl {{.*}} Load 'vector<element_type, element_count> (vector<int, [[LOAD_DIM]]>)'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<int, [[LOAD_DIM]]>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, element_count>' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_load_level' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<int, [[LOAD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<int, [[LOAD_DIM]]>'
// CHECK-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} Load 'vector<element_type, element_count> (vector<int, [[LOAD_DIM]]>, vector<int, [[DIM]]>)'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<int, [[LOAD_DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, element_count>' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_load_level' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<int, [[LOAD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<int, [[LOAD_DIM]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// CHECK-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} operator[] 'vector<element_type, element_count> const hlsl_device &([[INDEX_TYPE]]) const' inline
// CHECK-NEXT: ParmVarDecl {{.*}} Index '[[INDEX_TYPE]]'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: UnaryOperator {{.*}} 'vector<element_type, element_count> hlsl_device' lvalue prefix '*' cannot overflow
// CHECK-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, element_count> hlsl_device *' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_getpointer' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'const hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// CHECK-NEXT: DeclRefExpr {{.*}} '[[INDEX_TYPE]]' lvalue ParmVar {{.*}} 'Index' '[[INDEX_TYPE]]'
// CHECK-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} Sample 'vector<element_type, element_count> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, element_count>' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} Sample 'vector<element_type, element_count> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, vector<int, [[DIM]]>)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, element_count>' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// CHECK-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} Sample 'vector<element_type, element_count> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, vector<int, [[DIM]]>, float)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} Clamp 'float'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, element_count>' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'Clamp' 'float'
// CHECK-NEXT: AlwaysInlineAttr

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
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
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

// CHECK: CXXMethodDecl {{.*}} SampleBias 'vector<element_type, element_count> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, float, vector<int, [[DIM]]>)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} Bias 'float'
// CHECK-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, element_count>' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_bias' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'Bias' 'float'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// CHECK-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} SampleBias 'vector<element_type, element_count> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, float, vector<int, [[DIM]]>, float)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} Bias 'float'
// CHECK-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} Clamp 'float'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, element_count>' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_bias' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'Bias' 'float'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'Clamp' 'float'
// CHECK-NEXT: AlwaysInlineAttr

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
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
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

// CHECK: CXXMethodDecl {{.*}} SampleGrad 'vector<element_type, element_count> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, vector<float, [[DIM]]>, vector<float, [[DIM]]>, vector<int, [[DIM]]>)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} DDX 'vector<float, [[DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} DDY 'vector<float, [[DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, element_count>' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_grad' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
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
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// CHECK-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} SampleGrad 'vector<element_type, element_count> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, vector<float, [[DIM]]>, vector<float, [[DIM]]>, vector<int, [[DIM]]>, float)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} DDX 'vector<float, [[DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} DDY 'vector<float, [[DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} Clamp 'float'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, element_count>' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_grad' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
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
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'Clamp' 'float'
// CHECK-NEXT: AlwaysInlineAttr

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
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
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

// CHECK: CXXMethodDecl {{.*}} SampleLevel 'vector<element_type, element_count> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, float, vector<int, [[DIM]]>)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} LOD 'float'
// CHECK-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'vector<element_type, element_count>' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_level' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'LOD' 'float'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// CHECK-NEXT: AlwaysInlineAttr

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
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// CHECK-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} SampleCmp 'float (hlsl::SamplerComparisonState, vector<float, [[COORD_DIM]]>, float, vector<int, [[DIM]]>)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// CHECK-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'float' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_cmp' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// CHECK-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} SampleCmp 'float (hlsl::SamplerComparisonState, vector<float, [[COORD_DIM]]>, float, vector<int, [[DIM]]>, float)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// CHECK-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} Clamp 'float'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'float' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_cmp' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'Clamp' 'float'
// CHECK-NEXT: AlwaysInlineAttr

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
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// CHECK-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} SampleCmpLevelZero 'float (hlsl::SamplerComparisonState, vector<float, [[COORD_DIM]]>, float, vector<int, [[DIM]]>)'
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// CHECK-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CStyleCastExpr {{.*}} 'float' <Dependent>
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_sample_cmp_level_zero' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerComparisonState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerComparisonState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[COORD_DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'CompareValue' 'float'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// CHECK-NEXT: AlwaysInlineAttr

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
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
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
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("Sampler")]]
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::SamplerState' lvalue ParmVar {{.*}} 'Sampler' 'hlsl::SamplerState'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, [[DIM]]>' lvalue ParmVar {{.*}} 'Location' 'vector<float, [[DIM]]>'
// CHECK-NEXT: AlwaysInlineAttr

// CHECK: CXXMethodDecl {{.*}} GetDimensions 'void (out unsigned int, out unsigned int)'
// CHECK-NEXT: ParmVarDecl {{.*}} width 'unsigned int &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} out
// CHECK-NEXT: ParmVarDecl {{.*}} height 'unsigned int &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} out
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CallExpr {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} '__builtin_hlsl_resource_getdimensions_xy' 'void (__hlsl_resource_t, unsigned int &, unsigned int &) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'width' 'unsigned int &__restrict'
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'height' 'unsigned int &__restrict'

// CHECK: CXXMethodDecl {{.*}} GetDimensions 'void (unsigned int, out unsigned int, out unsigned int, out unsigned int)'
// CHECK-NEXT: ParmVarDecl {{.*}} mipLevel 'unsigned int'
// CHECK-NEXT: ParmVarDecl {{.*}} width 'unsigned int &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} out
// CHECK-NEXT: ParmVarDecl {{.*}} height 'unsigned int &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} out
// CHECK-NEXT: ParmVarDecl {{.*}} numberOfLevels 'unsigned int &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} out
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CallExpr {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} '__builtin_hlsl_resource_getdimensions_levels_xy' 'void (__hlsl_resource_t, unsigned int, unsigned int &, unsigned int &, unsigned int &) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'mipLevel' 'unsigned int'
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'width' 'unsigned int &__restrict'
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'height' 'unsigned int &__restrict'
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'numberOfLevels' 'unsigned int &__restrict'

// CHECK: CXXMethodDecl {{.*}} GetDimensions 'void (out float, out float)'
// CHECK-NEXT: ParmVarDecl {{.*}} width 'float &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} out
// CHECK-NEXT: ParmVarDecl {{.*}} height 'float &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} out
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CallExpr {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} '__builtin_hlsl_resource_getdimensions_xy_float' 'void (__hlsl_resource_t, float &, float &) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'width' 'float &__restrict'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'height' 'float &__restrict'

// CHECK: CXXMethodDecl {{.*}} GetDimensions 'void (unsigned int, out float, out float, out float)'
// CHECK-NEXT: ParmVarDecl {{.*}} mipLevel 'unsigned int'
// CHECK-NEXT: ParmVarDecl {{.*}} width 'float &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} out
// CHECK-NEXT: ParmVarDecl {{.*}} height 'float &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} out
// CHECK-NEXT: ParmVarDecl {{.*}} numberOfLevels 'float &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} out
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CallExpr {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} '__builtin_hlsl_resource_getdimensions_levels_xy_float' 'void (__hlsl_resource_t, unsigned int, float &, float &, float &) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class("SRV")]]
// ARRAY-SAME{LITERAL}: [[hlsl::is_array]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(vector<element_type, element_count>)]]
// CHECK-SAME: {{\[\[}}hlsl::dimension("[[DIM_NAME]]"){{\]\]}}
// CHECK-SAME: ' lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[TEXTURE]]<vector<element_type, element_count>>' lvalue implicit this
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'mipLevel' 'unsigned int'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'width' 'float &__restrict'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'height' 'float &__restrict'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'numberOfLevels' 'float &__restrict'

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

// CHECK: CXXMethodDecl {{.*}} Gather 'vector<element_type, 4> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, vector<int, [[DIM]]>)' inline
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
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
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// CHECK-NEXT: AlwaysInlineAttr

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

// CHECK: CXXMethodDecl {{.*}} GatherRed 'vector<element_type, 4> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, vector<int, [[DIM]]>)' inline
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
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
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// CHECK-NEXT: AlwaysInlineAttr

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

// CHECK: CXXMethodDecl {{.*}} GatherGreen 'vector<element_type, 4> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, vector<int, [[DIM]]>)' inline
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
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
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// CHECK-NEXT: AlwaysInlineAttr

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

// CHECK: CXXMethodDecl {{.*}} GatherBlue 'vector<element_type, 4> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, vector<int, [[DIM]]>)' inline
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
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
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// CHECK-NEXT: AlwaysInlineAttr

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

// CHECK: CXXMethodDecl {{.*}} GatherAlpha 'vector<element_type, 4> (hlsl::SamplerState, vector<float, [[COORD_DIM]]>, vector<int, [[DIM]]>)' inline
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
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
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// CHECK-NEXT: AlwaysInlineAttr

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

// CHECK: CXXMethodDecl {{.*}} GatherCmp 'vector<float, 4> (hlsl::SamplerComparisonState, vector<float, [[COORD_DIM]]>, float, vector<int, [[DIM]]>)' inline
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// CHECK-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
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
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// CHECK-NEXT: AlwaysInlineAttr

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

// CHECK: CXXMethodDecl {{.*}} GatherCmpAlpha 'vector<float, 4> (hlsl::SamplerComparisonState, vector<float, [[COORD_DIM]]>, float, vector<int, [[DIM]]>)' inline
// CHECK-NEXT: ParmVarDecl {{.*}} Sampler 'hlsl::SamplerComparisonState'
// CHECK-NEXT: ParmVarDecl {{.*}} Location 'vector<float, [[COORD_DIM]]>'
// CHECK-NEXT: ParmVarDecl {{.*}} CompareValue 'float'
// CHECK-NEXT: ParmVarDecl {{.*}} Offset 'vector<int, [[DIM]]>'
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
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned int' 3
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<int, [[DIM]]>' lvalue ParmVar {{.*}} 'Offset' 'vector<int, [[DIM]]>'
// CHECK-NEXT: AlwaysInlineAttr

TEXTURE<float4> t;
SamplerState s;
SamplerComparisonState scs;

void main(COORD_TYPE loc, float cmp) {
  t.Sample(s, loc);
  t.Sample(s, loc, OFFSET_ARG);
  t.Sample(s, loc, OFFSET_ARG, 1.0);
  t.SampleBias(s, loc, 0.0);
  t.SampleBias(s, loc, 0.0, OFFSET_ARG);
  t.SampleBias(s, loc, 0.0, OFFSET_ARG, 1.0);
  t.SampleGrad(s, loc, (GRAD_TYPE)0, (GRAD_TYPE)0);
  t.SampleGrad(s, loc, (GRAD_TYPE)0, (GRAD_TYPE)0, OFFSET_ARG);
  t.SampleGrad(s, loc, (GRAD_TYPE)0, (GRAD_TYPE)0, OFFSET_ARG, 1.0);
  t.SampleLevel(s, loc, 0.0);
  t.SampleLevel(s, loc, 0.0, OFFSET_ARG);
  t.SampleCmp(scs, loc, cmp);
  t.SampleCmp(scs, loc, cmp, OFFSET_ARG);
  t.SampleCmp(scs, loc, cmp, OFFSET_ARG, 1.0f);
  t.SampleCmpLevelZero(scs, loc, cmp);
  t.SampleCmpLevelZero(scs, loc, cmp, OFFSET_ARG);
  t.CalculateLevelOfDetail(s, LOD_LOCATION);
  t.CalculateLevelOfDetailUnclamped(s, LOD_LOCATION);
  t.Gather(s, loc);
  uint u_w, u_h, u_l;
  float f_w, f_h, f_l;
  t.GetDimensions(u_w, u_h);
  t.GetDimensions(0, u_w, u_h, u_l);
  t.GetDimensions(f_w, f_h);
  t.GetDimensions(0, f_w, f_h, f_l);
}
