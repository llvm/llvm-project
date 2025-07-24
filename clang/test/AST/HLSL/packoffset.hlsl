// RUN: %clang_cc1 -triple dxil-unknown-shadermodel6.3-library -S -finclude-default-header -fnative-half-type -ast-dump  -x hlsl %s | FileCheck %s


cbuffer A
{
    float4 A1 : packoffset(c);
    float A2 : packoffset(c1);
    float A3 : packoffset(c1.y);
}

// CHECK: |-HLSLBufferDecl {{.*}} cbuffer A
// CHECK-NEXT: | |-attrDetails: HLSLResourceClassAttr {{.*}} <<invalid sloc>> Implicit CBuffer
// CHECK-NEXT: | |-attrDetails: HLSLResourceBindingAttr {{.*}} <<invalid sloc>> Implicit "" "0"
// CHECK-NEXT: | |-VarDecl {{.*}} A1 'hlsl_constant float4':'vector<float hlsl_constant, 4>'
// CHECK-NEXT: | | |-qualTypeDetail: QualType {{.*}} 'hlsl_constant float4' hlsl_constant
// CHECK-NEXT: | | | `-typeDetails: ElaboratedType {{.*}} 'float4' sugar
// CHECK-NEXT: | | |   `-typeDetails: TypedefType {{.*}} 'hlsl::float4' sugar
// CHECK-NEXT: | | |     |-Typedef {{.*}} 'float4'
// CHECK-NEXT: | | |     `-typeDetails: ElaboratedType {{.*}} 'vector<float, 4>' sugar
// CHECK-NEXT: | | |       `-typeDetails: TemplateSpecializationType {{.*}} 'vector<float, 4>' sugar alias
// CHECK-NEXT: | | |         |-name: 'vector':'hlsl::vector' qualified
// CHECK-NEXT: | | |         | `-TypeAliasTemplateDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit vector
// CHECK-NEXT: | | |         |-TemplateArgument type 'float'
// CHECK-NEXT: | | |         | `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK-NEXT: | | |         |-TemplateArgument expr '4'
// CHECK-NEXT: | | |         | `-ConstantExpr {{.*}} 'int'
// CHECK-NEXT: | | |         |   |-value: Int 4
// CHECK-NEXT: | | |         |   `-IntegerLiteral {{.*}} 'int' 4
// CHECK-NEXT: | | |         `-typeDetails: ExtVectorType {{.*}} 'vector<float, 4>' 4
// CHECK-NEXT: | | |           `-typeDetails: SubstTemplateTypeParmType {{.*}} 'float' sugar class depth 0 index 0 element final
// CHECK-NEXT: | | |             |-TypeAliasTemplate {{.*}} 'vector'
// CHECK-NEXT: | | |             `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK-NEXT: | | `-attrDetails: HLSLPackOffsetAttr {{.*}} 0 0
// CHECK-NEXT: | |-VarDecl {{.*}} A2 'hlsl_constant float'
// CHECK-NEXT: | | |-qualTypeDetail: QualType {{.*}} 'hlsl_constant float' hlsl_constant
// CHECK-NEXT: | | | `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK-NEXT: | | `-attrDetails: HLSLPackOffsetAttr {{.*}} 1 0
// CHECK-NEXT: | |-VarDecl {{.*}} A3 'hlsl_constant float'
// CHECK-NEXT: | | |-qualTypeDetail: QualType {{.*}} 'hlsl_constant float' hlsl_constant
// CHECK-NEXT: | | | `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK-NEXT: | | `-attrDetails: HLSLPackOffsetAttr {{.*}} 1 1
// CHECK-NEXT: | `-CXXRecordDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit struct __cblayout_A definition
// CHECK-NEXT: |   |-DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
// CHECK-NEXT: |   | |-DefaultConstructor exists trivial needs_implicit
// CHECK-NEXT: |   | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: |   | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT: |   | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: |   | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT: |   | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT: |   |-attrDetails: PackedAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |   |-FieldDecl {{.*}} <<invalid sloc>> <invalid sloc> A1 'vector<float, 4>'
// CHECK-NEXT: |   |-FieldDecl {{.*}} <<invalid sloc>> <invalid sloc> A2 'float'
// CHECK-NEXT: |   `-FieldDecl {{.*}} <<invalid sloc>> <invalid sloc> A3 'float'

cbuffer B
{
  float B0 : packoffset(c0.g);
	double B1 : packoffset(c0.b);
	half B2 : packoffset(c0.r);
}

// CHECK: |-HLSLBufferDecl {{.*}} cbuffer B
// CHECK-NEXT: | |-attrDetails: HLSLResourceClassAttr {{.*}} <<invalid sloc>> Implicit CBuffer
// CHECK-NEXT: | |-attrDetails: HLSLResourceBindingAttr {{.*}} <<invalid sloc>> Implicit "" "0"
// CHECK-NEXT: | |-VarDecl {{.*}} B0 'hlsl_constant float'
// CHECK-NEXT: | | |-qualTypeDetail: QualType {{.*}} 'hlsl_constant float' hlsl_constant
// CHECK-NEXT: | | | `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK-NEXT: | | `-attrDetails: HLSLPackOffsetAttr {{.*}} 0 1
// CHECK-NEXT: | |-VarDecl {{.*}} B1 'hlsl_constant double'
// CHECK-NEXT: | | |-qualTypeDetail: QualType {{.*}} 'hlsl_constant double' hlsl_constant
// CHECK-NEXT: | | | `-typeDetails: BuiltinType {{.*}} 'double'
// CHECK-NEXT: | | `-attrDetails: HLSLPackOffsetAttr {{.*}} 0 2
// CHECK-NEXT: | |-VarDecl {{.*}} B2 'hlsl_constant half'
// CHECK-NEXT: | | |-qualTypeDetail: QualType {{.*}} 'hlsl_constant half' hlsl_constant
// CHECK-NEXT: | | | `-typeDetails: BuiltinType {{.*}} 'half'
// CHECK-NEXT: | | `-attrDetails: HLSLPackOffsetAttr {{.*}} 0 0
// CHECK-NEXT: | `-CXXRecordDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit struct __cblayout_B definition
// CHECK-NEXT: |   |-DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
// CHECK-NEXT: |   | |-DefaultConstructor exists trivial needs_implicit
// CHECK-NEXT: |   | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: |   | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT: |   | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: |   | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT: |   | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT: |   |-attrDetails: PackedAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |   |-FieldDecl {{.*}} <<invalid sloc>> <invalid sloc> B0 'float'
// CHECK-NEXT: |   |-FieldDecl {{.*}} <<invalid sloc>> <invalid sloc> B1 'double'
// CHECK-NEXT: |   `-FieldDecl {{.*}} <<invalid sloc>> <invalid sloc> B2 'half'

cbuffer C
{
  float C0 : packoffset(c0.y);
	float2 C1 : packoffset(c0.z);
	half C2 : packoffset(c0.x);
}

// CHECK: |-HLSLBufferDecl {{.*}} cbuffer C
// CHECK-NEXT: | |-attrDetails: HLSLResourceClassAttr {{.*}} <<invalid sloc>> Implicit CBuffer
// CHECK-NEXT: | |-attrDetails: HLSLResourceBindingAttr {{.*}} <<invalid sloc>> Implicit "" "0"
// CHECK-NEXT: | |-VarDecl {{.*}} C0 'hlsl_constant float'
// CHECK-NEXT: | | |-qualTypeDetail: QualType {{.*}} 'hlsl_constant float' hlsl_constant
// CHECK-NEXT: | | | `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK-NEXT: | | `-attrDetails: HLSLPackOffsetAttr {{.*}} 0 1
// CHECK-NEXT: | |-VarDecl {{.*}} C1 'hlsl_constant float2':'vector<float hlsl_constant, 2>'
// CHECK-NEXT: | | |-qualTypeDetail: QualType {{.*}} 'hlsl_constant float2' hlsl_constant
// CHECK-NEXT: | | | `-typeDetails: ElaboratedType {{.*}} 'float2' sugar
// CHECK-NEXT: | | |   `-typeDetails: TypedefType {{.*}} 'hlsl::float2' sugar
// CHECK-NEXT: | | |     |-Typedef {{.*}} 'float2'
// CHECK-NEXT: | | |     `-typeDetails: ElaboratedType {{.*}} 'vector<float, 2>' sugar
// CHECK-NEXT: | | |       `-typeDetails: TemplateSpecializationType {{.*}} 'vector<float, 2>' sugar alias
// CHECK-NEXT: | | |         |-name: 'vector':'hlsl::vector' qualified
// CHECK-NEXT: | | |         | `-TypeAliasTemplateDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit vector
// CHECK-NEXT: | | |         |-TemplateArgument type 'float'
// CHECK-NEXT: | | |         | `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK-NEXT: | | |         |-TemplateArgument expr '2'
// CHECK-NEXT: | | |         | `-ConstantExpr {{.*}} 'int'
// CHECK-NEXT: | | |         |   |-value: Int 2
// CHECK-NEXT: | | |         |   `-IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT: | | |         `-typeDetails: ExtVectorType {{.*}} 'vector<float, 2>' 2
// CHECK-NEXT: | | |           `-typeDetails: SubstTemplateTypeParmType {{.*}} 'float' sugar class depth 0 index 0 element final
// CHECK-NEXT: | | |             |-TypeAliasTemplate {{.*}} 'vector'
// CHECK-NEXT: | | |             `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK-NEXT: | | `-attrDetails: HLSLPackOffsetAttr {{.*}} 0 2
// CHECK-NEXT: | |-VarDecl {{.*}} C2 'hlsl_constant half'
// CHECK-NEXT: | | |-qualTypeDetail: QualType {{.*}} 'hlsl_constant half' hlsl_constant
// CHECK-NEXT: | | | `-typeDetails: BuiltinType {{.*}} 'half'
// CHECK-NEXT: | | `-attrDetails: HLSLPackOffsetAttr {{.*}} 0 0
// CHECK-NEXT: | `-CXXRecordDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit struct __cblayout_C definition
// CHECK-NEXT: |   |-DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
// CHECK-NEXT: |   | |-DefaultConstructor exists trivial needs_implicit
// CHECK-NEXT: |   | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: |   | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT: |   | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: |   | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT: |   | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT: |   |-attrDetails: PackedAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |   |-FieldDecl {{.*}} <<invalid sloc>> <invalid sloc> C0 'float'
// CHECK-NEXT: |   |-FieldDecl {{.*}} <<invalid sloc>> <invalid sloc> C1 'vector<float, 2>'
// CHECK-NEXT: |   `-FieldDecl {{.*}} <<invalid sloc>> <invalid sloc> C2 'half'

cbuffer D
{
  float D0 : packoffset(c0.y);
	float D1[2] : packoffset(c1.x);
	half3 D2 : packoffset(c2.y);
	double D3 : packoffset(c0.z);
}

struct ST {
  float a;
  float2 b;
  half c;
};

// CHECK: |-HLSLBufferDecl {{.*}} cbuffer D
// CHECK-NEXT: | |-attrDetails: HLSLResourceClassAttr {{.*}} <<invalid sloc>> Implicit CBuffer
// CHECK-NEXT: | |-attrDetails: HLSLResourceBindingAttr {{.*}} <<invalid sloc>> Implicit "" "0"
// CHECK-NEXT: | |-VarDecl {{.*}} D0 'hlsl_constant float'
// CHECK-NEXT: | | |-qualTypeDetail: QualType {{.*}} 'hlsl_constant float' hlsl_constant
// CHECK-NEXT: | | | `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK-NEXT: | | `-attrDetails: HLSLPackOffsetAttr {{.*}} 0 1
// CHECK-NEXT: | |-VarDecl {{.*}} D1 'hlsl_constant float[2]'
// CHECK-NEXT: | | |-qualTypeDetail: QualType {{.*}} 'hlsl_constant float[2]' hlsl_constant
// CHECK-NEXT: | | | `-typeDetails: ConstantArrayType {{.*}} 'float[2]' 2 
// CHECK-NEXT: | | |   `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK-NEXT: | | `-attrDetails: HLSLPackOffsetAttr {{.*}} 1 0
// CHECK-NEXT: | |-VarDecl {{.*}} D2 'hlsl_constant half3':'vector<half hlsl_constant, 3>'
// CHECK-NEXT: | | |-qualTypeDetail: QualType {{.*}} 'hlsl_constant half3' hlsl_constant
// CHECK-NEXT: | | | `-typeDetails: ElaboratedType {{.*}} 'half3' sugar
// CHECK-NEXT: | | |   `-typeDetails: TypedefType {{.*}} 'hlsl::half3' sugar
// CHECK-NEXT: | | |     |-Typedef {{.*}} 'half3'
// CHECK-NEXT: | | |     `-typeDetails: ElaboratedType {{.*}} 'vector<half, 3>' sugar
// CHECK-NEXT: | | |       `-typeDetails: TemplateSpecializationType {{.*}} 'vector<half, 3>' sugar alias
// CHECK-NEXT: | | |         |-name: 'vector':'hlsl::vector' qualified
// CHECK-NEXT: | | |         | `-TypeAliasTemplateDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit vector
// CHECK-NEXT: | | |         |-TemplateArgument type 'half'
// CHECK-NEXT: | | |         | `-typeDetails: BuiltinType {{.*}} 'half'
// CHECK-NEXT: | | |         |-TemplateArgument expr '3'
// CHECK-NEXT: | | |         | `-ConstantExpr {{.*}} 'int'
// CHECK-NEXT: | | |         |   |-value: Int 3
// CHECK-NEXT: | | |         |   `-IntegerLiteral {{.*}} 'int' 3
// CHECK-NEXT: | | |         `-typeDetails: ExtVectorType {{.*}} 'vector<half, 3>' 3
// CHECK-NEXT: | | |           `-typeDetails: SubstTemplateTypeParmType {{.*}} 'half' sugar class depth 0 index 0 element final
// CHECK-NEXT: | | |             |-TypeAliasTemplate {{.*}} 'vector'
// CHECK-NEXT: | | |             `-typeDetails: BuiltinType {{.*}} 'half'
// CHECK-NEXT: | | `-attrDetails: HLSLPackOffsetAttr {{.*}} 2 1
// CHECK-NEXT: | |-VarDecl {{.*}} D3 'hlsl_constant double'
// CHECK-NEXT: | | |-qualTypeDetail: QualType {{.*}} 'hlsl_constant double' hlsl_constant
// CHECK-NEXT: | | | `-typeDetails: BuiltinType {{.*}} 'double'
// CHECK-NEXT: | | `-attrDetails: HLSLPackOffsetAttr {{.*}} 0 2
// CHECK-NEXT: | `-CXXRecordDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit struct __cblayout_D definition
// CHECK-NEXT: |   |-DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
// CHECK-NEXT: |   | |-DefaultConstructor exists trivial needs_implicit
// CHECK-NEXT: |   | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: |   | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT: |   | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: |   | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT: |   | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT: |   |-attrDetails: PackedAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |   |-FieldDecl {{.*}} <<invalid sloc>> <invalid sloc> D0 'float'
// CHECK-NEXT: |   |-FieldDecl {{.*}} <<invalid sloc>> <invalid sloc> D1 'float[2]'
// CHECK-NEXT: |   |-FieldDecl {{.*}} <<invalid sloc>> <invalid sloc> D2 'vector<half, 3>'
// CHECK-NEXT: |   `-FieldDecl {{.*}} <<invalid sloc>> <invalid sloc> D3 'double'
// CHECK-NEXT: |-CXXRecordDecl {{.*}} referenced struct ST definition
// CHECK-NEXT: | |-DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
// CHECK-NEXT: | | |-DefaultConstructor exists trivial
// CHECK-NEXT: | | |-CopyConstructor simple trivial has_const_param implicit_has_const_param
// CHECK-NEXT: | | |-MoveConstructor exists simple trivial
// CHECK-NEXT: | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: | | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT: | | `-Destructor simple irrelevant trivial
// CHECK-NEXT: | |-attrDetails: PackedAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: | |-CXXRecordDecl {{.*}} implicit struct ST
// CHECK-NEXT: | |-FieldDecl {{.*}} a 'float'
// CHECK-NEXT: | |-FieldDecl {{.*}} b 'float2':'vector<float, 2>'
// CHECK-NEXT: | |-FieldDecl {{.*}} c 'half'
// CHECK-NEXT: | |-CXXConstructorDecl {{.*}} implicit used ST 'void () noexcept' inline default trivial
// CHECK-NEXT: | | `-CompoundStmt {{.*}} 
// CHECK-NEXT: | |-CXXConstructorDecl {{.*}} implicit constexpr ST 'void (const ST &)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT: | | `-ParmVarDecl {{.*}} 'const ST &'
// CHECK-NEXT: | |   `-typeDetails: LValueReferenceType {{.*}} 'const ST &'
// CHECK-NEXT: | |     `-qualTypeDetail: QualType {{.*}} 'const ST' const
// CHECK-NEXT: | |       `-typeDetails: ElaboratedType {{.*}} 'ST' sugar
// CHECK-NEXT: | |         `-typeDetails: RecordType {{.*}} 'ST'
// CHECK-NEXT: | |           `-CXXRecord {{.*}} 'ST'
// CHECK-NEXT: | |-CXXConstructorDecl {{.*}} implicit constexpr ST 'void (ST &&)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT: | | `-ParmVarDecl {{.*}} 'ST &&'
// CHECK-NEXT: | |   `-typeDetails: RValueReferenceType {{.*}} 'ST &&'
// CHECK-NEXT: | |     `-typeDetails: ElaboratedType {{.*}} 'ST' sugar
// CHECK-NEXT: | |       `-typeDetails: RecordType {{.*}} 'ST'
// CHECK-NEXT: | |         `-CXXRecord {{.*}} 'ST'
// CHECK-NEXT: | `-CXXDestructorDecl {{.*}} implicit ~ST 'void ()' inline default trivial noexcept-unevaluated {{.*}}

cbuffer S {
  float S0 : packoffset(c0.y);
  ST S1 : packoffset(c1);
  double2 S2 : packoffset(c2);
}

struct ST2 {
  float s0;
  ST s1;
  half s2;
};

// CHECK: |-HLSLBufferDecl {{.*}} cbuffer S
// CHECK-NEXT: | |-attrDetails: HLSLResourceClassAttr {{.*}} <<invalid sloc>> Implicit CBuffer
// CHECK-NEXT: | |-attrDetails: HLSLResourceBindingAttr {{.*}} <<invalid sloc>> Implicit "" "0"
// CHECK-NEXT: | |-VarDecl {{.*}} S0 'hlsl_constant float'
// CHECK-NEXT: | | |-qualTypeDetail: QualType {{.*}} 'hlsl_constant float' hlsl_constant
// CHECK-NEXT: | | | `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK-NEXT: | | `-attrDetails: HLSLPackOffsetAttr {{.*}} 0 1
// CHECK-NEXT: | |-VarDecl {{.*}} S1 'hlsl_constant ST' callinit
// CHECK-NEXT: | | |-CXXConstructExpr {{.*}} 'ST' 'void () noexcept'
// CHECK-NEXT: | | |-qualTypeDetail: QualType {{.*}} 'hlsl_constant ST' hlsl_constant
// CHECK-NEXT: | | | `-typeDetails: ElaboratedType {{.*}} 'ST' sugar
// CHECK-NEXT: | | |   `-typeDetails: RecordType {{.*}} 'ST'
// CHECK-NEXT: | | |     `-CXXRecord {{.*}} 'ST'
// CHECK-NEXT: | | `-attrDetails: HLSLPackOffsetAttr {{.*}} 1 0
// CHECK-NEXT: | |-VarDecl {{.*}} S2 'hlsl_constant double2':'vector<double hlsl_constant, 2>'
// CHECK-NEXT: | | |-qualTypeDetail: QualType {{.*}} 'hlsl_constant double2' hlsl_constant
// CHECK-NEXT: | | | `-typeDetails: ElaboratedType {{.*}} 'double2' sugar
// CHECK-NEXT: | | |   `-typeDetails: TypedefType {{.*}} 'hlsl::double2' sugar
// CHECK-NEXT: | | |     |-Typedef {{.*}} 'double2'
// CHECK-NEXT: | | |     `-typeDetails: ElaboratedType {{.*}} 'vector<double, 2>' sugar
// CHECK-NEXT: | | |       `-typeDetails: TemplateSpecializationType {{.*}} 'vector<double, 2>' sugar alias
// CHECK-NEXT: | | |         |-name: 'vector':'hlsl::vector' qualified
// CHECK-NEXT: | | |         | `-TypeAliasTemplateDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit vector
// CHECK-NEXT: | | |         |-TemplateArgument type 'double'
// CHECK-NEXT: | | |         | `-typeDetails: BuiltinType {{.*}} 'double'
// CHECK-NEXT: | | |         |-TemplateArgument expr '2'
// CHECK-NEXT: | | |         | `-ConstantExpr {{.*}} 'int'
// CHECK-NEXT: | | |         |   |-value: Int 2
// CHECK-NEXT: | | |         |   `-IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT: | | |         `-typeDetails: ExtVectorType {{.*}} 'vector<double, 2>' 2
// CHECK-NEXT: | | |           `-typeDetails: SubstTemplateTypeParmType {{.*}} 'double' sugar class depth 0 index 0 element final
// CHECK-NEXT: | | |             |-TypeAliasTemplate {{.*}} 'vector'
// CHECK-NEXT: | | |             `-typeDetails: BuiltinType {{.*}} 'double'
// CHECK-NEXT: | | `-attrDetails: HLSLPackOffsetAttr {{.*}} 2 0
// CHECK-NEXT: | `-CXXRecordDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit struct __cblayout_S definition
// CHECK-NEXT: |   |-DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
// CHECK-NEXT: |   | |-DefaultConstructor exists trivial needs_implicit
// CHECK-NEXT: |   | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: |   | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT: |   | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: |   | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT: |   | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT: |   |-attrDetails: PackedAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |   |-FieldDecl {{.*}} <<invalid sloc>> <invalid sloc> S0 'float'
// CHECK-NEXT: |   |-FieldDecl {{.*}} <<invalid sloc>> <invalid sloc> S1 'ST'
// CHECK-NEXT: |   `-FieldDecl {{.*}} <<invalid sloc>> <invalid sloc> S2 'vector<double, 2>'
// CHECK-NEXT: |-CXXRecordDecl {{.*}} referenced struct ST2 definition
// CHECK-NEXT: | |-DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
// CHECK-NEXT: | | |-DefaultConstructor exists trivial
// CHECK-NEXT: | | |-CopyConstructor simple trivial has_const_param implicit_has_const_param
// CHECK-NEXT: | | |-MoveConstructor exists simple trivial
// CHECK-NEXT: | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: | | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT: | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT: | |-attrDetails: PackedAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: | |-CXXRecordDecl {{.*}} implicit struct ST2
// CHECK-NEXT: | |-FieldDecl {{.*}} s0 'float'
// CHECK-NEXT: | |-FieldDecl {{.*}} s1 'ST'
// CHECK-NEXT: | |-FieldDecl {{.*}} s2 'half'
// CHECK-NEXT: | |-CXXConstructorDecl {{.*}} implicit used ST2 'void () noexcept' inline default trivial
// CHECK-NEXT: | | |-CXXCtorInitializer Field {{.*}} 's1' 'ST'
// CHECK-NEXT: | | | `-CXXConstructExpr {{.*}} 'ST' 'void () noexcept'
// CHECK-NEXT: | | `-CompoundStmt {{.*}} 
// CHECK-NEXT: | |-CXXConstructorDecl {{.*}} implicit constexpr ST2 'void (const ST2 &)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT: | | `-ParmVarDecl {{.*}} 'const ST2 &'
// CHECK-NEXT: | |   `-typeDetails: LValueReferenceType {{.*}} 'const ST2 &'
// CHECK-NEXT: | |     `-qualTypeDetail: QualType {{.*}} 'const ST2' const
// CHECK-NEXT: | |       `-typeDetails: ElaboratedType {{.*}} 'ST2' sugar
// CHECK-NEXT: | |         `-typeDetails: RecordType {{.*}} 'ST2'
// CHECK-NEXT: | |           `-CXXRecord {{.*}} 'ST2'
// CHECK-NEXT: | `-CXXConstructorDecl {{.*}} implicit constexpr ST2 'void (ST2 &&)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT: |   `-ParmVarDecl {{.*}} 'ST2 &&'
// CHECK-NEXT: |     `-typeDetails: RValueReferenceType {{.*}} 'ST2 &&'
// CHECK-NEXT: |       `-typeDetails: ElaboratedType {{.*}} 'ST2' sugar
// CHECK-NEXT: |         `-typeDetails: RecordType {{.*}} 'ST2'
// CHECK-NEXT: |           `-CXXRecord {{.*}} 'ST2'

cbuffer S2 {
  float S20 : packoffset(c0.a);
  ST2 S21 : packoffset(c1);
  half S22 : packoffset(c3.y);
}

// CHECK: `-HLSLBufferDecl {{.*}} cbuffer S2
// CHECK-NEXT:   |-attrDetails: HLSLResourceClassAttr {{.*}} <<invalid sloc>> Implicit CBuffer
// CHECK-NEXT:   |-attrDetails: HLSLResourceBindingAttr {{.*}} <<invalid sloc>> Implicit "" "0"
// CHECK-NEXT:   |-VarDecl {{.*}} S20 'hlsl_constant float'
// CHECK-NEXT:   | |-qualTypeDetail: QualType {{.*}} 'hlsl_constant float' hlsl_constant
// CHECK-NEXT:   | | `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK-NEXT:   | `-attrDetails: HLSLPackOffsetAttr {{.*}} 0 3
// CHECK-NEXT:   |-VarDecl {{.*}} S21 'hlsl_constant ST2' callinit
// CHECK-NEXT:   | |-CXXConstructExpr {{.*}} 'ST2' 'void () noexcept'
// CHECK-NEXT:   | |-qualTypeDetail: QualType {{.*}} 'hlsl_constant ST2' hlsl_constant
// CHECK-NEXT:   | | `-typeDetails: ElaboratedType {{.*}} 'ST2' sugar
// CHECK-NEXT:   | |   `-typeDetails: RecordType {{.*}} 'ST2'
// CHECK-NEXT:   | |     `-CXXRecord {{.*}} 'ST2'
// CHECK-NEXT:   | `-attrDetails: HLSLPackOffsetAttr {{.*}} 1 0
// CHECK-NEXT:   |-VarDecl {{.*}} S22 'hlsl_constant half'
// CHECK-NEXT:   | |-qualTypeDetail: QualType {{.*}} 'hlsl_constant half' hlsl_constant
// CHECK-NEXT:   | | `-typeDetails: BuiltinType {{.*}} 'half'
// CHECK-NEXT:   | `-attrDetails: HLSLPackOffsetAttr {{.*}} 3 1
// CHECK-NEXT:   `-CXXRecordDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit struct __cblayout_S2 definition
// CHECK-NEXT:     |-DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
// CHECK-NEXT:     | |-DefaultConstructor exists trivial needs_implicit
// CHECK-NEXT:     | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:     | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT:     | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:     | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:     | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:     |-attrDetails: PackedAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT:     |-FieldDecl {{.*}} <<invalid sloc>> <invalid sloc> S20 'float'
// CHECK-NEXT:     |-FieldDecl {{.*}} <<invalid sloc>> <invalid sloc> S21 'ST2'
// CHECK-NEXT:     `-FieldDecl {{.*}} <<invalid sloc>> <invalid sloc> S22 'half'