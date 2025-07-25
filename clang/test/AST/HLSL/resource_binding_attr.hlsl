// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -ast-dump -o - %s | FileCheck %s

cbuffer CB : register(b3, space2) {
  float a;
}

// CHECK: |-HLSLBufferDecl {{.*}} cbuffer CB
// CHECK-NEXT: | |-attrDetails: HLSLResourceClassAttr {{.*}} <<invalid sloc>> Implicit CBuffer
// CHECK-NEXT: | |-attrDetails: HLSLResourceBindingAttr {{.*}} "b3" "space2"
// CHECK-NEXT: | |-VarDecl {{.*}} used a 'hlsl_constant float'
// CHECK-NEXT: | | `-qualTypeDetail: QualType {{.*}} 'hlsl_constant float' hlsl_constant
// CHECK-NEXT: | |   `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK-NEXT: | `-CXXRecordDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit struct __cblayout_CB definition
// CHECK-NEXT: |   |-DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
// CHECK-NEXT: |   | |-DefaultConstructor exists trivial needs_implicit
// CHECK-NEXT: |   | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: |   | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT: |   | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: |   | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT: |   | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT: |   |-attrDetails: PackedAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |   `-FieldDecl {{.*}} <<invalid sloc>> <invalid sloc> a 'float'

tbuffer TB : register(t2, space1) {
  float b;
}

// CHECK-NEXT: |-HLSLBufferDecl {{.*}} tbuffer TB
// CHECK-NEXT: | |-attrDetails: HLSLResourceClassAttr {{.*}} <<invalid sloc>> Implicit SRV
// CHECK-NEXT: | |-attrDetails: HLSLResourceBindingAttr {{.*}} "t2" "space1"
// CHECK-NEXT: | |-VarDecl {{.*}} used b 'hlsl_constant float'
// CHECK-NEXT: | | `-qualTypeDetail: QualType {{.*}} 'hlsl_constant float' hlsl_constant
// CHECK-NEXT: | |   `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK-NEXT: | `-CXXRecordDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit struct __cblayout_TB definition
// CHECK-NEXT: |   |-DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
// CHECK-NEXT: |   | |-DefaultConstructor exists trivial needs_implicit
// CHECK-NEXT: |   | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: |   | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT: |   | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: |   | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT: |   | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT: |   |-attrDetails: PackedAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |   `-FieldDecl {{.*}} <<invalid sloc>> <invalid sloc> b 'float'


export float foo() {
  return a + b;
}

// CHECK-NEXT: |-ExportDecl {{.*}} 
// CHECK-NEXT: | `-FunctionDecl {{.*}} used foo 'float ()'
// CHECK-NEXT: |   `-CompoundStmt {{.*}} 
// CHECK-NEXT: |     `-ReturnStmt {{.*}} 
// CHECK-NEXT: |       `-BinaryOperator {{.*}} 'float' '+'
// CHECK-NEXT: |         |-ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: |         | `-DeclRefExpr {{.*}} 'hlsl_constant float' lvalue Var {{.*}} 'a' 'hlsl_constant float'
// CHECK-NEXT: |         `-ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: |           `-DeclRefExpr {{.*}} 'hlsl_constant float' lvalue Var {{.*}} 'b' 'hlsl_constant float'
// CHECK-NEXT: |-LinkageSpecDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit C
// CHECK-NEXT: | `-FunctionDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit __builtin_hlsl_resource_uninitializedhandle 'void (...) noexcept' extern
// CHECK-NEXT: |   |-attrDetails: BuiltinAttr {{.*}} <<invalid sloc>> Implicit 710
// CHECK-NEXT: |   `-attrDetails: NoThrowAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |-LinkageSpecDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit C
// CHECK-NEXT: | `-FunctionDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit used __builtin_hlsl_resource_handlefrombinding 'void (...) noexcept' extern
// CHECK-NEXT: |   |-attrDetails: BuiltinAttr {{.*}} <<invalid sloc>> Implicit 708
// CHECK-NEXT: |   `-attrDetails: NoThrowAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |-LinkageSpecDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit C
// CHECK-NEXT: | `-FunctionDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit used __builtin_hlsl_resource_handlefromimplicitbinding 'void (...) noexcept' extern
// CHECK-NEXT: |   |-attrDetails: BuiltinAttr {{.*}} <<invalid sloc>> Implicit 709
// CHECK-NEXT: |   `-attrDetails: NoThrowAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |-LinkageSpecDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit C
// CHECK-NEXT: | `-FunctionDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit __builtin_hlsl_resource_getpointer 'void (...) noexcept' extern
// CHECK-NEXT: |   |-attrDetails: BuiltinAttr {{.*}} <<invalid sloc>> Implicit 707
// CHECK-NEXT: |   `-attrDetails: NoThrowAttr {{.*}} <<invalid sloc>> Implicit

RWBuffer<float> UAV : register(u3);

RWBuffer<float> UAV1 : register(u2), UAV2 : register(u4);

RWBuffer<float> UAV3 : register(space5);

// CHECK-NEXT: |-VarDecl {{.*}} UAV 'RWBuffer<float>':'hlsl::RWBuffer<float>' static callinit
// CHECK-NEXT: | |-CXXConstructExpr {{.*}} 'RWBuffer<float>':'hlsl::RWBuffer<float>' 'void (unsigned int, unsigned int, int, unsigned int, const char *)'
// CHECK-NEXT: | | |-IntegerLiteral {{.*}} <<invalid sloc>> 'unsigned int' 3
// CHECK-NEXT: | | |-IntegerLiteral {{.*}} <<invalid sloc>> 'unsigned int' 0
// CHECK-NEXT: | | |-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT: | | |-IntegerLiteral {{.*}} <<invalid sloc>> 'unsigned int' 0
// CHECK-NEXT: | | `-ImplicitCastExpr {{.*}} <<invalid sloc>> 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT: | |   `-StringLiteral {{.*}} <<invalid sloc>> 'const char[4]' lvalue "UAV"
// CHECK-NEXT: | |-typeDetails: ElaboratedType {{.*}} 'RWBuffer<float>' sugar
// CHECK-NEXT: | | `-typeDetails: TemplateSpecializationType {{.*}} 'RWBuffer<float>' sugar
// CHECK-NEXT: | |   |-name: 'RWBuffer':'hlsl::RWBuffer' qualified
// CHECK-NEXT: | |   | `-ClassTemplateDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit RWBuffer
// CHECK-NEXT: | |   |-TemplateArgument type 'float'
// CHECK-NEXT: | |   | `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK-NEXT: | |   `-typeDetails: RecordType {{.*}} 'hlsl::RWBuffer<float>'
// CHECK-NEXT: | |     `-ClassTemplateSpecialization {{.*}} 'RWBuffer'
// CHECK-NEXT: | `-attrDetails: HLSLResourceBindingAttr {{.*}} "u3" "space0"
// CHECK-NEXT: |-VarDecl {{.*}} UAV1 'RWBuffer<float>':'hlsl::RWBuffer<float>' static callinit
// CHECK-NEXT: | |-CXXConstructExpr {{.*}} 'RWBuffer<float>':'hlsl::RWBuffer<float>' 'void (unsigned int, unsigned int, int, unsigned int, const char *)'
// CHECK-NEXT: | | |-IntegerLiteral {{.*}} <<invalid sloc>> 'unsigned int' 2
// CHECK-NEXT: | | |-IntegerLiteral {{.*}} <<invalid sloc>> 'unsigned int' 0
// CHECK-NEXT: | | |-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT: | | |-IntegerLiteral {{.*}} <<invalid sloc>> 'unsigned int' 0
// CHECK-NEXT: | | `-ImplicitCastExpr {{.*}} <<invalid sloc>> 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT: | |   `-StringLiteral {{.*}} <<invalid sloc>> 'const char[5]' lvalue "UAV1"
// CHECK-NEXT: | |-typeDetails: ElaboratedType {{.*}} 'RWBuffer<float>' sugar
// CHECK-NEXT: | | `-typeDetails: TemplateSpecializationType {{.*}} 'RWBuffer<float>' sugar
// CHECK-NEXT: | |   |-name: 'RWBuffer':'hlsl::RWBuffer' qualified
// CHECK-NEXT: | |   | `-ClassTemplateDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit RWBuffer
// CHECK-NEXT: | |   |-TemplateArgument type 'float'
// CHECK-NEXT: | |   | `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK-NEXT: | |   `-typeDetails: RecordType {{.*}} 'hlsl::RWBuffer<float>'
// CHECK-NEXT: | |     `-ClassTemplateSpecialization {{.*}} 'RWBuffer'
// CHECK-NEXT: | `-attrDetails: HLSLResourceBindingAttr {{.*}} "u2" "space0"
// CHECK-NEXT: |-VarDecl {{.*}} UAV2 'RWBuffer<float>':'hlsl::RWBuffer<float>' static callinit
// CHECK-NEXT: | |-CXXConstructExpr {{.*}} 'RWBuffer<float>':'hlsl::RWBuffer<float>' 'void (unsigned int, unsigned int, int, unsigned int, const char *)'
// CHECK-NEXT: | | |-IntegerLiteral {{.*}} <<invalid sloc>> 'unsigned int' 4
// CHECK-NEXT: | | |-IntegerLiteral {{.*}} <<invalid sloc>> 'unsigned int' 0
// CHECK-NEXT: | | |-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT: | | |-IntegerLiteral {{.*}} <<invalid sloc>> 'unsigned int' 0
// CHECK-NEXT: | | `-ImplicitCastExpr {{.*}} <<invalid sloc>> 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT: | |   `-StringLiteral {{.*}} <<invalid sloc>> 'const char[5]' lvalue "UAV2"
// CHECK-NEXT: | |-typeDetails: ElaboratedType {{.*}} 'RWBuffer<float>' sugar
// CHECK-NEXT: | | `-typeDetails: TemplateSpecializationType {{.*}} 'RWBuffer<float>' sugar
// CHECK-NEXT: | |   |-name: 'RWBuffer':'hlsl::RWBuffer' qualified
// CHECK-NEXT: | |   | `-ClassTemplateDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit RWBuffer
// CHECK-NEXT: | |   |-TemplateArgument type 'float'
// CHECK-NEXT: | |   | `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK-NEXT: | |   `-typeDetails: RecordType {{.*}} 'hlsl::RWBuffer<float>'
// CHECK-NEXT: | |     `-ClassTemplateSpecialization {{.*}} 'RWBuffer'
// CHECK-NEXT: | `-attrDetails: HLSLResourceBindingAttr {{.*}} "u4" "space0"
// CHECK-NEXT: |-VarDecl {{.*}} UAV3 'RWBuffer<float>':'hlsl::RWBuffer<float>' static callinit
// CHECK-NEXT: | |-CXXConstructExpr {{.*}} 'RWBuffer<float>':'hlsl::RWBuffer<float>' 'void (unsigned int, int, unsigned int, unsigned int, const char *)'
// CHECK-NEXT: | | |-IntegerLiteral {{.*}} <<invalid sloc>> 'unsigned int' 5
// CHECK-NEXT: | | |-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT: | | |-IntegerLiteral {{.*}} <<invalid sloc>> 'unsigned int' 0
// CHECK-NEXT: | | |-IntegerLiteral {{.*}} <<invalid sloc>> 'unsigned int' 0
// CHECK-NEXT: | | `-ImplicitCastExpr {{.*}} <<invalid sloc>> 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT: | |   `-StringLiteral {{.*}} <<invalid sloc>> 'const char[5]' lvalue "UAV3"
// CHECK-NEXT: | |-typeDetails: ElaboratedType {{.*}} 'RWBuffer<float>' sugar
// CHECK-NEXT: | | `-typeDetails: TemplateSpecializationType {{.*}} 'RWBuffer<float>' sugar
// CHECK-NEXT: | |   |-name: 'RWBuffer':'hlsl::RWBuffer' qualified
// CHECK-NEXT: | |   | `-ClassTemplateDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit RWBuffer
// CHECK-NEXT: | |   |-TemplateArgument type 'float'
// CHECK-NEXT: | |   | `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK-NEXT: | |   `-typeDetails: RecordType {{.*}} 'hlsl::RWBuffer<float>'
// CHECK-NEXT: | |     `-ClassTemplateSpecialization {{.*}} 'RWBuffer'
// CHECK-NEXT: | `-attrDetails: HLSLResourceBindingAttr {{.*}} "" "space5"

//
// Default constants ($Globals) layout annotations

float f : register(c5);

int4 intv : register(c2);

double dar[5] :  register(c3);

struct S {
  int a;
};

S s : register(c10);

// CHECK-NEXT: |-VarDecl {{.*}} f 'hlsl_constant float'
// CHECK-NEXT: | |-qualTypeDetail: QualType {{.*}} 'hlsl_constant float' hlsl_constant
// CHECK-NEXT: | | `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK-NEXT: | `-attrDetails: HLSLResourceBindingAttr {{.*}} "c5" "space0"
// CHECK-NEXT: |-VarDecl {{.*}} intv 'hlsl_constant int4':'vector<int hlsl_constant, 4>'
// CHECK-NEXT: | |-qualTypeDetail: QualType {{.*}} 'hlsl_constant int4' hlsl_constant
// CHECK-NEXT: | | `-typeDetails: ElaboratedType {{.*}} 'int4' sugar
// CHECK-NEXT: | |   `-typeDetails: TypedefType {{.*}} 'hlsl::int4' sugar
// CHECK-NEXT: | |     |-Typedef {{.*}} 'int4'
// CHECK-NEXT: | |     `-typeDetails: ElaboratedType {{.*}} 'vector<int, 4>' sugar
// CHECK-NEXT: | |       `-typeDetails: TemplateSpecializationType {{.*}} 'vector<int, 4>' sugar alias
// CHECK-NEXT: | |         |-name: 'vector':'hlsl::vector' qualified
// CHECK-NEXT: | |         | `-TypeAliasTemplateDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit vector
// CHECK-NEXT: | |         |-TemplateArgument type 'int'
// CHECK-NEXT: | |         | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: | |         |-TemplateArgument expr '4'
// CHECK-NEXT: | |         | `-ConstantExpr {{.*}} 'int'
// CHECK-NEXT: | |         |   |-value: Int 4
// CHECK-NEXT: | |         |   `-IntegerLiteral {{.*}} 'int' 4
// CHECK-NEXT: | |         `-typeDetails: ExtVectorType {{.*}} 'vector<int, 4>' 4
// CHECK-NEXT: | |           `-typeDetails: SubstTemplateTypeParmType {{.*}} 'int' sugar class depth 0 index 0 element final
// CHECK-NEXT: | |             |-TypeAliasTemplate {{.*}} 'vector'
// CHECK-NEXT: | |             `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: | `-attrDetails: HLSLResourceBindingAttr {{.*}} "c2" "space0"
// CHECK-NEXT: |-VarDecl {{.*}} dar 'hlsl_constant double[5]'
// CHECK-NEXT: | |-qualTypeDetail: QualType {{.*}} 'hlsl_constant double[5]' hlsl_constant
// CHECK-NEXT: | | `-typeDetails: ConstantArrayType {{.*}} 'double[5]' 5 
// CHECK-NEXT: | |   `-typeDetails: BuiltinType {{.*}} 'double'
// CHECK-NEXT: | `-attrDetails: HLSLResourceBindingAttr {{.*}} "c3" "space0"
// CHECK-NEXT: |-CXXRecordDecl {{.*}} referenced struct S definition
// CHECK-NEXT: | |-DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
// CHECK-NEXT: | | |-DefaultConstructor exists trivial needs_implicit
// CHECK-NEXT: | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT: | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: | | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT: | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT: | |-attrDetails: PackedAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: | |-CXXRecordDecl {{.*}} implicit struct S
// CHECK-NEXT: | `-FieldDecl {{.*}} a 'int'
// CHECK-NEXT: |-VarDecl {{.*}} s 'hlsl_constant S'
// CHECK-NEXT: | |-qualTypeDetail: QualType {{.*}} 'hlsl_constant S' hlsl_constant
// CHECK-NEXT: | | `-typeDetails: ElaboratedType {{.*}} 'S' sugar
// CHECK-NEXT: | |   `-typeDetails: RecordType {{.*}} 'S'
// CHECK-NEXT: | |     `-CXXRecord {{.*}} 'S'
// CHECK-NEXT: | `-attrDetails: HLSLResourceBindingAttr {{.*}} "c10" "space0"
// CHECK-NEXT: `-HLSLBufferDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit cbuffer $Globals
// CHECK-NEXT:   |-attrDetails: HLSLResourceBindingAttr {{.*}} <<invalid sloc>> Implicit "" "0"
// CHECK-NEXT:   `-CXXRecordDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit struct __cblayout_$Globals definition
// CHECK-NEXT:     |-DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
// CHECK-NEXT:     | |-DefaultConstructor exists trivial needs_implicit
// CHECK-NEXT:     | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:     | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT:     | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:     | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:     | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:     |-attrDetails: PackedAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT:     |-FieldDecl {{.*}} <<invalid sloc>> <invalid sloc> f 'float'
// CHECK-NEXT:     |-FieldDecl {{.*}} <<invalid sloc>> <invalid sloc> intv 'vector<int, 4>'
// CHECK-NEXT:     |-FieldDecl {{.*}} <<invalid sloc>> <invalid sloc> dar 'double[5]'
// CHECK-NEXT:     `-FieldDecl {{.*}} <<invalid sloc>> <invalid sloc> s 'S'
