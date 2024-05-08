// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -ast-dump %s | FileCheck %s 

// CHECK: NamespaceDecl {{.*}} implicit hlsl
// CHECK: CXXRecordDecl 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> <invalid sloc> implicit <undeserialized declarations> class Resource definition
// CHECK-NEXT: DefinitionData
// CHECK-NEXT: DefaultConstructor exists trivial needs_implicit
// CHECK-NEXT: CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT: CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT: Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT: FinalAttr 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> Implicit final
// CHECK-NEXT: FieldDecl 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> <invalid sloc>
// implicit h 'void *'
