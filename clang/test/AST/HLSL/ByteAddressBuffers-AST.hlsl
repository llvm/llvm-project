// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump -DEMPTY \
// RUN:  -DRESOURCE=ByteAddressBuffer %s | FileCheck -DRESOURCE=ByteAddressBuffer \
// RUN:  -check-prefix=EMPTY %s
//
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -DRESOURCE=ByteAddressBuffer %s | FileCheck -DRESOURCE=ByteAddressBuffer \
// RUN:   -check-prefixes=CHECK,CHECK-SRV,CHECK-NOSUBSCRIPT %s
//
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump -DEMPTY \
// RUN:  -DRESOURCE=RWByteAddressBuffer %s | FileCheck -DRESOURCE=RWByteAddressBuffer \
// RUN:  -check-prefix=EMPTY %s
//
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -DRESOURCE=RWByteAddressBuffer %s | FileCheck -DRESOURCE=RWByteAddressBuffer \
// RUN:   -check-prefixes=CHECK,CHECK-UAV,CHECK-NOSUBSCRIPT %s
//
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump -DEMPTY \
// RUN:  -DRESOURCE=RasterizerOrderedByteAddressBuffer %s | FileCheck -DRESOURCE=RasterizerOrderedByteAddressBuffer \
// RUN:  -check-prefix=EMPTY %s
//
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -DRESOURCE=RasterizerOrderedByteAddressBuffer %s | FileCheck -DRESOURCE=RasterizerOrderedByteAddressBuffer \
// RUN:   -check-prefixes=CHECK,CHECK-UAV,CHECK-NOSUBSCRIPT %s

// EMPTY: CXXRecordDecl {{.*}} implicit <undeserialized declarations> class [[RESOURCE]]
// EMPTY: FinalAttr {{.*}}  Implicit final

// There should be no more occurrences of RESOURCE
// EMPTY-NOT: {{[^[:alnum:]]}}[[RESOURCE]]

#ifndef EMPTY

RESOURCE Buffer;

#endif

// CHECK: CXXRecordDecl {{.*}} implicit referenced <undeserialized declarations> class [[RESOURCE]] definition
// CHECK: FinalAttr {{.*}}  Implicit final
// CHECK: FieldDecl {{.*}} implicit __handle '__hlsl_resource_t
// CHECK-SRV-SAME{LITERAL}: [[hlsl::resource_class(SRV)]]
// CHECK-UAV-SAME{LITERAL}: [[hlsl::resource_class(UAV)]]
// CHECK-SAME{LITERAL}: [[hlsl::raw_buffer]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(char8_t)]]

// Default constructor

// CHECK: CXXConstructorDecl {{.*}} [[RESOURCE]] 'void ()' inline
// CHECK: CompoundStmt
// CHECK: BinaryOperator {{.*}} '='
// CHECK: MemberExpr {{.*}} lvalue .__handle
// CHECK: CXXThisExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue implicit this
// CHECK: CallExpr {{.*}} '__hlsl_resource_t
// CHECK: ImplicitCastExpr {{.*}} <BuiltinFnToFnPtr>
// CHECK: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_uninitializedhandle'
// CHECK: MemberExpr {{.*}} lvalue .__handle
// CHECK: CXXThisExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue implicit this
// CHECK: AlwaysInlineAttr

// Constructor from binding

// CHECK: CXXConstructorDecl {{.*}} [[RESOURCE]] 'void (unsigned int, unsigned int, int, unsigned int, const char *)' inline
// CHECK: ParmVarDecl {{.*}} registerNo 'unsigned int'
// CHECK: ParmVarDecl {{.*}} spaceNo 'unsigned int'
// CHECK: ParmVarDecl {{.*}} range 'int'
// CHECK: ParmVarDecl {{.*}} index 'unsigned int'
// CHECK: ParmVarDecl {{.*}} name 'const char *'
// CHECK: CompoundStmt {{.*}}
// CHECK: BinaryOperator {{.*}} '='
// CHECK: MemberExpr {{.*}} lvalue .__handle
// CHECK: CXXThisExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue implicit this
// CHECK: CallExpr {{.*}} '__hlsl_resource_t
// CHECK: ImplicitCastExpr {{.*}} <BuiltinFnToFnPtr>
// CHECK: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_handlefrombinding'
// CHECK: MemberExpr {{.*}} lvalue .__handle
// CHECK: CXXThisExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue implicit this
// CHECK: DeclRefExpr {{.*}} 'unsigned int' ParmVar {{.*}} 'registerNo' 'unsigned int'
// CHECK: DeclRefExpr {{.*}} 'unsigned int' ParmVar {{.*}} 'spaceNo' 'unsigned int'
// CHECK: DeclRefExpr {{.*}} 'int' ParmVar {{.*}} 'range' 'int'
// CHECK: DeclRefExpr {{.*}} 'unsigned int' ParmVar {{.*}} 'index' 'unsigned int'
// CHECK: DeclRefExpr {{.*}} 'const char *' ParmVar {{.*}} 'name' 'const char *'
// CHECK: AlwaysInlineAttr

// Constructor from implicit binding

// CHECK: CXXConstructorDecl {{.*}} [[RESOURCE]] 'void (unsigned int, int, unsigned int, unsigned int, const char *)' inline
// CHECK: ParmVarDecl {{.*}} spaceNo 'unsigned int'
// CHECK: ParmVarDecl {{.*}} range 'int'
// CHECK: ParmVarDecl {{.*}} index 'unsigned int'
// CHECK: ParmVarDecl {{.*}} orderId 'unsigned int'
// CHECK: ParmVarDecl {{.*}} name 'const char *'
// CHECK: CompoundStmt {{.*}}
// CHECK: BinaryOperator {{.*}} '='
// CHECK: MemberExpr {{.*}} lvalue .__handle
// CHECK: CXXThisExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue implicit this
// CHECK: CallExpr {{.*}} '__hlsl_resource_t
// CHECK: ImplicitCastExpr {{.*}} <BuiltinFnToFnPtr>
// CHECK: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_handlefromimplicitbinding'
// CHECK: MemberExpr {{.*}} lvalue .__handle
// CHECK: CXXThisExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue implicit this
// CHECK: DeclRefExpr {{.*}} 'unsigned int' ParmVar {{.*}} 'spaceNo' 'unsigned int'
// CHECK: DeclRefExpr {{.*}} 'int' ParmVar {{.*}} 'range' 'int'
// CHECK: DeclRefExpr {{.*}} 'unsigned int' ParmVar {{.*}} 'index' 'unsigned int'
// CHECK: DeclRefExpr {{.*}} 'unsigned int' ParmVar {{.*}} 'orderId' 'unsigned int'
// CHECK: DeclRefExpr {{.*}} 'const char *' ParmVar {{.*}} 'name' 'const char *'
// CHECK: AlwaysInlineAttr

// CHECK-NOSUBSCRIPT-NOT: CXXMethodDecl {{.*}} operator[] 'const element_type &(unsigned int) const'
// CHECK-NOSUBSCRIPT-NOT: CXXMethodDecl {{.*}} operator[] 'element_type &(unsigned int)'
