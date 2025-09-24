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
// EMPTY-NEXT: FinalAttr {{.*}}  Implicit final

// There should be no more occurrences of RESOURCE
// EMPTY-NOT: {{[^[:alnum:]]}}[[RESOURCE]]

#ifndef EMPTY

RESOURCE Buffer;

#endif

// CHECK: CXXRecordDecl {{.*}} implicit referenced class [[RESOURCE]] definition
// CHECK: FinalAttr {{.*}}  Implicit final
// CHECK-NEXT: FieldDecl {{.*}} implicit __handle '__hlsl_resource_t
// CHECK-SRV-SAME{LITERAL}: [[hlsl::resource_class(SRV)]]
// CHECK-UAV-SAME{LITERAL}: [[hlsl::resource_class(UAV)]]
// CHECK-SAME{LITERAL}: [[hlsl::raw_buffer]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(char8_t)]]

// Default constructor

// CHECK: CXXConstructorDecl {{.*}} [[RESOURCE]] 'void ()' inline
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: BinaryOperator {{.*}} '='
// CHECK-NEXT: MemberExpr {{.*}} lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue implicit this
// CHECK-NEXT: CallExpr {{.*}} '__hlsl_resource_t
// CHECK-NEXT: ImplicitCastExpr {{.*}} <BuiltinFnToFnPtr>
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_uninitializedhandle'
// CHECK-NEXT: MemberExpr {{.*}} lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue implicit this
// CHECK-NEXT: AlwaysInlineAttr

// Copy constructor

// CHECK: CXXConstructorDecl {{.*}} [[RESOURCE]] 'void (const hlsl::[[RESOURCE]] &)' inline
// CHECK-NEXT: ParmVarDecl {{.*}} other 'const hlsl::[[RESOURCE]] &'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: BinaryOperator {{.*}} '='
// CHECK-NEXT: MemberExpr {{.*}} lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'const hlsl::[[RESOURCE]]' ParmVar {{.*}} 'other' 'const hlsl::[[RESOURCE]] &'
// CHECK-NEXT: AlwaysInlineAttr

// operator=

// CHECK: CXXMethodDecl {{.*}} operator= 'hlsl::[[RESOURCE]] &(const hlsl::[[RESOURCE]] &)'
// CHECK-NEXT: ParmVarDecl {{.*}} other 'const hlsl::[[RESOURCE]] &'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: BinaryOperator {{.*}} '='
// CHECK-NEXT: MemberExpr {{.*}} lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'const hlsl::[[RESOURCE]]' ParmVar {{.*}} 'other' 'const hlsl::[[RESOURCE]] &'
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue implicit this
// CHECK-NEXT: AlwaysInlineAttr

// Static __createFromBinding method

// CHECK: CXXMethodDecl {{.*}} __createFromBinding 'hlsl::[[RESOURCE]] (unsigned int, unsigned int, int, unsigned int, const char *)' static
// CHECK-NEXT: ParmVarDecl {{.*}} registerNo 'unsigned int'
// CHECK-NEXT: ParmVarDecl {{.*}} spaceNo 'unsigned int'
// CHECK-NEXT: ParmVarDecl {{.*}} range 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} index 'unsigned int'
// CHECK-NEXT: ParmVarDecl {{.*}} name 'const char *'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} tmp 'hlsl::[[RESOURCE]]'
// CHECK-NEXT: BinaryOperator {{.*}} '__hlsl_resource_t {{.*}}]]' '='
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t {{.*}}' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue Var {{.*}} 'tmp' 'hlsl::[[RESOURCE]]'
// CHECK-NEXT: CallExpr {{.*}} '__hlsl_resource_t {{.*}}'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(...) noexcept' <BuiltinFnToFnPtr>
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_handlefrombinding' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t {{.*}}' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue Var {{.*}} 'tmp' 'hlsl::[[RESOURCE]]'
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int' ParmVar {{.*}} 'registerNo' 'unsigned int'
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int' ParmVar {{.*}} 'spaceNo' 'unsigned int'
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' ParmVar {{.*}} 'range' 'int'
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int' ParmVar {{.*}} 'index' 'unsigned int'
// CHECK-NEXT: DeclRefExpr {{.*}} 'const char *' ParmVar {{.*}} 'name' 'const char *'
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CXXConstructExpr {{.*}} 'hlsl::[[RESOURCE]]' 'void (const hlsl::[[RESOURCE]] &)'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const hlsl::[[RESOURCE]]' xvalue <NoOp>
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue Var {{.*}} 'tmp' 'hlsl::[[RESOURCE]]'
// CHECK-NEXT: AlwaysInlineAttr {{.*}} Implicit always_inline

// Static __createFromImplicitBinding method

// CHECK: CXXMethodDecl {{.*}} __createFromImplicitBinding 'hlsl::[[RESOURCE]] (unsigned int, unsigned int, int, unsigned int, const char *)' static
// CHECK-NEXT: ParmVarDecl {{.*}} orderId 'unsigned int'
// CHECK-NEXT: ParmVarDecl {{.*}} spaceNo 'unsigned int'
// CHECK-NEXT: ParmVarDecl {{.*}} range 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} index 'unsigned int'
// CHECK-NEXT: ParmVarDecl {{.*}} name 'const char *'
// CHECK-NEXT: CompoundStmt {{.*}}
// CHECK-NEXT: DeclStmt {{.*}}
// CHECK-NEXT: VarDecl {{.*}} tmp 'hlsl::[[RESOURCE]]'
// CHECK-NEXT: BinaryOperator {{.*}} '__hlsl_resource_t {{.*}}]]' '='
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t {{.*}}' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue Var {{.*}} 'tmp' 'hlsl::[[RESOURCE]]'
// CHECK-NEXT: CallExpr {{.*}} '__hlsl_resource_t {{.*}}'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(...) noexcept' <BuiltinFnToFnPtr>
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_handlefromimplicitbinding' 'void (...) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t {{.*}}' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue Var {{.*}} 'tmp' 'hlsl::[[RESOURCE]]'
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int' ParmVar {{.*}} 'orderId' 'unsigned int'
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int' ParmVar {{.*}} 'spaceNo' 'unsigned int'
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' ParmVar {{.*}} 'range' 'int'
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int' ParmVar {{.*}} 'index' 'unsigned int'
// CHECK-NEXT: DeclRefExpr {{.*}} 'const char *' ParmVar {{.*}} 'name' 'const char *'
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CXXConstructExpr {{.*}} 'hlsl::[[RESOURCE]]' 'void (const hlsl::[[RESOURCE]] &)'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const hlsl::[[RESOURCE]]' xvalue <NoOp>
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue Var {{.*}} 'tmp' 'hlsl::[[RESOURCE]]'
// CHECK-NEXT: AlwaysInlineAttr {{.*}} Implicit always_inline

// CHECK-NOSUBSCRIPT-NOT: CXXMethodDecl {{.*}} operator[] 'const char8_t &(unsigned int) const'
// CHECK-NOSUBSCRIPT-NOT: CXXMethodDecl {{.*}} operator[] 'char8_t &(unsigned int)'
