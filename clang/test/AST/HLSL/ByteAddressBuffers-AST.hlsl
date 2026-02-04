// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump -DEMPTY \
// RUN:  -DRESOURCE=ByteAddressBuffer %s | FileCheck -DRESOURCE=ByteAddressBuffer \
// RUN:  -check-prefix=EMPTY %s
//
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -DRESOURCE=ByteAddressBuffer %s | FileCheck -DRESOURCE=ByteAddressBuffer \
// RUN:   -check-prefixes=CHECK,CHECK-SRV,CHECK-NOSUBSCRIPT,CHECK-LOAD %s
//
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump -DEMPTY \
// RUN:  -DRESOURCE=RWByteAddressBuffer %s | FileCheck -DRESOURCE=RWByteAddressBuffer \
// RUN:  -check-prefix=EMPTY %s
//
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump \
// RUN:   -DRESOURCE=RWByteAddressBuffer %s | FileCheck -DRESOURCE=RWByteAddressBuffer \
// RUN:   -check-prefixes=CHECK,CHECK-UAV,CHECK-NOSUBSCRIPT,CHECK-LOAD,CHECK-STORE %s
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
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__hlsl_resource_t (*)(__hlsl_resource_t) noexcept' <BuiltinFnToFnPtr>
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_uninitializedhandle' '__hlsl_resource_t (__hlsl_resource_t) noexcept'
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
// CHECK-NEXT: DeclRefExpr {{.*}} 'const hlsl::[[RESOURCE]]' lvalue ParmVar {{.*}} 'other' 'const hlsl::[[RESOURCE]] &'
// CHECK-NEXT: AlwaysInlineAttr

// operator=

// CHECK: CXXMethodDecl {{.*}} operator= 'hlsl::[[RESOURCE]] &(const hlsl::[[RESOURCE]] &)'
// CHECK-NEXT: ParmVarDecl {{.*}} other 'const hlsl::[[RESOURCE]] &'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: BinaryOperator {{.*}} '='
// CHECK-NEXT: MemberExpr {{.*}} lvalue .__handle
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue implicit this
// CHECK-NEXT: MemberExpr {{.*}} lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'const hlsl::[[RESOURCE]]' lvalue ParmVar {{.*}} 'other' 'const hlsl::[[RESOURCE]] &'
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
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__hlsl_resource_t (*)(__hlsl_resource_t, unsigned int, unsigned int, int, unsigned int, const char *) noexcept' <BuiltinFnToFnPtr>
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_handlefrombinding' '__hlsl_resource_t (__hlsl_resource_t, unsigned int, unsigned int, int, unsigned int, const char *) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t {{.*}}' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue Var {{.*}} 'tmp' 'hlsl::[[RESOURCE]]'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'registerNo' 'unsigned int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'spaceNo' 'unsigned int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'range' 'int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'index' 'unsigned int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const char *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'const char *' lvalue ParmVar {{.*}} 'name' 'const char *'
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
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__hlsl_resource_t (*)(__hlsl_resource_t, unsigned int, unsigned int, int, unsigned int, const char *) noexcept' <BuiltinFnToFnPtr>
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_handlefromimplicitbinding' '__hlsl_resource_t (__hlsl_resource_t, unsigned int, unsigned int, int, unsigned int, const char *) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t {{.*}}' lvalue .__handle
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue Var {{.*}} 'tmp' 'hlsl::[[RESOURCE]]'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'orderId' 'unsigned int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'spaceNo' 'unsigned int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'range' 'int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'index' 'unsigned int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const char *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'const char *' lvalue ParmVar {{.*}} 'name' 'const char *'
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CXXConstructExpr {{.*}} 'hlsl::[[RESOURCE]]' 'void (const hlsl::[[RESOURCE]] &)'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const hlsl::[[RESOURCE]]' xvalue <NoOp>
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue Var {{.*}} 'tmp' 'hlsl::[[RESOURCE]]'
// CHECK-NEXT: AlwaysInlineAttr {{.*}} Implicit always_inline

// Load methods

// CHECK-LOAD: CXXMethodDecl {{.*}} Load 'unsigned int (unsigned int)'
// CHECK-LOAD-NEXT: ParmVarDecl {{.*}} Index 'unsigned int'
// CHECK-LOAD-NEXT: CompoundStmt
// CHECK-LOAD-NEXT: ReturnStmt
// CHECK-LOAD-NEXT: UnaryOperator {{.*}} 'hlsl_device unsigned int' lvalue prefix '*' cannot overflow
// CHECK-LOAD-NEXT: CallExpr {{.*}} 'hlsl_device unsigned int *'
// CHECK-LOAD-NEXT: ImplicitCastExpr {{.*}} 'void (*)(...) noexcept' <BuiltinFnToFnPtr>
// CHECK-LOAD-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_getpointer_typed' 'void (...) noexcept'
// CHECK-LOAD-NEXT: ImplicitCastExpr {{.*}} '__hlsl_resource_t {{.*}}' <LValueToRValue>
// CHECK-LOAD-NEXT: MemberExpr {{.*}} '__hlsl_resource_t {{.*}}' lvalue .__handle
// CHECK-LOAD-NEXT: CXXThisExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue implicit this
// CHECK-LOAD-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <LValueToRValue>
// CHECK-LOAD-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'Index' 'unsigned int'
// CHECK-LOAD-NEXT: CXXScalarValueInitExpr {{.*}} 'unsigned int *'
// CHECK-LOAD-NEXT: AlwaysInlineAttr {{.*}} Implicit always_inline

// CHECK-LOAD: CXXMethodDecl {{.*}} Load 'unsigned int (unsigned int, out unsigned int)
// CHECK-LOAD-NEXT: ParmVarDecl {{.*}} Index 'unsigned int'
// CHECK-LOAD-NEXT: ParmVarDecl {{.*}} Status 'unsigned int &__restrict'
// CHECK-LOAD-NEXT: HLSLParamModifierAttr {{.*}} out
// CHECK-LOAD-NEXT: CompoundStmt
// CHECK-LOAD-NEXT: ReturnStmt
// CHECK-LOAD-NEXT: CallExpr {{.*}} 'unsigned int'
// CHECK-LOAD-NEXT: ImplicitCastExpr {{.*}} 'void (*)(...) noexcept' <BuiltinFnToFnPtr>
// CHECK-LOAD-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_load_with_status_typed' 'void (...) noexcept'
// CHECK-LOAD-NEXT: ImplicitCastExpr {{.*}} '__hlsl_resource_t {{.*}}' <LValueToRValue>
// CHECK-LOAD-NEXT: MemberExpr {{.*}} '__hlsl_resource_t {{.*}}' lvalue .__handle
// CHECK-LOAD-NEXT: CXXThisExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue implicit this
// CHECK-LOAD-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <LValueToRValue>
// CHECK-LOAD-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'Index' 'unsigned int'
// CHECK-LOAD-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <LValueToRValue>
// CHECK-LOAD-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'Status' 'unsigned int &__restrict'
// CHECK-LOAD-NEXT: CXXScalarValueInitExpr {{.*}} 'unsigned int *'
// CHECK-LOAD-NEXT: AlwaysInlineAttr {{.*}} Implicit always_inline

// CHECK-LOAD: CXXMethodDecl {{.*}} Load2 'vector<unsigned int (unsigned int), 2>'
// CHECK-LOAD-NEXT: ParmVarDecl {{.*}} Index 'unsigned int'
// CHECK-LOAD-NEXT: CompoundStmt
// CHECK-LOAD-NEXT: ReturnStmt
// CHECK-LOAD-NEXT: UnaryOperator {{.*}} 'vector<unsigned int hlsl_device, 2>' lvalue prefix '*' cannot overflow
// CHECK-LOAD-NEXT: CallExpr {{.*}} 'vector<unsigned int hlsl_device *, 2>'
// CHECK-LOAD-NEXT: ImplicitCastExpr {{.*}} 'void (*)(...) noexcept' <BuiltinFnToFnPtr>
// CHECK-LOAD-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_getpointer_typed' 'void (...) noexcept'
// CHECK-LOAD-NEXT: ImplicitCastExpr {{.*}} '__hlsl_resource_t {{.*}}' <LValueToRValue>
// CHECK-LOAD-NEXT: MemberExpr {{.*}} '__hlsl_resource_t {{.*}}' lvalue .__handle
// CHECK-LOAD-NEXT: CXXThisExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue implicit this
// CHECK-LOAD-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <LValueToRValue>
// CHECK-LOAD-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'Index' 'unsigned int'
// CHECK-LOAD-NEXT: CXXScalarValueInitExpr {{.*}} 'vector<unsigned int *, 2>'
// CHECK-LOAD-NEXT: AlwaysInlineAttr {{.*}} Implicit always_inline

// CHECK-LOAD: CXXMethodDecl {{.*}} Load2 'vector<unsigned int (unsigned int, out unsigned int), 2>'
// CHECK-LOAD-NEXT: ParmVarDecl {{.*}} Index 'unsigned int'
// CHECK-LOAD-NEXT: ParmVarDecl {{.*}} Status 'unsigned int &__restrict'
// CHECK-LOAD-NEXT: HLSLParamModifierAttr {{.*}} out
// CHECK-LOAD-NEXT: CompoundStmt
// CHECK-LOAD-NEXT: ReturnStmt
// CHECK-LOAD-NEXT: CallExpr {{.*}} 'vector<unsigned int, 2>'
// CHECK-LOAD-NEXT: ImplicitCastExpr {{.*}} 'void (*)(...) noexcept' <BuiltinFnToFnPtr>
// CHECK-LOAD-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_load_with_status_typed' 'void (...) noexcept'
// CHECK-LOAD-NEXT: ImplicitCastExpr {{.*}} '__hlsl_resource_t {{.*}}' <LValueToRValue>
// CHECK-LOAD-NEXT: MemberExpr {{.*}} '__hlsl_resource_t {{.*}}' lvalue .__handle
// CHECK-LOAD-NEXT: CXXThisExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue implicit this
// CHECK-LOAD-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <LValueToRValue>
// CHECK-LOAD-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'Index' 'unsigned int'
// CHECK-LOAD-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <LValueToRValue>
// CHECK-LOAD-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'Status' 'unsigned int &__restrict'
// CHECK-LOAD-NEXT: CXXScalarValueInitExpr {{.*}} 'vector<unsigned int *, 2>'
// CHECK-LOAD-NEXT: AlwaysInlineAttr {{.*}} Implicit always_inline

// CHECK-LOAD: CXXMethodDecl {{.*}} Load3 'vector<unsigned int (unsigned int), 3>'
// CHECK-LOAD-NEXT: ParmVarDecl {{.*}} Index 'unsigned int'
// CHECK-LOAD-NEXT: CompoundStmt
// CHECK-LOAD-NEXT: ReturnStmt
// CHECK-LOAD-NEXT: UnaryOperator {{.*}} 'vector<unsigned int hlsl_device, 3>' lvalue prefix '*' cannot overflow
// CHECK-LOAD-NEXT: CallExpr {{.*}} 'vector<unsigned int hlsl_device *, 3>'
// CHECK-LOAD-NEXT: ImplicitCastExpr {{.*}} 'void (*)(...) noexcept' <BuiltinFnToFnPtr>
// CHECK-LOAD-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_getpointer_typed' 'void (...) noexcept'
// CHECK-LOAD-NEXT: ImplicitCastExpr {{.*}} '__hlsl_resource_t {{.*}}' <LValueToRValue>
// CHECK-LOAD-NEXT: MemberExpr {{.*}} '__hlsl_resource_t {{.*}}' lvalue .__handle
// CHECK-LOAD-NEXT: CXXThisExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue implicit this
// CHECK-LOAD-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <LValueToRValue>
// CHECK-LOAD-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'Index' 'unsigned int'
// CHECK-LOAD-NEXT: CXXScalarValueInitExpr {{.*}} 'vector<unsigned int *, 3>'
// CHECK-LOAD-NEXT: AlwaysInlineAttr {{.*}} Implicit always_inline

// CHECK-LOAD: CXXMethodDecl {{.*}} Load3 'vector<unsigned int (unsigned int, out unsigned int), 3>'
// CHECK-LOAD-NEXT: ParmVarDecl {{.*}} Index 'unsigned int'
// CHECK-LOAD-NEXT: ParmVarDecl {{.*}} Status 'unsigned int &__restrict'
// CHECK-LOAD-NEXT: HLSLParamModifierAttr {{.*}} out
// CHECK-LOAD-NEXT: CompoundStmt
// CHECK-LOAD-NEXT: ReturnStmt
// CHECK-LOAD-NEXT: CallExpr {{.*}} 'vector<unsigned int, 3>'
// CHECK-LOAD-NEXT: ImplicitCastExpr {{.*}} 'void (*)(...) noexcept' <BuiltinFnToFnPtr>
// CHECK-LOAD-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_load_with_status_typed' 'void (...) noexcept'
// CHECK-LOAD-NEXT: ImplicitCastExpr {{.*}} '__hlsl_resource_t {{.*}}' <LValueToRValue>
// CHECK-LOAD-NEXT: MemberExpr {{.*}} '__hlsl_resource_t {{.*}}' lvalue .__handle
// CHECK-LOAD-NEXT: CXXThisExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue implicit this
// CHECK-LOAD-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <LValueToRValue>
// CHECK-LOAD-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'Index' 'unsigned int'
// CHECK-LOAD-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <LValueToRValue>
// CHECK-LOAD-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'Status' 'unsigned int &__restrict'
// CHECK-LOAD-NEXT: CXXScalarValueInitExpr {{.*}} 'vector<unsigned int *, 3>'
// CHECK-LOAD-NEXT: AlwaysInlineAttr {{.*}} Implicit always_inline

// CHECK-LOAD: CXXMethodDecl {{.*}} Load4 'vector<unsigned int (unsigned int), 4>'
// CHECK-LOAD-NEXT: ParmVarDecl {{.*}} Index 'unsigned int'
// CHECK-LOAD-NEXT: CompoundStmt
// CHECK-LOAD-NEXT: ReturnStmt
// CHECK-LOAD-NEXT: UnaryOperator {{.*}} 'vector<unsigned int hlsl_device, 4>' lvalue prefix '*' cannot overflow
// CHECK-LOAD-NEXT: CallExpr {{.*}} 'vector<unsigned int hlsl_device *, 4>'
// CHECK-LOAD-NEXT: ImplicitCastExpr {{.*}} 'void (*)(...) noexcept' <BuiltinFnToFnPtr>
// CHECK-LOAD-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_getpointer_typed' 'void (...) noexcept'
// CHECK-LOAD-NEXT: ImplicitCastExpr {{.*}} '__hlsl_resource_t {{.*}}' <LValueToRValue>
// CHECK-LOAD-NEXT: MemberExpr {{.*}} '__hlsl_resource_t {{.*}}' lvalue .__handle
// CHECK-LOAD-NEXT: CXXThisExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue implicit this
// CHECK-LOAD-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <LValueToRValue>
// CHECK-LOAD-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'Index' 'unsigned int'
// CHECK-LOAD-NEXT: CXXScalarValueInitExpr {{.*}} 'vector<unsigned int *, 4>'
// CHECK-LOAD-NEXT: AlwaysInlineAttr {{.*}} Implicit always_inline

// CHECK-LOAD: CXXMethodDecl {{.*}} Load4 'vector<unsigned int (unsigned int, out unsigned int), 4>'
// CHECK-LOAD-NEXT: ParmVarDecl {{.*}} Index 'unsigned int'
// CHECK-LOAD-NEXT: ParmVarDecl {{.*}} Status 'unsigned int &__restrict'
// CHECK-LOAD-NEXT: HLSLParamModifierAttr {{.*}} out
// CHECK-LOAD-NEXT: CompoundStmt
// CHECK-LOAD-NEXT: ReturnStmt
// CHECK-LOAD-NEXT: CallExpr {{.*}} 'vector<unsigned int, 4>'
// CHECK-LOAD-NEXT: ImplicitCastExpr {{.*}} 'void (*)(...) noexcept' <BuiltinFnToFnPtr>
// CHECK-LOAD-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_load_with_status_typed' 'void (...) noexcept'
// CHECK-LOAD-NEXT: ImplicitCastExpr {{.*}} '__hlsl_resource_t {{.*}}' <LValueToRValue>
// CHECK-LOAD-NEXT: MemberExpr {{.*}} '__hlsl_resource_t {{.*}}' lvalue .__handle
// CHECK-LOAD-NEXT: CXXThisExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue implicit this
// CHECK-LOAD-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <LValueToRValue>
// CHECK-LOAD-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'Index' 'unsigned int'
// CHECK-LOAD-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <LValueToRValue>
// CHECK-LOAD-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'Status' 'unsigned int &__restrict'
// CHECK-LOAD-NEXT: CXXScalarValueInitExpr {{.*}} 'vector<unsigned int *, 4>'
// CHECK-LOAD-NEXT: AlwaysInlineAttr {{.*}} Implicit always_inline

// CHECK-LOAD: CXXMethodDecl {{.*}} Load 'element_type (unsigned int)'
// CHECK-LOAD-NEXT: ParmVarDecl {{.*}} Index 'unsigned int'
// CHECK-LOAD-NEXT: CompoundStmt
// CHECK-LOAD-NEXT: ReturnStmt
// CHECK-LOAD-NEXT: UnaryOperator {{.*}} 'hlsl_device element_type' lvalue prefix '*' cannot overflow
// CHECK-LOAD-NEXT: CStyleCastExpr {{.*}} 'hlsl_device element_type *' <Dependent>
// CHECK-LOAD-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-LOAD-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_getpointer_typed' 'void (...) noexcept'
// CHECK-LOAD-NEXT: MemberExpr {{.*}} '__hlsl_resource_t {{.*}}' lvalue .__handle
// CHECK-LOAD-NEXT: CXXThisExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue implicit this
// CHECK-LOAD-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'Index' 'unsigned int'
// CHECK-LOAD-NEXT: CXXScalarValueInitExpr {{.*}} 'element_type *'
// CHECK-LOAD-NEXT: AlwaysInlineAttr {{.*}} Implicit always_inline

// CHECK-LOAD: CXXMethodDecl {{.*}} Load 'element_type (unsigned int, out unsigned int)
// CHECK-LOAD-NEXT: ParmVarDecl {{.*}} Index 'unsigned int'
// CHECK-LOAD-NEXT: ParmVarDecl {{.*}} Status 'unsigned int &__restrict'
// CHECK-LOAD-NEXT: HLSLParamModifierAttr {{.*}} out
// CHECK-LOAD-NEXT: CompoundStmt
// CHECK-LOAD-NEXT: ReturnStmt
// CHECK-LOAD-NEXT: CStyleCastExpr {{.*}} 'element_type' <Dependent>
// CHECK-LOAD-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-LOAD-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_load_with_status_typed' 'void (...) noexcept'
// CHECK-LOAD-NEXT: MemberExpr {{.*}} '__hlsl_resource_t {{.*}}' lvalue .__handle
// CHECK-LOAD-NEXT: CXXThisExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue implicit this
// CHECK-LOAD-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'Index' 'unsigned int'
// CHECK-LOAD-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'Status' 'unsigned int &__restrict'
// CHECK-LOAD-NEXT: CXXScalarValueInitExpr {{.*}} 'element_type *'
// CHECK-LOAD-NEXT: AlwaysInlineAttr {{.*}} Implicit always_inline

// Store method
// CHECK-STORE: CXXMethodDecl {{.*}} Store 'void (unsigned int, unsigned int)'
// CHECK-STORE-NEXT: ParmVarDecl {{.*}} Index 'unsigned int'
// CHECK-STORE-NEXT: ParmVarDecl {{.*}} Value 'unsigned int'
// CHECK-STORE-NEXT: CompoundStmt
// CHECK-STORE-NEXT: BinaryOperator {{.*}} 'hlsl_device unsigned int' '='
// CHECK-STORE-NEXT: UnaryOperator {{.*}} 'hlsl_device unsigned int' lvalue prefix '*' cannot overflow
// CHECK-STORE-NEXT: CallExpr {{.*}} 'hlsl_device unsigned int *'
// CHECK-STORE-NEXT: ImplicitCastExpr {{.*}} 'void (*)(...) noexcept' <BuiltinFnToFnPtr>
// CHECK-STORE-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_getpointer_typed' 'void (...) noexcept'
// CHECK-STORE-NEXT: ImplicitCastExpr {{.*}} '__hlsl_resource_t {{.*}}' <LValueToRValue>
// CHECK-STORE-NEXT: MemberExpr {{.*}} '__hlsl_resource_t {{.*}}' lvalue .__handle
// CHECK-STORE-NEXT: CXXThisExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue implicit this
// CHECK-STORE-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <LValueToRValue>
// CHECK-STORE-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'Index' 'unsigned int'
// CHECK-STORE-NEXT: CXXScalarValueInitExpr {{.*}} 'unsigned int *'
// CHECK-STORE-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'Value' 'unsigned int'
// CHECK-STORE-NEXT: AlwaysInlineAttr {{.*}} Implicit always_inline

// CHECK-STORE: CXXMethodDecl {{.*}} Store2 'void (unsigned int, vector<unsigned int, 2>)'
// CHECK-STORE-NEXT: ParmVarDecl {{.*}} Index 'unsigned int'
// CHECK-STORE-NEXT: ParmVarDecl {{.*}} Value 'vector<unsigned int, 2>'
// CHECK-STORE-NEXT: CompoundStmt
// CHECK-STORE-NEXT: BinaryOperator {{.*}} 'vector<unsigned int hlsl_device, 2>' '='
// CHECK-STORE-NEXT: UnaryOperator {{.*}} 'vector<unsigned int hlsl_device, 2>' lvalue prefix '*' cannot overflow
// CHECK-STORE-NEXT: CallExpr {{.*}} 'vector<unsigned int hlsl_device *, 2>'
// CHECK-STORE-NEXT: ImplicitCastExpr {{.*}} 'void (*)(...) noexcept' <BuiltinFnToFnPtr>
// CHECK-STORE-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_getpointer_typed' 'void (...) noexcept'
// CHECK-STORE-NEXT: ImplicitCastExpr {{.*}} '__hlsl_resource_t {{.*}}' <LValueToRValue>
// CHECK-STORE-NEXT: MemberExpr {{.*}} '__hlsl_resource_t {{.*}}' lvalue .__handle
// CHECK-STORE-NEXT: CXXThisExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue implicit this
// CHECK-STORE-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <LValueToRValue>
// CHECK-STORE-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'Index' 'unsigned int'
// CHECK-STORE-NEXT: CXXScalarValueInitExpr {{.*}} 'vector<unsigned int *, 2>'
// CHECK-STORE-NEXT: DeclRefExpr {{.*}} 'vector<unsigned int, 2>' lvalue ParmVar {{.*}} 'Value' 'vector<unsigned int, 2>'
// CHECK-STORE-NEXT: AlwaysInlineAttr {{.*}} Implicit always_inline

// CHECK-STORE: CXXMethodDecl {{.*}} Store3 'void (unsigned int, vector<unsigned int, 3>)'
// CHECK-STORE-NEXT: ParmVarDecl {{.*}} Index 'unsigned int'
// CHECK-STORE-NEXT: ParmVarDecl {{.*}} Value 'vector<unsigned int, 3>'
// CHECK-STORE-NEXT: CompoundStmt
// CHECK-STORE-NEXT: BinaryOperator {{.*}} 'vector<unsigned int hlsl_device, 3>' '='
// CHECK-STORE-NEXT: UnaryOperator {{.*}} 'vector<unsigned int hlsl_device, 3>' lvalue prefix '*' cannot overflow
// CHECK-STORE-NEXT: CallExpr {{.*}} 'vector<unsigned int hlsl_device *, 3>'
// CHECK-STORE-NEXT: ImplicitCastExpr {{.*}} 'void (*)(...) noexcept' <BuiltinFnToFnPtr>
// CHECK-STORE-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_getpointer_typed' 'void (...) noexcept'
// CHECK-STORE-NEXT: ImplicitCastExpr {{.*}} '__hlsl_resource_t {{.*}}' <LValueToRValue>
// CHECK-STORE-NEXT: MemberExpr {{.*}} '__hlsl_resource_t {{.*}}' lvalue .__handle
// CHECK-STORE-NEXT: CXXThisExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue implicit this
// CHECK-STORE-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <LValueToRValue>
// CHECK-STORE-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'Index' 'unsigned int'
// CHECK-STORE-NEXT: CXXScalarValueInitExpr {{.*}} 'vector<unsigned int *, 3>'
// CHECK-STORE-NEXT: DeclRefExpr {{.*}} 'vector<unsigned int, 3>' lvalue ParmVar {{.*}} 'Value' 'vector<unsigned int, 3>'
// CHECK-STORE-NEXT: AlwaysInlineAttr {{.*}} Implicit always_inline

// CHECK-STORE: CXXMethodDecl {{.*}} Store4 'void (unsigned int, vector<unsigned int, 4>)'
// CHECK-STORE-NEXT: ParmVarDecl {{.*}} Index 'unsigned int'
// CHECK-STORE-NEXT: ParmVarDecl {{.*}} Value 'vector<unsigned int, 4>'
// CHECK-STORE-NEXT: CompoundStmt
// CHECK-STORE-NEXT: BinaryOperator {{.*}} 'vector<unsigned int hlsl_device, 4>' '='
// CHECK-STORE-NEXT: UnaryOperator {{.*}} 'vector<unsigned int hlsl_device, 4>' lvalue prefix '*' cannot overflow
// CHECK-STORE-NEXT: CallExpr {{.*}} 'vector<unsigned int hlsl_device *, 4>'
// CHECK-STORE-NEXT: ImplicitCastExpr {{.*}} 'void (*)(...) noexcept' <BuiltinFnToFnPtr>
// CHECK-STORE-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_getpointer_typed' 'void (...) noexcept'
// CHECK-STORE-NEXT: ImplicitCastExpr {{.*}} '__hlsl_resource_t {{.*}}' <LValueToRValue>
// CHECK-STORE-NEXT: MemberExpr {{.*}} '__hlsl_resource_t {{.*}}' lvalue .__handle
// CHECK-STORE-NEXT: CXXThisExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue implicit this
// CHECK-STORE-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <LValueToRValue>
// CHECK-STORE-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'Index' 'unsigned int'
// CHECK-STORE-NEXT: CXXScalarValueInitExpr {{.*}} 'vector<unsigned int *, 4>'
// CHECK-STORE-NEXT: DeclRefExpr {{.*}} 'vector<unsigned int, 4>' lvalue ParmVar {{.*}} 'Value' 'vector<unsigned int, 4>'
// CHECK-STORE-NEXT: AlwaysInlineAttr {{.*}} Implicit always_inline

// CHECK-STORE: CXXMethodDecl {{.*}} Store 'void (unsigned int, element_type)'
// CHECK-STORE-NEXT: ParmVarDecl {{.*}} Index 'unsigned int'
// CHECK-STORE-NEXT: ParmVarDecl {{.*}} Value 'element_type'
// CHECK-STORE-NEXT: CompoundStmt
// CHECK-STORE-NEXT: BinaryOperator {{.*}} 'hlsl_device element_type' '='
// CHECK-STORE-NEXT: UnaryOperator {{.*}} 'hlsl_device element_type' lvalue prefix '*' cannot overflow
// CHECK-STORE-NEXT: CStyleCastExpr {{.*}} 'hlsl_device element_type *' <Dependent>
// CHECK-STORE-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-STORE-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_getpointer_typed' 'void (...) noexcept'
// CHECK-STORE-NEXT: MemberExpr {{.*}} '__hlsl_resource_t {{.*}}' lvalue .__handle
// CHECK-STORE-NEXT: CXXThisExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue implicit this
// CHECK-STORE-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'Index' 'unsigned int'
// CHECK-STORE-NEXT: CXXScalarValueInitExpr {{.*}} 'element_type *'
// CHECK-STORE-NEXT: DeclRefExpr {{.*}} 'element_type' lvalue ParmVar {{.*}} 'Value' 'element_type'
// CHECK-STORE-NEXT: AlwaysInlineAttr {{.*}} Implicit always_inline

// GetDimensions method

// CHECK: CXXMethodDecl {{.*}} GetDimensions 'void (out unsigned int)'
// CHECK-NEXT: ParmVarDecl {{.*}} dim 'unsigned int &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} out
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(__hlsl_resource_t, unsigned int &) noexcept' <BuiltinFnToFnPtr>
// CHECK-NEXT: DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_hlsl_resource_getdimensions_x' 'void (__hlsl_resource_t, unsigned int &) noexcept'
// CHECK-NEXT: MemberExpr {{.*}} '__hlsl_resource_t {{.*}}' lvalue .__handle {{.*}}
// CHECK-NEXT: CXXThisExpr {{.*}} 'hlsl::[[RESOURCE]]' lvalue implicit this
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}}  'dim' 'unsigned int &__restrict'
// CHECK-NEXT: AlwaysInlineAttr {{.*}} Implicit always_inline

// CHECK-NOSUBSCRIPT-NOT: CXXMethodDecl {{.*}} operator[] 'const char8_t &(unsigned int) const'
// CHECK-NOSUBSCRIPT-NOT: CXXMethodDecl {{.*}} operator[] 'char8_t &(unsigned int)'
