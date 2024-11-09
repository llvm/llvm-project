// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump -DEMPTY %s | FileCheck -check-prefix=EMPTY %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump %s | FileCheck %s


// EMPTY: CXXRecordDecl 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> <invalid sloc> implicit <undeserialized declarations> class ByteAddressBuffer
// EMPTY-NEXT: FinalAttr 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> Implicit final

// There should be no more occurrences of ByteAddressBuffer
// EMPTY-NOT: {{[^[:alnum:]]}}ByteAddressBuffer

#ifndef EMPTY

ByteAddressBuffer Buffer;

#endif

// CHECK: CXXRecordDecl 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> <invalid sloc> implicit referenced <undeserialized declarations> class ByteAddressBuffer definition


// CHECK: FinalAttr 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> Implicit final
// CHECK-NEXT: FieldDecl 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> <invalid sloc> implicit h '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(UAV)]]
// CHECK-SAME{LITERAL}: [[hlsl::raw_buffer]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(char8_t)]]
// CHECK-NEXT: HLSLResourceAttr 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> Implicit RawBuffer

// CHECK: CXXMethodDecl 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> <invalid sloc> operator[] 'char8_t &const (unsigned int) const'
// CHECK-NEXT: ParmVarDecl 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> <invalid sloc> Idx 'unsigned int'
// CHECK-NEXT: CompoundStmt 0x{{[0-9A-Fa-f]+}} <<invalid sloc>>
// CHECK-NEXT: ReturnStmt 0x{{[0-9A-Fa-f]+}} <<invalid sloc>>
// CHECK-NEXT: MemberExpr 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> 'char8_t' lvalue .e 0x{{[0-9A-Fa-f]+}}
// CHECK-NEXT: CXXThisExpr 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> 'const hlsl::ByteAddressBuffer' lvalue implicit this
// CHECK-NEXT: AlwaysInlineAttr 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> Implicit always_inline

// CHECK-NEXT: CXXMethodDecl 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> <invalid sloc> operator[] 'char8_t &(unsigned int)'
// CHECK-NEXT: ParmVarDecl 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> <invalid sloc> Idx 'unsigned int'
// CHECK-NEXT: CompoundStmt 0x{{[0-9A-Fa-f]+}} <<invalid sloc>>
// CHECK-NEXT: ReturnStmt 0x{{[0-9A-Fa-f]+}} <<invalid sloc>>
// CHECK-NEXT: MemberExpr 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> 'char8_t' lvalue .e 0x{{[0-9A-Fa-f]+}}
// CHECK-NEXT: CXXThisExpr 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> 'hlsl::ByteAddressBuffer' lvalue implicit this
// CHECK-NEXT: AlwaysInlineAttr 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> Implicit always_inline
