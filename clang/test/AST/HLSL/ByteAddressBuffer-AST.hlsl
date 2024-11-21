// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump -DEMPTY %s | FileCheck %s -DRESOURCE=ByteAddressBuffer -check-prefix=EMPTY 
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump %s | FileCheck %s -DRESOURCE=ByteAddressBuffer -check-prefixes=CHECK-ATTR


// EMPTY: CXXRecordDecl 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> <invalid sloc> implicit <undeserialized declarations> class [[RESOURCE]]
// EMPTY-NEXT: FinalAttr 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> Implicit final

// There should be no more occurrences of [[RESOURCE]]
// EMPTY-NOT: {{[^[:alnum:]]}}[[RESOURCE]]

#ifndef EMPTY

ByteAddressBuffer Buffer;

#endif

// CHECK: CXXRecordDecl 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> <invalid sloc> implicit referenced <undeserialized declarations> class [[RESOURCE]] definition


// CHECK-ATTR: FieldDecl 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> <invalid sloc> implicit h '__hlsl_resource_t
// CHECK-ATTR-SAME{LITERAL}: [[hlsl::resource_class(SRV)]]
// CHECK-ATTR-SAME{LITERAL}: [[hlsl::raw_buffer]]
// CHECK-ATTR-SAME{LITERAL}: [[hlsl::contained_type(char8_t)]]
// CHECK-ATTR-NEXT: HLSLResourceAttr 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> Implicit RawBuffer
