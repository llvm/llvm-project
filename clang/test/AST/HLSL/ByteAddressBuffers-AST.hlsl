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

// EMPTY: CXXRecordDecl 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> <invalid sloc> implicit <undeserialized declarations> class [[RESOURCE]]
// EMPTY-NEXT: FinalAttr 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> Implicit final

// There should be no more occurrences of RESOURCE
// EMPTY-NOT: {{[^[:alnum:]]}}[[RESOURCE]]

#ifndef EMPTY

RESOURCE Buffer;

#endif

// CHECK: CXXRecordDecl 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> <invalid sloc> implicit referenced <undeserialized declarations> class [[RESOURCE]] definition


// CHECK: FinalAttr 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> Implicit final
// CHECK-NEXT: FieldDecl 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> <invalid sloc> implicit __handle '__hlsl_resource_t
// CHECK-SRV-SAME{LITERAL}: [[hlsl::resource_class(SRV)]]
// CHECK-UAV-SAME{LITERAL}: [[hlsl::resource_class(UAV)]]
// CHECK-SAME{LITERAL}: [[hlsl::raw_buffer]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(char8_t)]]
// CHECK-NEXT: HLSLResourceAttr 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> Implicit RawBuffer

// CHECK-NOSUBSCRIPT-NOT: CXXMethodDecl 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> <invalid sloc> operator[] 'const element_type &(unsigned int) const'
// CHECK-NOSUBSCRIPT-NOT: CXXMethodDecl 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> <invalid sloc> operator[] 'element_type &(unsigned int)'
