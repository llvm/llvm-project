// RUN: %clang_cc1 -Wdocumentation -ast-dump=json -x hlsl -triple dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefix=JSON
// RUN: %clang_cc1 -Wdocumentation -ast-dump -x hlsl -triple dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefix=AST

// JSON:"kind": "HLSLBufferDecl",
// JSON:"name": "A",
// JSON-NEXT:"bufferKind": "cbuffer",
// JSON:"kind": "TextComment",
// JSON:"text": " CBuffer decl."

/// CBuffer decl.
cbuffer A {
    // JSON: "kind": "VarDecl",
    // JSON: "name": "a",
    // JSON: "qualType": "hlsl_constant float"
    float a;
    // JSON: "kind": "VarDecl",
    // JSON: "name": "b",
    // JSON: "qualType": "hlsl_constant int"
    int b;
}

// AST: HLSLBufferDecl {{.*}} line:11:9 cbuffer A
// AST-NEXT: HLSLResourceClassAttr {{.*}} Implicit CBuffer
// AST-NEXT: HLSLResourceAttr {{.*}} Implicit CBuffer
// AST-NEXT: FullComment
// AST-NEXT: ParagraphComment
// AST-NEXT: TextComment {{.*}} Text=" CBuffer decl."
// AST-NEXT: VarDecl {{.*}} a 'hlsl_constant float'
// AST-NEXT: VarDecl {{.*}} b 'hlsl_constant int'
// AST-NEXT: CXXRecordDecl {{.*}} implicit class __layout_A definition
// AST: FieldDecl {{.*}} a 'float'
// AST-NEXT: FieldDecl {{.*}} b 'int'
