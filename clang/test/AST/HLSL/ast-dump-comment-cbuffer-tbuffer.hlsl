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
    // JSON: "qualType": "float"
    float a;
    // JSON: "kind": "VarDecl",
    // JSON: "name": "b",
    // JSON: "qualType": "int"
    int b;
}

// JSON:"kind": "HLSLBufferDecl",
// JSON:"name": "B",
// JSON-NEXT:"bufferKind": "tbuffer",
// JSON:"kind": "TextComment",
// JSON:"text": " TBuffer decl."

/// TBuffer decl.
tbuffer B {
    // JSON: "kind": "VarDecl",
    // JSON: "name": "c",
    // JSON: "qualType": "float"
    float c;
    // JSON: "kind": "VarDecl",
    // JSON: "name": "d",
    // JSON: "qualType": "int"
    int d;
}

// AST: HLSLBufferDecl {{.*}} line:11:9 cbuffer A
// AST-NEXT: HLSLResourceClassAttr {{.*}} Implicit CBuffer
// AST-NEXT: HLSLResourceAttr {{.*}} Implicit CBuffer
// AST-NEXT: FullComment
// AST-NEXT: ParagraphComment
// AST-NEXT: TextComment {{.*}} Text=" CBuffer decl."
// AST-NEXT: VarDecl {{.*}} a 'float'
// AST-NEXT: VarDecl {{.*}} b 'int'
// AST-NEXT: CXXRecordDecl {{.*}} implicit class __layout_A definition
// AST: FieldDecl {{.*}} a 'float'
// AST-NEXT: FieldDecl {{.*}} b 'int'

// AST-NEXT: HLSLBufferDecl {{.*}} line:29:9 tbuffer B
// AST-NEXT: HLSLResourceClassAttr {{.*}} Implicit SRV
// AST-NEXT: HLSLResourceAttr {{.*}} Implicit TBuffer
// AST-NEXT: FullComment
// AST-NEXT: ParagraphComment
// AST-NEXT: TextComment {{.*}} Text=" TBuffer decl."
// AST-NEXT: VarDecl {{.*}} c 'float'
// AST-NEXT: VarDecl {{.*}} d 'int'
// AST-NEXT: CXXRecordDecl {{.*}} implicit class __layout_B definition
// AST: FieldDecl {{.*}} c 'float'
// AST-NEXT: FieldDecl {{.*}} d 'int'
