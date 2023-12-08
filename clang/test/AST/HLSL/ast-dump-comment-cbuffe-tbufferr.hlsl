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

// AST:HLSLBufferDecl {{.*}}:11:1, line:20:1> line:11:9 cbuffer A
// AST-NEXT:FullComment {{.*}}<line:10:4, col:17>
// AST-NEXT:`-ParagraphComment {{.*}}<col:4, col:17>
// AST-NEXT:`-TextComment {{.*}}<col:4, col:17> Text=" CBuffer decl."
// AST-NEXT:-VarDecl {{.*}}<line:15:5, col:11> col:11 a 'float'
// AST-NEXT:`-VarDecl {{.*}}<line:19:5, col:9> col:9 b 'int'
// AST-NEXT:HLSLBufferDecl {{.*}}<line:29:1, line:38:1> line:29:9 tbuffer B
// AST-NEXT:-FullComment {{.*}}<line:28:4, col:17>
// AST-NEXT: `-ParagraphComment {{.*}}<col:4, col:17>
// AST-NEXT:  `-TextComment {{.*}}<col:4, col:17> Text=" TBuffer decl."
// AST-NEXT:-VarDecl {{.*}}<line:33:5, col:11> col:11 c 'float'
// AST-NEXT:`-VarDecl {{.*}} <line:37:5, col:9> col:9 d 'int'
