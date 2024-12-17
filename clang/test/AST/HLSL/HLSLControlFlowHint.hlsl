// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -ast-dump %s | FileCheck %s

// CHECK: FunctionDecl 0x{{[0-9A-Fa-f]+}} <{{.*}}> {{.*}} used branch 'int (int)'
// CHECK: AttributedStmt 0x{{[0-9A-Fa-f]+}} <<invalid sloc>
// CHECK-NEXT: -HLSLControlFlowHintAttr 0x{{[0-9A-Fa-f]+}} <{{.*}}> branch
export int branch(int X){
    int resp;
    [branch] if (X > 0) {
        resp = -X;
    } else {
        resp = X * 2;
    }

    return resp;
}

// CHECK: FunctionDecl 0x{{[0-9A-Fa-f]+}} <{{.*}}> {{.*}} used flatten 'int (int)'
// CHECK: AttributedStmt 0x{{[0-9A-Fa-f]+}} <<invalid sloc>
// CHECK-NEXT: -HLSLControlFlowHintAttr 0x{{[0-9A-Fa-f]+}} <{{.*}}> flatten
export int flatten(int X){
    int resp;
    [flatten] if (X > 0) {
        resp = -X;
    } else {
        resp = X * 2;
    }

    return resp;
}

// CHECK: FunctionDecl 0x{{[0-9A-Fa-f]+}} <{{.*}}> {{.*}} used no_attr 'int (int)'
// CHECK-NOT: AttributedStmt 0x{{[0-9A-Fa-f]+}} <<invalid sloc>
// CHECK-NOT: -HLSLControlFlowHintAttr
export int no_attr(int X){
    int resp;
    if (X > 0) {
        resp = -X;
    } else {
        resp = X * 2;
    }

    return resp;
}
