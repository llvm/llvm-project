// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -ast-dump %s | FileCheck %s

// CHECK: FunctionDecl {{.*}} used branch 'int (int)'
// CHECK: AttributedStmt
// CHECK-NEXT: HLSLControlFlowHintAttr {{.*}} branch
export int branch(int X){
    int resp;
    [branch] if (X > 0) {
        resp = -X;
    } else {
        resp = X * 2;
    }

    return resp;
}

// CHECK: FunctionDecl {{.*}} used flatten 'int (int)'
// CHECK: AttributedStmt
// CHECK-NEXT: HLSLControlFlowHintAttr {{.*}} flatten
export int flatten(int X){
    int resp;
    [flatten] if (X > 0) {
        resp = -X;
    } else {
        resp = X * 2;
    }

    return resp;
}

// CHECK: FunctionDecl {{.*}} used no_attr 'int (int)'
// CHECK-NOT: AttributedStmt
// CHECK-NOT: HLSLControlFlowHintAttr
export int no_attr(int X){
    int resp;
    if (X > 0) {
        resp = -X;
    } else {
        resp = X * 2;
    }

    return resp;
}

// CHECK: FunctionDecl {{.*}} used flatten_switch 'int (int)'
// CHECK: AttributedStmt
// CHECK-NEXT: HLSLControlFlowHintAttr {{.*}} flatten
export int flatten_switch(int X){
    int resp;
    [flatten] 
    switch (X) {
        case 0:
        resp = -X;
        break;
    case 1:
         resp = X+X;
        break;
    case 2:
        resp = X * X; break;
    }

    return resp;
}

// CHECK: FunctionDecl {{.*}} used branch_switch 'int (int)'
// CHECK: AttributedStmt
// CHECK-NEXT: HLSLControlFlowHintAttr {{.*}} branch
export int branch_switch(int X){
    int resp;
    [branch] 
    switch (X) {
     case 0:
        resp = -X;
        break;
    case 1:
        resp = X+X;
        break;
    case 2:
        resp = X * X; break;
    }

    return resp;
}

// CHECK: FunctionDecl {{.*}} used no_attr_switch 'int (int)'
// CHECK-NOT: AttributedStmt
// CHECK-NOT: HLSLControlFlowHintAttr
export int no_attr_switch(int X){
    int resp;
    switch (X) {
        case 0:
        resp = -X;
        break;
    case 1:
         resp = X+X;
        break;
    case 2:
        resp = X * X; break;
    }

    return resp;
}
