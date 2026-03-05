// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --input-file=%t.og.ll %s --check-prefix=OGCG
//
// XFAIL: *
//
// CIR crashes when using computed goto (GNU extension).
//
// Computed goto allows taking the address of a label and jumping to it
// dynamically. This is implemented via the AddrLabelExpr AST node.
//
// Currently, CIR crashes with:
//   Assertion `0 && "NYI"' failed
//   at CIRGenExprConst.cpp:1634 in ConstantLValueEmitter::VisitAddrLabelExpr
//
// The issue is that CIR's constant expression emitter doesn't handle
// AddrLabelExpr (&&label syntax).
//
// This affects code using computed goto, which is common in interpreters,
// state machines, and performance-critical dispatch code.

int test_computed_goto(int x) {
    void* labels[] = {&&label0, &&label1, &&label2};

    if (x >= 0 && x <= 2)
        goto *labels[x];
    return -1;

label0:
    return 0;
label1:
    return 10;
label2:
    return 20;
}

// LLVM: Should generate indirectbr
// LLVM: define {{.*}} @_Z18test_computed_gotoi({{.*}})

// OGCG: Should use blockaddress and indirectbr
// OGCG: define {{.*}} @_Z18test_computed_gotoi({{.*}})
// OGCG: blockaddress(@_Z18test_computed_gotoi
// OGCG: indirectbr
