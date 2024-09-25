// RUN: %clang_cc1 -fclangir -emit-cir -triple spirv64-unknown-unknown %s -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=CIR


// CIR: #fn_attr[[KERNEL1:[0-9]*]] = {{.+}}cl.kernel = #cir.cl.kernel
// CIR-NEXT: #fn_attr[[FUNC1:[0-9]*]] =
// CIR-NOT: cl.kernel = #cir.cl.kernel

kernel void kernel1() {}
// CIR: cir.func @kernel1{{.+}} extra(#fn_attr[[KERNEL1]])

void func1() {}

// CIR: cir.func @func1{{.+}} extra(#fn_attr[[FUNC1]])
