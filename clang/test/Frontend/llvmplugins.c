// RUN: %clang_cc1 -load %llvmshlibdir/LLVMPrintFunctionNames%pluginext -o /dev/null -emit-llvm %s 2>&1 | FileCheck %s
// REQUIRES: plugins, examples

// CHECK: [PrintPass] Found function: x 
int x(int y){ return y; }
