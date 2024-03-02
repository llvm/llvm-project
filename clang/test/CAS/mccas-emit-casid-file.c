// REQUIRES: x86-registered-target
// RUN: rm -rf %t && mkdir %t

// RUN: %clang -cc1depscan -o %t/args.rsp  -cc1-args -cc1 -triple x86_64-apple-darwin10 \
// RUN:    -debug-info-kind=standalone -dwarf-version=4 -debugger-tuning=lldb \
// RUN:    -emit-obj -fcas-backend  -fcas-path %t/cas  -fcas-emit-casid-file %s

// RUN: %clang @%t/args.rsp -o %t/output1.o 

// cat %t/output1.o.casid | FileCheck %s

// CHECK: llvmcas://{{[a-z0-9]+}}

int foo(int x) {
    return x+1;
}
