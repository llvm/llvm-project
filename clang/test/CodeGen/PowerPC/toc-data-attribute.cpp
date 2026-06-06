// RUN: %clang_cc1 %s -triple powerpc-ibm-aix-xcoff -mtocdata -emit-llvm -o - 2>&1 | FileCheck %s -check-prefixes=COMMON,ALLTOC
// RUN: %clang_cc1 %s -triple powerpc-ibm-aix-xcoff -mtocdata=n,_ZN11MyNamespace10myVariableE,_ZL1s,_ZZ4testvE7counter -emit-llvm -o - 2>&1 | FileCheck %s -check-prefixes=COMMON,TOCLIST
// RUN: %clang_cc1 %s -triple powerpc64-ibm-aix-xcoff -mtocdata -emit-llvm -o - 2>&1 | FileCheck %s -check-prefixes=COMMON,ALLTOC
// RUN: %clang_cc1 %s -triple powerpc64-ibm-aix-xcoff -mtocdata=n,_ZN11MyNamespace10myVariableE,_ZL1s,_ZZ4testvE7counter -emit-llvm -o - 2>&1 | FileCheck %s -check-prefixes=COMMON,TOCLIST

extern int n;
static int s = 100;

inline int test() {
    static int counter = 0;
    counter++;
    return counter;
}

int a () {
    n = test();
    return 0;
}

namespace MyNamespace {
    int myVariable = 10;
}

int b(int x) {
    using namespace MyNamespace;
    return x + myVariable;
}

int c(int x) {
  s += x;
  return s;
}

// COMMON: @n = external global i32, align 4 #0
// COMMON: @_ZN11MyNamespace10myVariableE = global i32 10, align 4 #0
// COMMON-NOT: @_ZL1s = internal global i32 100, align 4 #0
// ALLTOC: @_ZZ4testvE7counter = linkonce_odr global i32 0, align 4 #0
// TOCLIST-NOT: @_ZZ4testvE7counter = linkonce_odr global i32 0, align 4 #0
// COMMON: attributes #0 = { "toc-data" }
