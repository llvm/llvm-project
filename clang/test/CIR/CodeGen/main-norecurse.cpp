// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

int f() { return 0; }

int main(int argc, char **argv) { return f(); }

// CIR: cir.func {{.*}} @_Z1fv(
// CIR-NOT: norecurse
// CIR: cir.func {{.*}} @main({{.*}} attributes {{.*}}norecurse

// LLVM: define {{.*}} @_Z1fv() #[[F_ATTR:[0-9]+]]
// LLVM: define {{.*}} @main({{.*}}) #[[MAIN_ATTR:[0-9]+]]
// LLVM-NOT: attributes #[[F_ATTR]] = {{.*}}norecurse
// LLVM: attributes #[[MAIN_ATTR]] = {{.*}}norecurse
