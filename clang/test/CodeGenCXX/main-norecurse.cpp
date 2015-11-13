// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

// CHECK: define i{{.*}} @main({{.*}}) #0
int main(int argc, char **argv) {
    return 1;
}

// CHECK: attributes #0 = { norecurse{{.*}} }
