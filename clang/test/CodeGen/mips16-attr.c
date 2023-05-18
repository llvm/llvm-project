// RUN: %clang_cc1 -triple mipsel-linux-gnu -emit-llvm  -o  - %s | FileCheck %s
void __attribute__((mips16)) foo (void) {

}

// CHECK: define{{.*}} void @foo() [[MIPS16:#[0-9]+]]

void __attribute__((nomips16)) nofoo (void) {

}

// CHECK: define{{.*}} void @nofoo() [[NOMIPS16:#[0-9]+]]

// CHECK: attributes [[MIPS16]] = { noinline nounwind {{.*}} "mips16" {{.*}} }

// CHECK: attributes [[NOMIPS16]]  = { noinline nounwind {{.*}} "nomips16" {{.*}} }

