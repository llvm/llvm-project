// RUN: %clang_cc1 -triple x86_64-linux-gnu -target-cpu x86-64 -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=HARD
// RUN: %clang_cc1 -msoft-float -triple x86_64-linux-gnu -target-cpu x86-64 -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=SOFT

int __attribute__((target("x87"))) foo(int a) { return 4; }
int __attribute__((target("no-x87"))) bar(int a) { return 4; }

// CHECK: foo{{.*}} #0
// CHECK: bar{{.*}} #1

// CHECK: #0 = {{.*}}"target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"
// HARD-NOT: "use-soft-float"
// SOFT: "use-soft-float"="true"

// CHECK: #1 = {{.*}}"target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,-x87"
// HARD-NOT: "use-soft-float"
// SOFT: "use-soft-float"="true"
