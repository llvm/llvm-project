// RUN: %clang_cc1 -triple i686-linux-gnu -target-cpu i686 -tune-cpu nehalem -emit-llvm %s -o - | FileCheck %s

int baz(int a) { return 4; }

// CHECK: baz{{.*}} #0
// CHECK: #0 = {{.*}}"target-cpu"="i686" "target-features"="+cmov,+cx8,+x87" "tune-cpu"="nehalem"
