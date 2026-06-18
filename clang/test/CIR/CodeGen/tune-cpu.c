// RUN: %clang_cc1 -triple i686-linux-gnu -target-cpu i686 -tune-cpu nehalem -fclangir -emit-cir %s -o - | FileCheck %s -check-prefix=CIR
// RUN: %clang_cc1 -triple i686-linux-gnu -target-cpu i686 -tune-cpu nehalem -fclangir -emit-llvm %s -o - | FileCheck %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple i686-linux-gnu -target-cpu i686 -tune-cpu nehalem -emit-llvm %s -o - | FileCheck %s -check-prefix=LLVM

int baz(int a) { return 4; }

// CIR: cir.func{{.*}} @baz
// CIR-SAME: "cir.target-cpu" = "i686"
// CIR-SAME: "cir.target-features" = "+cmov,+cx8,+x87"
// CIR-SAME: "cir.tune-cpu" = "nehalem"

// LLVM: baz{{.*}} #[[ATTR:[0-9]+]]
// LLVM: attributes #[[ATTR]] = {{.*}}"target-cpu"="i686" "target-features"="+cmov,+cx8,+x87" "tune-cpu"="nehalem"
