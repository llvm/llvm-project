// RUN: %clang_cc1 -triple loongarch64 -target-feature -lsx -emit-llvm %s -o - | FileCheck %s

__attribute__((target("lasx")))
// CHECK: #[[ATTR0:[0-9]+]] {
void testlasx() {}

// CHECK: attributes #[[ATTR0]] = { {{.*}}"target-features"="+64bit,+lasx,+lsx"{{.*}} }
