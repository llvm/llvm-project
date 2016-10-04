// RUN: %clang_cc1 -mrecip=!sqrtf,vec-divf:3 -emit-llvm %s -o - | FileCheck %s

int baz(int a) { return 4; }

// CHECK: baz{{.*}} #0
// CHECK: #0 = {{.*}}"reciprocal-estimates"="!sqrtf,vec-divf:3"

