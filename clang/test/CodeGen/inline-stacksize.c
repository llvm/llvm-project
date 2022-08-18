// RUN: %clang_cc1 -O2 -emit-llvm %s -o - | FileCheck %s --check-prefixes NOOPT
// RUN: %clang_cc1 -O2 -finline-max-stacksize=64 -emit-llvm %s -o - | FileCheck %s --check-prefix OPT

void foo() {}

// NOOPT-NOT: inline-max-stacksize
// OPT:       define {{.*}}@foo{{.*}}#[[ATTR:[0-9]+]]
// OPT:       attributes #[[ATTR]] = {{.*}}"inline-max-stacksize"="64"
