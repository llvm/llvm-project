// RUN: %clang -cc1 -triple %itanium_abi_triple %s -debug-info-kind=limited -fdynamic-debugging -o - \
// RUN:   -emit-llvm --discard-dynamic-debugging-debug-module \
// RUN: | FileCheck %s

// CHECK: define dso_local i32 @f() #0
// CHECK: attributes #0 = {{{.*}}"no-func-spec"{{.*}}}
int f() { return 0; }
