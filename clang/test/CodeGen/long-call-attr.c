// RUN: %clang_cc1 -triple mips-linux-gnu -emit-llvm  -o  - %s | FileCheck %s
// RUN: %clang_cc1 -triple mips64-linux-gnu -emit-llvm  -o  - %s | FileCheck %s

void __attribute__((long_call)) foo1 (void);
void __attribute__((short_call)) foo4 (void);

void __attribute__((far)) foo2 (void) {}

// CHECK: define{{.*}} void @foo2() [[FAR:#[0-9]+]]

void __attribute__((near)) foo3 (void) { foo1(); foo4(); }

// CHECK: define{{.*}} void @foo3() [[NEAR:#[0-9]+]]

// CHECK: declare void @foo1() [[LONGDECL:#[0-9]+]]
// CHECK: declare void @foo4() [[SHORTDECL:#[0-9]+]]

// CHECK: attributes [[FAR]] = { {{.*}} "long-call" {{.*}} }
// CHECK: attributes [[NEAR]] = { {{.*}} "short-call" {{.*}} }
// CHECK: attributes [[LONGDECL]] = { "long-call" {{.*}} }
// CHECK: attributes [[SHORTDECL]] = { {{.*}} "short-call" {{.*}} }
