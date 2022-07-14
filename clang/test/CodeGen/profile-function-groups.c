// RUN: %clang -fprofile-generate -fprofile-function-groups=3 -fprofile-selected-function-group=0 -emit-llvm -S %s -o - | FileCheck %s --check-prefixes=CHECK,SELECT0
// RUN: %clang -fprofile-generate -fprofile-function-groups=3 -fprofile-selected-function-group=1 -emit-llvm -S %s -o - | FileCheck %s --check-prefixes=CHECK,SELECT1
// RUN: %clang -fprofile-generate -fprofile-function-groups=3 -fprofile-selected-function-group=2 -emit-llvm -S %s -o - | FileCheck %s --check-prefixes=CHECK,SELECT2

// Group 0
// SELECT0-NOT: noprofile
// SELECT1: noprofile
// SELECT2: noprofile
// CHECK: define {{.*}} @hoo()
void hoo() {}

// Group 1
// SELECT0: noprofile
// SELECT1-NOT: noprofile
// SELECT2: noprofile
// CHECK: define {{.*}} @goo()
void goo() {}

// Group 2
// SELECT0: noprofile
// SELECT1: noprofile
// SELECT2-NOT: noprofile
// CHECK: define {{.*}} @boo()
void boo() {}
