// RUN: %clang -fprofile-generate -fprofile-function-groups=3 -fprofile-selected-function-group=0 -emit-llvm -S %s -o - | FileCheck %s --implicit-check-not="; {{.* (noprofile|skipprofile)}}" --check-prefixes=CHECK,SELECT0
// RUN: %clang -fprofile-generate -fprofile-function-groups=3 -fprofile-selected-function-group=1 -emit-llvm -S %s -o - | FileCheck %s --implicit-check-not="; {{.* (noprofile|skipprofile)}}" --check-prefixes=CHECK,SELECT1
// RUN: %clang -fprofile-generate -fprofile-function-groups=3 -fprofile-selected-function-group=2 -emit-llvm -S %s -o - | FileCheck %s --implicit-check-not="; {{.* (noprofile|skipprofile)}}" --check-prefixes=CHECK,SELECT2

// Group 0

// SELECT1: skipprofile
// SELECT2: skipprofile
// CHECK: define {{.*}} @hoo()
void hoo() {}

// Group 1
// SELECT0: skipprofile

// SELECT2: skipprofile
// CHECK: define {{.*}} @goo()
void goo() {}

// Group 2
// SELECT0: skipprofile
// SELECT1: skipprofile

// CHECK: define {{.*}} @boo()
void boo() {}
