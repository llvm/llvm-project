// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -disable-llvm-passes -o - %s -O1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s -O0 | FileCheck %s --check-prefix=CHECK_O0

// When optimizing, the builtin should be converted to metadata.
// When not optimizing, there should be no metadata created for the builtin.
// In both cases, the builtin should be removed from the code.

void foo();
void branch(int x) {
// CHECK-LABEL: define void @branch(

// CHECK-NOT: builtin_unpredictable
// CHECK: !unpredictable [[METADATA:.+]]

// CHECK_O0-NOT: builtin_unpredictable
// CHECK_O0-NOT: !unpredictable 

  if (__builtin_unpredictable(x > 0))
    foo ();
}

int unpredictable_switch(int x) {
// CHECK-LABEL: @unpredictable_switch(

// CHECK-NOT: builtin_unpredictable
// CHECK: !unpredictable [[METADATA:.+]]

// CHECK_O0-NOT: builtin_unpredictable
// CHECK_O0-NOT: !unpredictable 

  switch(__builtin_unpredictable(x)) {
  default:
    return 0;
  case 0:
  case 1:
  case 2:
    return 1;
  case 5:
    return 5;
  };

  return 0;
}

// CHECK: [[METADATA]] = !{}

