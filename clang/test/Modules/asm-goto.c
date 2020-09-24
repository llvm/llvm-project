// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -x c -I%S/Inputs/asm-goto -emit-module %S/Inputs/asm-goto/module.modulemap -fmodule-name=a -o %t/a.pcm
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -x c -I%S/Inputs/asm-goto -emit-llvm -o - %s -fmodule-file=%t/a.pcm | FileCheck %s
#include "a.h"

// CHECK-LABEL: define {{.*}} @foo(
// CHECK: callbr {{.*}} "=r,X{{.*}} blockaddress(@foo, %indirect))
// CHECK-NEXT: to label %asm.fallthrough [label %indirect]

int bar(void) {
  return foo();
}
