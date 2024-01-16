// REQUIRES: arm-registered-target
// RUN: %clang -target arm-none-eabi -emit-llvm -S -o - %s | FileCheck %s

// CHECK-LABEL: @v8() #0
__attribute__((target("arch=armv8-a")))
void v8() {}
// CHECK-LABEL: @v8crc() #1
__attribute__((target("arch=armv8-a+crc")))
void v8crc() {}

// CHECK: attributes #0 = { {{.*}} "target-features"="{{.*}}+armv8-a{{.*}}" }
// CHECK: attributes #1 = { {{.*}} "target-features"="{{.*}}+armv8-a,+crc{{.*}}" }

