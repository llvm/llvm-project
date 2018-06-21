// REQUIRES: arm-registered-target

// RUN: %clang_cc1 -triple thumbv7-linux-gnueabihf -emit-llvm -o - %s | FileCheck --check-prefix THUMB %s
// RUN: %clang_cc1 -triple thumbv7eb-linux-gnueabihf -emit-llvm -o - %s | FileCheck --check-prefix THUMB %s
// RUN: %clang -target armv7-linux-gnueabihf -mthumb -S -emit-llvm -o - %s | FileCheck --check-prefix THUMB-CLANG %s
// RUN: %clang_cc1 -triple armv7-linux-gnueabihf -emit-llvm -o - %s | FileCheck --check-prefix ARM %s
// RUN: %clang_cc1 -triple armv7eb-linux-gnueabihf -emit-llvm -o - %s | FileCheck --check-prefix ARM %s

void t1() {}

 __attribute__((target("no-thumb-mode")))
void t2() {}

 __attribute__((target("thumb-mode")))
void t3() {}

// THUMB: void @t1() [[ThumbAttr:#[0-7]]]
// THUMB: void @t2() [[NoThumbAttr:#[0-7]]]
// THUMB: void @t3() [[ThumbAttr:#[0-7]]]
// THUMB: attributes [[ThumbAttr]] = { {{.*}} "target-features"="+thumb-mode"
// THUMB: attributes [[NoThumbAttr]] = { {{.*}} "target-features"="-thumb-mode"
//
// THUMB-CLANG: void @t1() [[ThumbAttr:#[0-7]]]
// THUMB-CLANG: void @t2() [[NoThumbAttr:#[0-7]]]
// THUMB-CLANG: void @t3() [[ThumbAttr:#[0-7]]]
// THUMB-CLANG: attributes [[ThumbAttr]] = { {{.*}} "target-features"="{{.*}}+thumb-mode
// THUMB-CLANG: attributes [[NoThumbAttr]] = { {{.*}} "target-features"="{{.*}}-thumb-mode

// ARM: void @t1() [[NoThumbAtr:#[0-7]]]
// ARM: void @t2() [[NoThumbAttr:#[0-7]]]
// ARM: void @t3() [[ThumbAttr:#[0-7]]]
// ARM: attributes [[NoThumbAttr]] = { {{.*}} "target-features"="-thumb-mode"
// ARM: attributes [[ThumbAttr]] = { {{.*}} "target-features"="+thumb-mode"
