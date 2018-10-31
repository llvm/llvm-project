// RUN: %clang_cc1 %s -triple=i686-windows-msvc -emit-llvm -o - -mno-stack-arg-probe | FileCheck %s -check-prefix=NO-STACKPROBE
// RUN: %clang_cc1 %s -triple=x86_64-windows-msvc -emit-llvm -o - -mno-stack-arg-probe | FileCheck %s -check-prefix=NO-STACKPROBE
// RUN: %clang_cc1 %s -triple=armv7-windows-msvc -emit-llvm -o - -mno-stack-arg-probe | FileCheck %s -check-prefix=NO-STACKPROBE
// RUN: %clang_cc1 %s -triple=aarch64-windows-msvc -emit-llvm -o - -mno-stack-arg-probe | FileCheck %s -check-prefix=NO-STACKPROBE
// RUN: %clang_cc1 %s -triple=i686-windows-msvc -emit-llvm -o - | FileCheck %s -check-prefix=STACKPROBE
// RUN: %clang_cc1 %s -triple=x86_64-windows-msvc -emit-llvm -o - | FileCheck %s -check-prefix=STACKPROBE
// RUN: %clang_cc1 %s -triple=armv7-windows-msvc -emit-llvm -o - | FileCheck %s -check-prefix=STACKPROBE
// RUN: %clang_cc1 %s -triple=aarch64-windows-msvc -emit-llvm -o - | FileCheck %s -check-prefix=STACKPROBE


// NO-STACKPROBE: attributes #{{[0-9]+}} = {{{.*}} "no-stack-arg-probe"
// STACKPROBE-NOT: "no-stack-arg-probe"

void test1() {
}
