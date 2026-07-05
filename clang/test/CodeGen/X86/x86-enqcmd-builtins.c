// RUN: %clang_cc1 %s -ffreestanding -triple i386-unknown-unknown -target-feature +enqcmd -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -ffreestanding -triple x86_64-unknown-unknown -target-feature +enqcmd -emit-llvm -o - | FileCheck %s

#include <immintrin.h>

int test_enqcmd(void *dst, const void *src) {
// CHECK-LABEL: @test_enqcmd
// CHECK: %[[TMP0:.+]] = call i8 @llvm.x86.enqcmd(ptr %{{.+}}, ptr %{{.+}})
// CHECK: %[[RET:.+]] = zext i8 %[[TMP0]] to i32
// CHECK: ret i32 %[[RET]]
    return _enqcmd(dst, src);
}

int test_enqcmds(void *dst, const void *src) {
// CHECK-LABEL: @test_enqcmds
// CHECK: %[[TMP0:.+]] = call i8 @llvm.x86.enqcmds(ptr %{{.+}}, ptr %{{.+}})
// CHECK: %[[RET:.+]] = zext i8 %[[TMP0]] to i32
// CHECK: ret i32 %[[RET]]
    return _enqcmds(dst, src);
}
