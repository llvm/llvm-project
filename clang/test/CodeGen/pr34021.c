// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -no-opaque-pointers -fms-extensions %s -triple=i686-unknown-unknown -emit-llvm -o - | FileCheck %s --check-prefix=X86
// RUN: %clang_cc1 -no-opaque-pointers -fms-extensions %s -triple=x86_64-unknown-unknown -emit-llvm -o - | FileCheck %s --check-prefix=X64

typedef int v4si __attribute__ ((vector_size (16)));
v4si rep(void) {
// X86-LABEL: define{{.*}} <4 x i32> @rep
// X86: %[[ALLOCA0:.*]] = alloca <4 x i32>, align 16
// X86: %[[ALLOCA1:.*]] = alloca <4 x i32>, align 16
// X86: %[[BITCAST:.*]] = bitcast <4 x i32>* %[[ALLOCA0]] to i128*
// X86: %[[ASM:.*]] = call i64 asm sideeffect inteldialect "", "=A,~{dirflag},~{fpsr},~{flags}"()
// X86: %[[ZEXT:.*]] = zext i64 %[[ASM]] to i128
// X86: store i128 %[[ZEXT]], i128* %[[BITCAST]], align 16
// X86: %[[LOAD:.*]] = load <4 x i32>, <4 x i32>* %[[ALLOCA1]], align 16
// X86: ret <4 x i32> %[[LOAD]]
//
// X64-LABEL: define{{.*}} <4 x i32> @rep
// X64: %[[ALLOCA:.*]] = alloca <4 x i32>, align 16
// X64: call void asm sideeffect inteldialect "", "~{dirflag},~{fpsr},~{flags}"()
// X64: %[[LOAD:.*]] = load <4 x i32>, <4 x i32>* %[[ALLOCA]], align 16
// X64: ret <4 x i32> %[[LOAD]]
  v4si res;
  __asm {}
  return res;
}
