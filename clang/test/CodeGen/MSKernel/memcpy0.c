// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -fms-kernel -triple x86_64-windows-msvc -O2 -emit-llvm %s -o - | FileCheck %s --check-prefix=IR
// RUN: %clang_cc1 -fms-kernel -triple x86_64-windows-msvc -O2 -S %s -o - | FileCheck %s --check-prefix=ASM

// LLVM selects alignment of 1 << 32 for null pointer,
// which is a maximum allowed value according to https://llvm.org/docs/LangRef.html
// IR:      define {{.*}} void @builtin_copy_from_nullptr
// IR-NEXT: entry:
// IR-NEXT:  tail call void @llvm.memcpy.p0.p0.i64(ptr align 1 %dst, ptr align 4294967296 null, i64 %n, i1 false)
// IR-NEXT:  ret void

// ASM:      builtin_copy_from_nullptr:
// ASM-NEXT: # %bb.0:
// ASM-NEXT: movq %rdx, %r8
// ASM-NEXT: xorl %edx, %edx
// ASM-NEXT: jmp memcpy

void builtin_copy_from_nullptr(void* dst, long long n) {
    __builtin_memcpy(dst, (void*)0, n);
}

