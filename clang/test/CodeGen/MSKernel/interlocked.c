// Check that we don't fold no-op andl to memory barrier
// RUN: %clang_cc1 -fms-kernel -fms-extensions -Wno-implicit-function-declaration  -triple x86_64-pc-win32 -O2 -S -o - %s | FileCheck %s --check-prefix=X86
// RUN: %clang_cc1 -fms-kernel -fms-extensions -Wno-implicit-function-declaration  -triple aarch64-pc-win32 -O2 -S -o - %s | FileCheck %s --check-prefix=ARM64

// X86:      lock andl $-1, (%rcx)
// X86-NEXT: retq

// ARM64:      ldaxr
// ARM64-NEXT: stlxr
// ARM64-NEXT: cbnz

void access_via_interlocked(long volatile* addr) {
    _InterlockedAnd(addr, (long)-1);
}
