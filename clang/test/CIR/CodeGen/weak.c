// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

extern void B (void);
static __typeof(B) A __attribute__ ((__weakref__("B")));

void active (void)
{
  A();
}

// CIR:      cir.func extern_weak private @B()
// CIR:      cir.func @active()
// CIR-NEXT:   cir.call @B() : () -> ()

// LLVM:     declare !dbg !{{.}} extern_weak void @B()
// LLVM:     define void @active()
// LLVM-NEXT:  call void @B()
