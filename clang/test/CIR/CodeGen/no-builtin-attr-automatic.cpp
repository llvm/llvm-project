// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-builtin-memset -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefixes=CIR,CIR-STD
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-builtin-memset -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=LLVM,LLVM-STD
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-builtin-memset -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=LLVM,LLVM-STD
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu  -fno-builtin-memset -fno-builtin -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefixes=CIR,CIR-NB
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-builtin-memset -fno-builtin -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=LLVM,LLVM-NB
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-builtin-memset -fno-builtin -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=LLVM,LLVM-NB

extern "C" {
__attribute__((hot))
__attribute__((no_builtin))
void no_builtin() {}
// CIR: cir.func{{.*}}@no_builtin()
// CIR-SAME:  nobuiltins = []
// LLVM: define{{.*}}@no_builtin() #[[NO_BUILTIN_ATTRS:.*]] {

__attribute__((cold))
__attribute__((no_builtin("memcpy")))
void no_memcpy() {}
// CIR: cir.func{{.*}}@no_memcpy()
// CIR-STD-SAME:  nobuiltins = ["memset", "memcpy"]
// CIR-NB-SAME:  nobuiltins = []
// LLVM: define{{.*}}@no_memcpy() #[[NO_MEMCPY_ATTRS:.*]] {

__attribute__((noduplicate))
void memset() {}
// CIR: cir.func{{.*}}@memset()
// CIR-STD-SAME: nobuiltins = ["memset"]
// CIR-NB-SAME: nobuiltins = []
// LLVM: define{{.*}}@memset() #[[MEMSET_ATTRS:.*]] {

void caller() {
  no_builtin();
  // CIR: cir.call @no_builtin()
  // CIR-NB-SAME: nobuiltin
  // CIR-SAME: nobuiltins = []
  // LLVM: call void @no_builtin() #[[NO_BUILTIN_CALL_ATTRS:.*]]
  no_memcpy();
  // CIR: cir.call @no_memcpy()
  // CIR-STD-SAME: nobuiltins = ["memset", "memcpy"]
  // CIR-NB-SAME: nobuiltin
  // CIR-NB-SAME: nobuiltins = []
  // LLVM: call void @no_memcpy() #[[NO_MEMCPY_CALL_ATTRS:.*]]
  memset();
  // CIR: cir.call @memset()
  // CIR-STD-SAME: nobuiltins = ["memset"]
  // CIR-NB-SAME: nobuiltin
  // CIR-NB-SAME: nobuiltins = []
  // LLVM: call void @memset() #[[MEMSET_CALL_ATTRS:.*]]
}
}

// LLVM: attributes #[[NO_BUILTIN_ATTRS]]
// LLVM-SAME: no-builtins
// LLVM: attributes #[[NO_MEMCPY_ATTRS]]
// LLVM-STD-SAME: no-builtin-memcpy
// LLVM-STD-SAME: no-builtin-memset
// LLVM-NB-SAME: no-builtins
// LLVM: attributes #[[MEMSET_ATTRS]]
// LLVM-STD-SAME: no-builtin-memset
// LLVM-NB-SAME: no-builtins
// LLVM: attributes #[[NO_BUILTIN_CALL_ATTRS]]
// LLVM-NB-SAME: nobuiltin
// LLVM-SAME: no-builtins
// LLVM: attributes #[[NO_MEMCPY_CALL_ATTRS]]
// LLVM-STD-SAME: no-builtin-memcpy
// LLVM-STD-SAME: no-builtin-memset
// LLVM-NB-SAME: nobuiltin
// LLVM-NB-SAME: no-builtins
// LLVM: attributes #[[MEMSET_CALL_ATTRS]]
// LLVM-STD-SAME: no-builtin-memset
// LLVM-NB-SAME: nobuiltin
// LLVM-NB-SAME: no-builtins
