// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -O0 -o - %s | FileCheck %s --check-prefix=NOTBAA
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -O1 -o - %s | FileCheck %s --check-prefix=ERRNO-TBAA
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -x c++ -emit-llvm -O1 -o - %s | FileCheck %s --check-prefix=ERRNO-TBAA
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -O1 -relaxed-aliasing -o - %s | FileCheck %s --check-prefix=NOSTRICT

// Ensure !llvm.errno.tbaa metadata is emitted upon integer accesses, if TBAA is available.

int int_access(int *ptr) { return ptr ? *ptr : 0; }

// NOTBAA-NOT: !llvm.errno.tbaa
// ERRNO-TBAA: !llvm.errno.tbaa
// NOSTRICT-NOT: !llvm.errno.tbaa
