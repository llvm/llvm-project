// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -O0 -o - %s | FileCheck %s --check-prefix=NOTBAA
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -O1 -o - %s | FileCheck %s --check-prefix=ERRNO-TBAA
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -x c++ -emit-llvm -O1 -o - %s | FileCheck %s --check-prefix=ERRNO-TBAA
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -x c++ -emit-llvm -O1 -new-struct-path-tbaa -o - %s | FileCheck %s --check-prefix=ERRNO-TBAA-NEW
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -O1 -relaxed-aliasing -o - %s | FileCheck %s --check-prefix=NOSTRICT

// Ensure !llvm.errno.tbaa metadata is emitted upon integer accesses, if TBAA is available.

int int_access(int *ptr) { return ptr ? *ptr : 0; }

// NOTBAA-NOT: !llvm.errno.tbaa
// NOSTRICT-NOT: !llvm.errno.tbaa
// ERRNO-TBAA: !llvm.errno.tbaa = !{![[TAG:[0-9]+]]}
// ERRNO-TBAA: ![[TAG]] = !{![[BASE:[0-9]+]], ![[INT:[0-9]+]], i64 0}
// ERRNO-TBAA: ![[BASE]] = !{!"__libc_errno", ![[INT]], i64 0}
// ERRNO-TBAA: ![[INT]] = !{!"int", !{{.*}}, i64 0}
// ERRNO-TBAA-NEW: !llvm.errno.tbaa = !{![[TAG:[0-9]+]]}
// ERRNO-TBAA-NEW: ![[TAG]] = !{![[BASE:[0-9]+]], ![[INT:[0-9]+]], i64 0, i64 4}
// ERRNO-TBAA-NEW: ![[BASE]] = !{!{{.*}}, i64 4, !"__libc_errno", ![[INT]], i64 0, i64 4}
// ERRNO-TBAA-NEW: ![[INT]] = !{!{{.*}}, i64 4, !"int"}
