// The llvm.memcpy length for a cir.copy (aggregate pass-by-value) is size_t-wide:
// i32 on 32-bit ARM, not the i64 used by 64-bit targets.
//
// RUN: %clang_cc1 -std=c++20 -triple arm-linux-gnueabihf -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple arm-linux-gnueabihf -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=ARM --input-file=%t.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-x86.ll
// RUN: FileCheck --check-prefix=X86 --input-file=%t-x86.ll %s

struct P { int x; int y; };
int sum(P p);
int use() { P p; p.x = 1; p.y = 2; return sum(p); }

// The cir.copy memcpy length width is resolved later, during lowering to LLVM.
// CIR-LABEL: cir.func{{.*}} @_Z3usev()
// CIR: cir.copy {{.*}} : !cir.ptr<!rec_P>

// ARM: call void @llvm.memcpy.p0.p0.i32(ptr {{.*}}, ptr {{.*}}, i32 8, i1 false)

// X86: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}, ptr {{.*}}, i64 8, i1 false)
