// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t.ll %s

#define __maybe_undef __attribute__((maybe_undef))

struct S { int a, b; };
void scalar(int a, int __maybe_undef b, int c);
void variadic(int __maybe_undef x, ...);
void record(S __maybe_undef s);

void test() {
  int x, y, z;
  S s;
  scalar(x, y, z);
  variadic(x, y);
  record(s);
}

// CIR-LABEL: cir.func{{.*}} @_Z4testv()
// CIR:         %[[X:.*]] = cir.load{{.*}} : !cir.ptr<!s32i>, !s32i
// CIR:         %[[Y:.*]] = cir.load{{.*}} : !cir.ptr<!s32i>, !s32i
// CIR:         %[[Z:.*]] = cir.load{{.*}} : !cir.ptr<!s32i>, !s32i
// CIR:         %[[FY:.*]] = cir.freeze %[[Y]] : !s32i
// CIR-NEXT:    cir.call @_Z6scalariii(%[[X]], %[[FY]], %[[Z]])
// CIR:         %[[X2:.*]] = cir.load{{.*}} : !cir.ptr<!s32i>, !s32i
// CIR:         %[[Y2:.*]] = cir.load{{.*}} : !cir.ptr<!s32i>, !s32i
// CIR:         %[[FX2:.*]] = cir.freeze %[[X2]] : !s32i
// CIR-NEXT:    cir.call @_Z8variadiciz(%[[FX2]], %[[Y2]])
// CIR:         %[[SV:.*]] = cir.load{{.*}} : !cir.ptr<!rec_S>, !rec_S
// CIR:         cir.freeze %[[SV]] : !rec_S
// CIR:         cir.call @_Z6record1S(

// LLVM-LABEL: define{{.*}} void @_Z4testv()
// LLVM:         %[[X:.*]] = load i32
// LLVM:         %[[Y:.*]] = load i32
// LLVM:         %[[Z:.*]] = load i32
// LLVM:         %[[FY:.*]] = freeze i32 %[[Y]]
// LLVM-NEXT:    call void @_Z6scalariii(i32 noundef %[[X]], i32 noundef %[[FY]], i32 noundef %[[Z]])
// LLVM:         %[[X2:.*]] = load i32
// LLVM:         %[[Y2:.*]] = load i32
// LLVM:         %[[FX2:.*]] = freeze i32 %[[X2]]
// LLVM-NEXT:    call void (i32, ...) @_Z8variadiciz(i32 noundef %[[FX2]], i32 noundef %[[Y2]])
// LLVMCIR:      %[[SV:.*]] = load %struct.S
// LLVMCIR:      freeze %struct.S %[[SV]]
// OGCG:         %[[SV:.*]] = load i64
// OGCG:         freeze i64 %[[SV]]
// LLVM:         call void @_Z6record1S(i64
