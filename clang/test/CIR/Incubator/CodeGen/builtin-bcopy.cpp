// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

void foo(void) {
  // CIR-LABEL: cir.func {{.*}} @_Z3foov()
  // CIR: %[[V0:.*]] = cir.alloca !cir.array<!cir.float x 4>, !cir.ptr<!cir.array<!cir.float x 4>>, ["f4"] {alignment = 16 : i64}
  // CIR: %[[V1:.*]] = cir.alloca !cir.array<!cir.float x 8>, !cir.ptr<!cir.array<!cir.float x 8>>, ["f8"] {alignment = 16 : i64}
  // CIR: %[[V2:.*]] = cir.cast array_to_ptrdecay %[[V0]] : !cir.ptr<!cir.array<!cir.float x 4>> -> !cir.ptr<!cir.float>
  // CIR: %[[V3:.*]] = cir.cast bitcast %[[V2]] : !cir.ptr<!cir.float> -> !cir.ptr<!void>
  // CIR: %[[V4:.*]] = cir.cast array_to_ptrdecay %[[V1]] : !cir.ptr<!cir.array<!cir.float x 8>> -> !cir.ptr<!cir.float>
  // CIR: %[[V5:.*]] = cir.cast bitcast %[[V4]] : !cir.ptr<!cir.float> -> !cir.ptr<!void>
  // CIR: %[[V6:.*]] = cir.const #cir.int<4> : !u64i
  // CIR: %[[V7:.*]] = cir.const #cir.int<4> : !s32i
  // CIR: %[[V8:.*]] = cir.cast integral %[[V7]] : !s32i -> !u64i
  // CIR: %[[V9:.*]] = cir.binop(mul, %[[V6]], %[[V8]]) : !u64i
  // CIR: cir.libc.memmove %[[V9]] bytes from %[[V3]] to %[[V5]] : !cir.ptr<!void>, !u64i
  // CIR: cir.return

  // LLVM-LABEL: define dso_local void @_Z3foov()
  // LLVM: %[[V1:.*]] = alloca [4 x float], i64 1, align 16
  // LLVM: %[[V2:.*]] = alloca [8 x float], i64 1, align 16
  // LLVM: %[[V3:.*]] = getelementptr float, ptr %[[V1]], i32 0
  // LLVM: %[[V4:.*]] = getelementptr float, ptr %[[V2]], i32 0
  // LLVM: call void @llvm.memmove.p0.p0.i64(ptr %[[V4]], ptr %[[V3]], i64 16, i1 false)
  // LLVM: ret void

  float f4[4];
  float f8[8];
  __builtin_bcopy(f4, f8, sizeof(float) * 4);
}

void test_conditional_bcopy(void) {
  // CIR-LABEL: cir.func {{.*}} @_Z22test_conditional_bcopyv()
  // CIR: cir.libc.memmove {{.*}} bytes from {{.*}} to {{.*}} : !cir.ptr<!void>, !u64i
  // CIR: cir.libc.memmove {{.*}} bytes from {{.*}} to {{.*}} : !cir.ptr<!void>, !u64i

  // LLVM-LABEL: define{{.*}} void @_Z22test_conditional_bcopyv
  // LLVM: call void @llvm.memmove
  // LLVM: call void @llvm.memmove
  // LLVM-NOT: phi

  char dst[20];
  char src[20];
  int _sz = 20, len = 20;
  return (_sz ? ((_sz >= len) ? __builtin_bcopy(src, dst, len) : foo())
              : __builtin_bcopy(src, dst, len));
}

void another_conditional_bcopy(char *dst, char *src, int sz, int len) {
  // CIR-LABEL: cir.func {{.*}} @_Z25another_conditional_bcopyPcS_ii
  // CIR: cir.libc.memmove {{.*}} bytes from {{.*}} to {{.*}} : !cir.ptr<!void>, !u64i
  // CIR: cir.libc.memmove {{.*}} bytes from {{.*}} to {{.*}} : !cir.ptr<!void>, !u64i

  // LLVM-LABEL: define{{.*}} void @_Z25another_conditional_bcopyPcS_ii
  // LLVM: call void @llvm.memmove
  // LLVM: call void @llvm.memmove
  // LLVM-NOT: phi

  if (sz >= len)
    __builtin_bcopy(src, dst, len);
  else
    __builtin_bcopy(src, dst, len * 2);
}

#define size_t __SIZE_TYPE__

extern "C" void bcopy(const void *__src, void *__dest, size_t __n);

// LLVM: @_Z9testbcopyPKvPvm(
// LLVM:         call void @llvm.memmove.p0.p0.i64(ptr {{.*}}, ptr {{.*}}, i64 {{.*}}, i1 false)
// LLVM:    ret void

void testbcopy(const void *src, void *dest, size_t n) {
  bcopy(src, dest, n);
}
