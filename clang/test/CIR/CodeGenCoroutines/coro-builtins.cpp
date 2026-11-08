// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcoroutines -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefixes=CIR,CIR64
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcoroutines -fclangir -emit-llvm -disable-llvm-passes %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefixes=LLVM,LLVM64
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcoroutines -emit-llvm -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=LLVM,LLVM64
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fcoroutines -fclangir -emit-cir %s -o %t.32.cir
// RUN: FileCheck --input-file=%t.32.cir %s -check-prefixes=CIR,CIR32
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fcoroutines -fclangir -emit-llvm -disable-llvm-passes %s -o %t-cir.32.ll
// RUN: FileCheck --input-file=%t-cir.32.ll %s -check-prefixes=LLVM,LLVM32
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fcoroutines -emit-llvm -disable-llvm-passes %s -o %t.32.ll
// RUN: FileCheck --input-file=%t.32.ll %s -check-prefixes=LLVM,LLVM32

void *myAlloc(long long);

// CIR: cir.func {{.*}} @_Z1fi
// LLVM: void @_Z1fi
void f(int n) {
  int promise;
  // CIR: %[[ADDR:.*]] = cir.alloca "n"
  // CIR: %[[PROMISE:.*]] = cir.alloca "promise"

  // LLVM: %[[ADDR:.*]] = alloca i32
  // LLVM: %[[PROMISE:.*]] = alloca i32

  __builtin_coro_id(32, &promise, 0, 0);
  // CIR: %[[CORO_ID_ALIGN:.*]] = cir.const #cir.int<32>
  // CIR: %[[CAS_PROM:.*]] = cir.cast bitcast %[[PROMISE]]
  // CIR: %[[COROID:.*]] = cir.coro.intrinsic.id(%[[CORO_ID_ALIGN]], %[[CAS_PROM]], {{.*}}, {{.*}})

  // LLVM: %[[COROID:.*]] = call token @llvm.coro.id(i32 32, ptr %[[PROMISE]], ptr null, ptr null)

  __builtin_coro_alloc();
  // CIR: cir.coro.intrinsic.alloc(%[[COROID]])

  // LLVM: call i1 @llvm.coro.alloc(token %[[COROID]])

  __builtin_coro_noop();
  // CIR: %[[NOOP:.*]] = cir.coro.intrinsic.noop() : () -> !cir.ptr<!void>

  // LLVM: %[[NOOP:.*]] = call ptr @llvm.coro.noop()

  __builtin_coro_align();
  // CIR64: %[[ALIGN:.*]] = cir.coro.intrinsic.align() : () -> !u64i
  // CIR32: %[[ALIGN:.*]] = cir.coro.intrinsic.align() : () -> !u32i

  // LLVM64: %[[ALIGN:.*]] = call i64 @llvm.coro.align.i64()
  // LLVM32: %[[ALIGN:.*]] = call i32 @llvm.coro.align.i32()

  __builtin_coro_begin(myAlloc(__builtin_coro_size()));
  // CIR64: %[[SIZE:.*]] = cir.coro.intrinsic.size() : () -> !u64i
  // CIR64: %[[CAST_SIZE:.*]] = cir.cast integral %[[SIZE]] : !u64i -> !s64i
  // CIR32: %[[SIZE:.*]] = cir.coro.intrinsic.size() : () -> !u32i
  // CIR32: %[[CAST_SIZE:.*]] = cir.cast integral %[[SIZE]] : !u32i -> !s64i
  // CIR: %[[MEM:.*]] = cir.call @_Z7myAllocx(%[[CAST_SIZE]])
  // CIR: %[[FRAME:.*]] = cir.coro.intrinsic.begin(%[[COROID]], %[[MEM]])

  // LLVM64: %[[SIZE:.*]] = call i64 @llvm.coro.size.i64()
  // LLVM64: %[[MEM:.*]] = call noundef ptr @_Z7myAllocx(i64 noundef %[[SIZE]])
  // LLVM32: %[[SIZE:.*]] = call i32 @llvm.coro.size.i32()
  // LLVM32: %[[SIZE_EXT:.*]] = zext i32 %[[SIZE]] to i64
  // LLVM32: %[[MEM:.*]] = call noundef ptr @_Z7myAllocx(i64 noundef %[[SIZE_EXT]])
  // LLVM: %[[FRAME:.*]] = call ptr @llvm.coro.begin(token %[[COROID]], ptr %[[MEM]])

  __builtin_coro_resume(__builtin_coro_frame());
  // CIR: cir.coro.intrinsic.resume(%[[FRAME]]) : (!cir.ptr<!void>)
  // LLVM-NEXT: call void @llvm.coro.resume(ptr %[[FRAME]])

  __builtin_coro_destroy(__builtin_coro_frame());
  // CIR: cir.coro.intrinsic.destroy(%[[FRAME]]) : (!cir.ptr<!void>)
  // LLVM-NEXT: call void @llvm.coro.destroy(ptr %[[FRAME]])

  __builtin_coro_done(__builtin_coro_frame());
  // CIR: cir.coro.intrinsic.done(%[[FRAME]]) : (!cir.ptr<!void>) -> !cir.bool
  // LLVM-NEXT: call i1 @llvm.coro.done(ptr %[[FRAME]])

  __builtin_coro_promise(__builtin_coro_frame(), 48, 0);
  // CIR: %[[ALIGN:.*]] = cir.const #cir.int<48> : !s32i
  // CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
  // CIR: %[[FALSE:.*]] = cir.cast int_to_bool %[[ZERO]] : !s32i -> !cir.bool
  // CIR: cir.coro.intrinsic.promise(%[[FRAME]], %[[ALIGN]], %[[FALSE]])
  // LLVM: call ptr @llvm.coro.promise(ptr %[[FRAME]], i32 48, i1 false)

  __builtin_coro_free(__builtin_coro_frame());
  // CIR: cir.coro.intrinsic.free(%[[COROID]], %[[FRAME]])

  // LLVM: call ptr @llvm.coro.free(token %[[COROID]], ptr %[[FRAME]])

  __builtin_coro_end(__builtin_coro_frame(), false);
  // CIR: %[[FALSE:.*]] = cir.const #false
  // CIR: %[[TK_NONE:.*]] = cir.token.none
  // CIR: cir.coro.intrinsic.end(%[[FRAME]], %[[FALSE]], %[[TK_NONE]]) : (!cir.ptr<!void>, !cir.bool, token)

  // LLVM: call void @llvm.coro.end(ptr %[[FRAME]], i1 false, token none)

  __builtin_coro_suspend(true);
  // CIR: %[[TK_NONE2:.*]] = cir.token.none
  // CIR: %[[TRUE:.*]] = cir.const #true
  // CIR: cir.coro.intrinsic.suspend(%[[TK_NONE2]], %[[TRUE]])

  // LLVM: call i8 @llvm.coro.suspend(token none, i1 true)
}

void test_suspend_switch() {
  switch (__builtin_coro_suspend(false)) {
  case -1:
    return;
  case 0:
    break;
  }
}

// CIR: cir.func{{.*}} @_Z19test_suspend_switchv
// CIR: %[[TK_NONE:.*]] = cir.token.none
// CIR: %[[FALSE:.*]] = cir.const #false
// CIR: %[[SUSPEND:.*]] = cir.coro.intrinsic.suspend(%[[TK_NONE]], %[[FALSE]]) : (token, !cir.bool) -> !s8i
// CIR: %[[CAST:.*]] = cir.cast integral %[[SUSPEND]] : !s8i -> !s32i
// CIR: cir.switch(%[[CAST]] : !s32i)

// LLVM: define{{.*}} void @_Z19test_suspend_switchv
// LLVM: %[[SUSPEND:.*]] = call i8 @llvm.coro.suspend(token none, i1 false)
// LLVM: %[[CAST:.*]] = sext i8 %[[SUSPEND]] to i32
// LLVM: switch i32 %[[CAST]], label %{{.*}} [
// LLVM:   i32 -1, label %{{.*}}
// LLVM:   i32 0, label %{{.*}}
// LLVM: ]
