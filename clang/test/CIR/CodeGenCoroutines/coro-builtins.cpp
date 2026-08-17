// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcoroutines -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR

void *myAlloc(long long);

// CIR: cir.func {{.*}} @_Z1fi
void f(int n) {
  int promise;
  // CIR: %[[ADDR:.*]] = cir.alloca "n"
  // CIR: %[[PROMISE:.*]] = cir.alloca "promise"

  __builtin_coro_id(32, &promise, 0, 0);
  // CIR: %[[CORO_ID_ALIGN:.*]] = cir.const #cir.int<32>
  // CIR: %[[CAS_PROM:.*]] = cir.cast bitcast %[[PROMISE]]
  // CIR: %[[COROID:.*]] = cir.coro.intrinsic.id(%[[CORO_ID_ALIGN]], %[[CAS_PROM]], {{.*}}, {{.*}})

  __builtin_coro_alloc();
  // CIR: cir.coro.intrinsic.alloc(%[[COROID]])

  // TODO
  //__builtin_coro_noop();

  __builtin_coro_begin(myAlloc(__builtin_coro_size()));
  // TODO(CIR): Support both variants of the coroutine size intrinsic, matching
  // `llvm.coro.size.i32` and `llvm.coro.size.i64`.
  // CIR: %[[SIZE:.*]] = cir.coro.intrinsic.size()
  // CIR: %[[CAST_SIZE:.*]] = cir.cast integral %[[SIZE]] : !u64i -> !s64i
  // CIR: %[[MEM:.*]] = cir.call @_Z7myAllocx(%[[CAST_SIZE]])
  // CIR: %[[FRAME:.*]] = cir.coro.intrinsic.begin(%[[COROID]], %[[MEM]])

  // TODO(CIR):
  //__builtin_coro_resume(__builtin_coro_frame());

  // TODO(CIR):
  //__builtin_coro_destroy(__builtin_coro_frame());

  // TODO(CIR):
  //__builtin_coro_done(__builtin_coro_frame());

  // TODO(CIR):
  //__builtin_coro_promise(__builtin_coro_frame(), 48, 0);

  __builtin_coro_free(__builtin_coro_frame());
  // CIR: cir.coro.intrinsic.free(%[[COROID]], %[[FRAME]])

  __builtin_coro_end(__builtin_coro_frame(), false);
  // CIR: %[[FALSE:.*]] = cir.const #false
  // CIR: %[[TK_NONE:.*]] = cir.token.none
  // CIR: cir.coro.intrinsic.end(%[[FRAME]], %[[FALSE]], %[[TK_NONE]]) : (!cir.ptr<!void>, !cir.bool, token)

  // TODO(CIR):
  //__builtin_coro_suspend(1);
}
