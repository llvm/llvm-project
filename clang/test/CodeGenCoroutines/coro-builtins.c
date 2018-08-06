// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc18.0.0 -fcoroutines-ts -emit-llvm %s -o - -disable-llvm-passes | FileCheck %s

void *myAlloc(long long);

// CHECK-LABEL: f(
void f(int n) {
  // CHECK: %n.addr = alloca i32
  // CHECK: %n_copy = alloca i32
  // CHECK: %promise = alloca i32
  int n_copy;
  int promise;

  // CHECK: %[[PROM_ADDR:.+]] = bitcast i32* %promise to i8*
  // CHECK-NEXT: %[[COROID:.+]] = call token @llvm.coro.id(i32 32, i8* %[[PROM_ADDR]], i8* null, i8* null)
  __builtin_coro_id(32, &promise, 0, 0);

  // CHECK-NEXT: call i1 @llvm.coro.alloc(token %[[COROID]])
  __builtin_coro_alloc();

  // CHECK-NEXT: call i8* @llvm.coro.noop()
  __builtin_coro_noop();

  // CHECK-NEXT: %[[SIZE:.+]] = call i64 @llvm.coro.size.i64()
  // CHECK-NEXT: %[[MEM:.+]] = call i8* @myAlloc(i64 %[[SIZE]])
  // CHECK-NEXT: %[[FRAME:.+]] = call i8* @llvm.coro.begin(token %[[COROID]], i8* %[[MEM]])
  __builtin_coro_begin(myAlloc(__builtin_coro_size()));

  // CHECK-NEXT: call void @llvm.coro.resume(i8* %[[FRAME]])
  __builtin_coro_resume(__builtin_coro_frame());

  // CHECK-NEXT: call void @llvm.coro.destroy(i8* %[[FRAME]])
  __builtin_coro_destroy(__builtin_coro_frame());

  // CHECK-NEXT: call i1 @llvm.coro.done(i8* %[[FRAME]])
  __builtin_coro_done(__builtin_coro_frame());

  // CHECK-NEXT: call i8* @llvm.coro.promise(i8* %[[FRAME]], i32 48, i1 false)
  __builtin_coro_promise(__builtin_coro_frame(), 48, 0);

  // CHECK-NEXT: call i8* @llvm.coro.free(token %[[COROID]], i8* %[[FRAME]])
  __builtin_coro_free(__builtin_coro_frame());

  // CHECK-NEXT: call i1 @llvm.coro.end(i8* %[[FRAME]], i1 false)
  __builtin_coro_end(__builtin_coro_frame(), 0);

  // CHECK-NEXT: call i8 @llvm.coro.suspend(token none, i1 true)
  __builtin_coro_suspend(1);

  // CHECK-NEXT: %[[N_ADDR:.+]] = bitcast i32* %n.addr to i8*
  // CHECK-NEXT: %[[N_COPY_ADDR:.+]] = bitcast i32* %n_copy to i8*
  // CHECK-NEXT: call i1 @llvm.coro.param(i8* %[[N_ADDR]], i8* %[[N_COPY_ADDR]])
  __builtin_coro_param(&n, &n_copy);
}
