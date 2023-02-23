// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc18.0.0 -fcoroutines -emit-llvm %s -o - -disable-llvm-passes | FileCheck %s

void *myAlloc(long long);

// CHECK-LABEL: f(
void f(int n) {
  // CHECK: %n.addr = alloca i32
  // CHECK: %promise = alloca i32
  int promise;

  // CHECK: %[[COROID:.+]] = call token @llvm.coro.id(i32 32, ptr %promise, ptr null, ptr null)
  __builtin_coro_id(32, &promise, 0, 0);

  // CHECK-NEXT: call i1 @llvm.coro.alloc(token %[[COROID]])
  __builtin_coro_alloc();

  // CHECK-NEXT: call ptr @llvm.coro.noop()
  __builtin_coro_noop();

  // CHECK-NEXT: %[[SIZE:.+]] = call i64 @llvm.coro.size.i64()
  // CHECK-NEXT: %[[MEM:.+]] = call ptr @myAlloc(i64 noundef %[[SIZE]])
  // CHECK-NEXT: %[[FRAME:.+]] = call ptr @llvm.coro.begin(token %[[COROID]], ptr %[[MEM]])
  __builtin_coro_begin(myAlloc(__builtin_coro_size()));

  // CHECK-NEXT: call void @llvm.coro.resume(ptr %[[FRAME]])
  __builtin_coro_resume(__builtin_coro_frame());

  // CHECK-NEXT: call void @llvm.coro.destroy(ptr %[[FRAME]])
  __builtin_coro_destroy(__builtin_coro_frame());

  // CHECK-NEXT: call i1 @llvm.coro.done(ptr %[[FRAME]])
  __builtin_coro_done(__builtin_coro_frame());

  // CHECK-NEXT: call ptr @llvm.coro.promise(ptr %[[FRAME]], i32 48, i1 false)
  __builtin_coro_promise(__builtin_coro_frame(), 48, 0);

  // CHECK-NEXT: call ptr @llvm.coro.free(token %[[COROID]], ptr %[[FRAME]])
  __builtin_coro_free(__builtin_coro_frame());

  // CHECK-NEXT: call i1 @llvm.coro.end(ptr %[[FRAME]], i1 false)
  __builtin_coro_end(__builtin_coro_frame(), 0);

  // CHECK-NEXT: call i8 @llvm.coro.suspend(token none, i1 true)
  __builtin_coro_suspend(1);
}
