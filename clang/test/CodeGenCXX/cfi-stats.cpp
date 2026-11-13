// RUN: %clang_cc1 -triple x86_64-unknown-linux -fvisibility=hidden -fsanitize=cfi-vcall,cfi-nvcall,cfi-derived-cast,cfi-unrelated-cast,cfi-icall -fsanitize-stats -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -fvisibility=hidden -fsanitize=cfi-vcall,cfi-nvcall,cfi-derived-cast,cfi-unrelated-cast,cfi-icall -fsanitize-trap=cfi-vcall -fwhole-program-vtables -fsanitize-stats -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -flto -flto-unit -fvisibility=hidden -fsanitize=cfi-vcall,cfi-nvcall,cfi-derived-cast,cfi-unrelated-cast,cfi-icall -fwhole-program-vtables -fvirtual-function-elimination -fsanitize-stats -emit-llvm -o - %s | FileCheck --check-prefix=VFE %s

// CHECK: [[STATS:@[^ ]*]] = internal global { ptr, i32, [5 x [2 x ptr]] } { ptr null, i32 5, [5 x [2 x ptr]]
// VFE: [[STATS:@[^ ]*]] = internal global { ptr, i32, [5 x [2 x ptr]] } { ptr null, i32 5, [5 x [2 x ptr]]
// CHECK: {{\[\[}}2 x ptr] zeroinitializer,
// CHECK: [2 x ptr] [ptr null, ptr inttoptr (i64 2305843009213693952 to ptr)],
// CHECK: [2 x ptr] [ptr null, ptr inttoptr (i64 4611686018427387904 to ptr)],
// CHECK: [2 x ptr] [ptr null, ptr inttoptr (i64 6917529027641081856 to ptr)],
// CHECK: [2 x ptr] [ptr null, ptr inttoptr (i64 -9223372036854775808 to ptr)]] }

// CHECK: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr [[CTOR:@[^ ]*]], ptr null }]

struct A {
  virtual void vf();
  void nvf();
};
struct B : A {};

// CHECK: @vcall
// VFE-LABEL: @vcall(
extern "C" void vcall(A *a) {
  // CHECK: call void @__sanitizer_stat_report({{.*}}[[STATS]]{{.*}}i64 16)
  // VFE: call void @__sanitizer_stat_report({{.*}}[[STATS]]{{.*}}i64 16)
  a->vf();
}

// CHECK: @nvcall
extern "C" void nvcall(A *a) {
  // CHECK: call void @__sanitizer_stat_report({{.*}}[[STATS]]{{.*}}i64 32)
  a->nvf();
}

// CHECK: @dcast
extern "C" void dcast(A *a) {
  // CHECK: call void @__sanitizer_stat_report({{.*}}[[STATS]]{{.*}}i64 48)
  static_cast<B *>(a);
}

// CHECK: @ucast
extern "C" void ucast(void *a) {
  // CHECK: call void @__sanitizer_stat_report({{.*}}[[STATS]]{{.*}}i64 64)
  reinterpret_cast<A *>(a);
}

// CHECK: @icall
extern "C" void icall(void (*p)()) {
  // CHECK: call void @__sanitizer_stat_report({{.*}}[[STATS]]{{.*}}i64 80)
  p();
}

// CHECK-LABEL: @vcall_unsanitized(
// VFE-LABEL: @vcall_unsanitized(
extern "C" __attribute__((no_sanitize("cfi-vcall"))) void vcall_unsanitized(A *a) {
  // CHECK-NOT: @__sanitizer_stat_report
  // VFE-NOT: @__sanitizer_stat_report
  a->vf();
  // CHECK: ret void
  // VFE: ret void
}

// CHECK: define internal void [[CTOR]]()
// CHECK-NEXT: call void @__sanitizer_stat_init(ptr [[STATS]])
// CHECK-NEXT: ret void
