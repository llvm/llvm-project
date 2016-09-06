// RUN: %clang_cc1 -triple x86_64-unknown-linux -fvisibility hidden -fsanitize=cfi-vcall,cfi-nvcall,cfi-derived-cast,cfi-unrelated-cast,cfi-icall -fsanitize-stats -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -fvisibility hidden -fsanitize=cfi-vcall,cfi-nvcall,cfi-derived-cast,cfi-unrelated-cast,cfi-icall -fsanitize-trap=cfi-vcall -fwhole-program-vtables -fsanitize-stats -emit-llvm -o - %s | FileCheck %s

// CHECK: [[STATS:@[^ ]*]] = internal global { i8*, i32, [5 x [2 x i8*]] } { i8* null, i32 5, [5 x [2 x i8*]]
// CHECK: {{\[\[}}2 x i8*] zeroinitializer,
// CHECK: [2 x i8*] [i8* null, i8* inttoptr (i64 2305843009213693952 to i8*)],
// CHECK: [2 x i8*] [i8* null, i8* inttoptr (i64 4611686018427387904 to i8*)],
// CHECK: [2 x i8*] [i8* null, i8* inttoptr (i64 6917529027641081856 to i8*)],
// CHECK: [2 x i8*] [i8* null, i8* inttoptr (i64 -9223372036854775808 to i8*)]] }

// CHECK: @llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 0, void ()* [[CTOR:@[^ ]*]], i8* null }]

struct A {
  virtual void vf();
  void nvf();
};
struct B : A {};

// CHECK: @vcall
extern "C" void vcall(A *a) {
  // CHECK: call void @__sanitizer_stat_report({{.*}}[[STATS]]{{.*}}i64 0, i32 2, i64 0
  a->vf();
}

// CHECK: @nvcall
extern "C" void nvcall(A *a) {
  // CHECK: call void @__sanitizer_stat_report({{.*}}[[STATS]]{{.*}}i64 0, i32 2, i64 1
  a->nvf();
}

// CHECK: @dcast
extern "C" void dcast(A *a) {
  // CHECK: call void @__sanitizer_stat_report({{.*}}[[STATS]]{{.*}}i64 0, i32 2, i64 2
  static_cast<B *>(a);
}

// CHECK: @ucast
extern "C" void ucast(void *a) {
  // CHECK: call void @__sanitizer_stat_report({{.*}}[[STATS]]{{.*}}i64 0, i32 2, i64 3
  reinterpret_cast<A *>(a);
}

// CHECK: @icall
extern "C" void icall(void (*p)()) {
  // CHECK: call void @__sanitizer_stat_report({{.*}}[[STATS]]{{.*}}i64 0, i32 2, i64 4
  p();
}

// CHECK: define internal void [[CTOR]]()
// CHECK-NEXT: call void @__sanitizer_stat_init(i8* bitcast ({ i8*, i32, [5 x [2 x i8*]] }* [[STATS]] to i8*))
// CHECK-NEXT: ret void
