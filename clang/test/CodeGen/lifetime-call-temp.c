// RUN: %clang -cc1                  -triple x86_64-apple-macos -O1 -disable-llvm-passes %s -S -emit-llvm -o - | FileCheck %s --implicit-check-not=llvm.lifetime
// RUN: %clang -cc1 -xc++ -std=c++17 -triple x86_64-apple-macos -O1 -disable-llvm-passes %s -S -emit-llvm -o - -Wno-return-type-c-linkage | FileCheck %s --implicit-check-not=llvm.lifetime --check-prefix=CHECK --check-prefix=CXX
// RUN: %clang -cc1 -xobjective-c    -triple x86_64-apple-macos -O1 -disable-llvm-passes %s -S -emit-llvm -o - | FileCheck %s --implicit-check-not=llvm.lifetime --check-prefix=CHECK --check-prefix=OBJC

typedef struct { int x[100]; } aggregate;

#ifdef __cplusplus
extern "C" {
#endif

void takes_aggregate(aggregate);
aggregate gives_aggregate();

// CHECK-LABEL: define void @t1
void t1() {
  takes_aggregate(gives_aggregate());

  // CHECK: [[AGGTMP:%.*]] = alloca %struct.aggregate, align 8
  // CHECK: call void @llvm.lifetime.start.p0(i64 400, ptr [[AGGTMP]])
  // CHECK: call void{{.*}} @gives_aggregate(ptr sret(%struct.aggregate) align 4 [[AGGTMP]])
  // CHECK: call void @takes_aggregate(ptr noundef byval(%struct.aggregate) align 8 [[AGGTMP]])
  // CHECK: call void @llvm.lifetime.end.p0(i64 400, ptr [[AGGTMP]])
}

// CHECK: declare {{.*}}llvm.lifetime.start
// CHECK: declare {{.*}}llvm.lifetime.end

#ifdef __cplusplus
// CXX: define void @t2
void t2() {
  struct S {
    S(aggregate) {}
  };
  S{gives_aggregate()};

  // CXX: [[AGG:%.*]] = alloca %struct.aggregate
  // CXX: call void @llvm.lifetime.start.p0(i64 400, ptr
  // CXX: call void @gives_aggregate(ptr sret(%struct.aggregate) align 4 [[AGG]])
  // CXX: call void @_ZZ2t2EN1SC1E9aggregate(ptr {{.*}}, ptr {{.*}} byval(%struct.aggregate) align 8 [[AGG]])
  // CXX: call void @llvm.lifetime.end.p0(i64 400, ptr
}

struct Dtor {
  ~Dtor();
};

void takes_dtor(Dtor);
Dtor gives_dtor();

// CXX: define void @t3
void t3() {
  takes_dtor(gives_dtor());

  // CXX-NOT @llvm.lifetime
  // CXX: ret void
}

#endif

#ifdef __OBJC__

@interface X
-m:(aggregate)x;
@end

// OBJC: define void @t4
void t4(X *x) {
  [x m: gives_aggregate()];

  // OBJC: [[AGG:%.*]] = alloca %struct.aggregate
  // OBJC: call void @llvm.lifetime.start.p0(i64 400, ptr
  // OBJC: call void{{.*}} @gives_aggregate(ptr sret(%struct.aggregate) align 4 [[AGGTMP]])
  // OBJC: call {{.*}}@objc_msgSend
  // OBJC: call void @llvm.lifetime.end.p0(i64 400, ptr
}

#endif

#ifdef __cplusplus
}
#endif
