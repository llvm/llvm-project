// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-enable-irbuilder -verify -fopenmp -fopenmp-version=45 -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s 
// expected-no-diagnostics

struct S {
  int a, b;
};

struct P {
  int a, b;
};

void simple(float *a, float *b, int *c) {
  S s, *p;
  P pp;
#pragma omp simd
  for (int i = 3; i < 32; i += 5) {
    // llvm.access.group test
    // CHECK: %[[A_ADDR:.+]] = alloca float*, align 8
    // CHECK: %[[B_ADDR:.+]] = alloca float*, align 8
    // CHECK: %[[S:.+]] = alloca %struct.S, align 4
    // CHECK: %[[P:.+]] = alloca %struct.S*, align 8
    // CHECK: %[[I:.+]] = alloca i32, align 4
    // CHECK: %[[TMP3:.+]] = load float*, float** %[[B_ADDR:.+]], align 8, !llvm.access.group ![[META3:[0-9]+]]
    // CHECK-NEXT: %[[TMP4:.+]] = load i32, i32* %[[I:.+]], align 4, !llvm.access.group ![[META3:[0-9]+]]
    // CHECK-NEXT: %[[IDXPROM:.+]] = sext i32 %[[TMP4:.+]] to i64
    // CHECK-NEXT: %[[ARRAYIDX:.+]] = getelementptr inbounds float, float* %[[TMP3:.+]], i64 %[[IDXPROM:.+]]
    // CHECK-NEXT: %[[TMP5:.+]] = load float, float* %[[ARRAYIDX:.+]], align 4, !llvm.access.group ![[META3:[0-9]+]]
    // CHECK-NEXT: %[[A2:.+]] = getelementptr inbounds %struct.S, %struct.S* %[[S:.+]], i32 0, i32 0
    // CHECK-NEXT: %[[TMP6:.+]] = load i32, i32* %[[A2:.+]], align 4, !llvm.access.group ![[META3:[0-9]+]]
    // CHECK-NEXT: %[[CONV:.+]] = sitofp i32 %[[TMP6:.+]] to float
    // CHECK-NEXT: %[[ADD:.+]] = fadd float %[[TMP5:.+]], %[[CONV:.+]]
    // CHECK-NEXT: %[[TMP7:.+]] = load %struct.S*, %struct.S** %[[P:.+]], align 8, !llvm.access.group ![[META3:[0-9]+]]
    // CHECK-NEXT: %[[A3:.+]] = getelementptr inbounds %struct.S, %struct.S* %[[TMP7:.+]], i32 0, i32 0
    // CHECK-NEXT: %[[TMP8:.+]] = load i32, i32* %[[A3:.+]], align 4, !llvm.access.group ![[META3:[0-9]+]]
    // CHECK-NEXT: %[[CONV4:.+]] = sitofp i32 %[[TMP8:.+]] to float
    // CHECK-NEXT: %[[ADD5:.+]] = fadd float %[[ADD:.+]], %[[CONV4:.+]]
    // CHECK-NEXT: %[[TMP9:.+]] = load float*, float** %[[A_ADDR:.+]], align 8, !llvm.access.group ![[META3:[0-9]+]]
    // CHECK-NEXT: %[[TMP10:.+]] = load i32, i32* %[[I:.+]], align 4, !llvm.access.group ![[META3:[0-9]+]]
    // CHECK-NEXT: %[[IDXPROM6:.+]] = sext i32 %[[TMP10:.+]] to i64
    // CHECK-NEXT: %[[ARRAYIDX7:.+]] = getelementptr inbounds float, float* %[[TMP9:.+]], i64 %[[IDXPROM6:.+]]
    // CHECK-NEXT: store float %[[ADD5:.+]], float* %[[ARRAYIDX7:.+]], align 4, !llvm.access.group ![[META3:[0-9]+]]
    // llvm.loop test
    // CHECK: %[[OMP_LOOPDOTNEXT:.+]] = add nuw i32 %[[OMP_LOOPDOTIV:.+]], 1
    // CHECK-NEXT: br label %omp_loop.header, !llvm.loop ![[META4:[0-9]+]]
    a[i] = b[i] + s.a + p->a;
  }

#pragma omp simd
  for (int j = 3; j < 32; j += 5) {
    // test if unique access groups were used for a second loop
    // CHECK: %[[A22:.+]] = getelementptr inbounds %struct.P, %struct.P* %[[PP:.+]], i32 0, i32 0
    // CHECK-NEXT: %[[TMP14:.+]] = load i32, i32* %[[A22:.+]], align 4, !llvm.access.group ![[META7:[0-9]+]]
    // CHECK-NEXT: %[[TMP15:.+]] = load i32*, i32** %[[C_ADDR:.+]], align 8, !llvm.access.group ![[META7:[0-9]+]]
    // CHECK-NEXT: %[[TMP16:.+]] = load i32, i32* %[[J:.+]], align 4, !llvm.access.group ![[META7:[0-9]+]]
    // CHECK-NEXT: %[[IDXPROM23:.+]] = sext i32 %[[TMP16:.+]] to i64
    // CHECK-NEXT: %[[ARRAYIDX24:.+]] = getelementptr inbounds i32, i32* %[[TMP15:.+]], i64 %[[IDXPROM23:.+]]
    // CHECK-NEXT: store i32 %[[TMP14:.+]], i32* %[[ARRAYIDX24:.+]], align 4, !llvm.access.group ![[META7:[0-9]+]]
    // check llvm.loop metadata
    // CHECK: %[[OMP_LOOPDOTNEXT:.+]] = add nuw i32 %[[OMP_LOOPDOTIV:.+]], 1
    // CHECK-NEXT: br label %[[OMP_LLOP_BODY:.*]], !llvm.loop ![[META8:[0-9]+]]
    c[j] = pp.a;
  }
}

// CHECK: ![[META3:[0-9]+]] = distinct !{}
// CHECK-NEXT: ![[META4]]  = distinct !{![[META4]], ![[META5:[0-9]+]], ![[META6:[0-9]+]]}
// CHECK-NEXT: ![[META5]]  = !{!"llvm.loop.parallel_accesses", ![[META3]]}
// CHECK-NEXT: ![[META6]]  = !{!"llvm.loop.vectorize.enable", i1 true}
// CHECK-NEXT: ![[META7:[0-9]+]] = distinct !{}
// CHECK-NEXT: ![[META8]]  = distinct !{![[META8]], ![[META9:[0-9]+]], ![[META6]]}
// CHECK-NEXT: ![[META9]]  = !{!"llvm.loop.parallel_accesses", ![[META7]]}
