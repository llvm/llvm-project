// RUN: %clang_cc1 -triple hexagon -target-cpu hexagonv68 -target-feature +hvxv68 -target-feature +hvx-length128b -emit-llvm -o - %s | FileCheck %s
// REQUIRES: hexagon-registered-target

typedef long HEXAGON_Vect1024 __attribute__((__vector_size__(128)))
  __attribute__((aligned(128)));
typedef long HEXAGON_Vect2048 __attribute__((__vector_size__(256)))
  __attribute__((aligned(128)));

// CHECK-LABEL: test0
// CHECK: call <64 x i32> @llvm.hexagon.V6.v6mpyhubs10.128B(<64 x i32> %{{.*}}, <64 x i32> %{{.*}}, i32 0)
HEXAGON_Vect2048 test0(HEXAGON_Vect2048 Vuu, HEXAGON_Vect2048 Vvv) {
  return __builtin_HEXAGON_V6_v6mpyhubs10_128B(Vuu, Vvv, 0);
}

// CHECK-LABEL: test1
// CHECK: call <64 x i32> @llvm.hexagon.V6.v6mpyhubs10.vxx.128B(<64 x i32> %{{.*}}, <64 x i32> %{{.*}}, <64 x i32> %{{.*}}, i32 1)
HEXAGON_Vect2048 test1(HEXAGON_Vect2048 Vxx, HEXAGON_Vect2048 Vuu,
                       HEXAGON_Vect2048 Vvv) {
  return __builtin_HEXAGON_V6_v6mpyhubs10_vxx_128B(Vxx, Vuu, Vvv, 1);
}

// CHECK-LABEL: test2
// CHECK: call <64 x i32> @llvm.hexagon.V6.v6mpyvubs10.128B(<64 x i32> %{{.*}}, <64 x i32> %{{.*}}, i32 2)
HEXAGON_Vect2048 test2(HEXAGON_Vect2048 Vuu, HEXAGON_Vect2048 Vvv) {
  return __builtin_HEXAGON_V6_v6mpyvubs10_128B(Vuu, Vvv, 2);
}

// CHECK-LABEL: test3
// CHECK: call <64 x i32> @llvm.hexagon.V6.v6mpyvubs10.vxx.128B(<64 x i32> %{{.*}}, <64 x i32> %{{.*}}, <64 x i32> %{{.*}}, i32 3)
HEXAGON_Vect2048 test3(HEXAGON_Vect2048 Vxx, HEXAGON_Vect2048 Vuu,
                       HEXAGON_Vect2048 Vvv) {
  return __builtin_HEXAGON_V6_v6mpyvubs10_vxx_128B(Vxx, Vuu, Vvv, 3);
}

