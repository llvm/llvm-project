// RUN: %clang_cc1 -triple thumbv8m.main -O0 -mcmse  -S -emit-llvm \
// RUN:            -fallow-half-arguments-and-returns %s -o - | \
// RUN:    FileCheck %s --check-prefixes=CHECK,CHECK-NOPT-SOFT
// RUN: %clang_cc1 -triple thumbv8m.main -O2 -mcmse  -S -emit-llvm \
// RUN:            -fallow-half-arguments-and-returns %s -o - | \
// RUN:    FileCheck %s --check-prefixes=CHECK,CHECK-OPT-SOFT
// RUN: %clang_cc1 -triple thumbv8m.main -O0 -mcmse  -S -emit-llvm \
// RUN:            -fallow-half-arguments-and-returns -mfloat-abi hard %s -o - | \
// RUN:    FileCheck %s --check-prefixes=CHECK,CHECK-NOPT-HARD
// RUN: %clang_cc1 -triple thumbv8m.main -O2 -mcmse  -S -emit-llvm \
// RUN:            -fallow-half-arguments-and-returns -mfloat-abi hard %s -o - | \
// RUN:    FileCheck %s --check-prefixes=CHECK,CHECK-OPT-HARD

__fp16 g0();
__attribute__((cmse_nonsecure_entry)) __fp16 f0() {
  return g0();
}
// CHECK:           define {{.*}}@f0()

// CHECK-NOPT-SOFT: %[[V0:.*]] = load i32
// CHECK-NOPT-SOFT: %[[V1:.*]] = and i32 %[[V0]], 65535
// CHECK-NOPT-SOFT: ret i32 %[[V1]]

// CHECK-OPT-SOFT: %[[V0:.*]] = tail call {{.*}} @g0
// CHECK-OPT-SOFT: %[[V1:.*]] = and i32 %[[V0]], 65535
// CHECK-OPT-SOFT: ret i32 %[[V1]]

// CHECK-NOPT-HARD: %[[V0:.*]] = bitcast float {{.*}} to i32
// CHECK-NOPT-HARD: %[[V1:.*]] = and i32 %[[V0]], 65535
// CHECK-NOPT-HARD: %[[V2:.*]] = bitcast i32 %[[V1]] to float
// CHECK-NOPT-HARD: ret float %[[V2]]

// CHECK-OPT-HARD: %[[V0:.*]] = bitcast float {{.*}} to i32
// CHECK-OPT-HARD: %[[V1:.*]] = and i32 %[[V0]], 65535
// CHECK-OPT-HARD: %[[V2:.*]] = bitcast i32 %[[V1]] to float
// CHECK-OPT-HARD: ret float %[[V2]]

void __attribute__((cmse_nonsecure_call)) (*g1)(__fp16);
__fp16 x;
void f1() {
  g1(x);
}
// CHECK: define {{.*}}@f1()

// CHECK-NOPT-SOFT: %[[V0:.*]] = load i32
// CHECK-NOPT-SOFT: %[[V1:.*]] = and i32 %[[V0]], 65535
// CHECK-NOPT-SOFT: call {{.*}} void {{.*}}(i32 %[[V1]])

// CHECK-OPT-SOFT: %[[V1:.*]] = zext i16 {{.*}} to i32
// CHECK-OPT-SOFT: call {{.*}} void {{.*}}(i32 %[[V1]])

// CHECK-NOPT-HARD: %[[V0:.*]] = bitcast float {{.*}} to i32
// CHECK-NOPT-HARD: %[[V1:.*]] = and i32 %[[V0]], 65535
// CHECK-NOPT-HARD: %[[V2:.*]] = bitcast i32 %[[V1]] to float
// CHECK-NOPT-HARD: call {{.*}}(float %[[V2]])

// CHECK-OPT-HARD: %[[V0:.*]] = zext i16 {{.*}} to i32
// CHECK-OPT-HARD: %[[V1:.*]] = bitcast i32 %[[V0]] to float
// CHECK-OPT-HARD: call {{.*}}(float %[[V1]])
