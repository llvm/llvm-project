// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

#define __device__ __attribute__((device))

struct A { __device__ virtual void f(); };

// CHECK-LABEL: cir.func {{.*}} @_Z9cast_voidP1A
// CHECK:         %[[SRC:.*]] = cir.load {{.*}}, !cir.ptr<!rec_A, target_address_space(4)>
// CHECK:         %[[RAW:.*]] = cir.cast bitcast %[[SRC]] : !cir.ptr<!rec_A, target_address_space(4)> -> !cir.ptr<!u8i, target_address_space(4)>
// CHECK:         %[[ADJ:.*]] = cir.ptr_stride %[[RAW]], %{{.*}} : (!cir.ptr<!u8i, target_address_space(4)>, !s64i) -> !cir.ptr<!u8i, target_address_space(4)>
// CHECK:         cir.cast bitcast %[[ADJ]] : !cir.ptr<!u8i, target_address_space(4)> -> !cir.ptr<!void, target_address_space(4)>
// CHECK:       }) : (!cir.bool) -> !cir.ptr<!void, target_address_space(4)>
__device__ void *cast_void(A *p) { return dynamic_cast<void *>(p); }
