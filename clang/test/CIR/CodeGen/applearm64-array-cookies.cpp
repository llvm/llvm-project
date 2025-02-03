// RUN: %clang_cc1 -std=c++20 -triple=arm64e-apple-darwin -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

class C {
  public:
    ~C();
};

void t_constant_size_nontrivial() {
  auto p = new C[3];
}

// CHECK:  cir.func @_Z26t_constant_size_nontrivialv()
// CHECK:    %[[#NUM_ELEMENTS:]] = cir.const #cir.int<3> : !u64i
// CHECK:    %[[#SIZE_WITHOUT_COOKIE:]] = cir.const #cir.int<3> : !u64i
// CHECK:    %[[#ALLOCATION_SIZE:]] = cir.const #cir.int<19> : !u64i
// CHECK:    %[[#ALLOC_PTR:]] = cir.call @_Znam(%[[#ALLOCATION_SIZE]]) : (!u64i) -> !cir.ptr<!void>
// CHECK:    %[[#COOKIE_PTR:]] = cir.cast(bitcast, %[[#ALLOC_PTR]] : !cir.ptr<!void>), !cir.ptr<!u64i>
// CHECK:    %[[#ELEMENT_SIZE:]] = cir.const #cir.int<1> : !u64i
// CHECK:    cir.store %[[#ELEMENT_SIZE]], %[[#COOKIE_PTR]] : !u64i, !cir.ptr<!u64i>
// CHECK:    %[[#SECOND_COOKIE_OFFSET:]] = cir.const #cir.int<1> : !s32i
// CHECK:    %[[#COOKIE_PTR2:]] = cir.ptr_stride(%[[#COOKIE_PTR]] : !cir.ptr<!u64i>, %[[#SECOND_COOKIE_OFFSET]] : !s32i), !cir.ptr<!u64i>
// CHECK:    cir.store %[[#NUM_ELEMENTS]], %[[#COOKIE_PTR2]] : !u64i, !cir.ptr<!u64i>
// CHECK:    %[[#COOKIE_SIZE:]] = cir.const #cir.int<16> : !s32i
// CHECK:    %[[#ALLOC_AS_I8:]] = cir.cast(bitcast, %[[#ALLOC_PTR]] : !cir.ptr<!void>), !cir.ptr<!u8i>
// CHECK:    cir.ptr_stride(%[[#ALLOC_AS_I8]] : !cir.ptr<!u8i>, %[[#COOKIE_SIZE]] : !s32i), !cir.ptr<!u8i>

class D {
  public:
    int x;
    ~D();
};

void t_constant_size_nontrivial2() {
  auto p = new D[3];
}

// In this test SIZE_WITHOUT_COOKIE isn't used, but it would be if there were
// an initializer.

// CHECK:  cir.func @_Z27t_constant_size_nontrivial2v()
// CHECK:    %[[#NUM_ELEMENTS:]] = cir.const #cir.int<3> : !u64i
// CHECK:    %[[#SIZE_WITHOUT_COOKIE:]] = cir.const #cir.int<12> : !u64i
// CHECK:    %[[#ALLOCATION_SIZE:]] = cir.const #cir.int<28> : !u64i
// CHECK:    %[[#ALLOC_PTR:]] = cir.call @_Znam(%[[#ALLOCATION_SIZE]]) : (!u64i) -> !cir.ptr<!void>
// CHECK:    %[[#COOKIE_PTR:]] = cir.cast(bitcast, %[[#ALLOC_PTR]] : !cir.ptr<!void>), !cir.ptr<!u64i>
// CHECK:    %[[#ELEMENT_SIZE:]] = cir.const #cir.int<4> : !u64i
// CHECK:    cir.store %[[#ELEMENT_SIZE]], %[[#COOKIE_PTR]] : !u64i, !cir.ptr<!u64i>
// CHECK:    %[[#SECOND_COOKIE_OFFSET:]] = cir.const #cir.int<1> : !s32i
// CHECK:    %[[#COOKIE_PTR2:]] = cir.ptr_stride(%[[#COOKIE_PTR]] : !cir.ptr<!u64i>, %[[#SECOND_COOKIE_OFFSET]] : !s32i), !cir.ptr<!u64i>
// CHECK:    cir.store %[[#NUM_ELEMENTS]], %[[#COOKIE_PTR2]] : !u64i, !cir.ptr<!u64i>
// CHECK:    %[[#COOKIE_SIZE:]] = cir.const #cir.int<16> : !s32i
// CHECK:    %[[#ALLOC_AS_I8:]] = cir.cast(bitcast, %[[#ALLOC_PTR]] : !cir.ptr<!void>), !cir.ptr<!u8i>
// CHECK:    cir.ptr_stride(%[[#ALLOC_AS_I8]] : !cir.ptr<!u8i>, %[[#COOKIE_SIZE]] : !s32i), !cir.ptr<!u8i>
