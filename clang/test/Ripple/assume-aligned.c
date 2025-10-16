// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -Wall -Wpedantic -Wextra %s -O2 -fenable-ripple -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -Wall -Wpedantic -Wextra -x c++ %s -O2 -fenable-ripple -S -emit-llvm -o - | FileCheck %s

#include <ripple.h>

#ifdef __cplusplus
extern "C" {
#endif

void check_alignment_2D_slice_partial(size_t begin, size_t end, size_t chunks,
                     float *__restrict Arr1, float *__restrict Arr2) {
  ripple_block_t BS = ripple_set_block_shape(0, 32, 4);
  size_t RID = ripple_id(BS, 1) * ripple_get_block_size(BS,0) + ripple_id(BS, 0);
  size_t ninner = chunks / 128;
  for (size_t i = begin; i < end; i++) {
    float Sum = 0.f;

    Sum = 0.f;
    for (size_t j = 0; j < ninner; j++) {
      // Slice is 2, 0
      Sum += *ripple_ptr_alignment_slice(&Arr1[i * chunks + j * 128 + RID], 128, 2);
    }
    *ripple_ptr_alignment_slice(&Arr2[i * chunks + RID], 128, 2) += Sum;
  }
}

// We are saying that the slice at index 2 is aligned, not the one we use, hence the alignment should remain 4!

// CHECK-LABEL: void @check_alignment_2D_slice_partial
// CHECK: for.body{{[0-9]+}}:
// CHECK: [[LD:%.*]] = load <128 x float>, ptr {{.*}}, align 4

// After that test, everything should be aligned
// CHECK-LABEL: void @check_alignment_scalar
// CHECK-NOT: store <128 x float>{{.*}}, align 4
// CHECK-NOT: load <128 x float>{{.*}}, align 4

void check_alignment_scalar(size_t begin, size_t end, size_t chunks,
                     float *__restrict Arr1, float *__restrict Arr2) {
  ripple_block_t BS = ripple_set_block_shape(0, 128);
  size_t RID = ripple_id(BS, 0);
  size_t ninner = chunks / 128;
  for (size_t i = begin; i < end; i++) {
    float Sum = 0.f;

    Sum = 0.f;
    for (size_t j = 0; j < ninner; j++) {
      Sum += ripple_ptr_alignment(&Arr1[i * chunks + j * 128], 128)[RID];
    }
    ripple_ptr_alignment(&Arr2[i * chunks], 128)[RID] += Sum;
  }
}

void check_alignment(size_t begin, size_t end, size_t chunks,
                     float *__restrict Arr1, float *__restrict Arr2) {
  ripple_block_t BS = ripple_set_block_shape(0, 128);
  size_t RID = ripple_id(BS, 0);
  size_t ninner = chunks / 128;
  for (size_t i = begin; i < end; i++) {
    float Sum = 0.f;

    Sum = 0.f;
    for (size_t j = 0; j < ninner; j++) {
      Sum += *ripple_ptr_alignment(&Arr1[i * chunks + j * 128 + RID], 128);
    }
    *ripple_ptr_alignment(&Arr2[i * chunks + RID], 128) += Sum;
  }
}

void check_alignment_slice(size_t begin, size_t end, size_t chunks,
                           float *__restrict Arr1, float *__restrict Arr2) {
  ripple_block_t BS = ripple_set_block_shape(0, 128);
  size_t RID = ripple_id(BS, 0);
  size_t ninner = chunks / 128;
  for (size_t i = begin; i < end; i++) {
    float Sum = 0.f;

    Sum = 0.f;
    for (size_t j = 0; j < ninner; j++) {
      Sum += *ripple_ptr_alignment_slice(&Arr1[i * chunks + j * 128 + RID],
                                          128, 0);
    }
    *ripple_ptr_alignment_slice(&Arr2[i * chunks + RID], 128, 0) += Sum;
  }
}

void check_alignment_rparallel(size_t begin, size_t end, size_t chunks,
                               float *__restrict Arr1, float *__restrict Arr2) {
  ripple_block_t BS = ripple_set_block_shape(0, 128);
  size_t RID = ripple_id(BS, 0);
  for (size_t i = begin; i < end; i++) {
    float Sum = 0.f;

    // Option 3 ripple_parallel_full
    Sum = 0.f;
    ripple_parallel_full(BS, 0);
    for (size_t j = 0; j < chunks; j++) {
      Sum += *ripple_ptr_alignment(&Arr1[i * chunks + j], 128);
    }
    *ripple_ptr_alignment(&Arr2[i * chunks + RID], 128) += Sum;
  }
}

void check_alignment_rparallel_slice(size_t begin, size_t end, size_t chunks,
                                     float *__restrict Arr1,
                                     float *__restrict Arr2) {
  ripple_block_t BS = ripple_set_block_shape(0, 128);
  size_t RID = ripple_id(BS, 0);
  for (size_t i = begin; i < end; i++) {
    float Sum = 0.f;

    Sum = 0.f;
    ripple_parallel_full(BS, 0);
    for (size_t j = 0; j < chunks; j++) {
      Sum += *ripple_ptr_alignment_slice(&Arr1[i * chunks + j], 128, 0);
    }
    *ripple_ptr_alignment_slice(&Arr2[i * chunks + RID], 128, 0) += Sum;
  }
}

void check_alignment_2D(size_t begin, size_t end, size_t chunks,
                     float *__restrict Arr1, float *__restrict Arr2) {
  ripple_block_t BS = ripple_set_block_shape(0, 32, 4);
  size_t RID = ripple_id(BS, 1) * ripple_get_block_size(BS, 0) + ripple_id(BS, 0);
  size_t ninner = chunks / 128;
  for (size_t i = begin; i < end; i++) {
    float Sum = 0.f;

    Sum = 0.f;
    for (size_t j = 0; j < ninner; j++) {
      Sum += *ripple_ptr_alignment(&Arr1[i * chunks + j * 128 + RID], 128);
    }
    *ripple_ptr_alignment(&Arr2[i * chunks + RID], 128) += Sum;
  }
}

#ifdef __cplusplus
}
#endif
