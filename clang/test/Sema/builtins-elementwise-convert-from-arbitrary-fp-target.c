// RUN: %clang_cc1 -triple wasm32-unknown-unknown -DTEST_BF16 \
// RUN:   -emit-llvm -o /dev/null -verify=bf16 %s
// RUN: %clang_cc1 -triple i386-unknown-linux-gnu -DTEST_F16 \
// RUN:   -emit-llvm -o /dev/null -verify=f16 %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -fopenmp \
// RUN:   -fopenmp-is-target-device -DTEST_OMP_BF16 \
// RUN:   -emit-llvm -o /dev/null -verify=omp-bf16 %s

#if !__has_builtin(__builtin_elementwise_convert_from_f8e5m2_f16) ||          \
    !__has_builtin(__builtin_elementwise_convert_from_f8e5m2_bf16)
#error "target-independent builtin spellings must remain available"
#endif

#if defined(TEST_BF16)
void test_bf16(unsigned char src) {
  (void)__builtin_elementwise_convert_from_f8e5m2_bf16(src); // bf16-error {{__bf16 is not supported on this target}}
}
#endif

#if defined(TEST_F16)
void test_f16(unsigned char src) {
  (void)__builtin_elementwise_convert_from_f8e5m2_f16(src); // f16-error {{_Float16 is not supported on this target}}
}
#endif

#if defined(TEST_OMP_BF16)
void test_omp_bf16_not_emitted(unsigned char src) {
  (void)__builtin_elementwise_convert_from_f8e5m2_bf16(src);
}

#pragma omp declare target
void test_omp_bf16(unsigned char src) {
  (void)__builtin_elementwise_convert_from_f8e5m2_bf16(src); // omp-bf16-error {{__bf16 is not supported on this target}}
}

void test_omp_bf16_sizeof(unsigned char src) {
  (void)sizeof(__builtin_elementwise_convert_from_f8e5m2_bf16(src)); // omp-bf16-error {{__bf16 is not supported on this target}}
}

void test_omp_bf16_auto(unsigned char src) {
  __auto_type value = __builtin_elementwise_convert_from_f8e5m2_bf16(src); // omp-bf16-error {{__bf16 is not supported on this target}}
}
#pragma omp end declare target
#endif
