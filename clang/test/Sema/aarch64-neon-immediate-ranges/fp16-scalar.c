// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon  -target-feature +v8.2a -ffreestanding -fsyntax-only -verify %s

#include <arm_neon.h>
#include <arm_fp16.h>
// REQUIRES: aarch64-registered-target

// vcvth_n_f16_s16, vcvth_n_f16_s32, vcvth_n_f16_s64, vcvth_n_f16_u16, vcvth_n_f16_u32
// vcvth_n_s16_f16, vcvth_n_s32_f16, vcvth_n_s64_f16, vcvth_n_u16_f16, vcvth_n_u32_f16
// are tested under clang/test/Sema/aarch64-neon-fp16-ranges.c

void test_conversions_u64(uint64_t arg_u64) {
	vcvth_n_f16_u64(arg_u64, 1);
	vcvth_n_f16_u64(arg_u64, 16);
	vcvth_n_f16_u64(arg_u64, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvth_n_f16_u64(arg_u64, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

void test_conversions_f16(float16_t arg_f16) {
	vcvth_n_u64_f16(arg_f16, 1);
	vcvth_n_u64_f16(arg_f16, 16);
	vcvth_n_u64_f16(arg_f16, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvth_n_u64_f16(arg_f16, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

