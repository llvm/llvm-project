// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +sve -target-feature +neon -mvscale-min=1 -mvscale-max=1 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,HARD,AAPCS,LE,ARG8
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +sve -target-feature +neon -mvscale-min=1 -mvscale-max=1 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,HARD,AAPCS,LE,ARG8 --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -target-feature +sve -target-feature +neon -mvscale-min=1 -mvscale-max=1 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,HARD,AAPCS,BE,ARG64
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -target-feature +sve -target-feature +neon -mvscale-min=1 -mvscale-max=1 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,HARD,AAPCS,BE,ARG64 --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-abi darwinpcs -target-feature +sve -target-feature +neon -mvscale-min=1 -mvscale-max=1 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,HARD,OTHER,LE,ARG64
// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-abi darwinpcs -target-feature +sve -target-feature +neon -mvscale-min=1 -mvscale-max=1 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,HARD,OTHER,LE,ARG64 --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64-pc-windows-msvc -target-feature +sve -target-feature +neon -mvscale-min=1 -mvscale-max=1 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,HARD,OTHER,LE,ARG8
// RUN: %clang_cc1 -triple aarch64-pc-windows-msvc -target-feature +sve -target-feature +neon -mvscale-min=1 -mvscale-max=1 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,HARD,OTHER,LE,ARG8 --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple arm64ec-pc-windows-msvc -target-feature +sve -target-feature +neon -mvscale-min=1 -mvscale-max=1 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,HARD,OTHER,LE,ARG8
// RUN: %clang_cc1 -triple arm64ec-pc-windows-msvc -target-feature +sve -target-feature +neon -mvscale-min=1 -mvscale-max=1 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,HARD,OTHER,LE,ARG8 --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature -fp-armv8 -target-abi aapcs-soft -DSOFT_FLOAT -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,LE,ARG8
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature -fp-armv8 -target-abi aapcs-soft -DSOFT_FLOAT -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,LE,ARG8 --implicit-check-not="not yet implemented"

// The scalar is an opaque 8-bit float. It is passed and returned directly as
// <1 x i8>, which occupies a SIMD and floating-point register, and it is not
// promoted. A record containing it is an ordinary aggregate. Eight scalars
// fill the AAPCS FP register file, so a following pure scalable aggregate is
// passed indirectly.

// The soft-float ABI has no FPU, so this run has no SVE or Neon.
#ifndef SOFT_FLOAT
typedef __SVInt32_t fixed_int32_t __attribute__((arm_sve_vector_bits(128)));
typedef __SVMfloat8_t fixed_mf8_t __attribute__((arm_sve_vector_bits(128)));
typedef __mfp8 mfloat8x8_t __attribute__((neon_vector_type(8)));
typedef __mfp8 mfloat8x16_t __attribute__((neon_vector_type(16)));

typedef struct { fixed_int32_t x; } One;
#endif

struct OneByte { __mfp8 x; };
struct ThreeMFP8 { __mfp8 a[3]; };

// CHECK-LABEL: define{{.*}} void @arg_mfp8(<1 x i8> %{{.*}})
void arg_mfp8(__mfp8 x) {}

// CHECK-LABEL: define{{.*}} <1 x i8> @ret_mfp8(<1 x i8> %{{.*}})
__mfp8 ret_mfp8(__mfp8 x) { return x; }

// A one-byte record argument keeps its exact width on a little-endian
// non-Darwin target and is widened to i64 on a big-endian or Darwin target. A
// one-byte record return keeps its exact width on a little-endian target and is
// widened to a full register on a big-endian target.

// ARG8-LABEL: define{{.*}} void @arg_wrap(i8 %{{.*}})
// ARG64-LABEL: define{{.*}} void @arg_wrap(i64 %{{.*}})
void arg_wrap(struct OneByte s) {}

// LE-LABEL: define{{.*}} i8 @ret_wrap(
// BE-LABEL: define{{.*}} i64 @ret_wrap(
struct OneByte ret_wrap(struct OneByte s) { return s; }

// A three-byte record argument keeps its exact width on a little-endian
// non-Darwin target and is widened to i64 on a big-endian or Darwin target.
// The return keeps its exact width on every little-endian target.

// ARG8-LABEL: define{{.*}} void @arg_arr(i24 %{{.*}})
// ARG64-LABEL: define{{.*}} void @arg_arr(i64 %{{.*}})
void arg_arr(struct ThreeMFP8 s) {}

// LE-LABEL: define{{.*}} i24 @ret_arr(
// BE-LABEL: define{{.*}} i64 @ret_arr(
struct ThreeMFP8 ret_arr(struct ThreeMFP8 s) { return s; }

#ifndef SOFT_FLOAT
// HARD-LABEL: define{{.*}} void @arg_svmf8(<vscale x 16 x i8> %{{.*}})
void arg_svmf8(__SVMfloat8_t v) {}

// HARD-LABEL: define{{.*}} void @arg_fixed_mf8(<vscale x 16 x i8> %{{.*}})
void arg_fixed_mf8(fixed_mf8_t v) {}

// HARD-LABEL: define{{.*}} <vscale x 16 x i8> @ret_fixed_mf8(<vscale x 16 x i8> %{{.*}})
fixed_mf8_t ret_fixed_mf8(fixed_mf8_t v) { return v; }

// HARD-LABEL: define{{.*}} void @arg_neon8(<8 x i8> %{{.*}})
void arg_neon8(mfloat8x8_t v) {}

// HARD-LABEL: define{{.*}} void @arg_neon16(<16 x i8> %{{.*}})
void arg_neon16(mfloat8x16_t v) {}

// AAPCS-LABEL: define{{.*}} void @budget(<1 x i8> %{{.*}}, <1 x i8> %{{.*}}, <1 x i8> %{{.*}}, <1 x i8> %{{.*}}, <1 x i8> %{{.*}}, <1 x i8> %{{.*}}, <1 x i8> %{{.*}}, <1 x i8> %{{.*}}, ptr {{.*}}align 16 dead_on_return dereferenceable(16) %{{.*}})
// OTHER-LABEL: define{{.*}} void @budget(<1 x i8> %{{.*}}, <1 x i8> %{{.*}}, <1 x i8> %{{.*}}, <1 x i8> %{{.*}}, <1 x i8> %{{.*}}, <1 x i8> %{{.*}}, <1 x i8> %{{.*}}, <1 x i8> %{{.*}}, i128 %{{.*}})
void budget(__mfp8 a, __mfp8 b, __mfp8 c, __mfp8 d, __mfp8 e, __mfp8 f,
            __mfp8 g, __mfp8 h, One one) {}

// AAPCS-LABEL: define{{.*}} void @budget_room(<1 x i8> %{{.*}}, <1 x i8> %{{.*}}, <1 x i8> %{{.*}}, <1 x i8> %{{.*}}, <1 x i8> %{{.*}}, <1 x i8> %{{.*}}, <1 x i8> %{{.*}}, <vscale x 4 x i32> %{{.*}})
// OTHER-LABEL: define{{.*}} void @budget_room(<1 x i8> %{{.*}}, <1 x i8> %{{.*}}, <1 x i8> %{{.*}}, <1 x i8> %{{.*}}, <1 x i8> %{{.*}}, <1 x i8> %{{.*}}, <1 x i8> %{{.*}}, i128 %{{.*}})
void budget_room(__mfp8 a, __mfp8 b, __mfp8 c, __mfp8 d, __mfp8 e, __mfp8 f,
                 __mfp8 g, One one) {}
#endif
