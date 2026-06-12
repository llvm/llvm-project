// RUN: %clang_cc1 -triple i686-linux-gnu -target-cpu i686 -tune-cpu i686 \
// RUN:   -fclangir -emit-cir %s -o - | FileCheck %s -check-prefix=CIR

// RUN: %clang_cc1 -triple i686-linux-gnu -target-cpu i686 -tune-cpu i686 \
// RUN:   -fclangir -emit-llvm %s -o - | FileCheck %s -check-prefix=LLVM

// RUN: %clang_cc1 -triple i686-linux-gnu -target-cpu i686 -tune-cpu i686 \
// RUN:   -emit-llvm %s -o - | FileCheck %s -check-prefix=LLVM

// LLVM: define {{.*}}@f_default({{.*}} [[f_default:#[0-9]+]]
// LLVM: define {{.*}}@f_avx_sse4_2_ivybridge({{.*}} [[f_avx_sse4_2_ivybridge:#[0-9]+]]
// LLVM: define {{.*}}@f_fpmath_387({{.*}} [[f_default]]
// LLVM: define {{.*}}@f_no_sse2({{.*}} [[f_no_sse2:#[0-9]+]]
// LLVM: define {{.*}}@f_sse4({{.*}} [[f_sse4:#[0-9]+]]
// LLVM: define {{.*}}@f_no_sse4({{.*}} [[f_no_sse4:#[0-9]+]]
// LLVM: define {{.*}}@f_default2({{.*}} [[f_default]]
// LLVM: define {{.*}}@f_avx_sse4_2_ivybridge_2({{.*}} [[f_avx_sse4_2_ivybridge]]
// LLVM: define {{.*}}@f_no_aes_ivybridge({{.*}} [[f_no_aes_ivybridge:#[0-9]+]]
// LLVM: define {{.*}}@f_no_mmx({{.*}} [[f_no_mmx:#[0-9]+]]
// LLVM: define {{.*}}@f_lakemont_mmx({{.*}} [[f_lakemont_mmx:#[0-9]+]]
// LLVM: define {{.*}}@f_use_before_def({{.*}} [[f_lakemont_mmx]]
// LLVM: define {{.*}}@f_tune_sandybridge({{.*}} [[f_tune_sandybridge:#[0-9]+]]
// LLVM: define {{.*}}@f_x86_64_v2({{.*}} [[f_x86_64_v2:#[0-9]+]]
// LLVM: define {{.*}}@f_x86_64_v3({{.*}} [[f_x86_64_v3:#[0-9]+]]
// LLVM: define {{.*}}@f_x86_64_v4({{.*}} [[f_x86_64_v4:#[0-9]+]]
// LLVM: define {{.*}}@f_avx10_1{{.*}} [[f_avx10_1:#[0-9]+]]
// LLVM: define {{.*}}@f_prefer_256_bit({{.*}} [[f_prefer_256_bit:#[0-9]+]]
// LLVM: define {{.*}}@f_no_prefer_256_bit({{.*}} [[f_no_prefer_256_bit:#[0-9]+]]

// CIR:      cir.func{{.*}} @f_default()
// CIR-SAME: "cir.target-cpu" = "i686"
// CIR-SAME: "cir.target-features" = "+cmov,+cx8,+x87"
// CIR-SAME: "cir.tune-cpu" = "i686"

// LLVM: [[f_default]] = {{.*}}"target-cpu"="i686" "target-features"="+cmov,+cx8,+x87" "tune-cpu"="i686"
void f_default(void) {}

// CIR:      cir.func{{.*}} @f_avx_sse4_2_ivybridge()
// CIR-SAME: "cir.target-cpu" = "ivybridge"
// CIR-SAME: "cir.target-features" = "+avx,+cmov,+crc32,+cx16,+cx8,+f16c,+fsgsbase,+fxsr,+mmx,+pclmul,+popcnt,+rdrnd,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt"
// CIR-NOT:  "cir.tune-cpu"

// LLVM: [[f_avx_sse4_2_ivybridge]] = {{.*}}"target-cpu"="ivybridge" "target-features"="+avx,+cmov,+crc32,+cx16,+cx8,+f16c,+fsgsbase,+fxsr,+mmx,+pclmul,+popcnt,+rdrnd,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt"
__attribute__((target("avx,sse4.2,arch=ivybridge")))
void f_avx_sse4_2_ivybridge(void) {}

// fpmath= is currently ignored, so f_fpmath_387 has identical attributes to
// f_default.
// CIR: cir.func{{.*}} @f_fpmath_387()
__attribute__((target("fpmath=387")))
void f_fpmath_387(void) {}

// CIR:      cir.func{{.*}} @f_no_sse2()
// CIR-SAME: "cir.target-cpu" = "i686"
// CIR-SAME: "cir.target-features" = "+cmov,+cx8,+x87,-aes,-amx-avx512,-avx,-avx10.1,-avx10.2,-avx2,-avx512bf16,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512f,-avx512fp16,-avx512ifma,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vp2intersect,-avx512vpopcntdq,-avxifma,-avxneconvert,-avxvnni,-avxvnniint16,-avxvnniint8,-f16c,-fma,-fma4,-gfni,-kl,-pclmul,-sha,-sha512,-sm3,-sm4,-sse2,-sse3,-sse4.1,-sse4.2,-sse4a,-ssse3,-vaes,-vpclmulqdq,-widekl,-xop"
// CIR-SAME: "cir.tune-cpu" = "i686"

// LLVM-NOT: tune-cpu
// LLVM:     [[f_no_sse2]] = {{.*}}"target-cpu"="i686" "target-features"="+cmov,+cx8,+x87,-aes,-amx-avx512,-avx,-avx10.1,-avx10.2,-avx2,-avx512bf16,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512f,-avx512fp16,-avx512ifma,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vp2intersect,-avx512vpopcntdq,-avxifma,-avxneconvert,-avxvnni,-avxvnniint16,-avxvnniint8,-f16c,-fma,-fma4,-gfni,-kl,-pclmul,-sha,-sha512,-sm3,-sm4,-sse2,-sse3,-sse4.1,-sse4.2,-sse4a,-ssse3,-vaes,-vpclmulqdq,-widekl,-xop" "tune-cpu"="i686"
__attribute__((target("no-sse2")))
void f_no_sse2(void) {}

// CIR:      cir.func{{.*}} @f_sse4()
// CIR-SAME: "cir.target-cpu" = "i686"
// CIR-SAME: "cir.target-features" = "+cmov,+crc32,+cx8,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87"
// CIR-SAME: "cir.tune-cpu" = "i686"

// LLVM: [[f_sse4]] = {{.*}}"target-cpu"="i686" "target-features"="+cmov,+crc32,+cx8,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87" "tune-cpu"="i686"
__attribute__((target("sse4")))
void f_sse4(void) {}


// CIR:      cir.func{{.*}} @f_no_sse4()
// CIR-SAME: "cir.target-cpu" = "i686"
// CIR-SAME: "cir.target-features" = "+cmov,+cx8,+x87,-amx-avx512,-avx,-avx10.1,-avx10.2,-avx2,-avx512bf16,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512f,-avx512fp16,-avx512ifma,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vp2intersect,-avx512vpopcntdq,-avxifma,-avxneconvert,-avxvnni,-avxvnniint16,-avxvnniint8,-f16c,-fma,-fma4,-sha512,-sm3,-sm4,-sse4.1,-sse4.2,-vaes,-vpclmulqdq,-xop"
// CIR-SAME: "cir.tune-cpu" = "i686"

// LLVM: [[f_no_sse4]] = {{.*}}"target-cpu"="i686" "target-features"="+cmov,+cx8,+x87,-amx-avx512,-avx,-avx10.1,-avx10.2,-avx2,-avx512bf16,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512f,-avx512fp16,-avx512ifma,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vp2intersect,-avx512vpopcntdq,-avxifma,-avxneconvert,-avxvnni,-avxvnniint16,-avxvnniint8,-f16c,-fma,-fma4,-sha512,-sm3,-sm4,-sse4.1,-sse4.2,-vaes,-vpclmulqdq,-xop" "tune-cpu"="i686"
__attribute__((target("no-sse4")))
void f_no_sse4(void) {}

// f_default2: no per-function attribute, identical attributes to f_default
// (checked above).
void f_default2(void) {
  f_avx_sse4_2_ivybridge();
  return f_default();
}

// f_avx_sse4_2_ivybridge_2: same attributes as f_avx_sse4_2_ivybridge despite
// the extra whitespace in the target string.
__attribute__((target("avx,      sse4.2,      arch=   ivybridge")))
void f_avx_sse4_2_ivybridge_2(void) {}

// CIR:      cir.func{{.*}} @f_no_aes_ivybridge()
// CIR-SAME: "cir.target-cpu" = "ivybridge"
// CIR-SAME: "cir.target-features" = "+avx,+cmov,+crc32,+cx16,+cx8,+f16c,+fsgsbase,+fxsr,+mmx,+pclmul,+popcnt,+rdrnd,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-aes,-vaes"
// CIR-NOT:  "cir.tune-cpu"

// LLVM: [[f_no_aes_ivybridge]] = {{.*}}"target-cpu"="ivybridge" "target-features"="+avx,+cmov,+crc32,+cx16,+cx8,+f16c,+fsgsbase,+fxsr,+mmx,+pclmul,+popcnt,+rdrnd,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-aes,-vaes"
__attribute__((target("no-aes, arch=ivybridge")))
void f_no_aes_ivybridge(void) {}

// CIR:      cir.func{{.*}} @f_no_mmx()
// CIR-SAME: "cir.target-cpu" = "i686"
// CIR-SAME: "cir.target-features" = "+cmov,+cx8,+x87,-mmx"
// CIR-SAME: "cir.tune-cpu" = "i686"

// LLVM: [[f_no_mmx]] = {{.*}}"target-cpu"="i686" "target-features"="+cmov,+cx8,+x87,-mmx"{{.*}}"tune-cpu"="i686"
__attribute__((target("no-mmx")))
void f_no_mmx(void) {}

// CIR:      cir.func{{.*}} @f_lakemont_mmx()
// CIR-SAME: "cir.target-cpu" = "lakemont"
// CIR-SAME: "cir.target-features" = "+cx8,+mmx"

// LLVM: [[f_lakemont_mmx]] = {{.*}}"target-cpu"="lakemont" "target-features"="+cx8,+mmx"
__attribute__((target("arch=lakemont,mmx")))
void f_lakemont_mmx(void) {}

void f_use_before_def(void);
void usage(void){
  f_use_before_def();
}

// f_use_before_def: same attributes as f_lakemont_mmx (checked above) - the
// definition's attribute should be propagated to the earlier declaration.
__attribute__((target("arch=lakemont,mmx")))
void f_use_before_def(void) {}

// CIR:      cir.func{{.*}} @f_tune_sandybridge()
// CIR-SAME: "cir.target-cpu" = "i686"
// CIR-SAME: "cir.target-features" = "+cmov,+cx8,+x87"
// CIR-SAME: "cir.tune-cpu" = "sandybridge"

// LLVM: [[f_tune_sandybridge]] = {{.*}}"target-cpu"="i686" "target-features"="+cmov,+cx8,+x87" "tune-cpu"="sandybridge"
__attribute__((target("tune=sandybridge")))
void f_tune_sandybridge(void) {}

// CIR:      cir.func{{.*}} @f_x86_64_v2()
// CIR-SAME: "cir.target-cpu" = "x86-64-v2"
// CIR-SAME: "cir.target-features" = "+cmov,+crc32,+cx16,+cx8,+fxsr,+mmx,+popcnt,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87"

// LLVM:      [[f_x86_64_v2]] ={{.*}}"target-cpu"="x86-64-v2"
// LLVM-SAME: "target-features"="+cmov,+crc32,+cx16,+cx8,+fxsr,+mmx,+popcnt,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87"
__attribute__((target("arch=x86-64-v2")))
void f_x86_64_v2(void) {}

// CIR:      cir.func{{.*}} @f_x86_64_v3()
// CIR-SAME: "cir.target-cpu" = "x86-64-v3"
// CIR-SAME: "cir.target-features" = "+avx,+avx2,+bmi,+bmi2,+cmov,+crc32,+cx16,+cx8,+f16c,+fma,+fxsr,+lzcnt,+mmx,+movbe,+popcnt,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"

// LLVM:      [[f_x86_64_v3]] = {{.*}}"target-cpu"="x86-64-v3"
// LLVM-SAME: "target-features"="+avx,+avx2,+bmi,+bmi2,+cmov,+crc32,+cx16,+cx8,+f16c,+fma,+fxsr,+lzcnt,+mmx,+movbe,+popcnt,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("arch=x86-64-v3")))
void f_x86_64_v3(void) {}

// CIR:      cir.func{{.*}} @f_x86_64_v4()
// CIR-SAME: "cir.target-cpu" = "x86-64-v4"
// CIR-SAME: "cir.target-features" = "+avx,+avx2,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512vl,+bmi,+bmi2,+cmov,+crc32,+cx16,+cx8,+f16c,+fma,+fxsr,+lzcnt,+mmx,+movbe,+popcnt,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"

// LLVM:      [[f_x86_64_v4]] = {{.*}}"target-cpu"="x86-64-v4"
// LLVM-SAME: "target-features"="+avx,+avx2,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512vl,+bmi,+bmi2,+cmov,+crc32,+cx16,+cx8,+f16c,+fma,+fxsr,+lzcnt,+mmx,+movbe,+popcnt,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("arch=x86-64-v4")))
void f_x86_64_v4(void) {}

// CIR:      cir.func{{.*}} @f_avx10_1()
// CIR-SAME: "cir.target-cpu" = "i686"
// CIR-SAME: "cir.target-features" = "+avx,+avx10.1,+avx2,+avx512bf16,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512fp16,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+cmov,+crc32,+cx8,+f16c,+fma,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"

// LLVM: [[f_avx10_1]] = {{.*}}"target-cpu"="i686" "target-features"="+avx,+avx10.1,+avx2,+avx512bf16,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512fp16,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+cmov,+crc32,+cx8,+f16c,+fma,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("avx10.1")))
void f_avx10_1(void) {}

// CIR:      cir.func{{.*}} @f_prefer_256_bit()
// CIR-SAME: "cir.target-features" = "{{.*}}+prefer-256-bit{{.*}}"

// LLVM: [[f_prefer_256_bit]] = {{.*}}"target-features"="{{.*}}+prefer-256-bit
__attribute__((target("prefer-256-bit")))
void f_prefer_256_bit(void) {}

// CIR:      cir.func{{.*}} @f_no_prefer_256_bit()
// CIR-SAME: "cir.target-features" = "{{.*}}-prefer-256-bit{{.*}}"

// LLVM: [[f_no_prefer_256_bit]] = {{.*}}"target-features"="{{.*}}-prefer-256-bit
__attribute__((target("no-prefer-256-bit")))
void f_no_prefer_256_bit(void) {}
