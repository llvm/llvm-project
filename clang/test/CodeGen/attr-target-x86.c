// RUN: %clang_cc1 -triple i686-linux-gnu -target-cpu i686 -tune-cpu i686 -emit-llvm %s -o - | FileCheck %s

// CHECK: define {{.*}}@f_default({{.*}} [[f_default:#[0-9]+]]
// CHECK: define {{.*}}@f_avx_sse4_2_ivybridge({{.*}} [[f_avx_sse4_2_ivybridge:#[0-9]+]]
// CHECK: define {{.*}}@f_fpmath_387({{.*}} [[f_default]]
// CHECK: define {{.*}}@f_no_sse2({{.*}} [[f_no_sse2:#[0-9]+]]
// CHECK: define {{.*}}@f_sse4({{.*}} [[f_sse4:#[0-9]+]]
// CHECK: define {{.*}}@f_no_sse4({{.*}} [[f_no_sse4:#[0-9]+]]
// CHECK: define {{.*}}@f_default2({{.*}} [[f_default]]
// CHECK: define {{.*}}@f_avx_sse4_2_ivybridge_2({{.*}} [[f_avx_sse4_2_ivybridge]]
// CHECK: define {{.*}}@f_no_aes_ivybridge({{.*}} [[f_no_aes_ivybridge:#[0-9]+]]
// CHECK: define {{.*}}@f_no_mmx({{.*}} [[f_no_mmx:#[0-9]+]]
// CHECK: define {{.*}}@f_lakemont_mmx({{.*}} [[f_lakemont_mmx:#[0-9]+]]
// CHECK: define {{.*}}@f_use_before_def({{.*}} [[f_lakemont_mmx]]
// CHECK: define {{.*}}@f_tune_sandybridge({{.*}} [[f_tune_sandybridge:#[0-9]+]]
// CHECK: define {{.*}}@f_x86_64_v2({{.*}} [[f_x86_64_v2:#[0-9]+]]
// CHECK: define {{.*}}@f_x86_64_v3({{.*}} [[f_x86_64_v3:#[0-9]+]]
// CHECK: define {{.*}}@f_x86_64_v4({{.*}} [[f_x86_64_v4:#[0-9]+]]
// CHECK: define {{.*}}@f_avx10_1_256{{.*}} [[f_avx10_1_256:#[0-9]+]]
// CHECK: define {{.*}}@f_avx10_1_512{{.*}} [[f_avx10_1_512:#[0-9]+]]
// CHECK: define {{.*}}@f_prefer_256_bit({{.*}} [[f_prefer_256_bit:#[0-9]+]]
// CHECK: define {{.*}}@f_no_prefer_256_bit({{.*}} [[f_no_prefer_256_bit:#[0-9]+]]

// CHECK: [[f_default]] = {{.*}}"target-cpu"="i686" "target-features"="+cmov,+cx8,+x87" "tune-cpu"="i686"
void f_default(void) {}

// CHECK: [[f_avx_sse4_2_ivybridge]] = {{.*}}"target-cpu"="ivybridge" "target-features"="+avx,+cmov,+crc32,+cx16,+cx8,+f16c,+fsgsbase,+fxsr,+mmx,+pclmul,+popcnt,+rdrnd,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt"
__attribute__((target("avx,sse4.2,arch=ivybridge")))
void f_avx_sse4_2_ivybridge(void) {}

// We're currently ignoring the fpmath attribute. So checked above that
// attributes are identical to f_default.
__attribute__((target("fpmath=387")))
void f_fpmath_387(void) {}

// CHECK-NOT: tune-cpu
// CHECK: [[f_no_sse2]] = {{.*}}"target-cpu"="i686" "target-features"="+cmov,+cx8,+x87,-aes,-amx-avx512,-avx,-avx10.1-256,-avx10.1-512,-avx10.2-256,-avx10.2-512,-avx2,-avx512bf16,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512f,-avx512fp16,-avx512ifma,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vp2intersect,-avx512vpopcntdq,-avxifma,-avxneconvert,-avxvnni,-avxvnniint16,-avxvnniint8,-f16c,-fma,-fma4,-gfni,-kl,-pclmul,-sha,-sha512,-sm3,-sm4,-sse2,-sse3,-sse4.1,-sse4.2,-sse4a,-ssse3,-vaes,-vpclmulqdq,-widekl,-xop" "tune-cpu"="i686"
__attribute__((target("no-sse2")))
void f_no_sse2(void) {}

// CHECK: [[f_sse4]] = {{.*}}"target-cpu"="i686" "target-features"="+cmov,+crc32,+cx8,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87" "tune-cpu"="i686"
__attribute__((target("sse4")))
void f_sse4(void) {}

// CHECK: [[f_no_sse4]] = {{.*}}"target-cpu"="i686" "target-features"="+cmov,+cx8,+x87,-amx-avx512,-avx,-avx10.1-256,-avx10.1-512,-avx10.2-256,-avx10.2-512,-avx2,-avx512bf16,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512f,-avx512fp16,-avx512ifma,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vp2intersect,-avx512vpopcntdq,-avxifma,-avxneconvert,-avxvnni,-avxvnniint16,-avxvnniint8,-f16c,-fma,-fma4,-sha512,-sm3,-sm4,-sse4.1,-sse4.2,-vaes,-vpclmulqdq,-xop" "tune-cpu"="i686"
__attribute__((target("no-sse4")))
void f_no_sse4(void) {}

// checked above that attributes are identical to f_default
void f_default2(void) {
  f_avx_sse4_2_ivybridge();
  return f_default();
}

// Checked above to have same attributes as f_avx_sse4_2_ivybridge
__attribute__((target("avx,      sse4.2,      arch=   ivybridge")))
void f_avx_sse4_2_ivybridge_2(void) {}

// CHECK: [[f_no_aes_ivybridge]] = {{.*}}"target-cpu"="ivybridge" "target-features"="+avx,+cmov,+crc32,+cx16,+cx8,+f16c,+fsgsbase,+fxsr,+mmx,+pclmul,+popcnt,+rdrnd,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-aes,-amx-avx512,-avx10.1-256,-avx10.1-512,-avx10.2-256,-avx10.2-512,-vaes"
__attribute__((target("no-aes, arch=ivybridge")))
void f_no_aes_ivybridge(void) {}

// CHECK-NOT: tune-cpu
// CHECK: [[f_no_mmx]] = {{.*}}"target-cpu"="i686" "target-features"="+cmov,+cx8,+x87,-mmx"
__attribute__((target("no-mmx")))
void f_no_mmx(void) {}

// CHECK: [[f_lakemont_mmx]] = {{.*}}"target-cpu"="lakemont" "target-features"="+cx8,+mmx"
// Adding the attribute to a definition does update it in IR.
__attribute__((target("arch=lakemont,mmx")))
void f_lakemont_mmx(void) {}

void f_use_before_def(void);
void usage(void){
  f_use_before_def();
}

// Checked above to have same attributes as f_lakemont_mmx
__attribute__((target("arch=lakemont,mmx")))
void f_use_before_def(void) {}

// CHECK: [[f_tune_sandybridge]] = {{.*}}"target-cpu"="i686" "target-features"="+cmov,+cx8,+x87" "tune-cpu"="sandybridge"
__attribute__((target("tune=sandybridge")))
void f_tune_sandybridge(void) {}

// CHECK: [[f_x86_64_v2]] ={{.*}}"target-cpu"="x86-64-v2"
// CHECK-SAME: "target-features"="+cmov,+crc32,+cx16,+cx8,+fxsr,+mmx,+popcnt,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87"
__attribute__((target("arch=x86-64-v2")))
void f_x86_64_v2(void) {}

// CHECK: [[f_x86_64_v3]] = {{.*}}"target-cpu"="x86-64-v3"
// CHECK-SAME: "target-features"="+avx,+avx2,+bmi,+bmi2,+cmov,+crc32,+cx16,+cx8,+f16c,+fma,+fxsr,+lzcnt,+mmx,+movbe,+popcnt,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("arch=x86-64-v3")))
void f_x86_64_v3(void) {}

// CHECK: [[f_x86_64_v4]] = {{.*}}"target-cpu"="x86-64-v4"
// CHECK-SAME: "target-features"="+avx,+avx2,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512vl,+bmi,+bmi2,+cmov,+crc32,+cx16,+cx8,+evex512,+f16c,+fma,+fxsr,+lzcnt,+mmx,+movbe,+popcnt,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("arch=x86-64-v4")))
void f_x86_64_v4(void) {}

// CHECK: [[f_avx10_1_256]] = {{.*}}"target-cpu"="i686" "target-features"="+aes,+avx,+avx10.1-256,+avx2,+avx512bf16,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512fp16,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+cmov,+crc32,+cx8,+f16c,+fma,+mmx,+pclmul,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+vaes,+vpclmulqdq,+x87,+xsave,-amx-avx512,-avx10.1-512,-avx10.2-512,-evex512"
__attribute__((target("avx10.1-256")))
void f_avx10_1_256(void) {}

// CHECK: [[f_avx10_1_512]] = {{.*}}"target-cpu"="i686" "target-features"="+aes,+avx,+avx10.1-256,+avx10.1-512,+avx2,+avx512bf16,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512fp16,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+cmov,+crc32,+cx8,+evex512,+f16c,+fma,+mmx,+pclmul,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+vaes,+vpclmulqdq,+x87,+xsave"
__attribute__((target("avx10.1-512")))
void f_avx10_1_512(void) {}

// CHECK: [[f_prefer_256_bit]] = {{.*}}"target-features"="{{.*}}+prefer-256-bit
__attribute__((target("prefer-256-bit")))
void f_prefer_256_bit(void) {}

// CHECK: [[f_no_prefer_256_bit]] = {{.*}}"target-features"="{{.*}}-prefer-256-bit
__attribute__((target("no-prefer-256-bit")))
void f_no_prefer_256_bit(void) {}