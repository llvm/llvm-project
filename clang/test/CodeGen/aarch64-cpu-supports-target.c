// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -emit-llvm -o - %s | FileCheck %s

int check_all_feature() {
  if (__builtin_cpu_supports("rng+flagm+flagm2+fp16fml+dotprod+sm4"))
    return 1;
  else if (__builtin_cpu_supports("rdm+lse+fp+simd+crc+sha1+sha2+sha3"))
    return 2;
  else if (__builtin_cpu_supports("aes+pmull+fp16+dit+dpb+dpb2+jscvt"))
    return 3;
  else if (__builtin_cpu_supports("fcma+rcpc+rcpc2+rcpc3+frintts+dgh"))
    return 4;
  else if (__builtin_cpu_supports("i8mm+bf16+ebf16+rpres+sve+sve-bf16"))
    return 5;
  else if (__builtin_cpu_supports("sve-ebf16+sve-i8mm+f32mm+f64mm"))
    return 6;
  else if (__builtin_cpu_supports("sve2+sve2-aes+sve2-pmull128"))
    return 7;
  else if (__builtin_cpu_supports("sve2-bitperm+sve2-sha3+sve2-sm4"))
    return 8;
  else if (__builtin_cpu_supports("sme+memtag+memtag2+memtag3+sb"))
    return 9;
  else if (__builtin_cpu_supports("predres+ssbs+ssbs2+bti+ls64+ls64_v"))
    return 10;
  else if (__builtin_cpu_supports("ls64_accdata+wfxt+sme-f64f64"))
    return 11;
  else if (__builtin_cpu_supports("sme-i16i64+sme2"))
    return 12;
  else
    return 0;
}

// CHECK-LABEL: define dso_local i32 @neon_code() #1
int __attribute__((target("simd"))) neon_code() { return 1; }

// CHECK-LABEL: define dso_local i32 @sve_code() #2
int __attribute__((target("sve"))) sve_code() { return 2; }

// CHECK-LABEL: define dso_local i32 @code() #0
int code() { return 3; }

// CHECK-LABEL: define dso_local i32 @test_versions() #0
int test_versions() {
  if (__builtin_cpu_supports("sve"))
    return sve_code();
  else if (__builtin_cpu_supports("simd"))
    return neon_code();
  else
    return code();
}
// CHECK: attributes #0 = { noinline nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
// CHECK: attributes #1 = { noinline nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+neon" }
// CHECK: attributes #2 = { noinline nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+fp-armv8,+fullfp16,+neon,+sve" }
