// Test/document all of the dependencies between possible AArch64 FMV extensions.
// Also test the name mangling.

// RUN: %clang --target=aarch64-linux-gnu --rtlib=compiler-rt -emit-llvm -S -o - %s | FileCheck %s

// CHECK: define dso_local i32 @fmv._Maes() #[[aes:[0-9]+]] {
__attribute__((target_version("aes"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mbf16() #[[bf16:[0-9]+]] {
__attribute__((target_version("bf16"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mbti() #[[bti:[0-9]+]] {
__attribute__((target_version("bti"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mcrc() #[[crc:[0-9]+]] {
__attribute__((target_version("crc"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mdit() #[[dit:[0-9]+]] {
__attribute__((target_version("dit"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mdotprod() #[[dotprod:[0-9]+]] {
__attribute__((target_version("dotprod"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mdpb() #[[dpb:[0-9]+]] {
__attribute__((target_version("dpb"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mdpb2() #[[dpb2:[0-9]+]] {
__attribute__((target_version("dpb2"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mf32mm() #[[f32mm:[0-9]+]] {
__attribute__((target_version("f32mm"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mf64mm() #[[f64mm:[0-9]+]] {
__attribute__((target_version("f64mm"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mfcma() #[[fcma:[0-9]+]] {
__attribute__((target_version("fcma"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mflagm() #[[flagm:[0-9]+]] {
__attribute__((target_version("flagm"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mflagm2() #[[flagm2:[0-9]+]] {
__attribute__((target_version("flagm2"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mfp() #[[fp:[0-9]+]] {
__attribute__((target_version("fp"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mfp16() #[[fp16:[0-9]+]] {
__attribute__((target_version("fp16"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mfp16fml() #[[fp16fml:[0-9]+]] {
__attribute__((target_version("fp16fml"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mfrintts() #[[frintts:[0-9]+]] {
__attribute__((target_version("frintts"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mi8mm() #[[i8mm:[0-9]+]] {
__attribute__((target_version("i8mm"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mjscvt() #[[jscvt:[0-9]+]] {
__attribute__((target_version("jscvt"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mlse() #[[lse:[0-9]+]] {
__attribute__((target_version("lse"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mmemtag() #[[memtag:[0-9]+]] {
__attribute__((target_version("memtag"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mmops() #[[mops:[0-9]+]] {
__attribute__((target_version("mops"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mrcpc() #[[rcpc:[0-9]+]] {
__attribute__((target_version("rcpc"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mrcpc2() #[[rcpc2:[0-9]+]] {
__attribute__((target_version("rcpc2"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mrcpc3() #[[rcpc3:[0-9]+]] {
__attribute__((target_version("rcpc3"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mrdm() #[[rdm:[0-9]+]] {
__attribute__((target_version("rdm"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mrng() #[[rng:[0-9]+]] {
__attribute__((target_version("rng"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msb() #[[sb:[0-9]+]] {
__attribute__((target_version("sb"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msha2() #[[sha2:[0-9]+]] {
__attribute__((target_version("sha2"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msha3() #[[sha3:[0-9]+]] {
__attribute__((target_version("sha3"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msimd() #[[simd:[0-9]+]] {
__attribute__((target_version("simd"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msm4() #[[sm4:[0-9]+]] {
__attribute__((target_version("sm4"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msme() #[[sme:[0-9]+]] {
__attribute__((target_version("sme"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msme-f64f64() #[[sme_f64f64:[0-9]+]] {
__attribute__((target_version("sme-f64f64"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msme-i16i64() #[[sme_i16i64:[0-9]+]] {
__attribute__((target_version("sme-i16i64"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msme2() #[[sme2:[0-9]+]] {
__attribute__((target_version("sme2"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mssbs() #[[ssbs:[0-9]+]] {
__attribute__((target_version("ssbs"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msve() #[[sve:[0-9]+]] {
__attribute__((target_version("sve"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msve2() #[[sve2:[0-9]+]] {
__attribute__((target_version("sve2"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msve2-aes() #[[sve2_aes:[0-9]+]] {
__attribute__((target_version("sve2-aes"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msve2-bitperm() #[[sve2_bitperm:[0-9]+]] {
__attribute__((target_version("sve2-bitperm"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msve2-sha3() #[[sve2_sha3:[0-9]+]] {
__attribute__((target_version("sve2-sha3"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msve2-sm4() #[[sve2_sm4:[0-9]+]] {
__attribute__((target_version("sve2-sm4"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mwfxt() #[[wfxt:[0-9]+]] {
__attribute__((target_version("wfxt"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mcssc() #[[cssc:[0-9]+]] {
__attribute__((target_version("cssc"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mfp8() #[[fp8:[0-9]+]] {
__attribute__((target_version("fp8"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mf8f32mm() #[[f8f32mm:[0-9]+]] {
__attribute__((target_version("f8f32mm"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mfp8dot4() #[[fp8dot4:[0-9]+]] {
__attribute__((target_version("fp8dot4"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mfp8dot2() #[[fp8dot2:[0-9]+]] {
__attribute__((target_version("fp8dot2"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mfp8fma() #[[fp8fma:[0-9]+]] {
__attribute__((target_version("fp8fma"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msme-f8f32() #[[sme_f8f32:[0-9]+]] {
__attribute__((target_version("sme-f8f32"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mssve-fp8dot4() #[[ssve_fp8dot4:[0-9]+]] {
__attribute__((target_version("ssve-fp8dot4"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mssve-fp8fma() #[[ssve_fp8fma:[0-9]+]] {
__attribute__((target_version("ssve-fp8fma"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mssve-bitperm() #[[ssve_bitperm:[0-9]+]] {
__attribute__((target_version("ssve-bitperm"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mssve-fp8dot2() #[[ssve_fp8dot2:[0-9]+]] {
__attribute__((target_version("ssve-fp8dot2"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mssve-aes() #[[ssve_aes:[0-9]+]] {
__attribute__((target_version("ssve-aes"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mssve-fexpa() #[[ssve_fexpa:[0-9]+]] {
__attribute__((target_version("ssve-fexpa"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mlut() #[[lut:[0-9]+]] {
__attribute__((target_version("lut"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mfaminmax() #[[faminmax:[0-9]+]] {
__attribute__((target_version("faminmax"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msme-lutv2() #[[sme_lutv2:[0-9]+]] {
__attribute__((target_version("sme-lutv2"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msme2p1() #[[sme2p1:[0-9]+]] {
__attribute__((target_version("sme2p1"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msme2p2() #[[sme2p2:[0-9]+]] {
__attribute__((target_version("sme2p2"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msve2p1() #[[sve2p1:[0-9]+]] {
__attribute__((target_version("sve2p1"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msve2p2() #[[sve2p2:[0-9]+]] {
__attribute__((target_version("sve2p2"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msme-f16f16() #[[sme_f16f16:[0-9]+]] {
__attribute__((target_version("sme-f16f16"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mgcs() #[[gcs:[0-9]+]] {
__attribute__((target_version("gcs"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msme-f8f16() #[[sme_f8f16:[0-9]+]] {
__attribute__((target_version("sme-f8f16"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mf8f16mm() #[[f8f16mm:[0-9]+]] {
__attribute__((target_version("f8f16mm"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msve-aes2() #[[sve_aes2:[0-9]+]] {
__attribute__((target_version("sve-aes2"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msve-bfscale() #[[sve_bfscale:[0-9]+]] {
__attribute__((target_version("sve-bfscale"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msve-f16f32mm() #[[sve_f16f32mm:[0-9]+]] {
__attribute__((target_version("sve-f16f32mm"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msme-mop4() #[[sme_mop4:[0-9]+]] {
__attribute__((target_version("sme-mop4"))) int fmv(void) { return 0; }

// CHECK-NOT: define dso_local i32 @fmv._M{{.*}}
__attribute__((target_version("non_existent_extension"))) int fmv(void);

__attribute__((target_version("default"))) int fmv(void);

int caller() {
  return fmv();
}

// CHECK: attributes #[[aes]] = { {{.*}} "target-features"="+aes,+fp-armv8,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[bf16]] = { {{.*}} "target-features"="+bf16,+fp-armv8,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[bti]] = { {{.*}} "target-features"="+bti,+fp-armv8,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[crc]] = { {{.*}} "target-features"="+crc,+fp-armv8,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[dit]] = { {{.*}} "target-features"="+dit,+fp-armv8,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[dotprod]] = { {{.*}} "target-features"="+dotprod,+fp-armv8,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[dpb]] = { {{.*}} "target-features"="+ccpp,+fp-armv8,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[dpb2]] = { {{.*}} "target-features"="+ccdp,+ccpp,+fp-armv8,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[f32mm]] = { {{.*}} "target-features"="+f32mm,+fp-armv8,+fullfp16,+neon,+outline-atomics,+sve,+v8a"
// CHECK: attributes #[[f64mm]] = { {{.*}} "target-features"="+f64mm,+fp-armv8,+fullfp16,+neon,+outline-atomics,+sve,+v8a"
// CHECK: attributes #[[fcma]] = { {{.*}} "target-features"="+complxnum,+fp-armv8,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[flagm]] = { {{.*}} "target-features"="+flagm,+fp-armv8,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[flagm2]] = { {{.*}} "target-features"="+altnzcv,+flagm,+fp-armv8,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[fp]] = { {{.*}} "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[fp16]] = { {{.*}} "target-features"="+fp-armv8,+fullfp16,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[fp16fml]] = { {{.*}} "target-features"="+fp-armv8,+fp16fml,+fullfp16,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[frintts]] = { {{.*}} "target-features"="+fp-armv8,+fptoint,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[i8mm]] = { {{.*}} "target-features"="+fp-armv8,+i8mm,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[jscvt]] = { {{.*}} "target-features"="+fp-armv8,+jsconv,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[lse]] = { {{.*}} "target-features"="+fp-armv8,+lse,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[memtag]] = { {{.*}} "target-features"="+fp-armv8,+mte,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[mops]] = { {{.*}} "target-features"="+fp-armv8,+mops,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[rcpc]] = { {{.*}} "target-features"="+fp-armv8,+neon,+outline-atomics,+rcpc,+v8a"
// CHECK: attributes #[[rcpc2]] = { {{.*}} "target-features"="+fp-armv8,+neon,+outline-atomics,+rcpc,+rcpc-immo,+v8a"
// CHECK: attributes #[[rcpc3]] = { {{.*}} "target-features"="+fp-armv8,+neon,+outline-atomics,+rcpc,+rcpc-immo,+rcpc3,+v8a"
// CHECK: attributes #[[rdm]] = { {{.*}} "target-features"="+fp-armv8,+neon,+outline-atomics,+rdm,+v8a"
// CHECK: attributes #[[rng]] = { {{.*}} "target-features"="+fp-armv8,+neon,+outline-atomics,+rand,+v8a"
// CHECK: attributes #[[sb]] = { {{.*}} "target-features"="+fp-armv8,+neon,+outline-atomics,+sb,+v8a"
// CHECK: attributes #[[sha2]] = { {{.*}} "target-features"="+fp-armv8,+neon,+outline-atomics,+sha2,+v8a"
// CHECK: attributes #[[sha3]] = { {{.*}} "target-features"="+fp-armv8,+neon,+outline-atomics,+sha2,+sha3,+v8a"
// CHECK: attributes #[[simd]] = { {{.*}} "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[sm4]] = { {{.*}} "target-features"="+fp-armv8,+neon,+outline-atomics,+sm4,+v8a"
// CHECK: attributes #[[sme]] = { {{.*}} "target-features"="+bf16,+fp-armv8,+fullfp16,+neon,+outline-atomics,+sme,+v8a"
// CHECK: attributes #[[sme_f64f64]] = { {{.*}} "target-features"="+bf16,+fp-armv8,+fullfp16,+neon,+outline-atomics,+sme,+sme-f64f64,+v8a"
// CHECK: attributes #[[sme_i16i64]] = { {{.*}} "target-features"="+bf16,+fp-armv8,+fullfp16,+neon,+outline-atomics,+sme,+sme-i16i64,+v8a"
// CHECK: attributes #[[sme2]] = { {{.*}} "target-features"="+bf16,+fp-armv8,+fullfp16,+neon,+outline-atomics,+sme,+sme2,+v8a"
// CHECK: attributes #[[ssbs]] = { {{.*}} "target-features"="+fp-armv8,+neon,+outline-atomics,+ssbs,+v8a"
// CHECK: attributes #[[sve]] = { {{.*}} "target-features"="+fp-armv8,+fullfp16,+neon,+outline-atomics,+sve,+v8a"
// CHECK: attributes #[[sve2]] = { {{.*}} "target-features"="+fp-armv8,+fullfp16,+neon,+outline-atomics,+sve,+sve2,+v8a"
// CHECK: attributes #[[sve2_aes]] = { {{.*}} "target-features"="+aes,+fp-armv8,+fullfp16,+neon,+outline-atomics,+sve,+sve-aes,+sve2,+sve2-aes,+v8a"
// CHECK: attributes #[[sve2_bitperm]] = { {{.*}} "target-features"="+fp-armv8,+fullfp16,+neon,+outline-atomics,+sve,+sve-bitperm,+sve2,+sve2-bitperm,+v8a"
// CHECK: attributes #[[sve2_sha3]] = { {{.*}} "target-features"="+fp-armv8,+fullfp16,+neon,+outline-atomics,+sha2,+sha3,+sve,+sve-sha3,+sve2,+sve2-sha3,+v8a"
// CHECK: attributes #[[sve2_sm4]] = { {{.*}} "target-features"="+fp-armv8,+fullfp16,+neon,+outline-atomics,+sm4,+sve,+sve-sm4,+sve2,+sve2-sm4,+v8a"
// CHECK: attributes #[[wfxt]] = { {{.*}} "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,+wfxt"
// CHECK: attributes #[[cssc]] = { {{.*}} "target-features"="+cssc,+fp-armv8,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[fp8]] = { {{.*}} "target-features"="+fp-armv8,+fp8,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[f8f32mm]] = { {{.*}} "target-features"="+f8f32mm,+fp-armv8,+fp8,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[fp8dot4]] = { {{.*}} "target-features"="+fp-armv8,+fp8,+fp8dot4,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[fp8dot2]] = { {{.*}} "target-features"="+fp-armv8,+fp8,+fp8dot2,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[fp8fma]] = { {{.*}} "target-features"="+fp-armv8,+fp8,+fp8fma,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[sme_f8f32]] = { {{.*}} "target-features"="+bf16,+fp-armv8,+fp8,+fullfp16,+neon,+outline-atomics,+sme,+sme-f8f32,+sme2,+v8a"
// CHECK: attributes #[[ssve_fp8dot4]] = { {{.*}} "target-features"="+bf16,+fp-armv8,+fp8,+fullfp16,+neon,+outline-atomics,+sme,+sme2,+ssve-fp8dot4,+v8a"
// CHECK: attributes #[[ssve_fp8fma]] = { {{.*}} "target-features"="+bf16,+fp-armv8,+fp8,+fullfp16,+neon,+outline-atomics,+sme,+sme2,+ssve-fp8fma,+v8a"
// CHECK: attributes #[[ssve_bitperm]] = { {{.*}} "target-features"="+bf16,+fp-armv8,+fullfp16,+neon,+outline-atomics,+sme,+sme2,+ssve-bitperm,+sve-bitperm,+v8a"
// CHECK: attributes #[[ssve_fp8dot2]] = { {{.*}} "target-features"="+bf16,+fp-armv8,+fp8,+fullfp16,+neon,+outline-atomics,+sme,+sme2,+ssve-fp8dot2,+v8a"
// CHECK: attributes #[[ssve_aes]] = { {{.*}} "target-features"="+aes,+bf16,+fp-armv8,+fullfp16,+neon,+outline-atomics,+sme,+sme2,+ssve-aes,+sve-aes,+v8a"
// CHECK: attributes #[[ssve_fexpa]] = { {{.*}} "target-features"="+bf16,+fp-armv8,+fullfp16,+neon,+outline-atomics,+sme,+sme2,+ssve-fexpa,+v8a"
// CHECK: attributes #[[lut]] = { {{.*}} "target-features"="+fp-armv8,+lut,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[faminmax]] = { {{.*}} "target-features"="+faminmax,+fp-armv8,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[sme_lutv2]] = { {{.*}} "target-features"="+bf16,+fp-armv8,+fullfp16,+neon,+outline-atomics,+sme,+sme-lutv2,+sme2,+v8a"
// CHECK: attributes #[[sme2p1]] = { {{.*}} "target-features"="+bf16,+fp-armv8,+fullfp16,+neon,+outline-atomics,+sme,+sme2,+sme2p1,+v8a"
// CHECK: attributes #[[sme2p2]] = { {{.*}} "target-features"="+bf16,+fp-armv8,+fullfp16,+neon,+outline-atomics,+sme,+sme2,+sme2p1,+sme2p2,+v8a"
// CHECK: attributes #[[sve2p1]] = { {{.*}} "target-features"="+fp-armv8,+fullfp16,+neon,+outline-atomics,+sve,+sve2,+sve2p1,+v8a"
// CHECK: attributes #[[sve2p2]] = { {{.*}} "target-features"="+fp-armv8,+fullfp16,+neon,+outline-atomics,+sve,+sve2,+sve2p1,+sve2p2,+v8a"
// CHECK: attributes #[[sme_f16f16]] = { {{.*}} "target-features"="+bf16,+fp-armv8,+fullfp16,+neon,+outline-atomics,+sme,+sme-f16f16,+sme2,+v8a"
// CHECK: attributes #[[gcs]] = { {{.*}} "target-features"="+chk,+fp-armv8,+gcs,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[sme_f8f16]] = { {{.*}} "target-features"="+bf16,+fp-armv8,+fp8,+fullfp16,+neon,+outline-atomics,+sme,+sme-f8f16,+sme2,+v8a"
// CHECK: attributes #[[f8f16mm]] = { {{.*}} "target-features"="+f8f16mm,+fp-armv8,+fp8,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[sve_aes2]] = { {{.*}} "target-features"="+fp-armv8,+neon,+outline-atomics,+sve-aes2,+v8a"
// CHECK: attributes #[[sve_bfscale]] = { {{.*}} "target-features"="+fp-armv8,+neon,+outline-atomics,+sve-bfscale,+v8a"
// CHECK: attributes #[[sve_f16f32mm]] = { {{.*}} "target-features"="+fp-armv8,+fullfp16,+neon,+outline-atomics,+sve,+sve-f16f32mm,+v8a"
// CHECK: attributes #[[sme_mop4]] = { {{.*}} "target-features"="+bf16,+fp-armv8,+fullfp16,+neon,+outline-atomics,+sme,+sme-mop4,+sme2,+v8a"
