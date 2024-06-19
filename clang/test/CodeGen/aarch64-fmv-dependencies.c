// Test/document all of the dependencies between possible AArch64 FMV extensions.
// Also test the name mangling.

// RUN: %clang --target=aarch64-linux-gnu --rtlib=compiler-rt -emit-llvm -S -o - %s | FileCheck %s

// CHECK: define dso_local i32 @fmv._Maes() #[[ATTR0:[0-9]+]] {
__attribute__((target_version("aes"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mbf16() #[[bf16_ebf16:[0-9]+]] {
__attribute__((target_version("bf16"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mbti() #[[bti:[0-9]+]] {
__attribute__((target_version("bti"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mcrc() #[[crc:[0-9]+]] {
__attribute__((target_version("crc"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mdgh() #[[ATTR0:[0-9]+]] {
__attribute__((target_version("dgh"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mdit() #[[dit:[0-9]+]] {
__attribute__((target_version("dit"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mdotprod() #[[dotprod:[0-9]+]] {
__attribute__((target_version("dotprod"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mdpb() #[[dpb:[0-9]+]] {
__attribute__((target_version("dpb"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mdpb2() #[[dpb2:[0-9]+]] {
__attribute__((target_version("dpb2"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mebf16() #[[bf16_ebf16:[0-9]+]] {
__attribute__((target_version("ebf16"))) int fmv(void) { return 0; }

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

// CHECK: define dso_local i32 @fmv._Mfp() #[[ATTR0:[0-9]+]] {
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

// CHECK: define dso_local i32 @fmv._Mls64() #[[ATTR0:[0-9]+]] {
__attribute__((target_version("ls64"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mls64_accdata() #[[ls64_accdata:[0-9]+]] {
__attribute__((target_version("ls64_accdata"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mls64_v() #[[ATTR0:[0-9]+]] {
__attribute__((target_version("ls64_v"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mlse() #[[lse:[0-9]+]] {
__attribute__((target_version("lse"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mmemtag() #[[ATTR0:[0-9]+]] {
__attribute__((target_version("memtag"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mmemtag2() #[[memtag2:[0-9]+]] {
__attribute__((target_version("memtag2"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mmemtag3() #[[memtag2:[0-9]+]] {
__attribute__((target_version("memtag3"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mmops() #[[mops:[0-9]+]] {
__attribute__((target_version("mops"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mpmull() #[[pmull:[0-9]+]] {
__attribute__((target_version("pmull"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mpredres() #[[predres:[0-9]+]] {
__attribute__((target_version("predres"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mrcpc() #[[rcpc:[0-9]+]] {
__attribute__((target_version("rcpc"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mrcpc2() #[[rcpc:[0-9]+]] {
__attribute__((target_version("rcpc2"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mrcpc3() #[[rcpc3:[0-9]+]] {
__attribute__((target_version("rcpc3"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mrdm() #[[rdm:[0-9]+]] {
__attribute__((target_version("rdm"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mrng() #[[rng:[0-9]+]] {
__attribute__((target_version("rng"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mrpres() #[[ATTR0:[0-9]+]] {
__attribute__((target_version("rpres"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msb() #[[sb:[0-9]+]] {
__attribute__((target_version("sb"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msha1() #[[ATTR0:[0-9]+]] {
__attribute__((target_version("sha1"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msha2() #[[sha2:[0-9]+]] {
__attribute__((target_version("sha2"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msha3() #[[sha3:[0-9]+]] {
__attribute__((target_version("sha3"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msimd() #[[ATTR0:[0-9]+]] {
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

// CHECK: define dso_local i32 @fmv._Mssbs() #[[ATTR0:[0-9]+]] {
__attribute__((target_version("ssbs"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mssbs2() #[[ssbs2:[0-9]+]] {
__attribute__((target_version("ssbs2"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msve() #[[sve:[0-9]+]] {
__attribute__((target_version("sve"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msve-bf16() #[[sve_bf16_ebf16:[0-9]+]] {
__attribute__((target_version("sve-bf16"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msve-ebf16() #[[sve_bf16_ebf16:[0-9]+]] {
__attribute__((target_version("sve-ebf16"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msve-i8mm() #[[sve_i8mm:[0-9]+]] {
__attribute__((target_version("sve-i8mm"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msve2() #[[sve2:[0-9]+]] {
__attribute__((target_version("sve2"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msve2-aes() #[[sve2_aes_sve2_pmull128:[0-9]+]] {
__attribute__((target_version("sve2-aes"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msve2-bitperm() #[[sve2_bitperm:[0-9]+]] {
__attribute__((target_version("sve2-bitperm"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msve2-pmull128() #[[sve2_aes_sve2_pmull128:[0-9]+]] {
__attribute__((target_version("sve2-pmull128"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msve2-sha3() #[[sve2_sha3:[0-9]+]] {
__attribute__((target_version("sve2-sha3"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Msve2-sm4() #[[sve2_sm4:[0-9]+]] {
__attribute__((target_version("sve2-sm4"))) int fmv(void) { return 0; }

// CHECK: define dso_local i32 @fmv._Mwfxt() #[[wfxt:[0-9]+]] {
__attribute__((target_version("wfxt"))) int fmv(void) { return 0; }

// CHECK-NOT: define dso_local i32 @fmv._M{{.*}}
__attribute__((target_version("non_existent_extension"))) int fmv(void);

__attribute__((target_version("default"))) int fmv(void);

int caller() {
  return fmv();
}

// CHECK: attributes #[[ATTR0:[0-9]+]] = { {{.*}} "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[bf16_ebf16:[0-9]+]] = { {{.*}} "target-features"="+bf16,+fp-armv8,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[bti:[0-9]+]] = { {{.*}} "target-features"="+bti,+fp-armv8,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[crc:[0-9]+]] = { {{.*}} "target-features"="+crc,+fp-armv8,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[dit:[0-9]+]] = { {{.*}} "target-features"="+dit,+fp-armv8,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[dotprod:[0-9]+]] = { {{.*}} "target-features"="+dotprod,+fp-armv8,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[dpb:[0-9]+]] = { {{.*}} "target-features"="+ccpp,+fp-armv8,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[dpb2:[0-9]+]] = { {{.*}} "target-features"="+ccdp,+ccpp,+fp-armv8,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[f32mm:[0-9]+]] = { {{.*}} "target-features"="+f32mm,+fp-armv8,+fullfp16,+neon,+outline-atomics,+sve,+v8a"
// CHECK: attributes #[[f64mm:[0-9]+]] = { {{.*}} "target-features"="+f64mm,+fp-armv8,+fullfp16,+neon,+outline-atomics,+sve,+v8a"
// CHECK: attributes #[[fcma:[0-9]+]] = { {{.*}} "target-features"="+complxnum,+fp-armv8,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[flagm:[0-9]+]] = { {{.*}} "target-features"="+flagm,+fp-armv8,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[flagm2:[0-9]+]] = { {{.*}} "target-features"="+altnzcv,+flagm,+fp-armv8,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[fp16:[0-9]+]] = { {{.*}} "target-features"="+fp-armv8,+fullfp16,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[fp16fml:[0-9]+]] = { {{.*}} "target-features"="+fp-armv8,+fp16fml,+fullfp16,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[frintts:[0-9]+]] = { {{.*}} "target-features"="+fp-armv8,+fptoint,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[i8mm:[0-9]+]] = { {{.*}} "target-features"="+fp-armv8,+i8mm,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[jscvt:[0-9]+]] = { {{.*}} "target-features"="+fp-armv8,+jsconv,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[ls64_accdata:[0-9]+]] = { {{.*}} "target-features"="+fp-armv8,+ls64,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[lse:[0-9]+]] = { {{.*}} "target-features"="+fp-armv8,+lse,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[memtag2:[0-9]+]] = { {{.*}} "target-features"="+fp-armv8,+mte,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[mops:[0-9]+]] = { {{.*}} "target-features"="+fp-armv8,+mops,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[pmull:[0-9]+]] = { {{.*}} "target-features"="+aes,+fp-armv8,+neon,+outline-atomics,+v8a"
// CHECK: attributes #[[predres:[0-9]+]] = { {{.*}} "target-features"="+fp-armv8,+neon,+outline-atomics,+predres,+v8a"
// CHECK: attributes #[[rcpc:[0-9]+]] = { {{.*}} "target-features"="+fp-armv8,+neon,+outline-atomics,+rcpc,+v8a"
// CHECK: attributes #[[rcpc3:[0-9]+]] = { {{.*}} "target-features"="+fp-armv8,+neon,+outline-atomics,+rcpc,+rcpc3,+v8a"
// CHECK: attributes #[[rdm:[0-9]+]] = { {{.*}} "target-features"="+fp-armv8,+neon,+outline-atomics,+rdm,+v8a"
// CHECK: attributes #[[rng:[0-9]+]] = { {{.*}} "target-features"="+fp-armv8,+neon,+outline-atomics,+rand,+v8a"
// CHECK: attributes #[[sb:[0-9]+]] = { {{.*}} "target-features"="+fp-armv8,+neon,+outline-atomics,+sb,+v8a"
// CHECK: attributes #[[sha2:[0-9]+]] = { {{.*}} "target-features"="+fp-armv8,+neon,+outline-atomics,+sha2,+v8a"
// CHECK: attributes #[[sha3:[0-9]+]] = { {{.*}} "target-features"="+fp-armv8,+neon,+outline-atomics,+sha2,+sha3,+v8a"
// CHECK: attributes #[[sm4:[0-9]+]] = { {{.*}} "target-features"="+fp-armv8,+neon,+outline-atomics,+sm4,+v8a"
// CHECK: attributes #[[sme:[0-9]+]] = { {{.*}} "target-features"="+bf16,+fp-armv8,+neon,+outline-atomics,+sme,+v8a"
// CHECK: attributes #[[sme_f64f64:[0-9]+]] = { {{.*}} "target-features"="+bf16,+fp-armv8,+neon,+outline-atomics,+sme,+sme-f64f64,+v8a"
// CHECK: attributes #[[sme_i16i64:[0-9]+]] = { {{.*}} "target-features"="+bf16,+fp-armv8,+neon,+outline-atomics,+sme,+sme-i16i64,+v8a"
// CHECK: attributes #[[sme2:[0-9]+]] = { {{.*}} "target-features"="+bf16,+fp-armv8,+neon,+outline-atomics,+sme,+sme2,+v8a"
// CHECK: attributes #[[ssbs2:[0-9]+]] = { {{.*}} "target-features"="+fp-armv8,+neon,+outline-atomics,+ssbs,+v8a"
// CHECK: attributes #[[sve:[0-9]+]] = { {{.*}} "target-features"="+fp-armv8,+fullfp16,+neon,+outline-atomics,+sve,+v8a"
// CHECK: attributes #[[sve_bf16_ebf16:[0-9]+]] = { {{.*}} "target-features"="+bf16,+fp-armv8,+fullfp16,+neon,+outline-atomics,+sve,+v8a"
// CHECK: attributes #[[sve_i8mm:[0-9]+]] = { {{.*}} "target-features"="+fp-armv8,+fullfp16,+i8mm,+neon,+outline-atomics,+sve,+v8a"
// CHECK: attributes #[[sve2:[0-9]+]] = { {{.*}} "target-features"="+fp-armv8,+fullfp16,+neon,+outline-atomics,+sve,+sve2,+v8a"
// CHECK: attributes #[[sve2_aes_sve2_pmull128:[0-9]+]] = { {{.*}} "target-features"="+fp-armv8,+fullfp16,+neon,+outline-atomics,+sve,+sve2,+sve2-aes,+v8a"
// CHECK: attributes #[[sve2_bitperm:[0-9]+]] = { {{.*}} "target-features"="+fp-armv8,+fullfp16,+neon,+outline-atomics,+sve,+sve2,+sve2-bitperm,+v8a"
// CHECK: attributes #[[sve2_sha3:[0-9]+]] = { {{.*}} "target-features"="+fp-armv8,+fullfp16,+neon,+outline-atomics,+sve,+sve2,+sve2-sha3,+v8a"
// CHECK: attributes #[[sve2_sm4:[0-9]+]] = { {{.*}} "target-features"="+fp-armv8,+fullfp16,+neon,+outline-atomics,+sve,+sve2,+sve2-sm4,+v8a"
// CHECK: attributes #[[wfxt:[0-9]+]] = { {{.*}} "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,+wfxt"
