// Test all of the AArch64 fmv-features metadata without any dependency expansion.
// It is used to propagate the attribute string information from C/C++ source to LLVM IR.

// RUN: %clang --target=aarch64-linux-gnu --rtlib=compiler-rt -emit-llvm -S -o - %s | FileCheck %s

// CHECK: define dso_local i32 @fmv._Maes() #[[aes:[0-9]+]] {
// CHECK: define dso_local i32 @fmv._Mbf16() #[[bf16:[0-9]+]] {
__attribute__((target_clones("aes", "bf16"))) int fmv(void) { return 0; }

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

// CHECK: define dso_local i32 @fmv._MaesMbf16MbtiMcrc() #[[unordered_features_with_duplicates:[0-9]+]] {
__attribute__((target_version("crc+bti+bti+bti+aes+aes+bf16"))) int fmv(void) { return 0; }

// CHECK-NOT: define dso_local i32 @fmv._M{{.*}}
__attribute__((target_version("non_existent_extension"))) int fmv(void);

// CHECK: define dso_local i32 @fmv.default() #[[default:[0-9]+]] {
__attribute__((target_version("default"))) int fmv(void);

int caller() {
  return fmv();
}

// CHECK: attributes #[[aes]] = {{.*}} "fmv-features"="aes"
// CHECK: attributes #[[bf16]] = {{.*}} "fmv-features"="bf16"
// CHECK: attributes #[[bti]] = {{.*}} "fmv-features"="bti"
// CHECK: attributes #[[crc]] = {{.*}} "fmv-features"="crc"
// CHECK: attributes #[[dit]] = {{.*}} "fmv-features"="dit"
// CHECK: attributes #[[dotprod]] = {{.*}} "fmv-features"="dotprod"
// CHECK: attributes #[[dpb]] = {{.*}} "fmv-features"="dpb"
// CHECK: attributes #[[dpb2]] = {{.*}} "fmv-features"="dpb2"
// CHECK: attributes #[[f32mm]] = {{.*}} "fmv-features"="f32mm"
// CHECK: attributes #[[f64mm]] = {{.*}} "fmv-features"="f64mm"
// CHECK: attributes #[[fcma]] = {{.*}} "fmv-features"="fcma"
// CHECK: attributes #[[flagm]] = {{.*}} "fmv-features"="flagm"
// CHECK: attributes #[[flagm2]] = {{.*}} "fmv-features"="flagm2"
// CHECK: attributes #[[fp]] = {{.*}} "fmv-features"="fp"
// CHECK: attributes #[[fp16]] = {{.*}} "fmv-features"="fp16"
// CHECK: attributes #[[fp16fml]] = {{.*}} "fmv-features"="fp16fml"
// CHECK: attributes #[[frintts]] = {{.*}} "fmv-features"="frintts"
// CHECK: attributes #[[i8mm]] = {{.*}} "fmv-features"="i8mm"
// CHECK: attributes #[[jscvt]] = {{.*}} "fmv-features"="jscvt"
// CHECK: attributes #[[lse]] = {{.*}} "fmv-features"="lse"
// CHECK: attributes #[[memtag]] = {{.*}} "fmv-features"="memtag"
// CHECK: attributes #[[mops]] = {{.*}} "fmv-features"="mops"
// CHECK: attributes #[[rcpc]] = {{.*}} "fmv-features"="rcpc"
// CHECK: attributes #[[rcpc2]] = {{.*}} "fmv-features"="rcpc2"
// CHECK: attributes #[[rcpc3]] = {{.*}} "fmv-features"="rcpc3"
// CHECK: attributes #[[rdm]] = {{.*}} "fmv-features"="rdm"
// CHECK: attributes #[[rng]] = {{.*}} "fmv-features"="rng"
// CHECK: attributes #[[sb]] = {{.*}} "fmv-features"="sb"
// CHECK: attributes #[[sha2]] = {{.*}} "fmv-features"="sha2"
// CHECK: attributes #[[sha3]] = {{.*}} "fmv-features"="sha3"
// CHECK: attributes #[[simd]] = {{.*}} "fmv-features"="simd"
// CHECK: attributes #[[sm4]] = {{.*}} "fmv-features"="sm4"
// CHECK: attributes #[[sme]] = {{.*}} "fmv-features"="sme"
// CHECK: attributes #[[sme_f64f64]] = {{.*}} "fmv-features"="sme-f64f64"
// CHECK: attributes #[[sme_i16i64]] = {{.*}} "fmv-features"="sme-i16i64"
// CHECK: attributes #[[sme2]] = {{.*}} "fmv-features"="sme2"
// CHECK: attributes #[[ssbs]] = {{.*}} "fmv-features"="ssbs"
// CHECK: attributes #[[sve]] = {{.*}} "fmv-features"="sve"
// CHECK: attributes #[[sve2]] = {{.*}} "fmv-features"="sve2"
// CHECK: attributes #[[sve2_aes]] = {{.*}} "fmv-features"="sve2-aes"
// CHECK: attributes #[[sve2_bitperm]] = {{.*}} "fmv-features"="sve2-bitperm"
// CHECK: attributes #[[sve2_sha3]] = {{.*}} "fmv-features"="sve2-sha3"
// CHECK: attributes #[[sve2_sm4]] = {{.*}} "fmv-features"="sve2-sm4"
// CHECK: attributes #[[wfxt]] = {{.*}} "fmv-features"="wfxt"
// CHECK: attributes #[[unordered_features_with_duplicates]] = {{.*}} "fmv-features"="aes,bf16,bti,crc"
// CHECK: attributes #[[default]] = {{.*}} "fmv-features"
