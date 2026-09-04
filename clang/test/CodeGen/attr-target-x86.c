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
// CHECK: define {{.*}}@f_avx10_1{{.*}} [[f_avx10_1:#[0-9]+]]
// CHECK: define {{.*}}@f_prefer_256_bit({{.*}} [[f_prefer_256_bit:#[0-9]+]]
// CHECK: define {{.*}}@f_no_prefer_256_bit({{.*}} [[f_no_prefer_256_bit:#[0-9]+]]
// CHECK: define {{.*}}@f_apxf({{.*}} [[f_apxf:#[0-9]+]]
// CHECK: define {{.*}}@f_no_apxf({{.*}} [[f_no_apxf:#[0-9]+]]
// CHECK: define {{.*}}@f_egpr({{.*}} [[f_egpr:#[0-9]+]]
// CHECK: define {{.*}}@f_ndd({{.*}} [[f_ndd:#[0-9]+]]
// CHECK: define {{.*}}@f_ccmp({{.*}} [[f_ccmp:#[0-9]+]]
// CHECK: define {{.*}}@f_nf({{.*}} [[f_nf:#[0-9]+]]
// CHECK: define {{.*}}@f_cf({{.*}} [[f_cf:#[0-9]+]]
// CHECK: define {{.*}}@f_zu({{.*}} [[f_zu:#[0-9]+]]
// CHECK: define {{.*}}@f_push2pop2({{.*}} [[f_push2pop2:#[0-9]+]]
// CHECK: define {{.*}}@f_ppx({{.*}} [[f_ppx:#[0-9]+]]
// CHECK: define {{.*}}@f_jmpabs({{.*}} [[f_jmpabs:#[0-9]+]]
// CHECK: define {{.*}}@f_adx({{.*}} [[f_adx:#[0-9]+]]
// CHECK: define {{.*}}@f_aes({{.*}} [[f_aes:#[0-9]+]]
// CHECK: define {{.*}}@f_amx_avx512({{.*}} [[f_amx_avx512:#[0-9]+]]
// CHECK: define {{.*}}@f_amx_bf16({{.*}} [[f_amx_bf16:#[0-9]+]]
// CHECK: define {{.*}}@f_amx_complex({{.*}} [[f_amx_complex:#[0-9]+]]
// CHECK: define {{.*}}@f_amx_fp16({{.*}} [[f_amx_fp16:#[0-9]+]]
// CHECK: define {{.*}}@f_amx_fp8({{.*}} [[f_amx_fp8:#[0-9]+]]
// CHECK: define {{.*}}@f_amx_int8({{.*}} [[f_amx_int8:#[0-9]+]]
// CHECK: define {{.*}}@f_amx_movrs({{.*}} [[f_amx_movrs:#[0-9]+]]
// CHECK: define {{.*}}@f_amx_tile({{.*}} [[f_amx_tile:#[0-9]+]]
// CHECK: define {{.*}}@f_avx10_2({{.*}} [[f_avx10_2:#[0-9]+]]
// CHECK: define {{.*}}@f_avx2({{.*}} [[f_avx2:#[0-9]+]]
// CHECK: define {{.*}}@f_avx512f({{.*}} [[f_avx512f:#[0-9]+]]
// CHECK: define {{.*}}@f_avx512cd({{.*}} [[f_avx512cd:#[0-9]+]]
// CHECK: define {{.*}}@f_avx512vpopcntdq({{.*}} [[f_avx512vpopcntdq:#[0-9]+]]
// CHECK: define {{.*}}@f_avx512vnni({{.*}} [[f_avx512vnni:#[0-9]+]]
// CHECK: define {{.*}}@f_avx512bf16({{.*}} [[f_avx512bf16:#[0-9]+]]
// CHECK: define {{.*}}@f_avx512fp16({{.*}} [[f_avx512fp16:#[0-9]+]]
// CHECK: define {{.*}}@f_avx512dq({{.*}} [[f_avx512dq:#[0-9]+]]
// CHECK: define {{.*}}@f_avx512bitalg({{.*}} [[f_avx512bitalg:#[0-9]+]]
// CHECK: define {{.*}}@f_avx512bmm({{.*}} [[f_avx512bmm:#[0-9]+]]
// CHECK: define {{.*}}@f_avx512bw({{.*}} [[f_avx512bw:#[0-9]+]]
// CHECK: define {{.*}}@f_avx512vl({{.*}} [[f_avx512vl:#[0-9]+]]
// CHECK: define {{.*}}@f_avx512vbmi({{.*}} [[f_avx512vbmi:#[0-9]+]]
// CHECK: define {{.*}}@f_avx512vbmi2({{.*}} [[f_avx512vbmi2:#[0-9]+]]
// CHECK: define {{.*}}@f_avx512ifma({{.*}} [[f_avx512ifma:#[0-9]+]]
// CHECK: define {{.*}}@f_avx512vp2intersect({{.*}} [[f_avx512vp2intersect:#[0-9]+]]
// CHECK: define {{.*}}@f_avxifma({{.*}} [[f_avxifma:#[0-9]+]]
// CHECK: define {{.*}}@f_avxneconvert({{.*}} [[f_avxneconvert:#[0-9]+]]
// CHECK: define {{.*}}@f_avxvnni({{.*}} [[f_avxvnni:#[0-9]+]]
// CHECK: define {{.*}}@f_avxvnniint16({{.*}} [[f_avxvnniint16:#[0-9]+]]
// CHECK: define {{.*}}@f_avxvnniint8({{.*}} [[f_avxvnniint8:#[0-9]+]]
// CHECK: define {{.*}}@f_bmi({{.*}} [[f_bmi:#[0-9]+]]
// CHECK: define {{.*}}@f_bmi2({{.*}} [[f_bmi2:#[0-9]+]]
// CHECK: define {{.*}}@f_cldemote({{.*}} [[f_cldemote:#[0-9]+]]
// CHECK: define {{.*}}@f_clflushopt({{.*}} [[f_clflushopt:#[0-9]+]]
// CHECK: define {{.*}}@f_clwb({{.*}} [[f_clwb:#[0-9]+]]
// CHECK: define {{.*}}@f_clzero({{.*}} [[f_clzero:#[0-9]+]]
// CHECK: define {{.*}}@f_cmpccxadd({{.*}} [[f_cmpccxadd:#[0-9]+]]
// CHECK: define {{.*}}@f_crc32({{.*}} [[f_crc32:#[0-9]+]]
// CHECK: define {{.*}}@f_cx16({{.*}} [[f_cx16:#[0-9]+]]
// CHECK: define {{.*}}@f_enqcmd({{.*}} [[f_enqcmd:#[0-9]+]]
// CHECK: define {{.*}}@f_f16c({{.*}} [[f_f16c:#[0-9]+]]
// CHECK: define {{.*}}@f_fma({{.*}} [[f_fma:#[0-9]+]]
// CHECK: define {{.*}}@f_fma4({{.*}} [[f_fma4:#[0-9]+]]
// CHECK: define {{.*}}@f_fsgsbase({{.*}} [[f_fsgsbase:#[0-9]+]]
// CHECK: define {{.*}}@f_fxsr({{.*}} [[f_fxsr:#[0-9]+]]
// CHECK: define {{.*}}@f_general_regs_only({{.*}} [[f_general_regs_only:#[0-9]+]]
// CHECK: define {{.*}}@f_gfni({{.*}} [[f_gfni:#[0-9]+]]
// CHECK: define {{.*}}@f_hreset({{.*}} [[f_hreset:#[0-9]+]]
// CHECK: define {{.*}}@f_invpcid({{.*}} [[f_invpcid:#[0-9]+]]
// CHECK: define {{.*}}@f_kl({{.*}} [[f_kl:#[0-9]+]]
// CHECK: define {{.*}}@f_widekl({{.*}} [[f_widekl:#[0-9]+]]
// CHECK: define {{.*}}@f_lwp({{.*}} [[f_lwp:#[0-9]+]]
// CHECK: define {{.*}}@f_lzcnt({{.*}} [[f_lzcnt:#[0-9]+]]
// CHECK: define {{.*}}@f_movbe({{.*}} [[f_movbe:#[0-9]+]]
// CHECK: define {{.*}}@f_movrs({{.*}} [[f_movrs:#[0-9]+]]
// CHECK: define {{.*}}@f_movdiri({{.*}} [[f_movdiri:#[0-9]+]]
// CHECK: define {{.*}}@f_movdir64b({{.*}} [[f_movdir64b:#[0-9]+]]
// CHECK: define {{.*}}@f_mwaitx({{.*}} [[f_mwaitx:#[0-9]+]]
// CHECK: define {{.*}}@f_pclmul({{.*}} [[f_pclmul:#[0-9]+]]
// CHECK: define {{.*}}@f_pconfig({{.*}} [[f_pconfig:#[0-9]+]]
// CHECK: define {{.*}}@f_pku({{.*}} [[f_pku:#[0-9]+]]
// CHECK: define {{.*}}@f_popcnt({{.*}} [[f_popcnt:#[0-9]+]]
// CHECK: define {{.*}}@f_prefetchi({{.*}} [[f_prefetchi:#[0-9]+]]
// CHECK: define {{.*}}@f_prfchw({{.*}} [[f_prfchw:#[0-9]+]]
// CHECK: define {{.*}}@f_ptwrite({{.*}} [[f_ptwrite:#[0-9]+]]
// CHECK: define {{.*}}@f_raoint({{.*}} [[f_raoint:#[0-9]+]]
// CHECK: define {{.*}}@f_rdpid({{.*}} [[f_rdpid:#[0-9]+]]
// CHECK: define {{.*}}@f_rdpru({{.*}} [[f_rdpru:#[0-9]+]]
// CHECK: define {{.*}}@f_rdrnd({{.*}} [[f_rdrnd:#[0-9]+]]
// CHECK: define {{.*}}@f_rdseed({{.*}} [[f_rdseed:#[0-9]+]]
// CHECK: define {{.*}}@f_rtm({{.*}} [[f_rtm:#[0-9]+]]
// CHECK: define {{.*}}@f_sahf({{.*}} [[f_sahf:#[0-9]+]]
// CHECK: define {{.*}}@f_serialize({{.*}} [[f_serialize:#[0-9]+]]
// CHECK: define {{.*}}@f_sgx({{.*}} [[f_sgx:#[0-9]+]]
// CHECK: define {{.*}}@f_sha({{.*}} [[f_sha:#[0-9]+]]
// CHECK: define {{.*}}@f_sha512({{.*}} [[f_sha512:#[0-9]+]]
// CHECK: define {{.*}}@f_shstk({{.*}} [[f_shstk:#[0-9]+]]
// CHECK: define {{.*}}@f_sm3({{.*}} [[f_sm3:#[0-9]+]]
// CHECK: define {{.*}}@f_sm4({{.*}} [[f_sm4:#[0-9]+]]
// CHECK: define {{.*}}@f_sse({{.*}} [[f_sse:#[0-9]+]]
// CHECK: define {{.*}}@f_sse2({{.*}} [[f_sse2:#[0-9]+]]
// CHECK: define {{.*}}@f_sse3({{.*}} [[f_sse3:#[0-9]+]]
// CHECK: define {{.*}}@f_ssse3({{.*}} [[f_ssse3:#[0-9]+]]
// CHECK: define {{.*}}@f_sse4_1({{.*}} [[f_sse4_1:#[0-9]+]]
// CHECK: define {{.*}}@f_sse4a({{.*}} [[f_sse4a:#[0-9]+]]
// CHECK: define {{.*}}@f_tbm({{.*}} [[f_tbm:#[0-9]+]]
// CHECK: define {{.*}}@f_tsxldtrk({{.*}} [[f_tsxldtrk:#[0-9]+]]
// CHECK: define {{.*}}@f_uintr({{.*}} [[f_uintr:#[0-9]+]]
// CHECK: define {{.*}}@f_usermsr({{.*}} [[f_usermsr:#[0-9]+]]
// CHECK: define {{.*}}@f_vaes({{.*}} [[f_vaes:#[0-9]+]]
// CHECK: define {{.*}}@f_vpclmulqdq({{.*}} [[f_vpclmulqdq:#[0-9]+]]
// CHECK: define {{.*}}@f_wbnoinvd({{.*}} [[f_wbnoinvd:#[0-9]+]]
// CHECK: define {{.*}}@f_waitpkg({{.*}} [[f_waitpkg:#[0-9]+]]
// CHECK: define {{.*}}@f_x87({{.*}} [[f_default]]
// CHECK: define {{.*}}@f_xop({{.*}} [[f_xop:#[0-9]+]]
// CHECK: define {{.*}}@f_xsave({{.*}} [[f_xsave:#[0-9]+]]
// CHECK: define {{.*}}@f_xsavec({{.*}} [[f_xsavec:#[0-9]+]]
// CHECK: define {{.*}}@f_xsaves({{.*}} [[f_xsaves:#[0-9]+]]
// CHECK: define {{.*}}@f_xsaveopt({{.*}} [[f_xsaveopt:#[0-9]+]]

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
// CHECK: [[f_no_sse2]] = {{.*}}"target-cpu"="i686" "target-features"="+cmov,+cx8,+x87,-aes,-amx-avx512,-avx,-avx10.1,-avx10.2,-avx2,-avx512bf16,-avx512bitalg,-avx512bmm,-avx512bw,-avx512cd,-avx512dq,-avx512f,-avx512fp16,-avx512ifma,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vp2intersect,-avx512vpopcntdq,-avxifma,-avxneconvert,-avxvnni,-avxvnniint16,-avxvnniint8,-f16c,-fma,-fma4,-gfni,-kl,-pclmul,-sha,-sha512,-sm3,-sm4,-sse2,-sse3,-sse4.1,-sse4.2,-sse4a,-ssse3,-vaes,-vpclmulqdq,-widekl,-xop" "tune-cpu"="i686"
__attribute__((target("no-sse2")))
void f_no_sse2(void) {}

// CHECK: [[f_sse4]] = {{.*}}"target-cpu"="i686" "target-features"="+cmov,+crc32,+cx8,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87" "tune-cpu"="i686"
__attribute__((target("sse4")))
void f_sse4(void) {}

// CHECK: [[f_no_sse4]] = {{.*}}"target-cpu"="i686" "target-features"="+cmov,+cx8,+x87,-amx-avx512,-avx,-avx10.1,-avx10.2,-avx2,-avx512bf16,-avx512bitalg,-avx512bmm,-avx512bw,-avx512cd,-avx512dq,-avx512f,-avx512fp16,-avx512ifma,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vp2intersect,-avx512vpopcntdq,-avxifma,-avxneconvert,-avxvnni,-avxvnniint16,-avxvnniint8,-f16c,-fma,-fma4,-sha512,-sm3,-sm4,-sse4.1,-sse4.2,-vaes,-vpclmulqdq,-xop" "tune-cpu"="i686"
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

// CHECK: [[f_no_aes_ivybridge]] = {{.*}}"target-cpu"="ivybridge" "target-features"="+avx,+cmov,+crc32,+cx16,+cx8,+f16c,+fsgsbase,+fxsr,+mmx,+pclmul,+popcnt,+rdrnd,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-aes,-vaes"
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
// CHECK-SAME: "target-features"="+avx,+avx2,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512vl,+bmi,+bmi2,+cmov,+crc32,+cx16,+cx8,+f16c,+fma,+fxsr,+lzcnt,+mmx,+movbe,+popcnt,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("arch=x86-64-v4")))
void f_x86_64_v4(void) {}

// CHECK: [[f_avx10_1]] = {{.*}}"target-cpu"="i686" "target-features"="+avx,+avx10.1,+avx2,+avx512bf16,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512fp16,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+cmov,+crc32,+cx8,+f16c,+fma,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("avx10.1")))
void f_avx10_1(void) {}

// CHECK: [[f_prefer_256_bit]] = {{.*}}"target-features"="{{.*}}+prefer-256-bit
__attribute__((target("prefer-256-bit")))
void f_prefer_256_bit(void) {}

// CHECK: [[f_no_prefer_256_bit]] = {{.*}}"target-features"="{{.*}}-prefer-256-bit
__attribute__((target("no-prefer-256-bit")))
void f_no_prefer_256_bit(void) {}

// CHECK: [[f_apxf]] = {{.*}}"target-features"="{{.*}}+ccmp{{.*}}+egpr{{.*}}+jmpabs{{.*}}+ndd{{.*}}+nf{{.*}}+ppx{{.*}}+push2pop2{{.*}}+zu
__attribute__((target("apxf")))
void f_apxf(void) {}

// CHECK: [[f_no_apxf]] = {{.*}}"target-features"="{{.*}}-ccmp{{.*}}-egpr{{.*}}-jmpabs{{.*}}-ndd{{.*}}-nf{{.*}}-ppx{{.*}}-push2pop2{{.*}}-zu
__attribute__((target("no-apxf")))
void f_no_apxf(void) {}

// CHECK: [[f_egpr]] = {{.*}}"target-features"="{{.*}}+egpr
__attribute__((target("egpr")))
void f_egpr(void) {}

// CHECK: [[f_ndd]] = {{.*}}"target-features"="{{.*}}+ndd
__attribute__((target("ndd")))
void f_ndd(void) {}

// CHECK: [[f_ccmp]] = {{.*}}"target-features"="{{.*}}+ccmp
__attribute__((target("ccmp")))
void f_ccmp(void) {}

// CHECK: [[f_nf]] = {{.*}}"target-features"="{{.*}}+nf
__attribute__((target("nf")))
void f_nf(void) {}

// CHECK: [[f_cf]] = {{.*}}"target-features"="{{.*}}+cf
__attribute__((target("cf")))
void f_cf(void) {}

// CHECK: [[f_zu]] = {{.*}}"target-features"="{{.*}}+zu
__attribute__((target("zu")))
void f_zu(void) {}

// CHECK: [[f_push2pop2]] = {{.*}}"target-features"="{{.*}}+push2pop2
__attribute__((target("push2pop2")))
void f_push2pop2(void) {}

// CHECK: [[f_ppx]] = {{.*}}"target-features"="{{.*}}+ppx
__attribute__((target("ppx")))
void f_ppx(void) {}

// CHECK: [[f_jmpabs]] = {{.*}}"target-features"="{{.*}}+jmpabs
__attribute__((target("jmpabs")))
void f_jmpabs(void) {}

// CHECK: [[f_adx]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+adx,+cmov,+cx8,+x87"
__attribute__((target("adx")))
void f_adx(void) {}

// CHECK: [[f_aes]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+aes,+cmov,+cx8,+mmx,+sse,+sse2,+x87"
__attribute__((target("aes")))
void f_aes(void) {}

// CHECK: [[f_amx_avx512]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+amx-avx512,+amx-tile,+avx,+avx10.1,+avx10.2,+avx2,+avx512bf16,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512fp16,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+cmov,+crc32,+cx8,+f16c,+fma,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("amx-avx512")))
void f_amx_avx512(void) {}

// CHECK: [[f_amx_bf16]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+amx-bf16,+amx-tile,+cmov,+cx8,+x87"
__attribute__((target("amx-bf16")))
void f_amx_bf16(void) {}

// CHECK: [[f_amx_complex]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+amx-complex,+amx-tile,+cmov,+cx8,+x87"
__attribute__((target("amx-complex")))
void f_amx_complex(void) {}

// CHECK: [[f_amx_fp16]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+amx-fp16,+amx-tile,+cmov,+cx8,+x87"
__attribute__((target("amx-fp16")))
void f_amx_fp16(void) {}

// CHECK: [[f_amx_fp8]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+amx-fp8,+amx-tile,+cmov,+cx8,+x87"
__attribute__((target("amx-fp8")))
void f_amx_fp8(void) {}

// CHECK: [[f_amx_int8]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+amx-int8,+amx-tile,+cmov,+cx8,+x87"
__attribute__((target("amx-int8")))
void f_amx_int8(void) {}

// CHECK: [[f_amx_movrs]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+amx-movrs,+amx-tile,+cmov,+cx8,+x87"
__attribute__((target("amx-movrs")))
void f_amx_movrs(void) {}

// CHECK: [[f_amx_tile]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+amx-tile,+cmov,+cx8,+x87"
__attribute__((target("amx-tile")))
void f_amx_tile(void) {}

// CHECK: [[f_avx10_2]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+avx,+avx10.1,+avx10.2,+avx2,+avx512bf16,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512fp16,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+cmov,+crc32,+cx8,+f16c,+fma,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("avx10.2")))
void f_avx10_2(void) {}

// CHECK: [[f_avx2]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+avx,+avx2,+cmov,+crc32,+cx8,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("avx2")))
void f_avx2(void) {}

// CHECK: [[f_avx512f]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+avx,+avx2,+avx512f,+cmov,+crc32,+cx8,+f16c,+fma,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("avx512f")))
void f_avx512f(void) {}

// CHECK: [[f_avx512cd]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+avx,+avx2,+avx512cd,+avx512f,+cmov,+crc32,+cx8,+f16c,+fma,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("avx512cd")))
void f_avx512cd(void) {}

// CHECK: [[f_avx512vpopcntdq]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+avx,+avx2,+avx512f,+avx512vpopcntdq,+cmov,+crc32,+cx8,+f16c,+fma,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("avx512vpopcntdq")))
void f_avx512vpopcntdq(void) {}

// CHECK: [[f_avx512vnni]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+avx,+avx2,+avx512f,+avx512vnni,+cmov,+crc32,+cx8,+f16c,+fma,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("avx512vnni")))
void f_avx512vnni(void) {}

// CHECK: [[f_avx512bf16]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+avx,+avx2,+avx512bf16,+avx512bw,+avx512f,+cmov,+crc32,+cx8,+f16c,+fma,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("avx512bf16")))
void f_avx512bf16(void) {}

// CHECK: [[f_avx512fp16]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+avx,+avx2,+avx512bw,+avx512f,+avx512fp16,+cmov,+crc32,+cx8,+f16c,+fma,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("avx512fp16")))
void f_avx512fp16(void) {}

// CHECK: [[f_avx512dq]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+avx,+avx2,+avx512dq,+avx512f,+cmov,+crc32,+cx8,+f16c,+fma,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("avx512dq")))
void f_avx512dq(void) {}

// CHECK: [[f_avx512bitalg]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+avx,+avx2,+avx512bitalg,+avx512bw,+avx512f,+cmov,+crc32,+cx8,+f16c,+fma,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("avx512bitalg")))
void f_avx512bitalg(void) {}

// CHECK: [[f_avx512bmm]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+avx,+avx2,+avx512bmm,+avx512bw,+avx512f,+cmov,+crc32,+cx8,+f16c,+fma,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("avx512bmm")))
void f_avx512bmm(void) {}

// CHECK: [[f_avx512bw]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+avx,+avx2,+avx512bw,+avx512f,+cmov,+crc32,+cx8,+f16c,+fma,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("avx512bw")))
void f_avx512bw(void) {}

// CHECK: [[f_avx512vl]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+avx,+avx2,+avx512f,+avx512vl,+cmov,+crc32,+cx8,+f16c,+fma,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("avx512vl")))
void f_avx512vl(void) {}

// CHECK: [[f_avx512vbmi]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+avx,+avx2,+avx512bw,+avx512f,+avx512vbmi,+cmov,+crc32,+cx8,+f16c,+fma,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("avx512vbmi")))
void f_avx512vbmi(void) {}

// CHECK: [[f_avx512vbmi2]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+avx,+avx2,+avx512bw,+avx512f,+avx512vbmi2,+cmov,+crc32,+cx8,+f16c,+fma,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("avx512vbmi2")))
void f_avx512vbmi2(void) {}

// CHECK: [[f_avx512ifma]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+avx,+avx2,+avx512f,+avx512ifma,+cmov,+crc32,+cx8,+f16c,+fma,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("avx512ifma")))
void f_avx512ifma(void) {}

// CHECK: [[f_avx512vp2intersect]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+avx,+avx2,+avx512f,+avx512vp2intersect,+cmov,+crc32,+cx8,+f16c,+fma,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("avx512vp2intersect")))
void f_avx512vp2intersect(void) {}

// CHECK: [[f_avxifma]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+avx,+avx2,+avxifma,+cmov,+crc32,+cx8,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("avxifma")))
void f_avxifma(void) {}

// CHECK: [[f_avxneconvert]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+avx,+avx2,+avxneconvert,+cmov,+crc32,+cx8,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("avxneconvert")))
void f_avxneconvert(void) {}

// CHECK: [[f_avxvnni]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+avx,+avx2,+avxvnni,+cmov,+crc32,+cx8,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("avxvnni")))
void f_avxvnni(void) {}

// CHECK: [[f_avxvnniint16]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+avx,+avx2,+avxvnniint16,+cmov,+crc32,+cx8,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("avxvnniint16")))
void f_avxvnniint16(void) {}

// CHECK: [[f_avxvnniint8]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+avx,+avx2,+avxvnniint8,+cmov,+crc32,+cx8,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("avxvnniint8")))
void f_avxvnniint8(void) {}

// CHECK: [[f_bmi]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+bmi,+cmov,+cx8,+x87"
__attribute__((target("bmi")))
void f_bmi(void) {}

// CHECK: [[f_bmi2]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+bmi2,+cmov,+cx8,+x87"
__attribute__((target("bmi2")))
void f_bmi2(void) {}

// CHECK: [[f_cldemote]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cldemote,+cmov,+cx8,+x87"
__attribute__((target("cldemote")))
void f_cldemote(void) {}

// CHECK: [[f_clflushopt]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+clflushopt,+cmov,+cx8,+x87"
__attribute__((target("clflushopt")))
void f_clflushopt(void) {}

// CHECK: [[f_clwb]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+clwb,+cmov,+cx8,+x87"
__attribute__((target("clwb")))
void f_clwb(void) {}

// CHECK: [[f_clzero]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+clzero,+cmov,+cx8,+x87"
__attribute__((target("clzero")))
void f_clzero(void) {}

// CHECK: [[f_cmpccxadd]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cmpccxadd,+cx8,+x87"
__attribute__((target("cmpccxadd")))
void f_cmpccxadd(void) {}

// CHECK: [[f_crc32]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+crc32,+cx8,+x87"
__attribute__((target("crc32")))
void f_crc32(void) {}

// CHECK: [[f_cx16]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx16,+cx8,+x87"
__attribute__((target("cx16")))
void f_cx16(void) {}

// CHECK: [[f_enqcmd]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+enqcmd,+x87"
__attribute__((target("enqcmd")))
void f_enqcmd(void) {}

// CHECK: [[f_f16c]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+avx,+cmov,+crc32,+cx8,+f16c,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("f16c")))
void f_f16c(void) {}

// CHECK: [[f_fma]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+avx,+cmov,+crc32,+cx8,+fma,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("fma")))
void f_fma(void) {}

// CHECK: [[f_fma4]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+avx,+cmov,+crc32,+cx8,+fma4,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+sse4a,+ssse3,+x87,+xsave"
__attribute__((target("fma4")))
void f_fma4(void) {}

// CHECK: [[f_fsgsbase]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+fsgsbase,+x87"
__attribute__((target("fsgsbase")))
void f_fsgsbase(void) {}

// CHECK: [[f_fxsr]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+fxsr,+x87"
__attribute__((target("fxsr")))
void f_fxsr(void) {}

// CHECK: [[f_general_regs_only]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,-aes,-amx-avx512,-avx,-avx10.1,-avx10.2,-avx2,-avx512bf16,-avx512bitalg,-avx512bmm,-avx512bw,-avx512cd,-avx512dq,-avx512f,-avx512fp16,-avx512ifma,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vp2intersect,-avx512vpopcntdq,-avxifma,-avxneconvert,-avxvnni,-avxvnniint16,-avxvnniint8,-f16c,-fma,-fma4,-gfni,-kl,-mmx,-pclmul,-sha,-sha512,-sm3,-sm4,-sse,-sse2,-sse3,-sse4.1,-sse4.2,-sse4a,-ssse3,-vaes,-vpclmulqdq,-widekl,-x87,-xop"
__attribute__((target("general-regs-only")))
void f_general_regs_only(void) {}

// CHECK: [[f_gfni]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+gfni,+mmx,+sse,+sse2,+x87"
__attribute__((target("gfni")))
void f_gfni(void) {}

// CHECK: [[f_hreset]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+hreset,+x87"
__attribute__((target("hreset")))
void f_hreset(void) {}

// CHECK: [[f_invpcid]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+invpcid,+x87"
__attribute__((target("invpcid")))
void f_invpcid(void) {}

// CHECK: [[f_kl]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+kl,+mmx,+sse,+sse2,+x87"
__attribute__((target("kl")))
void f_kl(void) {}

// CHECK: [[f_widekl]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+kl,+mmx,+sse,+sse2,+widekl,+x87"
__attribute__((target("widekl")))
void f_widekl(void) {}

// CHECK: [[f_lwp]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+lwp,+x87"
__attribute__((target("lwp")))
void f_lwp(void) {}

// CHECK: [[f_lzcnt]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+lzcnt,+x87"
__attribute__((target("lzcnt")))
void f_lzcnt(void) {}

// CHECK: [[f_movbe]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+movbe,+x87"
__attribute__((target("movbe")))
void f_movbe(void) {}

// CHECK: [[f_movrs]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+movrs,+x87"
__attribute__((target("movrs")))
void f_movrs(void) {}

// CHECK: [[f_movdiri]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+movdiri,+x87"
__attribute__((target("movdiri")))
void f_movdiri(void) {}

// CHECK: [[f_movdir64b]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+movdir64b,+x87"
__attribute__((target("movdir64b")))
void f_movdir64b(void) {}

// CHECK: [[f_mwaitx]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+mwaitx,+x87"
__attribute__((target("mwaitx")))
void f_mwaitx(void) {}

// CHECK: [[f_pclmul]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+mmx,+pclmul,+sse,+sse2,+x87"
__attribute__((target("pclmul")))
void f_pclmul(void) {}

// CHECK: [[f_pconfig]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+pconfig,+x87"
__attribute__((target("pconfig")))
void f_pconfig(void) {}

// CHECK: [[f_pku]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+pku,+x87"
__attribute__((target("pku")))
void f_pku(void) {}

// CHECK: [[f_popcnt]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+popcnt,+x87"
__attribute__((target("popcnt")))
void f_popcnt(void) {}

// CHECK: [[f_prefetchi]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+prefetchi,+x87"
__attribute__((target("prefetchi")))
void f_prefetchi(void) {}

// CHECK: [[f_prfchw]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+prfchw,+x87"
__attribute__((target("prfchw")))
void f_prfchw(void) {}

// CHECK: [[f_ptwrite]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+ptwrite,+x87"
__attribute__((target("ptwrite")))
void f_ptwrite(void) {}

// CHECK: [[f_raoint]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+raoint,+x87"
__attribute__((target("raoint")))
void f_raoint(void) {}

// CHECK: [[f_rdpid]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+rdpid,+x87"
__attribute__((target("rdpid")))
void f_rdpid(void) {}

// CHECK: [[f_rdpru]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+rdpru,+x87"
__attribute__((target("rdpru")))
void f_rdpru(void) {}

// CHECK: [[f_rdrnd]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+rdrnd,+x87"
__attribute__((target("rdrnd")))
void f_rdrnd(void) {}

// CHECK: [[f_rdseed]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+rdseed,+x87"
__attribute__((target("rdseed")))
void f_rdseed(void) {}

// CHECK: [[f_rtm]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+rtm,+x87"
__attribute__((target("rtm")))
void f_rtm(void) {}

// CHECK: [[f_sahf]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+sahf,+x87"
__attribute__((target("sahf")))
void f_sahf(void) {}

// CHECK: [[f_serialize]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+serialize,+x87"
__attribute__((target("serialize")))
void f_serialize(void) {}

// CHECK: [[f_sgx]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+sgx,+x87"
__attribute__((target("sgx")))
void f_sgx(void) {}

// CHECK: [[f_sha]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+mmx,+sha,+sse,+sse2,+x87"
__attribute__((target("sha")))
void f_sha(void) {}

// CHECK: [[f_sha512]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+avx,+avx2,+cmov,+crc32,+cx8,+mmx,+popcnt,+sha512,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("sha512")))
void f_sha512(void) {}

// CHECK: [[f_shstk]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+shstk,+x87"
__attribute__((target("shstk")))
void f_shstk(void) {}

// CHECK: [[f_sm3]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+avx,+cmov,+crc32,+cx8,+mmx,+popcnt,+sm3,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("sm3")))
void f_sm3(void) {}

// CHECK: [[f_sm4]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+avx,+avx2,+cmov,+crc32,+cx8,+mmx,+popcnt,+sm4,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
__attribute__((target("sm4")))
void f_sm4(void) {}

// CHECK: [[f_sse]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+mmx,+sse,+x87"
__attribute__((target("sse")))
void f_sse(void) {}

// CHECK: [[f_sse2]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+mmx,+sse,+sse2,+x87"
__attribute__((target("sse2")))
void f_sse2(void) {}

// CHECK: [[f_sse3]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+mmx,+sse,+sse2,+sse3,+x87"
__attribute__((target("sse3")))
void f_sse3(void) {}

// CHECK: [[f_ssse3]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+mmx,+sse,+sse2,+sse3,+ssse3,+x87"
__attribute__((target("ssse3")))
void f_ssse3(void) {}

// CHECK: [[f_sse4_1]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+mmx,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87"
__attribute__((target("sse4.1")))
void f_sse4_1(void) {}

// CHECK: [[f_sse4a]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+mmx,+sse,+sse2,+sse3,+sse4a,+x87"
__attribute__((target("sse4a")))
void f_sse4a(void) {}

// CHECK: [[f_tbm]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+tbm,+x87"
__attribute__((target("tbm")))
void f_tbm(void) {}

// CHECK: [[f_tsxldtrk]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+tsxldtrk,+x87"
__attribute__((target("tsxldtrk")))
void f_tsxldtrk(void) {}

// CHECK: [[f_uintr]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+uintr,+x87"
__attribute__((target("uintr")))
void f_uintr(void) {}

// CHECK: [[f_usermsr]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+usermsr,+x87"
__attribute__((target("usermsr")))
void f_usermsr(void) {}

// CHECK: [[f_vaes]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+aes,+avx,+avx2,+cmov,+crc32,+cx8,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+vaes,+x87,+xsave"
__attribute__((target("vaes")))
void f_vaes(void) {}

// CHECK: [[f_vpclmulqdq]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+avx,+cmov,+crc32,+cx8,+mmx,+pclmul,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+vpclmulqdq,+x87,+xsave"
__attribute__((target("vpclmulqdq")))
void f_vpclmulqdq(void) {}

// CHECK: [[f_wbnoinvd]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+wbnoinvd,+x87"
__attribute__((target("wbnoinvd")))
void f_wbnoinvd(void) {}

// CHECK: [[f_waitpkg]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+waitpkg,+x87"
__attribute__((target("waitpkg")))
void f_waitpkg(void) {}

// x87 is already enabled for i686, so this is checked above to have the
// same attributes as f_default.
__attribute__((target("x87")))
void f_x87(void) {}

// CHECK: [[f_xop]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+avx,+cmov,+crc32,+cx8,+fma4,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+sse4a,+ssse3,+x87,+xop,+xsave"
__attribute__((target("xop")))
void f_xop(void) {}

// CHECK: [[f_xsave]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+x87,+xsave"
__attribute__((target("xsave")))
void f_xsave(void) {}

// CHECK: [[f_xsavec]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+x87,+xsave,+xsavec"
__attribute__((target("xsavec")))
void f_xsavec(void) {}

// CHECK: [[f_xsaves]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+x87,+xsave,+xsaves"
__attribute__((target("xsaves")))
void f_xsaves(void) {}

// CHECK: [[f_xsaveopt]] = {{.*}}"target-cpu"="i686"
// CHECK-SAME: "target-features"="+cmov,+cx8,+x87,+xsave,+xsaveopt"
__attribute__((target("xsaveopt")))
void f_xsaveopt(void) {}
