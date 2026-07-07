// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm < %s | FileCheck %s

// Test that __builtin_cpu_supports emits the correct field and bit for every
// feature listed in llvm/include/llvm/TargetParser/X86TargetParser.def. 
extern void a(const char *);

// CHECK: @__cpu_model = external dso_local global { i32, i32, i32, [1 x i32] }

#define TEST_CPU_SUPPORTS(NAME, STR)                                           \
  void test_##NAME(void) {                                                     \
    if (__builtin_cpu_supports(STR))                                           \
      a(STR);                                                                  \
  }

// CHECK-LABEL: define{{.*}} void @test_cmov(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 1
TEST_CPU_SUPPORTS(cmov, "cmov")

// CHECK-LABEL: define{{.*}} void @test_mmx(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 2
TEST_CPU_SUPPORTS(mmx, "mmx")

// CHECK-LABEL: define{{.*}} void @test_popcnt(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 4
TEST_CPU_SUPPORTS(popcnt, "popcnt")

// CHECK-LABEL: define{{.*}} void @test_sse(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 8
TEST_CPU_SUPPORTS(sse, "sse")

// CHECK-LABEL: define{{.*}} void @test_sse2(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 16
TEST_CPU_SUPPORTS(sse2, "sse2")

// CHECK-LABEL: define{{.*}} void @test_sse3(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 32
TEST_CPU_SUPPORTS(sse3, "sse3")

// CHECK-LABEL: define{{.*}} void @test_ssse3(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 64
TEST_CPU_SUPPORTS(ssse3, "ssse3")

// CHECK-LABEL: define{{.*}} void @test_sse4_1(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 128
TEST_CPU_SUPPORTS(sse4_1, "sse4.1")

// CHECK-LABEL: define{{.*}} void @test_sse4_2(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 256
TEST_CPU_SUPPORTS(sse4_2, "sse4.2")

// CHECK-LABEL: define{{.*}} void @test_avx(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 512
TEST_CPU_SUPPORTS(avx, "avx")

// CHECK-LABEL: define{{.*}} void @test_avx2(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 1024
TEST_CPU_SUPPORTS(avx2, "avx2")

// CHECK-LABEL: define{{.*}} void @test_sse4a(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 2048
TEST_CPU_SUPPORTS(sse4a, "sse4a")

// CHECK-LABEL: define{{.*}} void @test_fma4(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 4096
TEST_CPU_SUPPORTS(fma4, "fma4")

// CHECK-LABEL: define{{.*}} void @test_xop(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 8192
TEST_CPU_SUPPORTS(xop, "xop")

// CHECK-LABEL: define{{.*}} void @test_fma(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 16384
TEST_CPU_SUPPORTS(fma, "fma")

// CHECK-LABEL: define{{.*}} void @test_avx512f(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 32768
TEST_CPU_SUPPORTS(avx512f, "avx512f")

// CHECK-LABEL: define{{.*}} void @test_bmi(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 65536
TEST_CPU_SUPPORTS(bmi, "bmi")

// CHECK-LABEL: define{{.*}} void @test_bmi2(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 131072
TEST_CPU_SUPPORTS(bmi2, "bmi2")

// CHECK-LABEL: define{{.*}} void @test_aes(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 262144
TEST_CPU_SUPPORTS(aes, "aes")

// CHECK-LABEL: define{{.*}} void @test_pclmul(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 524288
TEST_CPU_SUPPORTS(pclmul, "pclmul")

// CHECK-LABEL: define{{.*}} void @test_avx512vl(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 1048576
TEST_CPU_SUPPORTS(avx512vl, "avx512vl")

// CHECK-LABEL: define{{.*}} void @test_avx512bw(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 2097152
TEST_CPU_SUPPORTS(avx512bw, "avx512bw")

// CHECK-LABEL: define{{.*}} void @test_avx512dq(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 4194304
TEST_CPU_SUPPORTS(avx512dq, "avx512dq")

// CHECK-LABEL: define{{.*}} void @test_avx512cd(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 8388608
TEST_CPU_SUPPORTS(avx512cd, "avx512cd")

// CHECK-LABEL: define{{.*}} void @test_avx512vbmi(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 67108864
TEST_CPU_SUPPORTS(avx512vbmi, "avx512vbmi")

// CHECK-LABEL: define{{.*}} void @test_avx512ifma(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 134217728
TEST_CPU_SUPPORTS(avx512ifma, "avx512ifma")

// CHECK-LABEL: define{{.*}} void @test_avx512vpopcntdq(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 1073741824
TEST_CPU_SUPPORTS(avx512vpopcntdq, "avx512vpopcntdq")

// CHECK-LABEL: define{{.*}} void @test_avx512vbmi2(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], -2147483648
TEST_CPU_SUPPORTS(avx512vbmi2, "avx512vbmi2")

// CHECK-LABEL: define{{.*}} void @test_gfni(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 1
TEST_CPU_SUPPORTS(gfni, "gfni")

// CHECK-LABEL: define{{.*}} void @test_vpclmulqdq(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 2
TEST_CPU_SUPPORTS(vpclmulqdq, "vpclmulqdq")

// CHECK-LABEL: define{{.*}} void @test_avx512vnni(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 4
TEST_CPU_SUPPORTS(avx512vnni, "avx512vnni")

// CHECK-LABEL: define{{.*}} void @test_avx512bitalg(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 8
TEST_CPU_SUPPORTS(avx512bitalg, "avx512bitalg")

// CHECK-LABEL: define{{.*}} void @test_avx512bf16(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 16
TEST_CPU_SUPPORTS(avx512bf16, "avx512bf16")

// CHECK-LABEL: define{{.*}} void @test_avx512vp2intersect(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 32
TEST_CPU_SUPPORTS(avx512vp2intersect, "avx512vp2intersect")

// CHECK-LABEL: define{{.*}} void @test_adx(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 256
TEST_CPU_SUPPORTS(adx, "adx")

// CHECK-LABEL: define{{.*}} void @test_cldemote(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 1024
TEST_CPU_SUPPORTS(cldemote, "cldemote")

// CHECK-LABEL: define{{.*}} void @test_clflushopt(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 2048
TEST_CPU_SUPPORTS(clflushopt, "clflushopt")

// CHECK-LABEL: define{{.*}} void @test_clwb(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 4096
TEST_CPU_SUPPORTS(clwb, "clwb")

// CHECK-LABEL: define{{.*}} void @test_clzero(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 8192
TEST_CPU_SUPPORTS(clzero, "clzero")

// CHECK-LABEL: define{{.*}} void @test_cx16(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 16384
TEST_CPU_SUPPORTS(cx16, "cx16")

// CHECK-LABEL: define{{.*}} void @test_enqcmd(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 65536
TEST_CPU_SUPPORTS(enqcmd, "enqcmd")

// CHECK-LABEL: define{{.*}} void @test_f16c(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 131072
TEST_CPU_SUPPORTS(f16c, "f16c")

// CHECK-LABEL: define{{.*}} void @test_fsgsbase(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 262144
TEST_CPU_SUPPORTS(fsgsbase, "fsgsbase")

// CHECK-LABEL: define{{.*}} void @test_sahf(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 4194304
TEST_CPU_SUPPORTS(sahf, "sahf")

// CHECK-LABEL: define{{.*}} void @test_64bit(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 8388608
TEST_CPU_SUPPORTS(64bit, "64bit")

// CHECK-LABEL: define{{.*}} void @test_lwp(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 16777216
TEST_CPU_SUPPORTS(lwp, "lwp")

// CHECK-LABEL: define{{.*}} void @test_lzcnt(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 33554432
TEST_CPU_SUPPORTS(lzcnt, "lzcnt")

// CHECK-LABEL: define{{.*}} void @test_movbe(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 67108864
TEST_CPU_SUPPORTS(movbe, "movbe")

// CHECK-LABEL: define{{.*}} void @test_movdir64b(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 134217728
TEST_CPU_SUPPORTS(movdir64b, "movdir64b")

// CHECK-LABEL: define{{.*}} void @test_movdiri(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 268435456
TEST_CPU_SUPPORTS(movdiri, "movdiri")

// CHECK-LABEL: define{{.*}} void @test_mwaitx(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 536870912
TEST_CPU_SUPPORTS(mwaitx, "mwaitx")

// CHECK-LABEL: define{{.*}} void @test_pconfig(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], -2147483648
TEST_CPU_SUPPORTS(pconfig, "pconfig")

// CHECK-LABEL: define{{.*}} void @test_pku(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 1
TEST_CPU_SUPPORTS(pku, "pku")

// CHECK-LABEL: define{{.*}} void @test_prfchw(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 4
TEST_CPU_SUPPORTS(prfchw, "prfchw")

// CHECK-LABEL: define{{.*}} void @test_ptwrite(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 8
TEST_CPU_SUPPORTS(ptwrite, "ptwrite")

// CHECK-LABEL: define{{.*}} void @test_rdpid(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 16
TEST_CPU_SUPPORTS(rdpid, "rdpid")

// CHECK-LABEL: define{{.*}} void @test_rdrnd(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 32
TEST_CPU_SUPPORTS(rdrnd, "rdrnd")

// CHECK-LABEL: define{{.*}} void @test_rdseed(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 64
TEST_CPU_SUPPORTS(rdseed, "rdseed")

// CHECK-LABEL: define{{.*}} void @test_rtm(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 128
TEST_CPU_SUPPORTS(rtm, "rtm")

// CHECK-LABEL: define{{.*}} void @test_serialize(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 256
TEST_CPU_SUPPORTS(serialize, "serialize")

// CHECK-LABEL: define{{.*}} void @test_sgx(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 512
TEST_CPU_SUPPORTS(sgx, "sgx")

// CHECK-LABEL: define{{.*}} void @test_sha(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 1024
TEST_CPU_SUPPORTS(sha, "sha")

// CHECK-LABEL: define{{.*}} void @test_shstk(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 2048
TEST_CPU_SUPPORTS(shstk, "shstk")

// CHECK-LABEL: define{{.*}} void @test_tbm(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 4096
TEST_CPU_SUPPORTS(tbm, "tbm")

// CHECK-LABEL: define{{.*}} void @test_tsxldtrk(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 8192
TEST_CPU_SUPPORTS(tsxldtrk, "tsxldtrk")

// CHECK-LABEL: define{{.*}} void @test_vaes(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 16384
TEST_CPU_SUPPORTS(vaes, "vaes")

// CHECK-LABEL: define{{.*}} void @test_waitpkg(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 32768
TEST_CPU_SUPPORTS(waitpkg, "waitpkg")

// CHECK-LABEL: define{{.*}} void @test_wbnoinvd(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 65536
TEST_CPU_SUPPORTS(wbnoinvd, "wbnoinvd")

// CHECK-LABEL: define{{.*}} void @test_xsave(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 131072
TEST_CPU_SUPPORTS(xsave, "xsave")

// CHECK-LABEL: define{{.*}} void @test_xsavec(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 262144
TEST_CPU_SUPPORTS(xsavec, "xsavec")

// CHECK-LABEL: define{{.*}} void @test_xsaveopt(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 524288
TEST_CPU_SUPPORTS(xsaveopt, "xsaveopt")

// CHECK-LABEL: define{{.*}} void @test_xsaves(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 1048576
TEST_CPU_SUPPORTS(xsaves, "xsaves")

// CHECK-LABEL: define{{.*}} void @test_amx_tile(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 2097152
TEST_CPU_SUPPORTS(amx_tile, "amx-tile")

// CHECK-LABEL: define{{.*}} void @test_amx_int8(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 4194304
TEST_CPU_SUPPORTS(amx_int8, "amx-int8")

// CHECK-LABEL: define{{.*}} void @test_amx_bf16(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 8388608
TEST_CPU_SUPPORTS(amx_bf16, "amx-bf16")

// CHECK-LABEL: define{{.*}} void @test_uintr(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 16777216
TEST_CPU_SUPPORTS(uintr, "uintr")

// CHECK-LABEL: define{{.*}} void @test_hreset(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 33554432
TEST_CPU_SUPPORTS(hreset, "hreset")

// CHECK-LABEL: define{{.*}} void @test_kl(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 67108864
TEST_CPU_SUPPORTS(kl, "kl")

// CHECK-LABEL: define{{.*}} void @test_widekl(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 268435456
TEST_CPU_SUPPORTS(widekl, "widekl")

// CHECK-LABEL: define{{.*}} void @test_avxvnni(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 536870912
TEST_CPU_SUPPORTS(avxvnni, "avxvnni")

// CHECK-LABEL: define{{.*}} void @test_avx512fp16(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 1073741824
TEST_CPU_SUPPORTS(avx512fp16, "avx512fp16")

// CHECK-LABEL: define{{.*}} void @test_x86_64(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], -2147483648
TEST_CPU_SUPPORTS(x86_64, "x86-64")

// CHECK-LABEL: define{{.*}} void @test_x86_64_v2(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 1
TEST_CPU_SUPPORTS(x86_64_v2, "x86-64-v2")

// CHECK-LABEL: define{{.*}} void @test_x86_64_v3(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 2
TEST_CPU_SUPPORTS(x86_64_v3, "x86-64-v3")

// CHECK-LABEL: define{{.*}} void @test_x86_64_v4(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 4
TEST_CPU_SUPPORTS(x86_64_v4, "x86-64-v4")

// CHECK-LABEL: define{{.*}} void @test_avxifma(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 8
TEST_CPU_SUPPORTS(avxifma, "avxifma")

// CHECK-LABEL: define{{.*}} void @test_avxvnniint8(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 16
TEST_CPU_SUPPORTS(avxvnniint8, "avxvnniint8")

// CHECK-LABEL: define{{.*}} void @test_avxneconvert(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 32
TEST_CPU_SUPPORTS(avxneconvert, "avxneconvert")

// CHECK-LABEL: define{{.*}} void @test_cmpccxadd(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 64
TEST_CPU_SUPPORTS(cmpccxadd, "cmpccxadd")

// CHECK-LABEL: define{{.*}} void @test_amx_fp16(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 128
TEST_CPU_SUPPORTS(amx_fp16, "amx-fp16")

// CHECK-LABEL: define{{.*}} void @test_prefetchi(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 256
TEST_CPU_SUPPORTS(prefetchi, "prefetchi")

// CHECK-LABEL: define{{.*}} void @test_raoint(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 512
TEST_CPU_SUPPORTS(raoint, "raoint")

// CHECK-LABEL: define{{.*}} void @test_amx_complex(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 1024
TEST_CPU_SUPPORTS(amx_complex, "amx-complex")

// CHECK-LABEL: define{{.*}} void @test_avxvnniint16(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 2048
TEST_CPU_SUPPORTS(avxvnniint16, "avxvnniint16")

// CHECK-LABEL: define{{.*}} void @test_sm3(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 4096
TEST_CPU_SUPPORTS(sm3, "sm3")

// CHECK-LABEL: define{{.*}} void @test_sha512(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 8192
TEST_CPU_SUPPORTS(sha512, "sha512")

// CHECK-LABEL: define{{.*}} void @test_sm4(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 16384
TEST_CPU_SUPPORTS(sm4, "sm4")

// CHECK-LABEL: define{{.*}} void @test_apxf(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 32768
TEST_CPU_SUPPORTS(apxf, "apxf")

// CHECK-LABEL: define{{.*}} void @test_usermsr(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 65536
TEST_CPU_SUPPORTS(usermsr, "usermsr")

// CHECK-LABEL: define{{.*}} void @test_avx10_1(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 262144
TEST_CPU_SUPPORTS(avx10_1, "avx10.1")

// CHECK-LABEL: define{{.*}} void @test_avx10_2(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 1048576
TEST_CPU_SUPPORTS(avx10_2, "avx10.2")

// CHECK-LABEL: define{{.*}} void @test_amx_avx512(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 2097152
TEST_CPU_SUPPORTS(amx_avx512, "amx-avx512")

// CHECK-LABEL: define{{.*}} void @test_amx_fp8(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 16777216
TEST_CPU_SUPPORTS(amx_fp8, "amx-fp8")

// CHECK-LABEL: define{{.*}} void @test_movrs(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 33554432
TEST_CPU_SUPPORTS(movrs, "movrs")

// CHECK-LABEL: define{{.*}} void @test_amx_movrs(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 67108864
TEST_CPU_SUPPORTS(amx_movrs, "amx-movrs")
