// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm < %s | FileCheck %s

// Test that __builtin_cpu_supports emits the correct field/bit for every
// feature listed in llvm/include/llvm/TargetParser/X86TargetParser.def. The
// ABI_VALUE is a bit index shared with compiler-rt/libgcc: bits 0-31 live in
// __cpu_model.__cpu_features (offset 12), bits >=32 in __cpu_features2[word-1].
extern void a(const char *);

// CHECK: @__cpu_model = external dso_local global { i32, i32, i32, [1 x i32] }

void test_cmov(void) {
  if (__builtin_cpu_supports("cmov"))
    a("cmov");

  // CHECK-LABEL: define{{.*}} void @test_cmov(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
  // CHECK: = and i32 [[LOAD]], 1
}

void test_mmx(void) {
  if (__builtin_cpu_supports("mmx"))
    a("mmx");

  // CHECK-LABEL: define{{.*}} void @test_mmx(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
  // CHECK: = and i32 [[LOAD]], 2
}

void test_popcnt(void) {
  if (__builtin_cpu_supports("popcnt"))
    a("popcnt");

  // CHECK-LABEL: define{{.*}} void @test_popcnt(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
  // CHECK: = and i32 [[LOAD]], 4
}

void test_sse(void) {
  if (__builtin_cpu_supports("sse"))
    a("sse");

  // CHECK-LABEL: define{{.*}} void @test_sse(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
  // CHECK: = and i32 [[LOAD]], 8
}

void test_sse2(void) {
  if (__builtin_cpu_supports("sse2"))
    a("sse2");

  // CHECK-LABEL: define{{.*}} void @test_sse2(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
  // CHECK: = and i32 [[LOAD]], 16
}

void test_sse3(void) {
  if (__builtin_cpu_supports("sse3"))
    a("sse3");

  // CHECK-LABEL: define{{.*}} void @test_sse3(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
  // CHECK: = and i32 [[LOAD]], 32
}

void test_ssse3(void) {
  if (__builtin_cpu_supports("ssse3"))
    a("ssse3");

  // CHECK-LABEL: define{{.*}} void @test_ssse3(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
  // CHECK: = and i32 [[LOAD]], 64
}

void test_sse4_1(void) {
  if (__builtin_cpu_supports("sse4.1"))
    a("sse4.1");

  // CHECK-LABEL: define{{.*}} void @test_sse4_1(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
  // CHECK: = and i32 [[LOAD]], 128
}

void test_sse4_2(void) {
  if (__builtin_cpu_supports("sse4.2"))
    a("sse4.2");

  // CHECK-LABEL: define{{.*}} void @test_sse4_2(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
  // CHECK: = and i32 [[LOAD]], 256
}

void test_avx(void) {
  if (__builtin_cpu_supports("avx"))
    a("avx");

  // CHECK-LABEL: define{{.*}} void @test_avx(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
  // CHECK: = and i32 [[LOAD]], 512
}

void test_avx2(void) {
  if (__builtin_cpu_supports("avx2"))
    a("avx2");

  // CHECK-LABEL: define{{.*}} void @test_avx2(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
  // CHECK: = and i32 [[LOAD]], 1024
}

void test_sse4a(void) {
  if (__builtin_cpu_supports("sse4a"))
    a("sse4a");

  // CHECK-LABEL: define{{.*}} void @test_sse4a(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
  // CHECK: = and i32 [[LOAD]], 2048
}

void test_fma4(void) {
  if (__builtin_cpu_supports("fma4"))
    a("fma4");

  // CHECK-LABEL: define{{.*}} void @test_fma4(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
  // CHECK: = and i32 [[LOAD]], 4096
}

void test_xop(void) {
  if (__builtin_cpu_supports("xop"))
    a("xop");

  // CHECK-LABEL: define{{.*}} void @test_xop(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
  // CHECK: = and i32 [[LOAD]], 8192
}

void test_fma(void) {
  if (__builtin_cpu_supports("fma"))
    a("fma");

  // CHECK-LABEL: define{{.*}} void @test_fma(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
  // CHECK: = and i32 [[LOAD]], 16384
}

void test_avx512f(void) {
  if (__builtin_cpu_supports("avx512f"))
    a("avx512f");

  // CHECK-LABEL: define{{.*}} void @test_avx512f(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
  // CHECK: = and i32 [[LOAD]], 32768
}

void test_bmi(void) {
  if (__builtin_cpu_supports("bmi"))
    a("bmi");

  // CHECK-LABEL: define{{.*}} void @test_bmi(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
  // CHECK: = and i32 [[LOAD]], 65536
}

void test_bmi2(void) {
  if (__builtin_cpu_supports("bmi2"))
    a("bmi2");

  // CHECK-LABEL: define{{.*}} void @test_bmi2(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
  // CHECK: = and i32 [[LOAD]], 131072
}

void test_aes(void) {
  if (__builtin_cpu_supports("aes"))
    a("aes");

  // CHECK-LABEL: define{{.*}} void @test_aes(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
  // CHECK: = and i32 [[LOAD]], 262144
}

void test_pclmul(void) {
  if (__builtin_cpu_supports("pclmul"))
    a("pclmul");

  // CHECK-LABEL: define{{.*}} void @test_pclmul(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
  // CHECK: = and i32 [[LOAD]], 524288
}

void test_avx512vl(void) {
  if (__builtin_cpu_supports("avx512vl"))
    a("avx512vl");

  // CHECK-LABEL: define{{.*}} void @test_avx512vl(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
  // CHECK: = and i32 [[LOAD]], 1048576
}

void test_avx512bw(void) {
  if (__builtin_cpu_supports("avx512bw"))
    a("avx512bw");

  // CHECK-LABEL: define{{.*}} void @test_avx512bw(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
  // CHECK: = and i32 [[LOAD]], 2097152
}

void test_avx512dq(void) {
  if (__builtin_cpu_supports("avx512dq"))
    a("avx512dq");

  // CHECK-LABEL: define{{.*}} void @test_avx512dq(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
  // CHECK: = and i32 [[LOAD]], 4194304
}

void test_avx512cd(void) {
  if (__builtin_cpu_supports("avx512cd"))
    a("avx512cd");

  // CHECK-LABEL: define{{.*}} void @test_avx512cd(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
  // CHECK: = and i32 [[LOAD]], 8388608
}

void test_avx512vbmi(void) {
  if (__builtin_cpu_supports("avx512vbmi"))
    a("avx512vbmi");

  // CHECK-LABEL: define{{.*}} void @test_avx512vbmi(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
  // CHECK: = and i32 [[LOAD]], 67108864
}

void test_avx512ifma(void) {
  if (__builtin_cpu_supports("avx512ifma"))
    a("avx512ifma");

  // CHECK-LABEL: define{{.*}} void @test_avx512ifma(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
  // CHECK: = and i32 [[LOAD]], 134217728
}

void test_avx512vpopcntdq(void) {
  if (__builtin_cpu_supports("avx512vpopcntdq"))
    a("avx512vpopcntdq");

  // CHECK-LABEL: define{{.*}} void @test_avx512vpopcntdq(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
  // CHECK: = and i32 [[LOAD]], 1073741824
}

void test_avx512vbmi2(void) {
  if (__builtin_cpu_supports("avx512vbmi2"))
    a("avx512vbmi2");

  // CHECK-LABEL: define{{.*}} void @test_avx512vbmi2(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
  // CHECK: = and i32 [[LOAD]], -2147483648
}

void test_gfni(void) {
  if (__builtin_cpu_supports("gfni"))
    a("gfni");

  // CHECK-LABEL: define{{.*}} void @test_gfni(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
  // CHECK: = and i32 [[LOAD]], 1
}

void test_vpclmulqdq(void) {
  if (__builtin_cpu_supports("vpclmulqdq"))
    a("vpclmulqdq");

  // CHECK-LABEL: define{{.*}} void @test_vpclmulqdq(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
  // CHECK: = and i32 [[LOAD]], 2
}

void test_avx512vnni(void) {
  if (__builtin_cpu_supports("avx512vnni"))
    a("avx512vnni");

  // CHECK-LABEL: define{{.*}} void @test_avx512vnni(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
  // CHECK: = and i32 [[LOAD]], 4
}

void test_avx512bitalg(void) {
  if (__builtin_cpu_supports("avx512bitalg"))
    a("avx512bitalg");

  // CHECK-LABEL: define{{.*}} void @test_avx512bitalg(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
  // CHECK: = and i32 [[LOAD]], 8
}

void test_avx512bf16(void) {
  if (__builtin_cpu_supports("avx512bf16"))
    a("avx512bf16");

  // CHECK-LABEL: define{{.*}} void @test_avx512bf16(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
  // CHECK: = and i32 [[LOAD]], 16
}

void test_avx512vp2intersect(void) {
  if (__builtin_cpu_supports("avx512vp2intersect"))
    a("avx512vp2intersect");

  // CHECK-LABEL: define{{.*}} void @test_avx512vp2intersect(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
  // CHECK: = and i32 [[LOAD]], 32
}

void test_adx(void) {
  if (__builtin_cpu_supports("adx"))
    a("adx");

  // CHECK-LABEL: define{{.*}} void @test_adx(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
  // CHECK: = and i32 [[LOAD]], 256
}

void test_cldemote(void) {
  if (__builtin_cpu_supports("cldemote"))
    a("cldemote");

  // CHECK-LABEL: define{{.*}} void @test_cldemote(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
  // CHECK: = and i32 [[LOAD]], 1024
}

void test_clflushopt(void) {
  if (__builtin_cpu_supports("clflushopt"))
    a("clflushopt");

  // CHECK-LABEL: define{{.*}} void @test_clflushopt(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
  // CHECK: = and i32 [[LOAD]], 2048
}

void test_clwb(void) {
  if (__builtin_cpu_supports("clwb"))
    a("clwb");

  // CHECK-LABEL: define{{.*}} void @test_clwb(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
  // CHECK: = and i32 [[LOAD]], 4096
}

void test_clzero(void) {
  if (__builtin_cpu_supports("clzero"))
    a("clzero");

  // CHECK-LABEL: define{{.*}} void @test_clzero(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
  // CHECK: = and i32 [[LOAD]], 8192
}

void test_cx16(void) {
  if (__builtin_cpu_supports("cx16"))
    a("cx16");

  // CHECK-LABEL: define{{.*}} void @test_cx16(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
  // CHECK: = and i32 [[LOAD]], 16384
}

void test_enqcmd(void) {
  if (__builtin_cpu_supports("enqcmd"))
    a("enqcmd");

  // CHECK-LABEL: define{{.*}} void @test_enqcmd(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
  // CHECK: = and i32 [[LOAD]], 65536
}

void test_f16c(void) {
  if (__builtin_cpu_supports("f16c"))
    a("f16c");

  // CHECK-LABEL: define{{.*}} void @test_f16c(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
  // CHECK: = and i32 [[LOAD]], 131072
}

void test_fsgsbase(void) {
  if (__builtin_cpu_supports("fsgsbase"))
    a("fsgsbase");

  // CHECK-LABEL: define{{.*}} void @test_fsgsbase(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
  // CHECK: = and i32 [[LOAD]], 262144
}

void test_sahf(void) {
  if (__builtin_cpu_supports("sahf"))
    a("sahf");

  // CHECK-LABEL: define{{.*}} void @test_sahf(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
  // CHECK: = and i32 [[LOAD]], 4194304
}

void test_64bit(void) {
  if (__builtin_cpu_supports("64bit"))
    a("64bit");

  // CHECK-LABEL: define{{.*}} void @test_64bit(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
  // CHECK: = and i32 [[LOAD]], 8388608
}

void test_lwp(void) {
  if (__builtin_cpu_supports("lwp"))
    a("lwp");

  // CHECK-LABEL: define{{.*}} void @test_lwp(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
  // CHECK: = and i32 [[LOAD]], 16777216
}

void test_lzcnt(void) {
  if (__builtin_cpu_supports("lzcnt"))
    a("lzcnt");

  // CHECK-LABEL: define{{.*}} void @test_lzcnt(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
  // CHECK: = and i32 [[LOAD]], 33554432
}

void test_movbe(void) {
  if (__builtin_cpu_supports("movbe"))
    a("movbe");

  // CHECK-LABEL: define{{.*}} void @test_movbe(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
  // CHECK: = and i32 [[LOAD]], 67108864
}

void test_movdir64b(void) {
  if (__builtin_cpu_supports("movdir64b"))
    a("movdir64b");

  // CHECK-LABEL: define{{.*}} void @test_movdir64b(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
  // CHECK: = and i32 [[LOAD]], 134217728
}

void test_movdiri(void) {
  if (__builtin_cpu_supports("movdiri"))
    a("movdiri");

  // CHECK-LABEL: define{{.*}} void @test_movdiri(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
  // CHECK: = and i32 [[LOAD]], 268435456
}

void test_mwaitx(void) {
  if (__builtin_cpu_supports("mwaitx"))
    a("mwaitx");

  // CHECK-LABEL: define{{.*}} void @test_mwaitx(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
  // CHECK: = and i32 [[LOAD]], 536870912
}

void test_pconfig(void) {
  if (__builtin_cpu_supports("pconfig"))
    a("pconfig");

  // CHECK-LABEL: define{{.*}} void @test_pconfig(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
  // CHECK: = and i32 [[LOAD]], -2147483648
}

void test_pku(void) {
  if (__builtin_cpu_supports("pku"))
    a("pku");

  // CHECK-LABEL: define{{.*}} void @test_pku(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
  // CHECK: = and i32 [[LOAD]], 1
}

void test_prfchw(void) {
  if (__builtin_cpu_supports("prfchw"))
    a("prfchw");

  // CHECK-LABEL: define{{.*}} void @test_prfchw(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
  // CHECK: = and i32 [[LOAD]], 4
}

void test_ptwrite(void) {
  if (__builtin_cpu_supports("ptwrite"))
    a("ptwrite");

  // CHECK-LABEL: define{{.*}} void @test_ptwrite(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
  // CHECK: = and i32 [[LOAD]], 8
}

void test_rdpid(void) {
  if (__builtin_cpu_supports("rdpid"))
    a("rdpid");

  // CHECK-LABEL: define{{.*}} void @test_rdpid(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
  // CHECK: = and i32 [[LOAD]], 16
}

void test_rdrnd(void) {
  if (__builtin_cpu_supports("rdrnd"))
    a("rdrnd");

  // CHECK-LABEL: define{{.*}} void @test_rdrnd(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
  // CHECK: = and i32 [[LOAD]], 32
}

void test_rdseed(void) {
  if (__builtin_cpu_supports("rdseed"))
    a("rdseed");

  // CHECK-LABEL: define{{.*}} void @test_rdseed(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
  // CHECK: = and i32 [[LOAD]], 64
}

void test_rtm(void) {
  if (__builtin_cpu_supports("rtm"))
    a("rtm");

  // CHECK-LABEL: define{{.*}} void @test_rtm(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
  // CHECK: = and i32 [[LOAD]], 128
}

void test_serialize(void) {
  if (__builtin_cpu_supports("serialize"))
    a("serialize");

  // CHECK-LABEL: define{{.*}} void @test_serialize(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
  // CHECK: = and i32 [[LOAD]], 256
}

void test_sgx(void) {
  if (__builtin_cpu_supports("sgx"))
    a("sgx");

  // CHECK-LABEL: define{{.*}} void @test_sgx(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
  // CHECK: = and i32 [[LOAD]], 512
}

void test_sha(void) {
  if (__builtin_cpu_supports("sha"))
    a("sha");

  // CHECK-LABEL: define{{.*}} void @test_sha(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
  // CHECK: = and i32 [[LOAD]], 1024
}

void test_shstk(void) {
  if (__builtin_cpu_supports("shstk"))
    a("shstk");

  // CHECK-LABEL: define{{.*}} void @test_shstk(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
  // CHECK: = and i32 [[LOAD]], 2048
}

void test_tbm(void) {
  if (__builtin_cpu_supports("tbm"))
    a("tbm");

  // CHECK-LABEL: define{{.*}} void @test_tbm(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
  // CHECK: = and i32 [[LOAD]], 4096
}

void test_tsxldtrk(void) {
  if (__builtin_cpu_supports("tsxldtrk"))
    a("tsxldtrk");

  // CHECK-LABEL: define{{.*}} void @test_tsxldtrk(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
  // CHECK: = and i32 [[LOAD]], 8192
}

void test_vaes(void) {
  if (__builtin_cpu_supports("vaes"))
    a("vaes");

  // CHECK-LABEL: define{{.*}} void @test_vaes(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
  // CHECK: = and i32 [[LOAD]], 16384
}

void test_waitpkg(void) {
  if (__builtin_cpu_supports("waitpkg"))
    a("waitpkg");

  // CHECK-LABEL: define{{.*}} void @test_waitpkg(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
  // CHECK: = and i32 [[LOAD]], 32768
}

void test_wbnoinvd(void) {
  if (__builtin_cpu_supports("wbnoinvd"))
    a("wbnoinvd");

  // CHECK-LABEL: define{{.*}} void @test_wbnoinvd(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
  // CHECK: = and i32 [[LOAD]], 65536
}

void test_xsave(void) {
  if (__builtin_cpu_supports("xsave"))
    a("xsave");

  // CHECK-LABEL: define{{.*}} void @test_xsave(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
  // CHECK: = and i32 [[LOAD]], 131072
}

void test_xsavec(void) {
  if (__builtin_cpu_supports("xsavec"))
    a("xsavec");

  // CHECK-LABEL: define{{.*}} void @test_xsavec(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
  // CHECK: = and i32 [[LOAD]], 262144
}

void test_xsaveopt(void) {
  if (__builtin_cpu_supports("xsaveopt"))
    a("xsaveopt");

  // CHECK-LABEL: define{{.*}} void @test_xsaveopt(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
  // CHECK: = and i32 [[LOAD]], 524288
}

void test_xsaves(void) {
  if (__builtin_cpu_supports("xsaves"))
    a("xsaves");

  // CHECK-LABEL: define{{.*}} void @test_xsaves(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
  // CHECK: = and i32 [[LOAD]], 1048576
}

void test_amx_tile(void) {
  if (__builtin_cpu_supports("amx-tile"))
    a("amx-tile");

  // CHECK-LABEL: define{{.*}} void @test_amx_tile(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
  // CHECK: = and i32 [[LOAD]], 2097152
}

void test_amx_int8(void) {
  if (__builtin_cpu_supports("amx-int8"))
    a("amx-int8");

  // CHECK-LABEL: define{{.*}} void @test_amx_int8(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
  // CHECK: = and i32 [[LOAD]], 4194304
}

void test_amx_bf16(void) {
  if (__builtin_cpu_supports("amx-bf16"))
    a("amx-bf16");

  // CHECK-LABEL: define{{.*}} void @test_amx_bf16(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
  // CHECK: = and i32 [[LOAD]], 8388608
}

void test_uintr(void) {
  if (__builtin_cpu_supports("uintr"))
    a("uintr");

  // CHECK-LABEL: define{{.*}} void @test_uintr(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
  // CHECK: = and i32 [[LOAD]], 16777216
}

void test_hreset(void) {
  if (__builtin_cpu_supports("hreset"))
    a("hreset");

  // CHECK-LABEL: define{{.*}} void @test_hreset(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
  // CHECK: = and i32 [[LOAD]], 33554432
}

void test_kl(void) {
  if (__builtin_cpu_supports("kl"))
    a("kl");

  // CHECK-LABEL: define{{.*}} void @test_kl(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
  // CHECK: = and i32 [[LOAD]], 67108864
}

void test_widekl(void) {
  if (__builtin_cpu_supports("widekl"))
    a("widekl");

  // CHECK-LABEL: define{{.*}} void @test_widekl(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
  // CHECK: = and i32 [[LOAD]], 268435456
}

void test_avxvnni(void) {
  if (__builtin_cpu_supports("avxvnni"))
    a("avxvnni");

  // CHECK-LABEL: define{{.*}} void @test_avxvnni(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
  // CHECK: = and i32 [[LOAD]], 536870912
}

void test_avx512fp16(void) {
  if (__builtin_cpu_supports("avx512fp16"))
    a("avx512fp16");

  // CHECK-LABEL: define{{.*}} void @test_avx512fp16(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
  // CHECK: = and i32 [[LOAD]], 1073741824
}

void test_avxifma(void) {
  if (__builtin_cpu_supports("avxifma"))
    a("avxifma");

  // CHECK-LABEL: define{{.*}} void @test_avxifma(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
  // CHECK: = and i32 [[LOAD]], 8
}

void test_avxvnniint8(void) {
  if (__builtin_cpu_supports("avxvnniint8"))
    a("avxvnniint8");

  // CHECK-LABEL: define{{.*}} void @test_avxvnniint8(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
  // CHECK: = and i32 [[LOAD]], 16
}

void test_avxneconvert(void) {
  if (__builtin_cpu_supports("avxneconvert"))
    a("avxneconvert");

  // CHECK-LABEL: define{{.*}} void @test_avxneconvert(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
  // CHECK: = and i32 [[LOAD]], 32
}

void test_cmpccxadd(void) {
  if (__builtin_cpu_supports("cmpccxadd"))
    a("cmpccxadd");

  // CHECK-LABEL: define{{.*}} void @test_cmpccxadd(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
  // CHECK: = and i32 [[LOAD]], 64
}

void test_amx_fp16(void) {
  if (__builtin_cpu_supports("amx-fp16"))
    a("amx-fp16");

  // CHECK-LABEL: define{{.*}} void @test_amx_fp16(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
  // CHECK: = and i32 [[LOAD]], 128
}

void test_prefetchi(void) {
  if (__builtin_cpu_supports("prefetchi"))
    a("prefetchi");

  // CHECK-LABEL: define{{.*}} void @test_prefetchi(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
  // CHECK: = and i32 [[LOAD]], 256
}

void test_raoint(void) {
  if (__builtin_cpu_supports("raoint"))
    a("raoint");

  // CHECK-LABEL: define{{.*}} void @test_raoint(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
  // CHECK: = and i32 [[LOAD]], 512
}

void test_amx_complex(void) {
  if (__builtin_cpu_supports("amx-complex"))
    a("amx-complex");

  // CHECK-LABEL: define{{.*}} void @test_amx_complex(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
  // CHECK: = and i32 [[LOAD]], 1024
}

void test_avxvnniint16(void) {
  if (__builtin_cpu_supports("avxvnniint16"))
    a("avxvnniint16");

  // CHECK-LABEL: define{{.*}} void @test_avxvnniint16(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
  // CHECK: = and i32 [[LOAD]], 2048
}

void test_sm3(void) {
  if (__builtin_cpu_supports("sm3"))
    a("sm3");

  // CHECK-LABEL: define{{.*}} void @test_sm3(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
  // CHECK: = and i32 [[LOAD]], 4096
}

void test_sha512(void) {
  if (__builtin_cpu_supports("sha512"))
    a("sha512");

  // CHECK-LABEL: define{{.*}} void @test_sha512(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
  // CHECK: = and i32 [[LOAD]], 8192
}

void test_sm4(void) {
  if (__builtin_cpu_supports("sm4"))
    a("sm4");

  // CHECK-LABEL: define{{.*}} void @test_sm4(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
  // CHECK: = and i32 [[LOAD]], 16384
}

void test_apxf(void) {
  if (__builtin_cpu_supports("apxf"))
    a("apxf");

  // CHECK-LABEL: define{{.*}} void @test_apxf(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
  // CHECK: = and i32 [[LOAD]], 32768
}

void test_usermsr(void) {
  if (__builtin_cpu_supports("usermsr"))
    a("usermsr");

  // CHECK-LABEL: define{{.*}} void @test_usermsr(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
  // CHECK: = and i32 [[LOAD]], 65536
}

void test_avx10_1(void) {
  if (__builtin_cpu_supports("avx10.1"))
    a("avx10.1");

  // CHECK-LABEL: define{{.*}} void @test_avx10_1(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
  // CHECK: = and i32 [[LOAD]], 262144
}

void test_avx10_2(void) {
  if (__builtin_cpu_supports("avx10.2"))
    a("avx10.2");

  // CHECK-LABEL: define{{.*}} void @test_avx10_2(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
  // CHECK: = and i32 [[LOAD]], 1048576
}

void test_amx_avx512(void) {
  if (__builtin_cpu_supports("amx-avx512"))
    a("amx-avx512");

  // CHECK-LABEL: define{{.*}} void @test_amx_avx512(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
  // CHECK: = and i32 [[LOAD]], 2097152
}

void test_amx_tf32(void) {
  if (__builtin_cpu_supports("amx-tf32"))
    a("amx-tf32");

  // CHECK-LABEL: define{{.*}} void @test_amx_tf32(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
  // CHECK: = and i32 [[LOAD]], 4194304
}

void test_amx_fp8(void) {
  if (__builtin_cpu_supports("amx-fp8"))
    a("amx-fp8");

  // CHECK-LABEL: define{{.*}} void @test_amx_fp8(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
  // CHECK: = and i32 [[LOAD]], 16777216
}

void test_movrs(void) {
  if (__builtin_cpu_supports("movrs"))
    a("movrs");

  // CHECK-LABEL: define{{.*}} void @test_movrs(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
  // CHECK: = and i32 [[LOAD]], 33554432
}

void test_amx_movrs(void) {
  if (__builtin_cpu_supports("amx-movrs"))
    a("amx-movrs");

  // CHECK-LABEL: define{{.*}} void @test_amx_movrs(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
  // CHECK: = and i32 [[LOAD]], 67108864
}

void test_x86_64(void) {
  if (__builtin_cpu_supports("x86-64"))
    a("x86-64");

  // CHECK-LABEL: define{{.*}} void @test_x86_64(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
  // CHECK: = and i32 [[LOAD]], -2147483648
}

void test_x86_64_v2(void) {
  if (__builtin_cpu_supports("x86-64-v2"))
    a("x86-64-v2");

  // CHECK-LABEL: define{{.*}} void @test_x86_64_v2(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
  // CHECK: = and i32 [[LOAD]], 1
}

void test_x86_64_v3(void) {
  if (__builtin_cpu_supports("x86-64-v3"))
    a("x86-64-v3");

  // CHECK-LABEL: define{{.*}} void @test_x86_64_v3(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
  // CHECK: = and i32 [[LOAD]], 2
}

void test_x86_64_v4(void) {
  if (__builtin_cpu_supports("x86-64-v4"))
    a("x86-64-v4");

  // CHECK-LABEL: define{{.*}} void @test_x86_64_v4(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
  // CHECK: = and i32 [[LOAD]], 4
}
