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

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 2
TEST_CPU_SUPPORTS(mmx, "mmx")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 4
TEST_CPU_SUPPORTS(popcnt, "popcnt")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 8
TEST_CPU_SUPPORTS(sse, "sse")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 16
TEST_CPU_SUPPORTS(sse2, "sse2")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 32
TEST_CPU_SUPPORTS(sse3, "sse3")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 64
TEST_CPU_SUPPORTS(ssse3, "ssse3")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 128
TEST_CPU_SUPPORTS(sse4_1, "sse4.1")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 256
TEST_CPU_SUPPORTS(sse4_2, "sse4.2")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 512
TEST_CPU_SUPPORTS(avx, "avx")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 1024
TEST_CPU_SUPPORTS(avx2, "avx2")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 2048
TEST_CPU_SUPPORTS(sse4a, "sse4a")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 4096
TEST_CPU_SUPPORTS(fma4, "fma4")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 8192
TEST_CPU_SUPPORTS(xop, "xop")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 16384
TEST_CPU_SUPPORTS(fma, "fma")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 32768
TEST_CPU_SUPPORTS(avx512f, "avx512f")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 65536
TEST_CPU_SUPPORTS(bmi, "bmi")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 131072
TEST_CPU_SUPPORTS(bmi2, "bmi2")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 262144
TEST_CPU_SUPPORTS(aes, "aes")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 524288
TEST_CPU_SUPPORTS(pclmul, "pclmul")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 1048576
TEST_CPU_SUPPORTS(avx512vl, "avx512vl")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 2097152
TEST_CPU_SUPPORTS(avx512bw, "avx512bw")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 4194304
TEST_CPU_SUPPORTS(avx512dq, "avx512dq")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 8388608
TEST_CPU_SUPPORTS(avx512cd, "avx512cd")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 67108864
TEST_CPU_SUPPORTS(avx512vbmi, "avx512vbmi")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 134217728
TEST_CPU_SUPPORTS(avx512ifma, "avx512ifma")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], 1073741824
TEST_CPU_SUPPORTS(avx512vpopcntdq, "avx512vpopcntdq")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// CHECK: = and i32 [[LOAD]], -2147483648
TEST_CPU_SUPPORTS(avx512vbmi2, "avx512vbmi2")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 1
TEST_CPU_SUPPORTS(gfni, "gfni")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 2
TEST_CPU_SUPPORTS(vpclmulqdq, "vpclmulqdq")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 4
TEST_CPU_SUPPORTS(avx512vnni, "avx512vnni")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 8
TEST_CPU_SUPPORTS(avx512bitalg, "avx512bitalg")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 16
TEST_CPU_SUPPORTS(avx512bf16, "avx512bf16")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 32
TEST_CPU_SUPPORTS(avx512vp2intersect, "avx512vp2intersect")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 256
TEST_CPU_SUPPORTS(adx, "adx")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 1024
TEST_CPU_SUPPORTS(cldemote, "cldemote")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 2048
TEST_CPU_SUPPORTS(clflushopt, "clflushopt")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 4096
TEST_CPU_SUPPORTS(clwb, "clwb")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 8192
TEST_CPU_SUPPORTS(clzero, "clzero")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 16384
TEST_CPU_SUPPORTS(cx16, "cx16")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 65536
TEST_CPU_SUPPORTS(enqcmd, "enqcmd")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 131072
TEST_CPU_SUPPORTS(f16c, "f16c")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 262144
TEST_CPU_SUPPORTS(fsgsbase, "fsgsbase")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 4194304
TEST_CPU_SUPPORTS(sahf, "sahf")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 8388608
TEST_CPU_SUPPORTS(64bit, "64bit")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 16777216
TEST_CPU_SUPPORTS(lwp, "lwp")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 33554432
TEST_CPU_SUPPORTS(lzcnt, "lzcnt")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 67108864
TEST_CPU_SUPPORTS(movbe, "movbe")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 134217728
TEST_CPU_SUPPORTS(movdir64b, "movdir64b")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 268435456
TEST_CPU_SUPPORTS(movdiri, "movdiri")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], 536870912
TEST_CPU_SUPPORTS(mwaitx, "mwaitx")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// CHECK: = and i32 [[LOAD]], -2147483648
TEST_CPU_SUPPORTS(pconfig, "pconfig")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 1
TEST_CPU_SUPPORTS(pku, "pku")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 4
TEST_CPU_SUPPORTS(prfchw, "prfchw")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 8
TEST_CPU_SUPPORTS(ptwrite, "ptwrite")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 16
TEST_CPU_SUPPORTS(rdpid, "rdpid")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 32
TEST_CPU_SUPPORTS(rdrnd, "rdrnd")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 64
TEST_CPU_SUPPORTS(rdseed, "rdseed")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 128
TEST_CPU_SUPPORTS(rtm, "rtm")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 256
TEST_CPU_SUPPORTS(serialize, "serialize")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 512
TEST_CPU_SUPPORTS(sgx, "sgx")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 1024
TEST_CPU_SUPPORTS(sha, "sha")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 2048
TEST_CPU_SUPPORTS(shstk, "shstk")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 4096
TEST_CPU_SUPPORTS(tbm, "tbm")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 8192
TEST_CPU_SUPPORTS(tsxldtrk, "tsxldtrk")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 16384
TEST_CPU_SUPPORTS(vaes, "vaes")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 32768
TEST_CPU_SUPPORTS(waitpkg, "waitpkg")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 65536
TEST_CPU_SUPPORTS(wbnoinvd, "wbnoinvd")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 131072
TEST_CPU_SUPPORTS(xsave, "xsave")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 262144
TEST_CPU_SUPPORTS(xsavec, "xsavec")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 524288
TEST_CPU_SUPPORTS(xsaveopt, "xsaveopt")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 1048576
TEST_CPU_SUPPORTS(xsaves, "xsaves")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 2097152
TEST_CPU_SUPPORTS(amx_tile, "amx-tile")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 4194304
TEST_CPU_SUPPORTS(amx_int8, "amx-int8")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 8388608
TEST_CPU_SUPPORTS(amx_bf16, "amx-bf16")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 16777216
TEST_CPU_SUPPORTS(uintr, "uintr")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 33554432
TEST_CPU_SUPPORTS(hreset, "hreset")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 67108864
TEST_CPU_SUPPORTS(kl, "kl")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 268435456
TEST_CPU_SUPPORTS(widekl, "widekl")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 536870912
TEST_CPU_SUPPORTS(avxvnni, "avxvnni")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], 1073741824
TEST_CPU_SUPPORTS(avx512fp16, "avx512fp16")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 8
TEST_CPU_SUPPORTS(avxifma, "avxifma")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 16
TEST_CPU_SUPPORTS(avxvnniint8, "avxvnniint8")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 32
TEST_CPU_SUPPORTS(avxneconvert, "avxneconvert")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 64
TEST_CPU_SUPPORTS(cmpccxadd, "cmpccxadd")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 128
TEST_CPU_SUPPORTS(amx_fp16, "amx-fp16")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 256
TEST_CPU_SUPPORTS(prefetchi, "prefetchi")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 512
TEST_CPU_SUPPORTS(raoint, "raoint")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 1024
TEST_CPU_SUPPORTS(amx_complex, "amx-complex")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 2048
TEST_CPU_SUPPORTS(avxvnniint16, "avxvnniint16")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 4096
TEST_CPU_SUPPORTS(sm3, "sm3")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 8192
TEST_CPU_SUPPORTS(sha512, "sha512")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 16384
TEST_CPU_SUPPORTS(sm4, "sm4")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 32768
TEST_CPU_SUPPORTS(apxf, "apxf")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 65536
TEST_CPU_SUPPORTS(usermsr, "usermsr")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 262144
TEST_CPU_SUPPORTS(avx10_1, "avx10.1")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 1048576
TEST_CPU_SUPPORTS(avx10_2, "avx10.2")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 2097152
TEST_CPU_SUPPORTS(amx_avx512, "amx-avx512")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 4194304
TEST_CPU_SUPPORTS(amx_tf32, "amx-tf32")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 16777216
TEST_CPU_SUPPORTS(amx_fp8, "amx-fp8")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 33554432
TEST_CPU_SUPPORTS(movrs, "movrs")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 67108864
TEST_CPU_SUPPORTS(amx_movrs, "amx-movrs")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// CHECK: = and i32 [[LOAD]], -2147483648
TEST_CPU_SUPPORTS(x86_64, "x86-64")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 1
TEST_CPU_SUPPORTS(x86_64_v2, "x86-64-v2")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 2
TEST_CPU_SUPPORTS(x86_64_v3, "x86-64-v3")

// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// CHECK: = and i32 [[LOAD]], 4
TEST_CPU_SUPPORTS(x86_64_v4, "x86-64-v4")
