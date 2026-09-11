// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm < %s | FileCheck %s

// Test that __builtin_cpu_is emits the correct ABI value for every CPU
// subtype, llvm/include/llvm/TargetParser/X86TargetParser.def.
extern void a(const char *);

// CHECK: @__cpu_model = external dso_local global { i32, i32, i32, [1 x i32] }

#define TEST_CPU_IS(NAME, STR)                                                 \
  void test_##NAME(void) {                                                     \
    if (__builtin_cpu_is(STR))                                                 \
      a(STR);                                                                  \
  }

// CHECK-LABEL: define{{.*}} void @test_nehalem(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 1
TEST_CPU_IS(nehalem, "nehalem")

// CHECK-LABEL: define{{.*}} void @test_westmere(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 2
TEST_CPU_IS(westmere, "westmere")

// CHECK-LABEL: define{{.*}} void @test_sandybridge(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 3
TEST_CPU_IS(sandybridge, "sandybridge")

// CHECK-LABEL: define{{.*}} void @test_barcelona(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 4
TEST_CPU_IS(barcelona, "barcelona")

// CHECK-LABEL: define{{.*}} void @test_shanghai(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 5
TEST_CPU_IS(shanghai, "shanghai")

// CHECK-LABEL: define{{.*}} void @test_istanbul(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 6
TEST_CPU_IS(istanbul, "istanbul")

// CHECK-LABEL: define{{.*}} void @test_bdver1(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 7
TEST_CPU_IS(bdver1, "bdver1")

// CHECK-LABEL: define{{.*}} void @test_bdver2(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 8
TEST_CPU_IS(bdver2, "bdver2")

// CHECK-LABEL: define{{.*}} void @test_bdver3(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 9
TEST_CPU_IS(bdver3, "bdver3")

// CHECK-LABEL: define{{.*}} void @test_bdver4(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 10
TEST_CPU_IS(bdver4, "bdver4")

// CHECK-LABEL: define{{.*}} void @test_znver1(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 11
TEST_CPU_IS(znver1, "znver1")

// CHECK-LABEL: define{{.*}} void @test_ivybridge(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 12
TEST_CPU_IS(ivybridge, "ivybridge")

// CHECK-LABEL: define{{.*}} void @test_haswell(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 13
TEST_CPU_IS(haswell, "haswell")

// CHECK-LABEL: define{{.*}} void @test_broadwell(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 14
TEST_CPU_IS(broadwell, "broadwell")

// CHECK-LABEL: define{{.*}} void @test_skylake(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 15
TEST_CPU_IS(skylake, "skylake")

// CHECK-LABEL: define{{.*}} void @test_skylake_avx512(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 16
TEST_CPU_IS(skylake_avx512, "skylake-avx512")

// CHECK-LABEL: define{{.*}} void @test_cannonlake(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 17
TEST_CPU_IS(cannonlake, "cannonlake")

// CHECK-LABEL: define{{.*}} void @test_icelake_client(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 18
TEST_CPU_IS(icelake_client, "icelake-client")

// CHECK-LABEL: define{{.*}} void @test_icelake_server(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 19
TEST_CPU_IS(icelake_server, "icelake-server")

// CHECK-LABEL: define{{.*}} void @test_znver2(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 20
TEST_CPU_IS(znver2, "znver2")

// CHECK-LABEL: define{{.*}} void @test_cascadelake(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 21
TEST_CPU_IS(cascadelake, "cascadelake")

// CHECK-LABEL: define{{.*}} void @test_tigerlake(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 22
TEST_CPU_IS(tigerlake, "tigerlake")

// CHECK-LABEL: define{{.*}} void @test_cooperlake(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 23
TEST_CPU_IS(cooperlake, "cooperlake")

// CHECK-LABEL: define{{.*}} void @test_sapphirerapids(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 24
TEST_CPU_IS(sapphirerapids, "sapphirerapids")

// CHECK-LABEL: define{{.*}} void @test_alderlake(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 25
TEST_CPU_IS(alderlake, "alderlake")

// CHECK-LABEL: define{{.*}} void @test_znver3(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 26
TEST_CPU_IS(znver3, "znver3")

// CHECK-LABEL: define{{.*}} void @test_rocketlake(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 27
TEST_CPU_IS(rocketlake, "rocketlake")

// CHECK-LABEL: define{{.*}} void @test_zhaoxin_fam7h_lujiazui(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 28
TEST_CPU_IS(zhaoxin_fam7h_lujiazui, "zhaoxin_fam7h_lujiazui")

// CHECK-LABEL: define{{.*}} void @test_znver4(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 29
TEST_CPU_IS(znver4, "znver4")

// CHECK-LABEL: define{{.*}} void @test_graniterapids(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 30
TEST_CPU_IS(graniterapids, "graniterapids")

// CHECK-LABEL: define{{.*}} void @test_graniterapids_d(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 31
TEST_CPU_IS(graniterapids_d, "graniterapids-d")

// CHECK-LABEL: define{{.*}} void @test_arrowlake(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 32
TEST_CPU_IS(arrowlake, "arrowlake")

// CHECK-LABEL: define{{.*}} void @test_arrowlake_s(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 33
TEST_CPU_IS(arrowlake_s, "arrowlake-s")

// CHECK-LABEL: define{{.*}} void @test_pantherlake(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 34
TEST_CPU_IS(pantherlake, "pantherlake")

// CHECK-LABEL: define{{.*}} void @test_znver5(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 36
TEST_CPU_IS(znver5, "znver5")

// CHECK-LABEL: define{{.*}} void @test_diamondrapids(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 38
TEST_CPU_IS(diamondrapids, "diamondrapids")

// CHECK-LABEL: define{{.*}} void @test_novalake(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 39
TEST_CPU_IS(novalake, "novalake")

// CHECK-LABEL: define{{.*}} void @test_znver6(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 40
TEST_CPU_IS(znver6, "znver6")

// CHECK-LABEL: define{{.*}} void @test_c86_4g_m4(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 41
TEST_CPU_IS(c86_4g_m4, "c86-4g-m4")

// CHECK-LABEL: define{{.*}} void @test_c86_4g_m6(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 42
TEST_CPU_IS(c86_4g_m6, "c86-4g-m6")

// CHECK-LABEL: define{{.*}} void @test_c86_4g_m7(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 43
TEST_CPU_IS(c86_4g_m7, "c86-4g-m7")

// CHECK-LABEL: define{{.*}} void @test_c86_4g_m8(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 44
TEST_CPU_IS(c86_4g_m8, "c86-4g-m8")

// Aliases

// CHECK-LABEL: define{{.*}} void @test_emeraldrapids(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 24
TEST_CPU_IS(emeraldrapids, "emeraldrapids")

// CHECK-LABEL: define{{.*}} void @test_raptorlake(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 25
TEST_CPU_IS(raptorlake, "raptorlake")

// CHECK-LABEL: define{{.*}} void @test_meteorlake(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 25
TEST_CPU_IS(meteorlake, "meteorlake")

// CHECK-LABEL: define{{.*}} void @test_gracemont(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 25
TEST_CPU_IS(gracemont, "gracemont")

// CHECK-LABEL: define{{.*}} void @test_lunarlake(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 33
TEST_CPU_IS(lunarlake, "lunarlake")

// CHECK-LABEL: define{{.*}} void @test_wildcatlake(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// CHECK: = icmp eq i32 [[LOAD]], 34
TEST_CPU_IS(wildcatlake, "wildcatlake")
