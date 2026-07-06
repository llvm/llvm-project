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

// CHECK: = icmp eq i32 {{.*}}, 2
TEST_CPU_IS(westmere, "westmere")

// CHECK: = icmp eq i32 {{.*}}, 3
TEST_CPU_IS(sandybridge, "sandybridge")

// CHECK: = icmp eq i32 {{.*}}, 4
TEST_CPU_IS(barcelona, "barcelona")

// CHECK: = icmp eq i32 {{.*}}, 5
TEST_CPU_IS(shanghai, "shanghai")

// CHECK: = icmp eq i32 {{.*}}, 6
TEST_CPU_IS(istanbul, "istanbul")

// CHECK: = icmp eq i32 {{.*}}, 7
TEST_CPU_IS(bdver1, "bdver1")

// CHECK: = icmp eq i32 {{.*}}, 8
TEST_CPU_IS(bdver2, "bdver2")

// CHECK: = icmp eq i32 {{.*}}, 9
TEST_CPU_IS(bdver3, "bdver3")

// CHECK: = icmp eq i32 {{.*}}, 10
TEST_CPU_IS(bdver4, "bdver4")

// CHECK: = icmp eq i32 {{.*}}, 11
TEST_CPU_IS(znver1, "znver1")

// CHECK: = icmp eq i32 {{.*}}, 12
TEST_CPU_IS(ivybridge, "ivybridge")

// CHECK: = icmp eq i32 {{.*}}, 13
TEST_CPU_IS(haswell, "haswell")

// CHECK: = icmp eq i32 {{.*}}, 14
TEST_CPU_IS(broadwell, "broadwell")

// CHECK: = icmp eq i32 {{.*}}, 15
TEST_CPU_IS(skylake, "skylake")

// CHECK: = icmp eq i32 {{.*}}, 16
TEST_CPU_IS(skylake_avx512, "skylake-avx512")

// CHECK: = icmp eq i32 {{.*}}, 17
TEST_CPU_IS(cannonlake, "cannonlake")

// CHECK: = icmp eq i32 {{.*}}, 18
TEST_CPU_IS(icelake_client, "icelake-client")

// CHECK: = icmp eq i32 {{.*}}, 19
TEST_CPU_IS(icelake_server, "icelake-server")

// CHECK: = icmp eq i32 {{.*}}, 20
TEST_CPU_IS(znver2, "znver2")

// CHECK: = icmp eq i32 {{.*}}, 21
TEST_CPU_IS(cascadelake, "cascadelake")

// CHECK: = icmp eq i32 {{.*}}, 22
TEST_CPU_IS(tigerlake, "tigerlake")

// CHECK: = icmp eq i32 {{.*}}, 23
TEST_CPU_IS(cooperlake, "cooperlake")

// CHECK: = icmp eq i32 {{.*}}, 24
TEST_CPU_IS(sapphirerapids, "sapphirerapids")

// CHECK: = icmp eq i32 {{.*}}, 25
TEST_CPU_IS(alderlake, "alderlake")

// CHECK: = icmp eq i32 {{.*}}, 26
TEST_CPU_IS(znver3, "znver3")

// CHECK: = icmp eq i32 {{.*}}, 27
TEST_CPU_IS(rocketlake, "rocketlake")

// CHECK: = icmp eq i32 {{.*}}, 28
TEST_CPU_IS(zhaoxin_fam7h_lujiazui, "zhaoxin_fam7h_lujiazui")

// CHECK: = icmp eq i32 {{.*}}, 29
TEST_CPU_IS(znver4, "znver4")

// CHECK: = icmp eq i32 {{.*}}, 30
TEST_CPU_IS(graniterapids, "graniterapids")

// CHECK: = icmp eq i32 {{.*}}, 31
TEST_CPU_IS(graniterapids_d, "graniterapids-d")

// CHECK: = icmp eq i32 {{.*}}, 32
TEST_CPU_IS(arrowlake, "arrowlake")

// CHECK: = icmp eq i32 {{.*}}, 33
TEST_CPU_IS(arrowlake_s, "arrowlake-s")

// CHECK: = icmp eq i32 {{.*}}, 34
TEST_CPU_IS(pantherlake, "pantherlake")

// CHECK: = icmp eq i32 {{.*}}, 36
TEST_CPU_IS(znver5, "znver5")

// CHECK: = icmp eq i32 {{.*}}, 38
TEST_CPU_IS(diamondrapids, "diamondrapids")

// CHECK: = icmp eq i32 {{.*}}, 39
TEST_CPU_IS(novalake, "novalake")

// CHECK: = icmp eq i32 {{.*}}, 40
TEST_CPU_IS(znver6, "znver6")

// CHECK: = icmp eq i32 {{.*}}, 41
TEST_CPU_IS(c86_4g_m4, "c86-4g-m4")

// CHECK: = icmp eq i32 {{.*}}, 42
TEST_CPU_IS(c86_4g_m6, "c86-4g-m6")

// CHECK: = icmp eq i32 {{.*}}, 43
TEST_CPU_IS(c86_4g_m7, "c86-4g-m7")

// CHECK: = icmp eq i32 {{.*}}, 44
TEST_CPU_IS(c86_4g_m8, "c86-4g-m8")

// CHECK: = icmp eq i32 {{.*}}, 25
TEST_CPU_IS(raptorlake, "raptorlake")

// CHECK: = icmp eq i32 {{.*}}, 25
TEST_CPU_IS(meteorlake, "meteorlake")

// CHECK: = icmp eq i32 {{.*}}, 24
TEST_CPU_IS(emeraldrapids, "emeraldrapids")

// CHECK: = icmp eq i32 {{.*}}, 33
TEST_CPU_IS(lunarlake, "lunarlake")

// CHECK: = icmp eq i32 {{.*}}, 25
TEST_CPU_IS(gracemont, "gracemont")

// CHECK: = icmp eq i32 {{.*}}, 34
TEST_CPU_IS(wildcatlake, "wildcatlake")
