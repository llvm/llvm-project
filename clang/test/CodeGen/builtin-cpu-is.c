// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm < %s| FileCheck %s

// Test that __builtin_cpu_is emits the correct ABI value and field offset
// for every vendor, cpu type, and cpu subtype (including aliases) listed in
// llvm/include/llvm/TargetParser/X86TargetParser.def. These values are an ABI
// contract shared with compiler-rt/libgcc.
extern void a(const char *);

// CHECK: @__cpu_model = external dso_local global { i32, i32, i32, [1 x i32] }

// Vendors (field offset 0).
void test_intel(void) {
  if (__builtin_cpu_is("intel"))
    a("intel");

  // CHECK-LABEL: define{{.*}} void @test_intel(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_model
  // CHECK: = icmp eq i32 [[LOAD]], 1
}

void test_amd(void) {
  if (__builtin_cpu_is("amd"))
    a("amd");

  // CHECK-LABEL: define{{.*}} void @test_amd(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_model
  // CHECK: = icmp eq i32 [[LOAD]], 2
}

void test_other(void) {
  if (__builtin_cpu_is("other"))
    a("other");

  // CHECK-LABEL: define{{.*}} void @test_other(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_model
  // CHECK: = icmp eq i32 [[LOAD]], 4
}

// CPU types (field offset 4).
void test_bonnell(void) {
  if (__builtin_cpu_is("bonnell"))
    a("bonnell");

  // CHECK-LABEL: define{{.*}} void @test_bonnell(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
  // CHECK: = icmp eq i32 [[LOAD]], 1
}

void test_core2(void) {
  if (__builtin_cpu_is("core2"))
    a("core2");

  // CHECK-LABEL: define{{.*}} void @test_core2(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
  // CHECK: = icmp eq i32 [[LOAD]], 2
}

void test_corei7(void) {
  if (__builtin_cpu_is("corei7"))
    a("corei7");

  // CHECK-LABEL: define{{.*}} void @test_corei7(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
  // CHECK: = icmp eq i32 [[LOAD]], 3
}

void test_amdfam10h(void) {
  if (__builtin_cpu_is("amdfam10h"))
    a("amdfam10h");

  // CHECK-LABEL: define{{.*}} void @test_amdfam10h(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
  // CHECK: = icmp eq i32 [[LOAD]], 4
}

void test_amdfam15h(void) {
  if (__builtin_cpu_is("amdfam15h"))
    a("amdfam15h");

  // CHECK-LABEL: define{{.*}} void @test_amdfam15h(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
  // CHECK: = icmp eq i32 [[LOAD]], 5
}

void test_silvermont(void) {
  if (__builtin_cpu_is("silvermont"))
    a("silvermont");

  // CHECK-LABEL: define{{.*}} void @test_silvermont(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
  // CHECK: = icmp eq i32 [[LOAD]], 6
}

void test_knl(void) {
  if (__builtin_cpu_is("knl"))
    a("knl");

  // CHECK-LABEL: define{{.*}} void @test_knl(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
  // CHECK: = icmp eq i32 [[LOAD]], 7
}

void test_btver1(void) {
  if (__builtin_cpu_is("btver1"))
    a("btver1");

  // CHECK-LABEL: define{{.*}} void @test_btver1(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
  // CHECK: = icmp eq i32 [[LOAD]], 8
}

void test_btver2(void) {
  if (__builtin_cpu_is("btver2"))
    a("btver2");

  // CHECK-LABEL: define{{.*}} void @test_btver2(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
  // CHECK: = icmp eq i32 [[LOAD]], 9
}

void test_amdfam17h(void) {
  if (__builtin_cpu_is("amdfam17h"))
    a("amdfam17h");

  // CHECK-LABEL: define{{.*}} void @test_amdfam17h(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
  // CHECK: = icmp eq i32 [[LOAD]], 10
}

void test_knm(void) {
  if (__builtin_cpu_is("knm"))
    a("knm");

  // CHECK-LABEL: define{{.*}} void @test_knm(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
  // CHECK: = icmp eq i32 [[LOAD]], 11
}

void test_goldmont(void) {
  if (__builtin_cpu_is("goldmont"))
    a("goldmont");

  // CHECK-LABEL: define{{.*}} void @test_goldmont(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
  // CHECK: = icmp eq i32 [[LOAD]], 12
}

void test_goldmont_plus(void) {
  if (__builtin_cpu_is("goldmont-plus"))
    a("goldmont-plus");

  // CHECK-LABEL: define{{.*}} void @test_goldmont_plus(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
  // CHECK: = icmp eq i32 [[LOAD]], 13
}

void test_tremont(void) {
  if (__builtin_cpu_is("tremont"))
    a("tremont");

  // CHECK-LABEL: define{{.*}} void @test_tremont(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
  // CHECK: = icmp eq i32 [[LOAD]], 14
}

void test_amdfam19h(void) {
  if (__builtin_cpu_is("amdfam19h"))
    a("amdfam19h");

  // CHECK-LABEL: define{{.*}} void @test_amdfam19h(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
  // CHECK: = icmp eq i32 [[LOAD]], 15
}

void test_zhaoxin_fam7h(void) {
  if (__builtin_cpu_is("zhaoxin_fam7h"))
    a("zhaoxin_fam7h");

  // CHECK-LABEL: define{{.*}} void @test_zhaoxin_fam7h(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
  // CHECK: = icmp eq i32 [[LOAD]], 16
}

void test_sierraforest(void) {
  if (__builtin_cpu_is("sierraforest"))
    a("sierraforest");

  // CHECK-LABEL: define{{.*}} void @test_sierraforest(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
  // CHECK: = icmp eq i32 [[LOAD]], 17
}

void test_grandridge(void) {
  if (__builtin_cpu_is("grandridge"))
    a("grandridge");

  // CHECK-LABEL: define{{.*}} void @test_grandridge(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
  // CHECK: = icmp eq i32 [[LOAD]], 18
}

void test_clearwaterforest(void) {
  if (__builtin_cpu_is("clearwaterforest"))
    a("clearwaterforest");

  // CHECK-LABEL: define{{.*}} void @test_clearwaterforest(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
  // CHECK: = icmp eq i32 [[LOAD]], 19
}

void test_amdfam1ah(void) {
  if (__builtin_cpu_is("amdfam1ah"))
    a("amdfam1ah");

  // CHECK-LABEL: define{{.*}} void @test_amdfam1ah(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
  // CHECK: = icmp eq i32 [[LOAD]], 20
}

void test_hygonfam18h(void) {
  if (__builtin_cpu_is("hygonfam18h"))
    a("hygonfam18h");

  // CHECK-LABEL: define{{.*}} void @test_hygonfam18h(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
  // CHECK: = icmp eq i32 [[LOAD]], 21
}

void test_atom(void) {
  if (__builtin_cpu_is("atom"))
    a("atom");

  // CHECK-LABEL: define{{.*}} void @test_atom(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
  // CHECK: = icmp eq i32 [[LOAD]], 1
}

void test_amdfam10(void) {
  if (__builtin_cpu_is("amdfam10"))
    a("amdfam10");

  // CHECK-LABEL: define{{.*}} void @test_amdfam10(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
  // CHECK: = icmp eq i32 [[LOAD]], 4
}

void test_amdfam15(void) {
  if (__builtin_cpu_is("amdfam15"))
    a("amdfam15");

  // CHECK-LABEL: define{{.*}} void @test_amdfam15(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
  // CHECK: = icmp eq i32 [[LOAD]], 5
}

void test_amdfam1a(void) {
  if (__builtin_cpu_is("amdfam1a"))
    a("amdfam1a");

  // CHECK-LABEL: define{{.*}} void @test_amdfam1a(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
  // CHECK: = icmp eq i32 [[LOAD]], 20
}

void test_slm(void) {
  if (__builtin_cpu_is("slm"))
    a("slm");

  // CHECK-LABEL: define{{.*}} void @test_slm(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
  // CHECK: = icmp eq i32 [[LOAD]], 6
}

// CPU subtypes (field offset 8).
void test_nehalem(void) {
  if (__builtin_cpu_is("nehalem"))
    a("nehalem");

  // CHECK-LABEL: define{{.*}} void @test_nehalem(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 1
}

void test_westmere(void) {
  if (__builtin_cpu_is("westmere"))
    a("westmere");

  // CHECK-LABEL: define{{.*}} void @test_westmere(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 2
}

void test_sandybridge(void) {
  if (__builtin_cpu_is("sandybridge"))
    a("sandybridge");

  // CHECK-LABEL: define{{.*}} void @test_sandybridge(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 3
}

void test_barcelona(void) {
  if (__builtin_cpu_is("barcelona"))
    a("barcelona");

  // CHECK-LABEL: define{{.*}} void @test_barcelona(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 4
}

void test_shanghai(void) {
  if (__builtin_cpu_is("shanghai"))
    a("shanghai");

  // CHECK-LABEL: define{{.*}} void @test_shanghai(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 5
}

void test_istanbul(void) {
  if (__builtin_cpu_is("istanbul"))
    a("istanbul");

  // CHECK-LABEL: define{{.*}} void @test_istanbul(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 6
}

void test_bdver1(void) {
  if (__builtin_cpu_is("bdver1"))
    a("bdver1");

  // CHECK-LABEL: define{{.*}} void @test_bdver1(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 7
}

void test_bdver2(void) {
  if (__builtin_cpu_is("bdver2"))
    a("bdver2");

  // CHECK-LABEL: define{{.*}} void @test_bdver2(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 8
}

void test_bdver3(void) {
  if (__builtin_cpu_is("bdver3"))
    a("bdver3");

  // CHECK-LABEL: define{{.*}} void @test_bdver3(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 9
}

void test_bdver4(void) {
  if (__builtin_cpu_is("bdver4"))
    a("bdver4");

  // CHECK-LABEL: define{{.*}} void @test_bdver4(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 10
}

void test_znver1(void) {
  if (__builtin_cpu_is("znver1"))
    a("znver1");

  // CHECK-LABEL: define{{.*}} void @test_znver1(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 11
}

void test_ivybridge(void) {
  if (__builtin_cpu_is("ivybridge"))
    a("ivybridge");

  // CHECK-LABEL: define{{.*}} void @test_ivybridge(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 12
}

void test_haswell(void) {
  if (__builtin_cpu_is("haswell"))
    a("haswell");

  // CHECK-LABEL: define{{.*}} void @test_haswell(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 13
}

void test_broadwell(void) {
  if (__builtin_cpu_is("broadwell"))
    a("broadwell");

  // CHECK-LABEL: define{{.*}} void @test_broadwell(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 14
}

void test_skylake(void) {
  if (__builtin_cpu_is("skylake"))
    a("skylake");

  // CHECK-LABEL: define{{.*}} void @test_skylake(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 15
}

void test_skylake_avx512(void) {
  if (__builtin_cpu_is("skylake-avx512"))
    a("skylake-avx512");

  // CHECK-LABEL: define{{.*}} void @test_skylake_avx512(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 16
}

void test_cannonlake(void) {
  if (__builtin_cpu_is("cannonlake"))
    a("cannonlake");

  // CHECK-LABEL: define{{.*}} void @test_cannonlake(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 17
}

void test_icelake_client(void) {
  if (__builtin_cpu_is("icelake-client"))
    a("icelake-client");

  // CHECK-LABEL: define{{.*}} void @test_icelake_client(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 18
}

void test_icelake_server(void) {
  if (__builtin_cpu_is("icelake-server"))
    a("icelake-server");

  // CHECK-LABEL: define{{.*}} void @test_icelake_server(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 19
}

void test_znver2(void) {
  if (__builtin_cpu_is("znver2"))
    a("znver2");

  // CHECK-LABEL: define{{.*}} void @test_znver2(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 20
}

void test_cascadelake(void) {
  if (__builtin_cpu_is("cascadelake"))
    a("cascadelake");

  // CHECK-LABEL: define{{.*}} void @test_cascadelake(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 21
}

void test_tigerlake(void) {
  if (__builtin_cpu_is("tigerlake"))
    a("tigerlake");

  // CHECK-LABEL: define{{.*}} void @test_tigerlake(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 22
}

void test_cooperlake(void) {
  if (__builtin_cpu_is("cooperlake"))
    a("cooperlake");

  // CHECK-LABEL: define{{.*}} void @test_cooperlake(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 23
}

void test_sapphirerapids(void) {
  if (__builtin_cpu_is("sapphirerapids"))
    a("sapphirerapids");

  // CHECK-LABEL: define{{.*}} void @test_sapphirerapids(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 24
}

void test_alderlake(void) {
  if (__builtin_cpu_is("alderlake"))
    a("alderlake");

  // CHECK-LABEL: define{{.*}} void @test_alderlake(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 25
}

void test_znver3(void) {
  if (__builtin_cpu_is("znver3"))
    a("znver3");

  // CHECK-LABEL: define{{.*}} void @test_znver3(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 26
}

void test_rocketlake(void) {
  if (__builtin_cpu_is("rocketlake"))
    a("rocketlake");

  // CHECK-LABEL: define{{.*}} void @test_rocketlake(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 27
}

void test_zhaoxin_fam7h_lujiazui(void) {
  if (__builtin_cpu_is("zhaoxin_fam7h_lujiazui"))
    a("zhaoxin_fam7h_lujiazui");

  // CHECK-LABEL: define{{.*}} void @test_zhaoxin_fam7h_lujiazui(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 28
}

void test_znver4(void) {
  if (__builtin_cpu_is("znver4"))
    a("znver4");

  // CHECK-LABEL: define{{.*}} void @test_znver4(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 29
}

void test_graniterapids(void) {
  if (__builtin_cpu_is("graniterapids"))
    a("graniterapids");

  // CHECK-LABEL: define{{.*}} void @test_graniterapids(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 30
}

void test_graniterapids_d(void) {
  if (__builtin_cpu_is("graniterapids-d"))
    a("graniterapids-d");

  // CHECK-LABEL: define{{.*}} void @test_graniterapids_d(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 31
}

void test_arrowlake(void) {
  if (__builtin_cpu_is("arrowlake"))
    a("arrowlake");

  // CHECK-LABEL: define{{.*}} void @test_arrowlake(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 32
}

void test_arrowlake_s(void) {
  if (__builtin_cpu_is("arrowlake-s"))
    a("arrowlake-s");

  // CHECK-LABEL: define{{.*}} void @test_arrowlake_s(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 33
}

void test_pantherlake(void) {
  if (__builtin_cpu_is("pantherlake"))
    a("pantherlake");

  // CHECK-LABEL: define{{.*}} void @test_pantherlake(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 34
}

void test_znver5(void) {
  if (__builtin_cpu_is("znver5"))
    a("znver5");

  // CHECK-LABEL: define{{.*}} void @test_znver5(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 36
}

void test_diamondrapids(void) {
  if (__builtin_cpu_is("diamondrapids"))
    a("diamondrapids");

  // CHECK-LABEL: define{{.*}} void @test_diamondrapids(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 38
}

void test_novalake(void) {
  if (__builtin_cpu_is("novalake"))
    a("novalake");

  // CHECK-LABEL: define{{.*}} void @test_novalake(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 39
}

void test_znver6(void) {
  if (__builtin_cpu_is("znver6"))
    a("znver6");

  // CHECK-LABEL: define{{.*}} void @test_znver6(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 40
}

void test_c86_4g_m4(void) {
  if (__builtin_cpu_is("c86-4g-m4"))
    a("c86-4g-m4");

  // CHECK-LABEL: define{{.*}} void @test_c86_4g_m4(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 41
}

void test_c86_4g_m6(void) {
  if (__builtin_cpu_is("c86-4g-m6"))
    a("c86-4g-m6");

  // CHECK-LABEL: define{{.*}} void @test_c86_4g_m6(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 42
}

void test_c86_4g_m7(void) {
  if (__builtin_cpu_is("c86-4g-m7"))
    a("c86-4g-m7");

  // CHECK-LABEL: define{{.*}} void @test_c86_4g_m7(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 43
}

void test_c86_4g_m8(void) {
  if (__builtin_cpu_is("c86-4g-m8"))
    a("c86-4g-m8");

  // CHECK-LABEL: define{{.*}} void @test_c86_4g_m8(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 44
}

void test_raptorlake(void) {
  if (__builtin_cpu_is("raptorlake"))
    a("raptorlake");

  // CHECK-LABEL: define{{.*}} void @test_raptorlake(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 25
}

void test_meteorlake(void) {
  if (__builtin_cpu_is("meteorlake"))
    a("meteorlake");

  // CHECK-LABEL: define{{.*}} void @test_meteorlake(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 25
}

void test_emeraldrapids(void) {
  if (__builtin_cpu_is("emeraldrapids"))
    a("emeraldrapids");

  // CHECK-LABEL: define{{.*}} void @test_emeraldrapids(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 24
}

void test_lunarlake(void) {
  if (__builtin_cpu_is("lunarlake"))
    a("lunarlake");

  // CHECK-LABEL: define{{.*}} void @test_lunarlake(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 33
}

void test_gracemont(void) {
  if (__builtin_cpu_is("gracemont"))
    a("gracemont");

  // CHECK-LABEL: define{{.*}} void @test_gracemont(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 25
}

void test_wildcatlake(void) {
  if (__builtin_cpu_is("wildcatlake"))
    a("wildcatlake");

  // CHECK-LABEL: define{{.*}} void @test_wildcatlake(
  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 34
}
