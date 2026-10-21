// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm < %s | FileCheck %s

// Test that __builtin_cpu_is emits the correct ABI value for every CPU type,
// in llvm/include/llvm/TargetParser/X86TargetParser.def.
extern void a(const char *);

// CHECK: @__cpu_model = external dso_local global { i32, i32, i32, [1 x i32] }

#define TEST_CPU_IS(NAME, STR)                                                 \
  void test_##NAME(void) {                                                     \
    if (__builtin_cpu_is(STR))                                                 \
      a(STR);                                                                  \
  }

// CHECK-LABEL: define{{.*}} void @test_bonnell(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// CHECK: = icmp eq i32 [[LOAD]], 1
TEST_CPU_IS(bonnell, "bonnell")

// CHECK-LABEL: define{{.*}} void @test_core2(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// CHECK: = icmp eq i32 [[LOAD]], 2
TEST_CPU_IS(core2, "core2")

// CHECK-LABEL: define{{.*}} void @test_corei7(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// CHECK: = icmp eq i32 [[LOAD]], 3
TEST_CPU_IS(corei7, "corei7")

// CHECK-LABEL: define{{.*}} void @test_amdfam10h(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// CHECK: = icmp eq i32 [[LOAD]], 4
TEST_CPU_IS(amdfam10h, "amdfam10h")

// CHECK-LABEL: define{{.*}} void @test_amdfam15h(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// CHECK: = icmp eq i32 [[LOAD]], 5
TEST_CPU_IS(amdfam15h, "amdfam15h")

// CHECK-LABEL: define{{.*}} void @test_silvermont(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// CHECK: = icmp eq i32 [[LOAD]], 6
TEST_CPU_IS(silvermont, "silvermont")

// CHECK-LABEL: define{{.*}} void @test_knl(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// CHECK: = icmp eq i32 [[LOAD]], 7
TEST_CPU_IS(knl, "knl")

// CHECK-LABEL: define{{.*}} void @test_btver1(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// CHECK: = icmp eq i32 [[LOAD]], 8
TEST_CPU_IS(btver1, "btver1")

// CHECK-LABEL: define{{.*}} void @test_btver2(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// CHECK: = icmp eq i32 [[LOAD]], 9
TEST_CPU_IS(btver2, "btver2")

// CHECK-LABEL: define{{.*}} void @test_amdfam17h(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// CHECK: = icmp eq i32 [[LOAD]], 10
TEST_CPU_IS(amdfam17h, "amdfam17h")

// CHECK-LABEL: define{{.*}} void @test_knm(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// CHECK: = icmp eq i32 [[LOAD]], 11
TEST_CPU_IS(knm, "knm")

// CHECK-LABEL: define{{.*}} void @test_goldmont(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// CHECK: = icmp eq i32 [[LOAD]], 12
TEST_CPU_IS(goldmont, "goldmont")

// CHECK-LABEL: define{{.*}} void @test_goldmont_plus(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// CHECK: = icmp eq i32 [[LOAD]], 13
TEST_CPU_IS(goldmont_plus, "goldmont-plus")

// CHECK-LABEL: define{{.*}} void @test_tremont(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// CHECK: = icmp eq i32 [[LOAD]], 14
TEST_CPU_IS(tremont, "tremont")

// CHECK-LABEL: define{{.*}} void @test_amdfam19h(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// CHECK: = icmp eq i32 [[LOAD]], 15
TEST_CPU_IS(amdfam19h, "amdfam19h")

// CHECK-LABEL: define{{.*}} void @test_zhaoxin_fam7h(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// CHECK: = icmp eq i32 [[LOAD]], 16
TEST_CPU_IS(zhaoxin_fam7h, "zhaoxin_fam7h")

// CHECK-LABEL: define{{.*}} void @test_sierraforest(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// CHECK: = icmp eq i32 [[LOAD]], 17
TEST_CPU_IS(sierraforest, "sierraforest")

// CHECK-LABEL: define{{.*}} void @test_grandridge(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// CHECK: = icmp eq i32 [[LOAD]], 18
TEST_CPU_IS(grandridge, "grandridge")

// CHECK-LABEL: define{{.*}} void @test_clearwaterforest(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// CHECK: = icmp eq i32 [[LOAD]], 19
TEST_CPU_IS(clearwaterforest, "clearwaterforest")

// CHECK-LABEL: define{{.*}} void @test_amdfam1ah(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// CHECK: = icmp eq i32 [[LOAD]], 20
TEST_CPU_IS(amdfam1ah, "amdfam1ah")

// CHECK-LABEL: define{{.*}} void @test_hygonfam18h(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// CHECK: = icmp eq i32 [[LOAD]], 21
TEST_CPU_IS(hygonfam18h, "hygonfam18h")

// Aliases

// CHECK-LABEL: define{{.*}} void @test_atom(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// CHECK: = icmp eq i32 [[LOAD]], 1
TEST_CPU_IS(atom, "atom")

// CHECK-LABEL: define{{.*}} void @test_amdfam10(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// CHECK: = icmp eq i32 [[LOAD]], 4
TEST_CPU_IS(amdfam10, "amdfam10")

// CHECK-LABEL: define{{.*}} void @test_amdfam15(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// CHECK: = icmp eq i32 [[LOAD]], 5
TEST_CPU_IS(amdfam15, "amdfam15")

// CHECK-LABEL: define{{.*}} void @test_slm(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// CHECK: = icmp eq i32 [[LOAD]], 6
TEST_CPU_IS(slm, "slm")

// CHECK-LABEL: define{{.*}} void @test_amdfam1a(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// CHECK: = icmp eq i32 [[LOAD]], 20
TEST_CPU_IS(amdfam1a, "amdfam1a")
