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

// CHECK: = icmp eq i32 {{.*}}, 2
TEST_CPU_IS(core2, "core2")

// CHECK: = icmp eq i32 {{.*}}, 3
TEST_CPU_IS(corei7, "corei7")

// CHECK: = icmp eq i32 {{.*}}, 4
TEST_CPU_IS(amdfam10h, "amdfam10h")

// CHECK: = icmp eq i32 {{.*}}, 5
TEST_CPU_IS(amdfam15h, "amdfam15h")

// CHECK: = icmp eq i32 {{.*}}, 6
TEST_CPU_IS(silvermont, "silvermont")

// CHECK: = icmp eq i32 {{.*}}, 7
TEST_CPU_IS(knl, "knl")

// CHECK: = icmp eq i32 {{.*}}, 8
TEST_CPU_IS(btver1, "btver1")

// CHECK: = icmp eq i32 {{.*}}, 9
TEST_CPU_IS(btver2, "btver2")

// CHECK: = icmp eq i32 {{.*}}, 10
TEST_CPU_IS(amdfam17h, "amdfam17h")

// CHECK: = icmp eq i32 {{.*}}, 11
TEST_CPU_IS(knm, "knm")

// CHECK: = icmp eq i32 {{.*}}, 12
TEST_CPU_IS(goldmont, "goldmont")

// CHECK: = icmp eq i32 {{.*}}, 13
TEST_CPU_IS(goldmont_plus, "goldmont-plus")

// CHECK: = icmp eq i32 {{.*}}, 14
TEST_CPU_IS(tremont, "tremont")

// CHECK: = icmp eq i32 {{.*}}, 15
TEST_CPU_IS(amdfam19h, "amdfam19h")

// CHECK: = icmp eq i32 {{.*}}, 16
TEST_CPU_IS(zhaoxin_fam7h, "zhaoxin_fam7h")

// CHECK: = icmp eq i32 {{.*}}, 17
TEST_CPU_IS(sierraforest, "sierraforest")

// CHECK: = icmp eq i32 {{.*}}, 18
TEST_CPU_IS(grandridge, "grandridge")

// CHECK: = icmp eq i32 {{.*}}, 19
TEST_CPU_IS(clearwaterforest, "clearwaterforest")

// CHECK: = icmp eq i32 {{.*}}, 20
TEST_CPU_IS(amdfam1ah, "amdfam1ah")

// CHECK: = icmp eq i32 {{.*}}, 21
TEST_CPU_IS(hygonfam18h, "hygonfam18h")

// CHECK: = icmp eq i32 {{.*}}, 1
TEST_CPU_IS(atom, "atom")

// CHECK: = icmp eq i32 {{.*}}, 4
TEST_CPU_IS(amdfam10, "amdfam10")

// CHECK: = icmp eq i32 {{.*}}, 5
TEST_CPU_IS(amdfam15, "amdfam15")

// CHECK: = icmp eq i32 {{.*}}, 20
TEST_CPU_IS(amdfam1a, "amdfam1a")

// CHECK: = icmp eq i32 {{.*}}, 6
TEST_CPU_IS(slm, "slm")
