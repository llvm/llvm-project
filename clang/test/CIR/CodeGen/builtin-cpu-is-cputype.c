// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

// Test that __builtin_cpu_is emits the correct ABI value for every CPU type,
// in llvm/include/llvm/TargetParser/X86TargetParser.def.
extern void a(const char *);

// CIR: ![[MODEL_TY:.*]] = !cir.struct<{!u32i, !u32i, !u32i, !cir.array<!u32i x 1>}>
// CIR: cir.global "private" external dso_local @__cpu_model : ![[MODEL_TY]]
// LLVM: @__cpu_model = external dso_local global { i32, i32, i32, [1 x i32] }

#define TEST_CPU_IS(NAME, STR)                                                 \
  void test_##NAME(void) {                                                     \
    if (__builtin_cpu_is(STR))                                                 \
      a(STR);                                                                  \
  }

// CIR-LABEL: cir.func{{.*}}@test_bonnell(
// CIR: %[[GET_MODEL:.*]] = cir.get_global @__cpu_model : !cir.ptr<![[MODEL_TY]]>
// CIR: %[[GET_MEM_PTR:.*]] = cir.get_member %[[GET_MODEL]][1] {name = "__cpu_type"} : !cir.ptr<![[MODEL_TY]]> -> !cir.ptr<!u32i>
// CIR: %[[LOAD_TYPE:.*]]  = cir.load {{.*}}%[[GET_MEM_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[MASK:.*]] = cir.const #cir.int<1> : !u32i
// CIR: cir.cmp eq %[[LOAD_TYPE]], %[[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_bonnell(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 1
TEST_CPU_IS(bonnell, "bonnell")

// CIR-LABEL: cir.func{{.*}}@test_core2(
// CIR: %[[GET_MODEL:.*]] = cir.get_global @__cpu_model : !cir.ptr<![[MODEL_TY]]>
// CIR: %[[GET_MEM_PTR:.*]] = cir.get_member %[[GET_MODEL]][1] {name = "__cpu_type"} : !cir.ptr<![[MODEL_TY]]> -> !cir.ptr<!u32i>
// CIR: %[[LOAD_TYPE:.*]]  = cir.load {{.*}}%[[GET_MEM_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[MASK:.*]] = cir.const #cir.int<2> : !u32i
// CIR: cir.cmp eq %[[LOAD_TYPE]], %[[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_core2(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 2
TEST_CPU_IS(core2, "core2")

// CIR-LABEL: cir.func{{.*}}@test_corei7(
// CIR: %[[GET_MODEL:.*]] = cir.get_global @__cpu_model : !cir.ptr<![[MODEL_TY]]>
// CIR: %[[GET_MEM_PTR:.*]] = cir.get_member %[[GET_MODEL]][1] {name = "__cpu_type"} : !cir.ptr<![[MODEL_TY]]> -> !cir.ptr<!u32i>
// CIR: %[[LOAD_TYPE:.*]]  = cir.load {{.*}}%[[GET_MEM_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[MASK:.*]] = cir.const #cir.int<3> : !u32i
// CIR: cir.cmp eq %[[LOAD_TYPE]], %[[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_corei7(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 3
TEST_CPU_IS(corei7, "corei7")

// CIR-LABEL: cir.func{{.*}}@test_amdfam10h(
// CIR: %[[GET_MODEL:.*]] = cir.get_global @__cpu_model : !cir.ptr<![[MODEL_TY]]>
// CIR: %[[GET_MEM_PTR:.*]] = cir.get_member %[[GET_MODEL]][1] {name = "__cpu_type"} : !cir.ptr<![[MODEL_TY]]> -> !cir.ptr<!u32i>
// CIR: %[[LOAD_TYPE:.*]]  = cir.load {{.*}}%[[GET_MEM_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[MASK:.*]] = cir.const #cir.int<4> : !u32i
// CIR: cir.cmp eq %[[LOAD_TYPE]], %[[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_amdfam10h(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 4
TEST_CPU_IS(amdfam10h, "amdfam10h")

// CIR-LABEL: cir.func{{.*}}@test_amdfam15h(
// CIR: %[[GET_MODEL:.*]] = cir.get_global @__cpu_model : !cir.ptr<![[MODEL_TY]]>
// CIR: %[[GET_MEM_PTR:.*]] = cir.get_member %[[GET_MODEL]][1] {name = "__cpu_type"} : !cir.ptr<![[MODEL_TY]]> -> !cir.ptr<!u32i>
// CIR: %[[LOAD_TYPE:.*]]  = cir.load {{.*}}%[[GET_MEM_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[MASK:.*]] = cir.const #cir.int<5> : !u32i
// CIR: cir.cmp eq %[[LOAD_TYPE]], %[[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_amdfam15h(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 5
TEST_CPU_IS(amdfam15h, "amdfam15h")

// CIR-LABEL: cir.func{{.*}}@test_silvermont(
// CIR: %[[GET_MODEL:.*]] = cir.get_global @__cpu_model : !cir.ptr<![[MODEL_TY]]>
// CIR: %[[GET_MEM_PTR:.*]] = cir.get_member %[[GET_MODEL]][1] {name = "__cpu_type"} : !cir.ptr<![[MODEL_TY]]> -> !cir.ptr<!u32i>
// CIR: %[[LOAD_TYPE:.*]]  = cir.load {{.*}}%[[GET_MEM_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[MASK:.*]] = cir.const #cir.int<6> : !u32i
// CIR: cir.cmp eq %[[LOAD_TYPE]], %[[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_silvermont(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 6
TEST_CPU_IS(silvermont, "silvermont")

// CIR-LABEL: cir.func{{.*}}@test_knl(
// CIR: %[[GET_MODEL:.*]] = cir.get_global @__cpu_model : !cir.ptr<![[MODEL_TY]]>
// CIR: %[[GET_MEM_PTR:.*]] = cir.get_member %[[GET_MODEL]][1] {name = "__cpu_type"} : !cir.ptr<![[MODEL_TY]]> -> !cir.ptr<!u32i>
// CIR: %[[LOAD_TYPE:.*]]  = cir.load {{.*}}%[[GET_MEM_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[MASK:.*]] = cir.const #cir.int<7> : !u32i
// CIR: cir.cmp eq %[[LOAD_TYPE]], %[[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_knl(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 7
TEST_CPU_IS(knl, "knl")

// CIR-LABEL: cir.func{{.*}}@test_btver1(
// CIR: %[[GET_MODEL:.*]] = cir.get_global @__cpu_model : !cir.ptr<![[MODEL_TY]]>
// CIR: %[[GET_MEM_PTR:.*]] = cir.get_member %[[GET_MODEL]][1] {name = "__cpu_type"} : !cir.ptr<![[MODEL_TY]]> -> !cir.ptr<!u32i>
// CIR: %[[LOAD_TYPE:.*]]  = cir.load {{.*}}%[[GET_MEM_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[MASK:.*]] = cir.const #cir.int<8> : !u32i
// CIR: cir.cmp eq %[[LOAD_TYPE]], %[[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_btver1(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 8
TEST_CPU_IS(btver1, "btver1")

// CIR-LABEL: cir.func{{.*}}@test_btver2(
// CIR: %[[GET_MODEL:.*]] = cir.get_global @__cpu_model : !cir.ptr<![[MODEL_TY]]>
// CIR: %[[GET_MEM_PTR:.*]] = cir.get_member %[[GET_MODEL]][1] {name = "__cpu_type"} : !cir.ptr<![[MODEL_TY]]> -> !cir.ptr<!u32i>
// CIR: %[[LOAD_TYPE:.*]]  = cir.load {{.*}}%[[GET_MEM_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[MASK:.*]] = cir.const #cir.int<9> : !u32i
// CIR: cir.cmp eq %[[LOAD_TYPE]], %[[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_btver2(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 9
TEST_CPU_IS(btver2, "btver2")

// CIR-LABEL: cir.func{{.*}}@test_amdfam17h(
// CIR: %[[GET_MODEL:.*]] = cir.get_global @__cpu_model : !cir.ptr<![[MODEL_TY]]>
// CIR: %[[GET_MEM_PTR:.*]] = cir.get_member %[[GET_MODEL]][1] {name = "__cpu_type"} : !cir.ptr<![[MODEL_TY]]> -> !cir.ptr<!u32i>
// CIR: %[[LOAD_TYPE:.*]]  = cir.load {{.*}}%[[GET_MEM_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[MASK:.*]] = cir.const #cir.int<10> : !u32i
// CIR: cir.cmp eq %[[LOAD_TYPE]], %[[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_amdfam17h(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 10
TEST_CPU_IS(amdfam17h, "amdfam17h")

// CIR-LABEL: cir.func{{.*}}@test_knm(
// CIR: %[[GET_MODEL:.*]] = cir.get_global @__cpu_model : !cir.ptr<![[MODEL_TY]]>
// CIR: %[[GET_MEM_PTR:.*]] = cir.get_member %[[GET_MODEL]][1] {name = "__cpu_type"} : !cir.ptr<![[MODEL_TY]]> -> !cir.ptr<!u32i>
// CIR: %[[LOAD_TYPE:.*]]  = cir.load {{.*}}%[[GET_MEM_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[MASK:.*]] = cir.const #cir.int<11> : !u32i
// CIR: cir.cmp eq %[[LOAD_TYPE]], %[[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_knm(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 11
TEST_CPU_IS(knm, "knm")

// CIR-LABEL: cir.func{{.*}}@test_goldmont(
// CIR: %[[GET_MODEL:.*]] = cir.get_global @__cpu_model : !cir.ptr<![[MODEL_TY]]>
// CIR: %[[GET_MEM_PTR:.*]] = cir.get_member %[[GET_MODEL]][1] {name = "__cpu_type"} : !cir.ptr<![[MODEL_TY]]> -> !cir.ptr<!u32i>
// CIR: %[[LOAD_TYPE:.*]]  = cir.load {{.*}}%[[GET_MEM_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[MASK:.*]] = cir.const #cir.int<12> : !u32i
// CIR: cir.cmp eq %[[LOAD_TYPE]], %[[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_goldmont(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 12
TEST_CPU_IS(goldmont, "goldmont")

// CIR-LABEL: cir.func{{.*}}@test_goldmont_plus(
// CIR: %[[GET_MODEL:.*]] = cir.get_global @__cpu_model : !cir.ptr<![[MODEL_TY]]>
// CIR: %[[GET_MEM_PTR:.*]] = cir.get_member %[[GET_MODEL]][1] {name = "__cpu_type"} : !cir.ptr<![[MODEL_TY]]> -> !cir.ptr<!u32i>
// CIR: %[[LOAD_TYPE:.*]]  = cir.load {{.*}}%[[GET_MEM_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[MASK:.*]] = cir.const #cir.int<13> : !u32i
// CIR: cir.cmp eq %[[LOAD_TYPE]], %[[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_goldmont_plus(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 13
TEST_CPU_IS(goldmont_plus, "goldmont-plus")

// CIR-LABEL: cir.func{{.*}}@test_tremont(
// CIR: %[[GET_MODEL:.*]] = cir.get_global @__cpu_model : !cir.ptr<![[MODEL_TY]]>
// CIR: %[[GET_MEM_PTR:.*]] = cir.get_member %[[GET_MODEL]][1] {name = "__cpu_type"} : !cir.ptr<![[MODEL_TY]]> -> !cir.ptr<!u32i>
// CIR: %[[LOAD_TYPE:.*]]  = cir.load {{.*}}%[[GET_MEM_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[MASK:.*]] = cir.const #cir.int<14> : !u32i
// CIR: cir.cmp eq %[[LOAD_TYPE]], %[[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_tremont(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 14
TEST_CPU_IS(tremont, "tremont")

// CIR-LABEL: cir.func{{.*}}@test_amdfam19h(
// CIR: %[[GET_MODEL:.*]] = cir.get_global @__cpu_model : !cir.ptr<![[MODEL_TY]]>
// CIR: %[[GET_MEM_PTR:.*]] = cir.get_member %[[GET_MODEL]][1] {name = "__cpu_type"} : !cir.ptr<![[MODEL_TY]]> -> !cir.ptr<!u32i>
// CIR: %[[LOAD_TYPE:.*]]  = cir.load {{.*}}%[[GET_MEM_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[MASK:.*]] = cir.const #cir.int<15> : !u32i
// CIR: cir.cmp eq %[[LOAD_TYPE]], %[[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_amdfam19h(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 15
TEST_CPU_IS(amdfam19h, "amdfam19h")

// CIR-LABEL: cir.func{{.*}}@test_zhaoxin_fam7h(
// CIR: %[[GET_MODEL:.*]] = cir.get_global @__cpu_model : !cir.ptr<![[MODEL_TY]]>
// CIR: %[[GET_MEM_PTR:.*]] = cir.get_member %[[GET_MODEL]][1] {name = "__cpu_type"} : !cir.ptr<![[MODEL_TY]]> -> !cir.ptr<!u32i>
// CIR: %[[LOAD_TYPE:.*]]  = cir.load {{.*}}%[[GET_MEM_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[MASK:.*]] = cir.const #cir.int<16> : !u32i
// CIR: cir.cmp eq %[[LOAD_TYPE]], %[[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_zhaoxin_fam7h(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 16
TEST_CPU_IS(zhaoxin_fam7h, "zhaoxin_fam7h")

// CIR-LABEL: cir.func{{.*}}@test_sierraforest(
// CIR: %[[GET_MODEL:.*]] = cir.get_global @__cpu_model : !cir.ptr<![[MODEL_TY]]>
// CIR: %[[GET_MEM_PTR:.*]] = cir.get_member %[[GET_MODEL]][1] {name = "__cpu_type"} : !cir.ptr<![[MODEL_TY]]> -> !cir.ptr<!u32i>
// CIR: %[[LOAD_TYPE:.*]]  = cir.load {{.*}}%[[GET_MEM_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[MASK:.*]] = cir.const #cir.int<17> : !u32i
// CIR: cir.cmp eq %[[LOAD_TYPE]], %[[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_sierraforest(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 17
TEST_CPU_IS(sierraforest, "sierraforest")

// CIR-LABEL: cir.func{{.*}}@test_grandridge(
// CIR: %[[GET_MODEL:.*]] = cir.get_global @__cpu_model : !cir.ptr<![[MODEL_TY]]>
// CIR: %[[GET_MEM_PTR:.*]] = cir.get_member %[[GET_MODEL]][1] {name = "__cpu_type"} : !cir.ptr<![[MODEL_TY]]> -> !cir.ptr<!u32i>
// CIR: %[[LOAD_TYPE:.*]]  = cir.load {{.*}}%[[GET_MEM_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[MASK:.*]] = cir.const #cir.int<18> : !u32i
// CIR: cir.cmp eq %[[LOAD_TYPE]], %[[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_grandridge(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 18
TEST_CPU_IS(grandridge, "grandridge")

// CIR-LABEL: cir.func{{.*}}@test_clearwaterforest(
// CIR: %[[GET_MODEL:.*]] = cir.get_global @__cpu_model : !cir.ptr<![[MODEL_TY]]>
// CIR: %[[GET_MEM_PTR:.*]] = cir.get_member %[[GET_MODEL]][1] {name = "__cpu_type"} : !cir.ptr<![[MODEL_TY]]> -> !cir.ptr<!u32i>
// CIR: %[[LOAD_TYPE:.*]]  = cir.load {{.*}}%[[GET_MEM_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[MASK:.*]] = cir.const #cir.int<19> : !u32i
// CIR: cir.cmp eq %[[LOAD_TYPE]], %[[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_clearwaterforest(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 19
TEST_CPU_IS(clearwaterforest, "clearwaterforest")

// CIR-LABEL: cir.func{{.*}}@test_amdfam1ah(
// CIR: %[[GET_MODEL:.*]] = cir.get_global @__cpu_model : !cir.ptr<![[MODEL_TY]]>
// CIR: %[[GET_MEM_PTR:.*]] = cir.get_member %[[GET_MODEL]][1] {name = "__cpu_type"} : !cir.ptr<![[MODEL_TY]]> -> !cir.ptr<!u32i>
// CIR: %[[LOAD_TYPE:.*]]  = cir.load {{.*}}%[[GET_MEM_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[MASK:.*]] = cir.const #cir.int<20> : !u32i
// CIR: cir.cmp eq %[[LOAD_TYPE]], %[[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_amdfam1ah(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 20
TEST_CPU_IS(amdfam1ah, "amdfam1ah")

// CIR-LABEL: cir.func{{.*}}@test_hygonfam18h(
// CIR: %[[GET_MODEL:.*]] = cir.get_global @__cpu_model : !cir.ptr<![[MODEL_TY]]>
// CIR: %[[GET_MEM_PTR:.*]] = cir.get_member %[[GET_MODEL]][1] {name = "__cpu_type"} : !cir.ptr<![[MODEL_TY]]> -> !cir.ptr<!u32i>
// CIR: %[[LOAD_TYPE:.*]]  = cir.load {{.*}}%[[GET_MEM_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[MASK:.*]] = cir.const #cir.int<21> : !u32i
// CIR: cir.cmp eq %[[LOAD_TYPE]], %[[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_hygonfam18h(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 21
TEST_CPU_IS(hygonfam18h, "hygonfam18h")

// Aliases

// CIR-LABEL: cir.func{{.*}}@test_atom(
// CIR: %[[GET_MODEL:.*]] = cir.get_global @__cpu_model : !cir.ptr<![[MODEL_TY]]>
// CIR: %[[GET_MEM_PTR:.*]] = cir.get_member %[[GET_MODEL]][1] {name = "__cpu_type"} : !cir.ptr<![[MODEL_TY]]> -> !cir.ptr<!u32i>
// CIR: %[[LOAD_TYPE:.*]]  = cir.load {{.*}}%[[GET_MEM_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[MASK:.*]] = cir.const #cir.int<1> : !u32i
// CIR: cir.cmp eq %[[LOAD_TYPE]], %[[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_atom(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 1
TEST_CPU_IS(atom, "atom")

// CIR-LABEL: cir.func{{.*}}@test_amdfam10(
// CIR: %[[GET_MODEL:.*]] = cir.get_global @__cpu_model : !cir.ptr<![[MODEL_TY]]>
// CIR: %[[GET_MEM_PTR:.*]] = cir.get_member %[[GET_MODEL]][1] {name = "__cpu_type"} : !cir.ptr<![[MODEL_TY]]> -> !cir.ptr<!u32i>
// CIR: %[[LOAD_TYPE:.*]]  = cir.load {{.*}}%[[GET_MEM_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[MASK:.*]] = cir.const #cir.int<4> : !u32i
// CIR: cir.cmp eq %[[LOAD_TYPE]], %[[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_amdfam10(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 4
TEST_CPU_IS(amdfam10, "amdfam10")

// CIR-LABEL: cir.func{{.*}}@test_amdfam15(
// CIR: %[[GET_MODEL:.*]] = cir.get_global @__cpu_model : !cir.ptr<![[MODEL_TY]]>
// CIR: %[[GET_MEM_PTR:.*]] = cir.get_member %[[GET_MODEL]][1] {name = "__cpu_type"} : !cir.ptr<![[MODEL_TY]]> -> !cir.ptr<!u32i>
// CIR: %[[LOAD_TYPE:.*]]  = cir.load {{.*}}%[[GET_MEM_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[MASK:.*]] = cir.const #cir.int<5> : !u32i
// CIR: cir.cmp eq %[[LOAD_TYPE]], %[[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_amdfam15(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 5
TEST_CPU_IS(amdfam15, "amdfam15")

// CIR-LABEL: cir.func{{.*}}@test_slm(
// CIR: %[[GET_MODEL:.*]] = cir.get_global @__cpu_model : !cir.ptr<![[MODEL_TY]]>
// CIR: %[[GET_MEM_PTR:.*]] = cir.get_member %[[GET_MODEL]][1] {name = "__cpu_type"} : !cir.ptr<![[MODEL_TY]]> -> !cir.ptr<!u32i>
// CIR: %[[LOAD_TYPE:.*]]  = cir.load {{.*}}%[[GET_MEM_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[MASK:.*]] = cir.const #cir.int<6> : !u32i
// CIR: cir.cmp eq %[[LOAD_TYPE]], %[[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_slm(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 6
TEST_CPU_IS(slm, "slm")

// CIR-LABEL: cir.func{{.*}}@test_amdfam1a(
// CIR: %[[GET_MODEL:.*]] = cir.get_global @__cpu_model : !cir.ptr<![[MODEL_TY]]>
// CIR: %[[GET_MEM_PTR:.*]] = cir.get_member %[[GET_MODEL]][1] {name = "__cpu_type"} : !cir.ptr<![[MODEL_TY]]> -> !cir.ptr<!u32i>
// CIR: %[[LOAD_TYPE:.*]]  = cir.load {{.*}}%[[GET_MEM_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[MASK:.*]] = cir.const #cir.int<20> : !u32i
// CIR: cir.cmp eq %[[LOAD_TYPE]], %[[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_amdfam1a(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 20
TEST_CPU_IS(amdfam1a, "amdfam1a")
