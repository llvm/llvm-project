// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// CIR: cir.global "private" internal dso_local @f.s = #cir.const_record<{#cir.block_addr_diff<@f, "A", "B"> : !s32i, #cir.block_addr_diff<@f, "B", "A"> : !s32i}> : !rec_S2
// CIR: cir.global "private" internal dso_local @e.arr = #cir.const_array<[#cir.block_addr_diff<@e, "l2", "l1"> : !s32i, #cir.block_addr_diff<@e, "l3", "l2"> : !s32i]> : !cir.array<!s32i x 2>
// CIR: cir.global "private" internal dso_local @d.s = #cir.const_record<{#cir.block_addr_info<@d, "A"> : !cir.ptr<!void>, #cir.block_addr_info<@d, "B"> : !cir.ptr<!void>}> : !rec_S
// CIR: cir.global "private" internal dso_local @c.tbl = #cir.const_array<[#cir.block_addr_info<@c, "A"> : !cir.ptr<!void>, #cir.block_addr_info<@c, "A"> : !cir.ptr<!void>, #cir.block_addr_info<@c, "B"> : !cir.ptr<!void>]> : !cir.array<!cir.ptr<!void> x 3>
// CIR: cir.global "private" internal dso_local @b.ar = #cir.block_addr_diff<@b, "l2", "l1"> : !s32i
// CIR: cir.global "private" internal dso_local @a.a = #cir.block_addr_info<@a, "A"> : !cir.ptr<!void>

// LLVM-DAG: @e.arr = internal global [2 x i32] [i32 trunc (i{{..}} sub (i{{..}} ptrtoint (ptr blockaddress(@e, %[[E_L2:.*]]) to i{{..}}), i{{..}} ptrtoint (ptr blockaddress(@e, %[[E_L1:.*]]) to i{{..}})) to i32), i32 trunc (i{{..}} sub (i{{..}} ptrtoint (ptr blockaddress(@e, %[[E_L3:.*]]) to i{{..}}), i{{..}} ptrtoint (ptr blockaddress(@e, %[[E_L2]]) to i{{..}})) to i32)], align 4
// LLVM-DAG: @a.a = internal global ptr blockaddress(@a, %[[A_BLOCK:.*]]), align 8
// LLVM-DAG: @b.ar = internal global {{.*}} sub (i{{..}} ptrtoint (ptr blockaddress(@b, %[[LABEL_L2:.*]]) to i{{..}}), i{{..}} ptrtoint (ptr blockaddress(@b, %[[LABEL_L1:.*]]) to i{{..}}))
// LLVM-DAG: @c.tbl = internal global [3 x ptr] [ptr blockaddress(@c, %[[C_A:.*]]), ptr blockaddress(@c, %[[C_A]]), ptr blockaddress(@c, %[[C_B:.*]])], align 16
// LLVM-DAG: @d.s = internal global %struct.S { ptr blockaddress(@d, %[[D_A:.*]]), ptr blockaddress(@d, %[[D_B:.*]]) }, align 8
// LLVM-DAG: @f.s = internal global %struct.S2 { i32 trunc (i{{..}} sub (i{{..}} ptrtoint (ptr blockaddress(@f, %[[F_A:.*]]) to i{{..}}), i{{..}} ptrtoint (ptr blockaddress(@f, %[[F_B:.*]]) to i{{..}})) to i32), i32 trunc (i{{..}} sub (i{{..}} ptrtoint (ptr blockaddress(@f, %[[F_B]]) to i{{..}}), i{{..}} ptrtoint (ptr blockaddress(@f, %[[F_A]]) to i{{..}})) to i32) }, align 4

void a(void) {
A:;
  static void *a = &&A;
}

// CIR: cir.func{{.*}} @a()
// CIR:   cir.br ^[[A_BLOCK:bb[0-9]+]]
// CIR: ^[[A_BLOCK]]:
// CIR:   cir.label "A"
// CIR:   %[[STATIC_A:.*]] = cir.get_global @a.a : !cir.ptr<!cir.ptr<!void>>
// CIR:   cir.return

// LLVM: define dso_local void @a()
// LLVM:   br label %[[A_BLOCK]]
// LLVM: [[A_BLOCK]]:
// LLVM:   ret void

// PR14005
int b(void) {
  static int ar = &&l2 - &&l1;
l1:
  return 10;
l2:
  return 11;
}

// CIR: cir.func{{.*}} @b
// CIR:   %[[B_AR:.*]] = cir.get_global @b.ar
// CIR: [[LABEL_L1:.*]]:
// CIR:   cir.label "l1"
// CIR: [[LABEL_L2:.*]]:
// CIR:   cir.label "l2"

// LLVM: define dso_local i32 @b()
// LLVM:   br label %[[LABEL_L1]]
// LLVM: [[LABEL_L1]]:
// LLVM: [[LABEL_L2]]:

void c(int x) {
  static void *tbl[3] = {&&A, &&A, &&B};
  int idx = x > 2 ? 2 : x;
A:
  void *p = tbl[idx];
B:
}

// CIR: cir.func{{.*}} @c
// CIR:   %[[C_TBL:.*]] = cir.get_global @c.tbl
// CIR: [[LABEL_A:.*]]:
// CIR:   cir.label "A"
// CIR:   %[[P:.*]] = cir.get_element %[[C_TBL]][%{{.*}}]
// CIR: [[LABEL_B:.*]]:
// CIR:   cir.label "B"

// LLVM: define dso_local void @c(i32 noundef %{{.*}})
// LLVM:   br label %[[C_A]]
// LLVM: [[C_A]]:
// LLVM:   %[[TARGET:.*]] = getelementptr{{.*}} [3 x ptr], ptr @c.tbl
// LLVM:   %[[P:.*]] = load ptr, ptr %[[TARGET]], align 8
// LLVM:   br label %[[C_B]]
// LLVM: [[C_B]]:
// LLVM:   ret void

struct S { void *a, *b; };
void d(void) {
A:;
B:;
  static struct S s = {&&A, &&B};
}

// CIR: cir.func{{.*}} @d
// CIR: [[LABEL_A:.*]]:
// CIR:   cir.label "A"
// CIR: [[LABEL_B:.*]]:
// CIR:   cir.label "B"
// CIR:   %[[S:.*]] = cir.get_global @d.s
// CIR:   cir.return

// LLVM: define dso_local void @d()
// LLVM:   br label %[[D_A]]
// LLVM: [[D_A]]:
// LLVM:   br label %[[D_B]]
// LLVM: [[D_B]]:
// LLVM:   ret void

int e(void) {
  static int arr[2] = {&&l2 - &&l1, &&l3 - &&l2};
l1:
  return 10;
l2:
  return 11;
l3:
  return 11;
}

// CIR: cir.func{{.*}} @e
// CIR:   %[[E_ARR:.*]] = cir.get_global @e.arr
// CIR: [[E_L1:.*]]:
// CIR:   cir.label "l1"
// CIR: [[E_L2:.*]]:
// CIR:   cir.label "l2"
// CIR: [[E_L3:.*]]:
// CIR:   cir.label "l3"

// LLVM: define dso_local i32 @e()
// LLVM:   br label %[[E_L1]]
// LLVM: [[E_L1]]:
// LLVM: [[E_L2]]:
// LLVM: [[E_L3]]:

struct S2 { int a, b; };
void f(void) {
A:;
B:;
  static struct S2 s = {&&A - &&B, &&B - &&A};
}

// CIR: cir.func{{.*}} @f
// CIR: [[F_A:.*]]:
// CIR:   cir.label "A"
// CIR: [[F_B:.*]]:
// CIR:   cir.label "B"
// CIR:   %[[S2:.*]] = cir.get_global @f.s
// CIR:   cir.return

// LLVM: define dso_local void @f()
// LLVM:   br label %[[F_A]]
// LLVM: [[F_A]]:
// LLVM:   br label %[[F_B]]
// LLVM: [[F_B]]:
// LLVM:   ret void
