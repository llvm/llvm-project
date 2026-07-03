// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=LLVM,OGCG

// CIR-DAG: cir.global "private" internal dso_local @f.tbl = #cir.const_array<[#cir.block_addr_info<@f, "L1"> : !cir.ptr<!void>, #cir.block_addr_info<@f, "L2"> : !cir.ptr<!void>]> : !cir.array<!cir.ptr<!void> x 2>
// CIR-DAG: cir.global "private" internal dso_local @g.tbl = #cir.const_array<[#cir.block_addr_info<@g, "A"> : !cir.ptr<!void>, #cir.block_addr_info<@g, "A"> : !cir.ptr<!void>, #cir.block_addr_info<@g, "B"> : !cir.ptr<!void>]> : !cir.array<!cir.ptr<!void> x 3>
// CIR-DAG: cir.global "private" internal dso_local @h.tbl = #cir.const_array<[#cir.block_addr_info<@h, "L1"> : !cir.ptr<!void>]> : !cir.array<!cir.ptr<!void> x 1>
// CIR-DAG: cir.global "private" internal dso_local @m.ctbl = #cir.const_array<[#cir.block_addr_info<@m, "A2"> : !cir.ptr<!void>]> : !cir.array<!cir.ptr<!void> x 1>

// LLVM-DAG: @f.tbl = internal global [2 x ptr] [ptr blockaddress(@f, %[[FL1:[0-9a-zA-Z_.]+]]), ptr blockaddress(@f, %[[FL2:[0-9a-zA-Z_.]+]])], align 16
// LLVM-DAG: @g.tbl = internal global [3 x ptr] [ptr blockaddress(@g, %[[GA:[0-9a-zA-Z_.]+]]), ptr blockaddress(@g, %[[GA]]), ptr blockaddress(@g, %[[GB:[0-9a-zA-Z_.]+]])], align 16
// LLVM-DAG: @h.tbl = internal global [1 x ptr] [ptr blockaddress(@h, %{{[0-9a-zA-Z_.]+}})], align 8
// LLVM-DAG: @m.ctbl = internal global [1 x ptr] [ptr blockaddress(@m, %[[MA2:[0-9a-zA-Z_.]+]])], align 8

int f(int x) {
  static const void *tbl[] = {&&L1, &&L2};
  goto *tbl[x];
L1:
  return 1;
L2:
  return 2;
}

// CIR-LABEL: cir.func {{.*}} @f
// CIR:   %[[TBL:.*]] = cir.get_global @f.tbl
// CIR:   cir.indirect_br %{{.*}} : !cir.ptr<!void>, [
// CIR-NEXT: ^[[L1BB:.*]],
// CIR-NEXT: ^[[L2BB:.*]]
// CIR:   ]
// CIR: ^[[L1BB]]:
// CIR:   cir.label "L1"
// CIR: ^[[L2BB]]:
// CIR:   cir.label "L2"

// LLVM-LABEL: define dso_local i32 @f(
// LLVM:   indirectbr ptr %{{.*}}, [label %[[FL1]], label %[[FL2]]]

// A appears twice in g's table, but a block only needs to be listed once as an
// indirect-branch successor, so CIR drops the duplicate (classic keeps it).
int g(int x) {
  static const void *tbl[] = {&&A, &&A, &&B};
  goto *tbl[x];
A:
  return 1;
B:
  return 2;
}

// CIR-LABEL: cir.func {{.*}} @g
// CIR:   cir.indirect_br %{{.*}} : !cir.ptr<!void>, [
// CIR-NEXT: ^[[ABB:.*]],
// CIR-NEXT: ^[[BBB:.*]]
// CIR:   ]
// CIR: ^[[ABB]]:
// CIR:   cir.label "A"
// CIR: ^[[BBB]]:
// CIR:   cir.label "B"

// LLVM-LABEL: define dso_local i32 @g(
// LLVMCIR:   indirectbr ptr %{{.*}}, [label %[[GA]], label %[[GB]]]
// OGCG:   indirectbr ptr %{{.*}}, [label %[[GA]], label %[[GA]], label %[[GB]]]

// h takes a label address but never executes a `goto *`, so CIR emits no
// indirect branch (classic still emits a dead poisoned indirectbr).
int h(int x) {
  static const void *tbl[] = {&&L1};
  (void)tbl;
  return x;
L1:
  return 0;
}

// CIR-LABEL: cir.func {{.*}} @h
// CIR-NOT: cir.indirect_br

// LLVM-LABEL: define dso_local i32 @h(
// LLVMCIR-NOT: indirectbr
// OGCG:   indirectbr ptr poison, [label %{{.+}}]

// A2's address comes from a constant table, B2's from a runtime block-address
// op; both feed the same indirect branch.
int m(int sel) {
  static const void *ctbl[] = {&&A2};
  void *p = &&B2;
  void *t = (void *)ctbl[0];
  void *dest = sel ? t : p;
  goto *dest;
A2:
  return 1;
B2:
  return 2;
}

// CIR-LABEL: cir.func {{.*}} @m
// CIR:   cir.block_address <@m, "B2">
// CIR:   cir.indirect_br
// CIR-DAG:   cir.label "A2"
// CIR-DAG:   cir.label "B2"

// LLVM-LABEL: define dso_local i32 @m(
// LLVM:   indirectbr ptr %{{.*}}, [label %[[MA2]], label %{{.+}}]
