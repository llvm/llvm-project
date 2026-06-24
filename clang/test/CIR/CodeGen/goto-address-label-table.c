// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

int f(int x) {
  static const void *tbl[] = {&&L1, &&L2};
  goto *tbl[x];
L1:
  return 1;
L2:
  return 2;
}

// A appears twice in g's table; both occurrences are kept as indirect-branch
// successors, matching classic codegen.
int g(int x) {
  static const void *tbl[] = {&&A, &&A, &&B};
  goto *tbl[x];
A:
  return 1;
B:
  return 2;
}

// A constant label address with no indirect goto reaching it: the indirect-goto
// block is created but has no predecessors, so it is left poisoned.
int h(int x) {
  static const void *tbl[] = {&&L1};
  (void)tbl;
  return x;
L1:
  return 0;
}

// A's address comes from a constant table, B's from a runtime block-address op;
// both feed the same indirect branch.
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

// CIR-DAG: cir.global "private" internal dso_local @f.tbl = #cir.const_array<[#cir.block_addr_info<@f, "L1"> : !cir.ptr<!void>, #cir.block_addr_info<@f, "L2"> : !cir.ptr<!void>]> : !cir.array<!cir.ptr<!void> x 2>
// CIR-DAG: cir.global "private" internal dso_local @g.tbl = #cir.const_array<[#cir.block_addr_info<@g, "A"> : !cir.ptr<!void>, #cir.block_addr_info<@g, "A"> : !cir.ptr<!void>, #cir.block_addr_info<@g, "B"> : !cir.ptr<!void>]> : !cir.array<!cir.ptr<!void> x 3>
// CIR-DAG: cir.global "private" internal dso_local @h.tbl = #cir.const_array<[#cir.block_addr_info<@h, "L1"> : !cir.ptr<!void>]> : !cir.array<!cir.ptr<!void> x 1>
// CIR-DAG: cir.global "private" internal dso_local @m.ctbl = #cir.const_array<[#cir.block_addr_info<@m, "A2"> : !cir.ptr<!void>]> : !cir.array<!cir.ptr<!void> x 1>

// CIR: cir.func {{.*}} @f
// CIR:   %[[TBL:.*]] = cir.get_global @f.tbl
// CIR:   cir.indirect_br %{{.*}} : !cir.ptr<!void>, [
// CIR-NEXT: ^[[L1BB:.*]],
// CIR-NEXT: ^[[L2BB:.*]]
// CIR:   ]
// CIR: ^[[L1BB]]:
// CIR:   cir.label "L1"
// CIR: ^[[L2BB]]:
// CIR:   cir.label "L2"

// CIR: cir.func {{.*}} @g
// CIR:   cir.indirect_br %{{.*}} : !cir.ptr<!void>, [
// CIR-NEXT: ^[[ABB:.*]],
// CIR-NEXT: ^[[ABB]],
// CIR-NEXT: ^[[BBB:.*]]
// CIR:   ]
// CIR: ^[[ABB]]:
// CIR:   cir.label "A"
// CIR: ^[[BBB]]:
// CIR:   cir.label "B"

// No indirect goto reaches the label, so the goto block is poisoned.
// CIR: cir.func {{.*}} @h
// CIR:   cir.indirect_br %{{.*}} poison : !cir.ptr<!void>, [
// CIR-NEXT: ^[[HL1:.*]]
// CIR:   ]
// CIR: ^[[HL1]]:
// CIR:   cir.label "L1"

// A2 comes from the constant table, B2 from a runtime block-address op; both
// are successors of the one indirect branch.
// CIR: cir.func {{.*}} @m
// CIR:   cir.block_address <@m, "B2">
// CIR:   cir.indirect_br
// CIR-DAG:   cir.label "A2"
// CIR-DAG:   cir.label "B2"

// LLVM-DAG: @f.tbl = internal global [2 x ptr] [ptr blockaddress(@f, %[[L1:[0-9]+]]), ptr blockaddress(@f, %[[L2:[0-9]+]])], align 16
// LLVM-DAG: @g.tbl = internal global [3 x ptr] [ptr blockaddress(@g, %[[GA:[0-9]+]]), ptr blockaddress(@g, %[[GA]]), ptr blockaddress(@g, %[[GB:[0-9]+]])], align 16
// LLVM-DAG: @h.tbl = internal global [1 x ptr] [ptr blockaddress(@h, %{{[0-9]+}})], align 8
// LLVM-DAG: @m.ctbl = internal global [1 x ptr] [ptr blockaddress(@m, %{{[0-9]+}})], align 8

// LLVM: define dso_local i32 @f(i32 noundef %{{.*}})
// LLVM:   indirectbr ptr %{{.*}}, [label %[[L1]], label %[[L2]]]

// LLVM: define dso_local i32 @g(i32 noundef %{{.*}})
// LLVM:   indirectbr ptr %{{.*}}, [label %[[GA]], label %[[GA]], label %[[GB]]]

// LLVM: define dso_local i32 @h(i32 noundef %{{.*}})
// LLVM:   indirectbr ptr poison, [label %{{[0-9]+}}]

// LLVM: define dso_local i32 @m(i32 noundef %{{.*}})
// LLVM:   indirectbr ptr %{{.*}}, [label %{{[0-9]+}}, label %{{[0-9]+}}]

// OGCG-DAG: @f.tbl = internal global [2 x ptr] [ptr blockaddress(@f, %L1), ptr blockaddress(@f, %L2)], align 16
// OGCG-DAG: @g.tbl = internal global [3 x ptr] [ptr blockaddress(@g, %A), ptr blockaddress(@g, %A), ptr blockaddress(@g, %B)], align 16
// OGCG-DAG: @h.tbl = internal global [1 x ptr] [ptr blockaddress(@h, %L1)], align 8
// OGCG-DAG: @m.ctbl = internal global [1 x ptr] [ptr blockaddress(@m, %A2)], align 8

// OGCG: define dso_local i32 @f(i32 noundef %{{.*}})
// OGCG:   indirectbr ptr %indirect.goto.dest, [label %L1, label %L2]

// OGCG: define dso_local i32 @g(i32 noundef %{{.*}})
// OGCG:   indirectbr ptr %indirect.goto.dest, [label %A, label %A, label %B]

// OGCG: define dso_local i32 @h(i32 noundef %{{.*}})
// OGCG:   indirectbr ptr poison, [label %L1]

// OGCG: define dso_local i32 @m(i32 noundef %{{.*}})
// OGCG:   indirectbr ptr %indirect.goto.dest, [label %{{.*}}, label %{{.*}}]
