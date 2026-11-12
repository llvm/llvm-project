// RUN: %clang_cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:   -emit-cir -mmlir --mlir-print-ir-before=cir-target-lowering %s \
// RUN:   -o %t.cir 2> %t.pre.cir
// RUN: FileCheck %s --check-prefix=PRE --input-file=%t.pre.cir
// RUN: FileCheck %s --check-prefix=POST --input-file=%t.cir
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:   -emit-llvm -O0 %s -o %t-cir.ll
// RUN: FileCheck %s --check-prefixes=LLVM,CIR-LLVM --input-file=%t-cir.ll
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 \
// RUN:   -emit-llvm -O0 %s -o %t.ll
// RUN: FileCheck %s --check-prefixes=LLVM,OGCG --input-file=%t.ll

// Language address spaces in record members are lowered; other records are
// kept.

struct S { __global int *p; int n; };
struct Outer { struct S s; __local int *lp; };
struct Node { __global struct Node *next; int v; };
union U { __global int *p; long l; };
struct Plain { int a; float b; };

// PRE-DAG: !rec_S = !cir.struct<"S" {data !cir.ptr<!s32i, lang_address_space(offload_global)>, data !s32i}>
// PRE-DAG: !rec_Outer = !cir.struct<"Outer" {data !rec_S, data !cir.ptr<!s32i, lang_address_space(offload_local)>}>
// PRE-DAG: !rec_Node = !cir.struct<"Node" {data !cir.ptr<!cir.struct<"Node">, lang_address_space(offload_global)>, data !s32i}>
// PRE-DAG: !rec_U = !cir.union<"U" {data !cir.ptr<!s32i, lang_address_space(offload_global)>, data !s64i}>
// PRE-DAG: !rec_Plain = !cir.struct<"Plain" {data !s32i, data !cir.float}>

// POST-DAG: !rec_S = !cir.struct<"S" {data !cir.ptr<!s32i, target_address_space(1)>, data !s32i}>
// POST-DAG: !rec_Outer = !cir.struct<"Outer" {data !rec_S, data !cir.ptr<!s32i, target_address_space(3)>}>
// POST-DAG: !rec_Node = !cir.struct<"Node" {data !cir.ptr<!cir.struct<"Node">, target_address_space(1)>, data !s32i}>
// POST-DAG: !rec_U = !cir.union<"U" {data !cir.ptr<!s32i, target_address_space(1)>, data !s64i}>
// POST-DAG: !rec_Plain = !cir.struct<"Plain" {data !s32i, data !cir.float}>

// LLVM-DAG: %struct.S = type { ptr addrspace(1), i32 }
// LLVM-DAG: %struct.Outer = type { %struct.S, ptr addrspace(3) }
// LLVM-DAG: %struct.Node = type { ptr addrspace(1), i32 }
// LLVM-DAG: %union.U = type { ptr addrspace(1) }
// LLVM-DAG: %struct.Plain = type { i32, float }

__global int gi;
__global struct S gs = {&gi, 4};

// The initializer type follows the record.

// POST: cir.global external target_address_space(1) @gs = #cir.const_record<{#cir.global_view<@gi> : !cir.ptr<!s32i, target_address_space(1)>, #cir.int<4> : !s32i}> : !rec_S

// CIR-LLVM: @gs = addrspace(1) global %struct.S { ptr addrspace(1) @gi, i32 4 }, align 8
// OGCG: @gs = addrspace(1) global { ptr addrspace(1), i32, [4 x i8] } { ptr addrspace(1) @gi, i32 4, [4 x i8] zeroinitializer }, align 8

kernel void k(__global int *g, __local int *l, __global struct Node *n) {
  struct S s;
  s.p = g;
  struct Outer o;
  o.lp = l;
  struct Node node;
  node.next = n;
  union U u;
  u.p = g;
  struct Plain plain;
  plain.a = 1;
}

// POST: cir.get_member %{{.*}}[0] {name = "p"} : !cir.ptr<!rec_S> -> !cir.ptr<!cir.ptr<!s32i, target_address_space(1)>>
// POST: cir.get_member %{{.*}}[1] {name = "lp"} : !cir.ptr<!rec_Outer> -> !cir.ptr<!cir.ptr<!s32i, target_address_space(3)>>
// POST: cir.get_member %{{.*}}[0] {name = "next"} : !cir.ptr<!rec_Node> -> !cir.ptr<!cir.ptr<!rec_Node, target_address_space(1)>>
// POST: cir.get_member %{{.*}}[0] {name = "p"} : !cir.ptr<!rec_U> -> !cir.ptr<!cir.ptr<!s32i, target_address_space(1)>>
// POST: cir.get_member %{{.*}}[0] {name = "a"} : !cir.ptr<!rec_Plain> -> !cir.ptr<!s32i>

// LLVM: %[[P:.*]] = getelementptr inbounds nuw %struct.S, ptr %{{.*}}, i32 0, i32 0
// LLVM: store ptr addrspace(1) %{{.*}}, ptr %[[P]], align 8
// LLVM: %[[LP:.*]] = getelementptr inbounds nuw %struct.Outer, ptr %{{.*}}, i32 0, i32 1
// LLVM: store ptr addrspace(3) %{{.*}}, ptr %[[LP]], align 8
// LLVM: %[[NEXT:.*]] = getelementptr inbounds nuw %struct.Node, ptr %{{.*}}, i32 0, i32 0
// LLVM: store ptr addrspace(1) %{{.*}}, ptr %[[NEXT]], align 8
// LLVM: store ptr addrspace(1) %{{.*}}, ptr %{{.*}}, align 8
// LLVM: %[[A:.*]] = getelementptr inbounds nuw %struct.Plain, ptr %{{.*}}, i32 0, i32 0
// LLVM: store i32 1, ptr %[[A]], align 4
