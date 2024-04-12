// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -S -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM


typedef struct {
  int *arr;
} S;

S a = {
  .arr = (int[]){}
};

// CIR: cir.global "private" internal @".compoundLiteral.0" = #cir.zero : !cir.array<!s32i x 0> {alignment = 4 : i64}
// CIR: cir.global external @a = #cir.const_struct<{#cir.global_view<@".compoundLiteral.0"> : !cir.ptr<!s32i>}> : !ty_22S22

// LLVM: @.compoundLiteral.0 = internal global [0 x i32] zeroinitializer
// LLVM: @a = global %struct.S { ptr @.compoundLiteral.0 }

S b = {
  .arr = (int[]){1}
};

// CIR: cir.global "private" internal @".compoundLiteral.1" = #cir.const_array<[#cir.int<1> : !s32i]> : !cir.array<!s32i x 1> {alignment = 4 : i64}
// CIR: cir.global external @b = #cir.const_struct<{#cir.global_view<@".compoundLiteral.1"> : !cir.ptr<!s32i>}> : !ty_22S22

// LLVM: @.compoundLiteral.1 = internal global [1 x i32] [i32 1]
// LLVM: @b = global %struct.S { ptr @.compoundLiteral.1 }

int foo() {
  return (struct {
           int i;
         }){1}
      .i;
}

// CIR:  cir.func no_proto @foo() -> !s32i
// CIR:    [[RET_MEM:%.*]] = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"] {alignment = 4 : i64}
// CIR:    [[COMPLITERAL_MEM:%.*]] = cir.alloca !ty_22anon2E122, cir.ptr <!ty_22anon2E122>, [".compoundliteral"] {alignment = 4 : i64}
// CIR:    [[FIELD:%.*]] = cir.get_member [[COMPLITERAL_MEM]][0] {name = "i"} : !cir.ptr<!ty_22anon2E122> -> !cir.ptr<!s32i>
// CIR:    [[ONE:%.*]] = cir.const(#cir.int<1> : !s32i) : !s32i
// CIR:    cir.store [[ONE]], [[FIELD]] : !s32i, cir.ptr <!s32i>
// CIR:    [[ONE:%.*]] = cir.const(#cir.int<1> : !s32i) : !s32i
// CIR:    cir.store [[ONE]], [[RET_MEM]] : !s32i, cir.ptr <!s32i>
// CIR:    [[RET:%.*]] = cir.load [[RET_MEM]] : cir.ptr <!s32i>, !s32i
// CIR:    cir.return [[RET]] : !s32i

struct G { short x, y, z; };
struct G g(int x, int y, int z) {
  return (struct G) { x, y, z };
}

// CIR:  cir.func @g
// CIR:    %[[RETVAL:.*]] = cir.alloca !ty_22G22, cir.ptr <!ty_22G22>, ["__retval"] {alignment = 2 : i64} loc(#loc18)
// CIR:    %[[X:.*]] = cir.get_member %[[RETVAL]][0] {name = "x"}
// CIR:    cir.store {{.*}}, %[[X]] : !s16i
// CIR:    %[[Y:.*]] = cir.get_member %[[RETVAL]][1] {name = "y"}
// CIR:    cir.store {{.*}}, %[[Y]] : !s16i
// CIR:    %[[Z:.*]] = cir.get_member %[[RETVAL]][2] {name = "z"}
// CIR:    cir.store {{.*}}, %[[Z]] : !s16i
// CIR:    %[[RES:.*]] = cir.load %[[RETVAL]]
// CIR:    cir.return %[[RES]]

// Nothing meaningful to test for LLVM codegen here.
// FIXME: ABI note, LLVM lowering differs from traditional LLVM codegen here,
// because the former does a memcopy + i48 load.