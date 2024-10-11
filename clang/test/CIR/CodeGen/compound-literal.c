// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-call-conv-lowering -Wno-unused-value -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-call-conv-lowering -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM


typedef struct {
  int *arr;
} S;

S a = {
  .arr = (int[]){}
};

// CIR: cir.global "private" internal @".compoundLiteral.0" = #cir.zero : !cir.array<!s32i x 0> {alignment = 4 : i64}
// CIR: cir.global external @a = #cir.const_struct<{#cir.global_view<@".compoundLiteral.0"> : !cir.ptr<!s32i>}> : !ty_S

// LLVM: @.compoundLiteral.0 = internal global [0 x i32] zeroinitializer
// LLVM: @a = global %struct.S { ptr @.compoundLiteral.0 }

S b = {
  .arr = (int[]){1}
};

// CIR: cir.global "private" internal @".compoundLiteral.1" = #cir.const_array<[#cir.int<1> : !s32i]> : !cir.array<!s32i x 1> {alignment = 4 : i64}
// CIR: cir.global external @b = #cir.const_struct<{#cir.global_view<@".compoundLiteral.1"> : !cir.ptr<!s32i>}> : !ty_S

// LLVM: @.compoundLiteral.1 = internal global [1 x i32] [i32 1]
// LLVM: @b = global %struct.S { ptr @.compoundLiteral.1 }

int foo() {
  return (struct {
           int i;
         }){1}
      .i;
}

// CIR:  cir.func no_proto @foo() -> !s32i
// CIR:    [[RET_MEM:%.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CIR:    [[COMPLITERAL_MEM:%.*]] = cir.alloca !ty_anon2E0_, !cir.ptr<!ty_anon2E0_>, [".compoundliteral"] {alignment = 4 : i64}
// CIR:    [[FIELD:%.*]] = cir.get_member [[COMPLITERAL_MEM]][0] {name = "i"} : !cir.ptr<!ty_anon2E0_> -> !cir.ptr<!s32i>
// CIR:    [[ONE:%.*]] = cir.const #cir.int<1> : !s32i
// CIR:    cir.store [[ONE]], [[FIELD]] : !s32i, !cir.ptr<!s32i>
// CIR:    [[ONE:%.*]] = cir.const #cir.int<1> : !s32i
// CIR:    cir.store [[ONE]], [[RET_MEM]] : !s32i, !cir.ptr<!s32i>
// CIR:    [[RET:%.*]] = cir.load [[RET_MEM]] : !cir.ptr<!s32i>, !s32i
// CIR:    cir.return [[RET]] : !s32i

struct G { short x, y, z; };
struct G g(int x, int y, int z) {
  return (struct G) { x, y, z };
}

// CIR:  cir.func @g
// CIR:    %[[RETVAL:.*]] = cir.alloca !ty_G, !cir.ptr<!ty_G>, ["__retval"] {alignment = 2 : i64}
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

typedef struct { unsigned long pgprot; } pgprot_t;
void split_large_page(unsigned long addr, pgprot_t prot)
{
  (addr ? prot : ((pgprot_t) { 0x001 } )).pgprot;
}

// CIR-LABEL: @split_large_page
// CIR:   %[[VAL_2:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["addr", init] {alignment = 8 : i64}
// CIR:   %[[VAL_3:.*]] = cir.alloca !ty_pgprot_t, !cir.ptr<!ty_pgprot_t>, ["prot", init] {alignment = 8 : i64}
// CIR:   %[[VAL_4:.*]] = cir.alloca !ty_pgprot_t, !cir.ptr<!ty_pgprot_t>, ["tmp"] {alignment = 8 : i64}
// CIR:   cir.store {{.*}}, %[[VAL_2]] : !u64i, !cir.ptr<!u64i>
// CIR:   cir.store {{.*}}, %[[VAL_3]] : !ty_pgprot_t, !cir.ptr<!ty_pgprot_t>
// CIR:   %[[VAL_5:.*]] = cir.load %[[VAL_2]] : !cir.ptr<!u64i>, !u64i
// CIR:   %[[VAL_6:.*]] = cir.cast(int_to_bool, %[[VAL_5]] : !u64i), !cir.bool
// CIR:   cir.if %[[VAL_6]] {
// CIR:     cir.copy %[[VAL_3]] to %[[VAL_4]] : !cir.ptr<!ty_pgprot_t>
// CIR:   } else {
// CIR:     %[[VAL_7:.*]] = cir.get_member %[[VAL_4]][0] {name = "pgprot"} : !cir.ptr<!ty_pgprot_t> -> !cir.ptr<!u64i>
// CIR:     %[[VAL_8:.*]] = cir.const #cir.int<1> : !s32i
// CIR:     %[[VAL_9:.*]] = cir.cast(integral, %[[VAL_8]] : !s32i), !u64i
// CIR:     cir.store %[[VAL_9]], %[[VAL_7]] : !u64i, !cir.ptr<!u64i>
// CIR:   }
// CIR:   %[[VAL_10:.*]] = cir.get_member %[[VAL_4]][0] {name = "pgprot"} : !cir.ptr<!ty_pgprot_t> -> !cir.ptr<!u64i>
// CIR:   %[[VAL_11:.*]] = cir.load %[[VAL_10]] : !cir.ptr<!u64i>, !u64i
// CIR:   cir.return
// CIR: }

// CHECK-LABEL: @split_large_page
// CHECK:    br i1 {{.*}}, label %[[TRUE:[a-z0-9]+]], label %[[FALSE:[a-z0-9]+]]
// CHECK:  [[FALSE]]:
// CHECK:    %[[GEP:.*]] = getelementptr {{.*}}, ptr %[[ADDR:.*]], i32 0, i32 0
// CHECK:    store i64 1, ptr %[[GEP]], align 8
// CHECK:    br label %[[EXIT:[a-z0-9]+]]
// CHECK:  [[TRUE]]:
// CHECK:    call void @llvm.memcpy.p0.p0.i32(ptr %[[ADDR]], ptr {{.*}}, i32 8, i1 false)
// CHECK:    br label %[[EXIT]]
// CHECK:  [[EXIT]]:
// CHECK:    ret void
