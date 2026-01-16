// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2> %t-before-lp.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void binary_assign(void) {
    bool b;
    char c;
    float f;
    int i;

    b = true;
    c = 65;
    f = 3.14f;
    i = 42;
}

// CIR-LABEL: cir.func{{.*}} @binary_assign()
// CIR:         %[[B:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["b"]
// CIR:         %[[C:.*]] = cir.alloca !s8i, !cir.ptr<!s8i>, ["c"]
// CIR:         %[[F:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["f"]
// CIR:         %[[I:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i"]
// CIR:         %[[TRUE:.*]] = cir.const #true
// CIR:         cir.store{{.*}} %[[TRUE]], %[[B]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:         %[[CHAR_VAL:.*]] = cir.const #cir.int<65> : !s8i
// CIR:         cir.store{{.*}} %[[CHAR_VAL]], %[[C]] : !s8i, !cir.ptr<!s8i>
// CIR:         %[[FLOAT_VAL:.*]] = cir.const #cir.fp<3.140000e+00> : !cir.float
// CIR:         cir.store{{.*}} %[[FLOAT_VAL]], %[[F]] : !cir.float, !cir.ptr<!cir.float>
// CIR:         %[[INT_VAL:.*]] = cir.const #cir.int<42> : !s32i
// CIR:         cir.store{{.*}} %[[INT_VAL]], %[[I]] : !s32i, !cir.ptr<!s32i>
// CIR:         cir.return

// LLVM-LABEL: define {{.*}}void @binary_assign(){{.*}} {
// LLVM:         %[[B_PTR:.*]] = alloca i8
// LLVM:         %[[C_PTR:.*]] = alloca i8
// LLVM:         %[[F_PTR:.*]] = alloca float
// LLVM:         %[[I_PTR:.*]] = alloca i32
// LLVM:         store i8 1, ptr %[[B_PTR]]
// LLVM:         store i8 65, ptr %[[C_PTR]]
// LLVM:         store float 0x40091EB860000000, ptr %[[F_PTR]]
// LLVM:         store i32 42, ptr %[[I_PTR]]
// LLVM:         ret void

// OGCG-LABEL: define {{.*}}void @binary_assign()
// OGCG:         %[[B_PTR:.*]] = alloca i8
// OGCG:         %[[C_PTR:.*]] = alloca i8
// OGCG:         %[[F_PTR:.*]] = alloca float
// OGCG:         %[[I_PTR:.*]] = alloca i32
// OGCG:         store i8 1, ptr %[[B_PTR]]
// OGCG:         store i8 65, ptr %[[C_PTR]]
// OGCG:         store float 0x40091EB860000000, ptr %[[F_PTR]]
// OGCG:         store i32 42, ptr %[[I_PTR]]
// OGCG:         ret void

struct S {
  int a;
  float b;
};

struct SV {
  int a;
  volatile float b;
};

struct S gs;
struct SV gsv;

void binary_assign_struct() {
  // Test normal struct assignment
  struct S ls;
  ls = gs;

  // Test assignment of a struct with a volatile member
  struct SV lsv;
  lsv = gsv;
}

// CIR: cir.func{{.*}} @binary_assign_struct()
// CIR:   %[[LS:.*]] = cir.alloca ![[REC_S:.*]], !cir.ptr<![[REC_S]]>, ["ls"]
// CIR:   %[[LSV:.*]] = cir.alloca ![[REC_SV:.*]], !cir.ptr<![[REC_SV]]>, ["lsv"]
// CIR:   %[[GS_PTR:.*]] = cir.get_global @gs : !cir.ptr<![[REC_S]]>
// CIR:   cir.copy %[[GS_PTR]] to %[[LS]] : !cir.ptr<![[REC_S]]>
// CIR:   %[[GSV_PTR:.*]] = cir.get_global @gsv : !cir.ptr<![[REC_SV]]>
// CIR:   cir.copy %[[GSV_PTR]] to %[[LSV]] volatile : !cir.ptr<![[REC_SV]]>
// CIR:   cir.return

// LLVM: define {{.*}}void @binary_assign_struct()
// LLVM:   %[[LS_PTR:.*]] = alloca %struct.S
// LLVM:   %[[LSV_PTR:.*]] = alloca %struct.SV
// LLVM:   call void @llvm.memcpy.p0.p0.i32(ptr %[[LS_PTR]], ptr @gs, i32 8, i1 false)
// LLVM:   call void @llvm.memcpy.p0.p0.i32(ptr %[[LSV_PTR]], ptr @gsv, i32 8, i1 true)
// LLVM:   ret void

// OGCG: define {{.*}}void @binary_assign_struct()
// OGCG:   %[[LS_PTR:.*]] = alloca %struct.S
// OGCG:   %[[LSV_PTR:.*]] = alloca %struct.SV
// OGCG:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[LS_PTR]], ptr align 4 @gs, i64 8, i1 false)
// OGCG:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[LSV_PTR]], ptr align 4 @gsv, i64 8, i1 true)
// OGCG:   ret void

int ignore_result_assign() {
  int arr[10];
  int i, j;
  j = i = 123, 0;
  j = arr[i = 5];
  int *p, *q = 0;
  if(p = q)
    return 1;
  return 0;
}

// CIR-LABEL: cir.func{{.*}} @ignore_result_assign() -> !s32i
// CIR:         %[[RETVAL:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:         %[[ARR:.*]] = cir.alloca !cir.array<!s32i x 10>, !cir.ptr<!cir.array<!s32i x 10>>, ["arr"]
// CIR:         %[[I:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i"]
// CIR:         %[[J:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["j"]
// CIR:         %[[P:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["p"]
// CIR:         %[[Q:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["q", init]
// CIR:         %[[VAL_123:.*]] = cir.const #cir.int<123> : !s32i
// CIR:         cir.store{{.*}} %[[VAL_123]], %[[I]] : !s32i, !cir.ptr<!s32i>
// CIR:         cir.store{{.*}} %[[VAL_123]], %[[J]] : !s32i, !cir.ptr<!s32i>
// CIR:         %[[VAL_0:.*]] = cir.const #cir.int<0> : !s32i
// CIR:         %[[VAL_5:.*]] = cir.const #cir.int<5> : !s32i
// CIR:         cir.store{{.*}} %[[VAL_5]], %[[I]] : !s32i, !cir.ptr<!s32i>
// CIR:         %[[ARR_ELEM:.*]] = cir.get_element %[[ARR]][%[[VAL_5]] : !s32i] : !cir.ptr<!cir.array<!s32i x 10>> -> !cir.ptr<!s32i>
// CIR:         %[[ARR_LOAD:.*]] = cir.load{{.*}} %[[ARR_ELEM]] : !cir.ptr<!s32i>, !s32i
// CIR:         cir.store{{.*}} %[[ARR_LOAD]], %[[J]] : !s32i, !cir.ptr<!s32i>
// CIR:         %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s32i>
// CIR:         cir.store{{.*}} %[[NULL]], %[[Q]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR:         cir.scope {
// CIR:           %[[Q_VAL:.*]] = cir.load{{.*}} %[[Q]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR:           cir.store{{.*}} %[[Q_VAL]], %[[P]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR:           %[[COND:.*]] = cir.cast ptr_to_bool %[[Q_VAL]] : !cir.ptr<!s32i> -> !cir.bool
// CIR:           cir.if %[[COND]] {
// CIR:             %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR:             cir.store %[[ONE]], %[[RETVAL]] : !s32i, !cir.ptr<!s32i>
// CIR:             %{{.*}} = cir.load %[[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR:             cir.return
// CIR:           }
// CIR:         }
// CIR:         %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:         cir.store %[[ZERO]], %[[RETVAL]] : !s32i, !cir.ptr<!s32i>
// CIR:         %{{.*}} = cir.load %[[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR:         cir.return

// LLVM-LABEL: define {{.*}}i32 @ignore_result_assign()
// LLVM:         %[[RETVAL_PTR:.*]] = alloca i32
// LLVM:         %[[ARR_PTR:.*]] = alloca [10 x i32]
// LLVM:         %[[I_PTR:.*]] = alloca i32
// LLVM:         %[[J_PTR:.*]] = alloca i32
// LLVM:         %[[P_PTR:.*]] = alloca ptr
// LLVM:         %[[Q_PTR:.*]] = alloca ptr
// LLVM:         store i32 123, ptr %[[I_PTR]]
// LLVM:         store i32 123, ptr %[[J_PTR]]
// LLVM:         store i32 5, ptr %[[I_PTR]]
// LLVM:         %[[GEP:.*]] = getelementptr [10 x i32], ptr %[[ARR_PTR]], i32 0, i64 5
// LLVM:         %[[ARR_VAL:.*]] = load i32, ptr %[[GEP]]
// LLVM:         store i32 %[[ARR_VAL]], ptr %[[J_PTR]]
// LLVM:         store ptr null, ptr %[[Q_PTR]]
// LLVM:         br label
// LLVM:         %[[Q_VAL:.*]] = load ptr, ptr %[[Q_PTR]]
// LLVM:         store ptr %[[Q_VAL]], ptr %[[P_PTR]]
// LLVM:         %[[CMP:.*]] = icmp ne ptr %[[Q_VAL]], null
// LLVM:         br i1 %[[CMP]], label %[[THEN:.*]], label %[[ELSE:.*]]
// LLVM:       [[THEN]]:
// LLVM:         store i32 1, ptr %[[RETVAL_PTR]]
// LLVM:         %{{.*}} = load i32, ptr %[[RETVAL_PTR]]
// LLVM:         ret i32
// LLVM:       [[ELSE]]:
// LLVM:         br label
// LLVM:         store i32 0, ptr %[[RETVAL_PTR]]
// LLVM:         %{{.*}} = load i32, ptr %[[RETVAL_PTR]]
// LLVM:         ret i32

// OGCG-LABEL: define {{.*}}i32 @ignore_result_assign()
// OGCG:         %[[RETVAL:.*]] = alloca i32
// OGCG:         %[[ARR:.*]] = alloca [10 x i32]
// OGCG:         %[[I:.*]] = alloca i32
// OGCG:         %[[J:.*]] = alloca i32
// OGCG:         %[[P:.*]] = alloca ptr
// OGCG:         %[[Q:.*]] = alloca ptr
// OGCG:         store i32 123, ptr %[[I]]
// OGCG:         store i32 123, ptr %[[J]]
// OGCG:         store i32 5, ptr %[[I]]
// OGCG:         %[[ARRAYIDX:.*]] = getelementptr inbounds [10 x i32], ptr %[[ARR]], i64 0, i64 5
// OGCG:         %[[ARR_VAL:.*]] = load i32, ptr %[[ARRAYIDX]]
// OGCG:         store i32 %[[ARR_VAL]], ptr %[[J]]
// OGCG:         store ptr null, ptr %[[Q]]
// OGCG:         %[[Q_VAL:.*]] = load ptr, ptr %[[Q]]
// OGCG:         store ptr %[[Q_VAL]], ptr %[[P]]
// OGCG:         %[[TOBOOL:.*]] = icmp ne ptr %[[Q_VAL]], null
// OGCG:         br i1 %[[TOBOOL]], label %[[IF_THEN:.*]], label %[[IF_END:.*]]
// OGCG:       [[IF_THEN]]:
// OGCG:         store i32 1, ptr %[[RETVAL]]
// OGCG:         br label %[[RETURN:.*]]
// OGCG:       [[IF_END]]:
// OGCG:         store i32 0, ptr %[[RETVAL]]
// OGCG:         br label %[[RETURN]]
// OGCG:       [[RETURN]]:
// OGCG:         %{{.*}} = load i32, ptr %[[RETVAL]]
// OGCG:         ret i32
