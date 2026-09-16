// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=LLVM,OGCG

struct Bar {
  float f1;
  float f2;
  unsigned u;
};

struct Bar varargs_aggregate_mixed_pair(int count, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, count);
  struct Bar res = __builtin_va_arg(args, struct Bar);
  __builtin_va_end(args);
  return res;
}

struct LongPair {
  long a;
  long b;
};

struct LongPair varargs_aggregate_gp_pair(int count, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, count);
  struct LongPair res = __builtin_va_arg(args, struct LongPair);
  __builtin_va_end(args);
  return res;
}

struct DoublePair {
  double a;
  double b;
};

struct DoublePair varargs_aggregate_sse_pair(int count, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, count);
  struct DoublePair res = __builtin_va_arg(args, struct DoublePair);
  __builtin_va_end(args);
  return res;
}

struct Big {
  long a;
  long b;
  long c;
};

struct Big varargs_aggregate_memory(int count, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, count);
  struct Big res = __builtin_va_arg(args, struct Big);
  __builtin_va_end(args);
  return res;
}

// Mixed pair: one SSE eightbyte, one INTEGER eightbyte, reassembled via a temp.
// CIR-LABEL: cir.func {{.*}} @varargs_aggregate_mixed_pair(
// CIR-SAME: -> !rec_anon_struct{{[0-9]*}}
// CIR:   %[[COERCE:.+]] = cir.alloca "coerce" {{.*}} : !cir.ptr<!rec_anon_struct{{[0-9]*}}>
// CIR:   %[[RET_ADDR:.+]] = cir.alloca "__retval" {{.*}} init : !cir.ptr<!rec_Bar>
// CIR:   %[[VAAREA:.+]] = cir.alloca "args" {{.*}} : !cir.ptr<!cir.array<!rec___va_list_tag x 1>>
// CIR:   %[[TMP_ADDR:.+]] = cir.alloca "vaarg.tmp" {{.*}} : !cir.ptr<!rec_Bar>
// CIR:   %[[VA_PTR0:.+]] = cir.cast array_to_ptrdecay %[[VAAREA]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>> -> !cir.ptr<!rec___va_list_tag>
// CIR:   cir.va_start %[[VA_PTR0]] : !cir.ptr<!rec___va_list_tag>
// CIR:   %[[VA_PTR1:.+]] = cir.cast array_to_ptrdecay %[[VAAREA]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>> -> !cir.ptr<!rec___va_list_tag>
// CIR:   %[[GP_OFFSET_P:.+]] = cir.get_member %[[VA_PTR1]][0] {name = "gp_offset"} : !cir.ptr<!rec___va_list_tag> -> !cir.ptr<!u32i>
// CIR:   %[[GP_OFFSET:.+]] = cir.load %[[GP_OFFSET_P]] : !cir.ptr<!u32i>, !u32i
// CIR:   %[[GP_LIMIT:.+]] = cir.const #cir.int<40> : !u32i
// CIR:   %[[FITS_GP:.+]] = cir.cmp le %[[GP_OFFSET]], %[[GP_LIMIT]] : !u32i
// CIR:   %[[FP_OFFSET_P:.+]] = cir.get_member %[[VA_PTR1]][1] {name = "fp_offset"} : !cir.ptr<!rec___va_list_tag> -> !cir.ptr<!u32i>
// CIR:   %[[FP_OFFSET:.+]] = cir.load %[[FP_OFFSET_P]] : !cir.ptr<!u32i>, !u32i
// CIR:   %[[FP_LIMIT:.+]] = cir.const #cir.int<160> : !u32i
// CIR:   %[[FITS_FP:.+]] = cir.cmp le %[[FP_OFFSET]], %[[FP_LIMIT]] : !u32i
// CIR:   %[[IN_REGS:.+]] = cir.select if %[[FITS_GP]] then %[[FITS_FP]] else %{{.+}} : (!cir.bool, !cir.bool, !cir.bool) -> !cir.bool
// CIR:   %[[REG_TMP:.+]] = cir.alloca "vaarg.reg" {{.*}} : !cir.ptr<!rec_anon_struct{{[0-9]*}}>
// CIR:   %[[VA_ARG:.+]] = cir.ternary(%[[IN_REGS]], true {
// CIR:     %[[REG_SAVE_P:.+]] = cir.get_member %[[VA_PTR1]][3] {name = "reg_save_area"}
// CIR:     %[[REG_SAVE:.+]] = cir.load %[[REG_SAVE_P]]
// CIR:     %[[REG_SAVE_B:.+]] = cir.cast bitcast %[[REG_SAVE]] : !cir.ptr<!void> -> !cir.ptr<!u8i>
// CIR:     %[[FP_ADDR:.+]] = cir.ptr_stride %[[REG_SAVE_B]], %[[FP_OFFSET]] : (!cir.ptr<!u8i>, !u32i) -> !cir.ptr<!u8i>
// CIR:     %[[FP_ADDR_V:.+]] = cir.cast bitcast %[[FP_ADDR]] : !cir.ptr<!u8i> -> !cir.ptr<!cir.vector<2 x !cir.float>>
// CIR:     %[[SSE_VAL:.+]] = cir.load %[[FP_ADDR_V]] : !cir.ptr<!cir.vector<2 x !cir.float>>, !cir.vector<2 x !cir.float>
// CIR:     %[[TMP_SSE:.+]] = cir.get_member %[[REG_TMP]][0] {{.*}} -> !cir.ptr<!cir.vector<2 x !cir.float>>
// CIR:     cir.store %[[SSE_VAL]], %[[TMP_SSE]] : !cir.vector<2 x !cir.float>, !cir.ptr<!cir.vector<2 x !cir.float>>
// CIR:     %[[GP_ADDR:.+]] = cir.ptr_stride %[[REG_SAVE_B]], %[[GP_OFFSET]] : (!cir.ptr<!u8i>, !u32i) -> !cir.ptr<!u8i>
// CIR:     %[[GP_ADDR_V:.+]] = cir.cast bitcast %[[GP_ADDR]] : !cir.ptr<!u8i> -> !cir.ptr<!u32i>
// CIR:     %[[INT_VAL:.+]] = cir.load %[[GP_ADDR_V]] : !cir.ptr<!u32i>, !u32i
// CIR:     %[[TMP_INT:.+]] = cir.get_member %[[REG_TMP]][1] {{.*}} -> !cir.ptr<!u32i>
// CIR:     cir.store %[[INT_VAL]], %[[TMP_INT]] : !u32i, !cir.ptr<!u32i>
// CIR:     %[[REG_TMP_B:.+]] = cir.cast bitcast %[[REG_TMP]] : !cir.ptr<!rec_anon_struct{{[0-9]*}}> -> !cir.ptr<!u8i>
// CIR:     %[[GP_BUMP:.+]] = cir.const #cir.int<8> : !u32i
// CIR:     %[[GP_NEXT:.+]] = cir.add %[[GP_OFFSET]], %[[GP_BUMP]] : !u32i
// CIR:     cir.store %[[GP_NEXT]], %[[GP_OFFSET_P]] : !u32i, !cir.ptr<!u32i>
// CIR:     %[[FP_BUMP:.+]] = cir.const #cir.int<16> : !u32i
// CIR:     %[[FP_NEXT:.+]] = cir.add %[[FP_OFFSET]], %[[FP_BUMP]] : !u32i
// CIR:     cir.store %[[FP_NEXT]], %[[FP_OFFSET_P]] : !u32i, !cir.ptr<!u32i>
// CIR:     cir.yield %[[REG_TMP_B]] : !cir.ptr<!u8i>
// CIR:   }, false {
// CIR:     %[[OVERFLOW_P:.+]] = cir.get_member %[[VA_PTR1]][2] {name = "overflow_arg_area"}
// CIR:     %[[OVERFLOW:.+]] = cir.load %[[OVERFLOW_P]]
// CIR:     %[[OVERFLOW_B:.+]] = cir.cast bitcast %[[OVERFLOW]] : !cir.ptr<!void> -> !cir.ptr<!u8i>
// CIR:     %[[STRIDE:.+]] = cir.const #cir.int<16> : !s32i
// CIR:     %[[OVERFLOW_NEXT:.+]] = cir.ptr_stride %[[OVERFLOW_B]], %[[STRIDE]] : (!cir.ptr<!u8i>, !s32i) -> !cir.ptr<!u8i>
// CIR:     cir.store %[[OVERFLOW_NEXT]], %{{.+}} : !cir.ptr<!u8i>, !cir.ptr<!cir.ptr<!u8i>>
// CIR:     cir.yield %[[OVERFLOW_B]] : !cir.ptr<!u8i>
// CIR:   }) : (!cir.bool) -> !cir.ptr<!u8i>
// CIR:   %[[VA_ARG_B:.+]] = cir.cast bitcast %[[VA_ARG]] : !cir.ptr<!u8i> -> !cir.ptr<!rec_Bar>
// CIR:   %[[VA_ARG_V:.+]] = cir.load %[[VA_ARG_B]] : !cir.ptr<!rec_Bar>, !rec_Bar
// CIR:   cir.store{{.*}} %[[VA_ARG_V]], %[[TMP_ADDR]] : !rec_Bar, !cir.ptr<!rec_Bar>
// CIR:   cir.copy %[[TMP_ADDR]] align(4) to %[[RET_ADDR]] align(4) : !cir.ptr<!rec_Bar>
// CIR:   %[[VA_PTR2:.+]] = cir.cast array_to_ptrdecay %[[VAAREA]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>> -> !cir.ptr<!rec___va_list_tag>
// CIR:   cir.va_end %[[VA_PTR2]] : !cir.ptr<!rec___va_list_tag>
// CIR:   %[[RETVAL:.+]] = cir.load{{.*}} %[[RET_ADDR]] : !cir.ptr<!rec_Bar>, !rec_Bar
// CIR:   %[[SLOT:.+]] = cir.cast bitcast %[[COERCE]] : !cir.ptr<!rec_anon_struct{{[0-9]*}}> -> !cir.ptr<!rec_Bar>
// CIR:   cir.store %[[RETVAL]], %[[SLOT]] : !rec_Bar, !cir.ptr<!rec_Bar>
// CIR:   %[[COERCED:.+]] = cir.load %[[COERCE]] : !cir.ptr<!rec_anon_struct{{[0-9]*}}>, !rec_anon_struct{{[0-9]*}}
// CIR:   cir.return %[[COERCED]] : !rec_anon_struct{{[0-9]*}}

// GP pair: both eightbytes INTEGER, contiguous, so a single load suffices.
// CIR-LABEL: cir.func {{.*}} @varargs_aggregate_gp_pair(
// CIR:   %[[GP_OFFSET_P:.+]] = cir.get_member %{{.+}}[0] {name = "gp_offset"} : !cir.ptr<!rec___va_list_tag> -> !cir.ptr<!u32i>
// CIR:   %[[GP_OFFSET:.+]] = cir.load %[[GP_OFFSET_P]] : !cir.ptr<!u32i>, !u32i
// CIR:   %[[GP_LIMIT:.+]] = cir.const #cir.int<32> : !u32i
// CIR:   %[[FITS_GP:.+]] = cir.cmp le %[[GP_OFFSET]], %[[GP_LIMIT]] : !u32i
// CIR:   %[[VA_ARG:.+]] = cir.ternary(%[[FITS_GP]], true {
// CIR:     %[[REG_SAVE:.+]] = cir.load %{{.+}}
// CIR:     %[[REG_SAVE_B:.+]] = cir.cast bitcast %[[REG_SAVE]] : !cir.ptr<!void> -> !cir.ptr<!u8i>
// CIR:     %[[REG_ADDR:.+]] = cir.ptr_stride %[[REG_SAVE_B]], %[[GP_OFFSET]] : (!cir.ptr<!u8i>, !u32i) -> !cir.ptr<!u8i>
// CIR:     %[[GP_BUMP:.+]] = cir.const #cir.int<16> : !u32i
// CIR:     %[[GP_NEXT:.+]] = cir.add %[[GP_OFFSET]], %[[GP_BUMP]] : !u32i
// CIR:     cir.store %[[GP_NEXT]], %[[GP_OFFSET_P]] : !u32i, !cir.ptr<!u32i>
// CIR:     cir.yield %[[REG_ADDR]] : !cir.ptr<!u8i>
// CIR:   }, false {
// CIR:     %[[OVERFLOW:.+]] = cir.load %{{.+}}
// CIR:     %[[OVERFLOW_B:.+]] = cir.cast bitcast %[[OVERFLOW]] : !cir.ptr<!void> -> !cir.ptr<!u8i>
// CIR:     %[[STRIDE:.+]] = cir.const #cir.int<16> : !s32i
// CIR:     %[[OVERFLOW_NEXT:.+]] = cir.ptr_stride %[[OVERFLOW_B]], %[[STRIDE]] : (!cir.ptr<!u8i>, !s32i) -> !cir.ptr<!u8i>
// CIR:     cir.yield %[[OVERFLOW_B]] : !cir.ptr<!u8i>
// CIR:   }) : (!cir.bool) -> !cir.ptr<!u8i>
// CIR:   %[[VA_ARG_B:.+]] = cir.cast bitcast %[[VA_ARG]] : !cir.ptr<!u8i> -> !cir.ptr<!rec_LongPair>
// CIR:   %[[VA_ARG_V:.+]] = cir.load %[[VA_ARG_B]] : !cir.ptr<!rec_LongPair>, !rec_LongPair

// SSE pair: both eightbytes SSE, 16 bytes apart, reassembled via a temp.
// CIR-LABEL: cir.func {{.*}} @varargs_aggregate_sse_pair(
// CIR:   %[[FP_OFFSET_P:.+]] = cir.get_member %{{.+}}[1] {name = "fp_offset"} : !cir.ptr<!rec___va_list_tag> -> !cir.ptr<!u32i>
// CIR:   %[[FP_OFFSET:.+]] = cir.load %[[FP_OFFSET_P]] : !cir.ptr<!u32i>, !u32i
// CIR:   %[[FP_LIMIT:.+]] = cir.const #cir.int<144> : !u32i
// CIR:   %[[FITS_FP:.+]] = cir.cmp le %[[FP_OFFSET]], %[[FP_LIMIT]] : !u32i
// CIR:   %[[REG_TMP:.+]] = cir.alloca "vaarg.reg" {{.*}} : !cir.ptr<!rec_anon_struct{{[0-9]*}}>
// CIR:   %[[VA_ARG:.+]] = cir.ternary(%[[FITS_FP]], true {
// CIR:     %[[REG_SAVE:.+]] = cir.load %{{.+}}
// CIR:     %[[REG_SAVE_B:.+]] = cir.cast bitcast %[[REG_SAVE]] : !cir.ptr<!void> -> !cir.ptr<!u8i>
// CIR:     %[[LO_ADDR:.+]] = cir.ptr_stride %[[REG_SAVE_B]], %[[FP_OFFSET]] : (!cir.ptr<!u8i>, !u32i) -> !cir.ptr<!u8i>
// CIR:     %[[LO_ADDR_V:.+]] = cir.cast bitcast %[[LO_ADDR]] : !cir.ptr<!u8i> -> !cir.ptr<!cir.double>
// CIR:     %[[LO_VAL:.+]] = cir.load %[[LO_ADDR_V]] : !cir.ptr<!cir.double>, !cir.double
// CIR:     %[[TMP_LO:.+]] = cir.get_member %[[REG_TMP]][0] {{.*}} -> !cir.ptr<!cir.double>
// CIR:     cir.store %[[LO_VAL]], %[[TMP_LO]] : !cir.double, !cir.ptr<!cir.double>
// CIR:     %[[HI_BUMP:.+]] = cir.const #cir.int<16> : !u32i
// CIR:     %[[HI_OFF:.+]] = cir.add %[[FP_OFFSET]], %[[HI_BUMP]] : !u32i
// CIR:     %[[HI_ADDR:.+]] = cir.ptr_stride %[[REG_SAVE_B]], %[[HI_OFF]] : (!cir.ptr<!u8i>, !u32i) -> !cir.ptr<!u8i>
// CIR:     %[[HI_ADDR_V:.+]] = cir.cast bitcast %[[HI_ADDR]] : !cir.ptr<!u8i> -> !cir.ptr<!cir.double>
// CIR:     %[[HI_VAL:.+]] = cir.load %[[HI_ADDR_V]] : !cir.ptr<!cir.double>, !cir.double
// CIR:     %[[TMP_HI:.+]] = cir.get_member %[[REG_TMP]][1] {{.*}} -> !cir.ptr<!cir.double>
// CIR:     cir.store %[[HI_VAL]], %[[TMP_HI]] : !cir.double, !cir.ptr<!cir.double>
// CIR:     %[[REG_TMP_B:.+]] = cir.cast bitcast %[[REG_TMP]] : !cir.ptr<!rec_anon_struct{{[0-9]*}}> -> !cir.ptr<!u8i>
// CIR:     %[[FP_BUMP:.+]] = cir.const #cir.int<32> : !u32i
// CIR:     %[[FP_NEXT:.+]] = cir.add %[[FP_OFFSET]], %[[FP_BUMP]] : !u32i
// CIR:     cir.store %[[FP_NEXT]], %[[FP_OFFSET_P]] : !u32i, !cir.ptr<!u32i>
// CIR:     cir.yield %[[REG_TMP_B]] : !cir.ptr<!u8i>
// CIR:   }, false {
// CIR:     %[[OVERFLOW:.+]] = cir.load %{{.+}}
// CIR:     cir.yield %{{.+}} : !cir.ptr<!u8i>
// CIR:   }) : (!cir.bool) -> !cir.ptr<!u8i>
// CIR:   %[[VA_ARG_B:.+]] = cir.cast bitcast %[[VA_ARG]] : !cir.ptr<!u8i> -> !cir.ptr<!rec_DoublePair>
// CIR:   %[[VA_ARG_V:.+]] = cir.load %[[VA_ARG_B]] : !cir.ptr<!rec_DoublePair>, !rec_DoublePair

// MEMORY class: three eightbytes never fit in registers.
// CIR-LABEL: cir.func {{.*}} @varargs_aggregate_memory(
// CIR:   %[[OVERFLOW_P:.+]] = cir.get_member %{{.+}}[2] {name = "overflow_arg_area"} : !cir.ptr<!rec___va_list_tag> -> !cir.ptr<!cir.ptr<!void>>
// CIR:   %[[OVERFLOW:.+]] = cir.load %[[OVERFLOW_P]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR:   %[[OVERFLOW_B:.+]] = cir.cast bitcast %[[OVERFLOW]] : !cir.ptr<!void> -> !cir.ptr<!u8i>
// CIR:   %[[STRIDE:.+]] = cir.const #cir.int<24> : !s32i
// CIR:   %[[OVERFLOW_NEXT:.+]] = cir.ptr_stride %[[OVERFLOW_B]], %[[STRIDE]] : (!cir.ptr<!u8i>, !s32i) -> !cir.ptr<!u8i>
// CIR:   cir.store %[[OVERFLOW_NEXT]], %{{.+}} : !cir.ptr<!u8i>, !cir.ptr<!cir.ptr<!u8i>>
// CIR:   %[[VA_ARG_B:.+]] = cir.cast bitcast %[[OVERFLOW_B]] : !cir.ptr<!u8i> -> !cir.ptr<!rec_Big>
// CIR:   %[[VA_ARG_V:.+]] = cir.load %[[VA_ARG_B]] : !cir.ptr<!rec_Big>, !rec_Big

// LLVM-LABEL: define dso_local { <2 x float>, i32 } @varargs_aggregate_mixed_pair(i32 noundef %{{.*}}, ...)
// LLVM:   call void @llvm.va_start.p0(ptr %{{.*}})
// LLVM:   %[[GP_OFFSET_P:.+]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr %{{.*}}, i32 0, i32 0
// LLVM:   %[[GP_OFFSET:.+]] = load i32, ptr %[[GP_OFFSET_P]], align {{[0-9]+}}
// LLVM:   %[[FITS_GP:.+]] = icmp ule i32 %[[GP_OFFSET]], 40
// LLVM:   %[[FP_OFFSET_P:.+]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr %{{.*}}, i32 0, i32 1
// LLVM:   %[[FP_OFFSET:.+]] = load i32, ptr %[[FP_OFFSET_P]], align {{[0-9]+}}
// LLVM:   %[[FITS_FP:.+]] = icmp ule i32 %[[FP_OFFSET]], 160
// LLVM:   %[[IN_REGS:.+]] = and i1 %[[FITS_GP]], %[[FITS_FP]]
// LLVMCIR: %[[REG_TMP:.+]] = alloca { <2 x float>, i32 }, align 8
// LLVM:   br i1 %[[IN_REGS]], label %[[REG_BB:.+]], label %[[MEM_BB:.+]]
// LLVM: [[REG_BB]]:
// LLVM:   %[[REG_SAVE:.+]] = load ptr, ptr %{{.*}}, align {{[0-9]+}}
// LLVM:   %[[SSE_VAL:.+]] = load <2 x float>, ptr %{{.+}}, align {{[0-9]+}}
// LLVM:   store <2 x float> %[[SSE_VAL]], ptr %{{.*}}, align {{[0-9]+}}
// LLVM:   %[[INT_VAL:.+]] = load i32, ptr %{{.+}}, align {{[0-9]+}}
// LLVM:   store i32 %[[INT_VAL]], ptr %{{.*}}, align {{[0-9]+}}
// LLVM:   %[[GP_NEXT:.+]] = add i32 %[[GP_OFFSET]], 8
// LLVM:   store i32 %[[GP_NEXT]], ptr %[[GP_OFFSET_P]], align {{[0-9]+}}
// LLVM:   %[[FP_NEXT:.+]] = add i32 %[[FP_OFFSET]], 16
// LLVM:   store i32 %[[FP_NEXT]], ptr %[[FP_OFFSET_P]], align {{[0-9]+}}
// LLVM:   br label %[[END_BB:.+]]
// LLVM: [[MEM_BB]]:
// LLVM:   %[[OVERFLOW_P:.+]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr %{{.*}}, i32 0, i32 2
// LLVM:   %[[OVERFLOW:.+]] = load ptr, ptr %[[OVERFLOW_P]], align 8
// LLVM:   %[[OVERFLOW_NEXT:.+]] = getelementptr i8, ptr %[[OVERFLOW]], i{{32|64}} 16
// LLVM:   store ptr %[[OVERFLOW_NEXT]], ptr %[[OVERFLOW_P]], align 8
// LLVM:   br label %[[END_BB]]
// LLVM: [[END_BB]]:
// The fetched value is materialized by loading the record on one side and by
// copying it on the other, so only the address feeding it is shared.
// LLVMCIR: %[[ADDR:.+]] = phi ptr [ %[[OVERFLOW]], %[[MEM_BB]] ], [ %[[REG_TMP]], %[[REG_BB]] ]
// LLVMCIR: %[[VA_ARG:.+]] = load %struct.Bar, ptr %[[ADDR]], align 4
// LLVMCIR: store %struct.Bar %[[VA_ARG]], ptr %{{.*}}, align 4
// OGCG:   %[[ADDR:.+]] = phi ptr [ %{{.*}}, %[[REG_BB]] ], [ %{{.*}}, %[[MEM_BB]] ]
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 %{{.*}}, i64 12, i1 false)

// LLVM-LABEL: define dso_local { i64, i64 } @varargs_aggregate_gp_pair(i32 noundef %{{.*}}, ...)
// LLVM:   %[[GP_OFFSET_P:.+]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr %{{.*}}, i32 0, i32 0
// LLVM:   %[[GP_OFFSET:.+]] = load i32, ptr %[[GP_OFFSET_P]], align {{[0-9]+}}
// LLVM:   %[[FITS_GP:.+]] = icmp ule i32 %[[GP_OFFSET]], 32
// LLVM:   br i1 %[[FITS_GP]], label %[[REG_BB:.+]], label %[[MEM_BB:.+]]
// LLVM: [[REG_BB]]:
// LLVM:   %[[REG_SAVE:.+]] = load ptr, ptr %{{.*}}, align {{[0-9]+}}
// LLVM:   %[[REG_ADDR:.+]] = getelementptr i8, ptr %[[REG_SAVE]], i{{32|64}} %{{.+}}
// LLVM:   %[[GP_NEXT:.+]] = add i32 %[[GP_OFFSET]], 16
// LLVM:   store i32 %[[GP_NEXT]], ptr %[[GP_OFFSET_P]], align {{[0-9]+}}
// LLVM:   br label %[[END_BB:.+]]
// LLVM: [[MEM_BB]]:
// LLVM:   %[[OVERFLOW_P:.+]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr %{{.*}}, i32 0, i32 2
// LLVM:   %[[OVERFLOW:.+]] = load ptr, ptr %[[OVERFLOW_P]], align 8
// LLVM:   %[[OVERFLOW_NEXT:.+]] = getelementptr i8, ptr %[[OVERFLOW]], i{{32|64}} 16
// LLVM:   store ptr %[[OVERFLOW_NEXT]], ptr %[[OVERFLOW_P]], align 8
// LLVM:   br label %[[END_BB]]
// LLVM: [[END_BB]]:
// LLVMCIR: %[[ADDR:.+]] = phi ptr [ %[[OVERFLOW]], %[[MEM_BB]] ], [ %[[REG_ADDR]], %[[REG_BB]] ]
// LLVMCIR: %[[VA_ARG:.+]] = load %struct.LongPair, ptr %[[ADDR]], align 8
// LLVMCIR: store %struct.LongPair %[[VA_ARG]], ptr %{{.*}}, align 8
// OGCG:   %[[ADDR:.+]] = phi ptr [ %{{.*}}, %[[REG_BB]] ], [ %{{.*}}, %[[MEM_BB]] ]
// OGCG:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %{{.*}}, ptr align 8 %[[ADDR]], i64 16, i1 false)

// LLVM-LABEL: define dso_local { double, double } @varargs_aggregate_sse_pair(i32 noundef %{{.*}}, ...)
// LLVM:   %[[FP_OFFSET_P:.+]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr %{{.*}}, i32 0, i32 1
// LLVM:   %[[FP_OFFSET:.+]] = load i32, ptr %[[FP_OFFSET_P]], align {{[0-9]+}}
// LLVM:   %[[FITS_FP:.+]] = icmp ule i32 %[[FP_OFFSET]], 144
// LLVMCIR: %[[REG_TMP:.+]] = alloca { double, double }, align 8
// LLVM:   br i1 %[[FITS_FP]], label %[[REG_BB:.+]], label %[[MEM_BB:.+]]
// LLVM: [[REG_BB]]:
// LLVM:   %[[REG_SAVE:.+]] = load ptr, ptr %{{.*}}, align {{[0-9]+}}
// LLVM:   %[[LO_VAL:.+]] = load double, ptr %{{.+}}, align {{[0-9]+}}
// LLVM:   store double %[[LO_VAL]], ptr %{{.*}}, align {{[0-9]+}}
// The high eightbyte sits one 16-byte vector slot past the low one, reached by
// bumping the offset on one side and by folding it into the address on the
// other.
// LLVMCIR: %[[HI_OFF:.+]] = add i32 %[[FP_OFFSET]], 16
// LLVM:   %[[HI_VAL:.+]] = load double, ptr %{{.+}}, align {{[0-9]+}}
// LLVM:   store double %[[HI_VAL]], ptr %{{.*}}, align {{[0-9]+}}
// LLVM:   %[[FP_NEXT:.+]] = add i32 %[[FP_OFFSET]], 32
// LLVM:   store i32 %[[FP_NEXT]], ptr %[[FP_OFFSET_P]], align {{[0-9]+}}
// LLVM:   br label %[[END_BB:.+]]
// LLVM: [[MEM_BB]]:
// LLVM:   br label %[[END_BB]]
// LLVM: [[END_BB]]:
// LLVMCIR: %[[ADDR:.+]] = phi ptr [ %{{.*}}, %[[MEM_BB]] ], [ %[[REG_TMP]], %[[REG_BB]] ]
// LLVMCIR: %[[VA_ARG:.+]] = load %struct.DoublePair, ptr %[[ADDR]], align 8
// LLVMCIR: store %struct.DoublePair %[[VA_ARG]], ptr %{{.*}}, align 8
// OGCG:   %[[ADDR:.+]] = phi ptr [ %{{.*}}, %[[REG_BB]] ], [ %{{.*}}, %[[MEM_BB]] ]
// OGCG:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %{{.*}}, ptr align 8 %[[ADDR]], i64 16, i1 false)

// LLVM-LABEL: define dso_local void @varargs_aggregate_memory(ptr dead_on_unwind noalias writable sret(%struct.Big) align 8 %{{.*}}, i32 noundef %{{.*}}, ...)
// LLVM:   %[[OVERFLOW_P:.+]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr %{{.*}}, i32 0, i32 2
// LLVM:   %[[OVERFLOW:.+]] = load ptr, ptr %[[OVERFLOW_P]], align 8
// LLVM:   %[[OVERFLOW_NEXT:.+]] = getelementptr i8, ptr %[[OVERFLOW]], i{{32|64}} 24
// LLVM:   store ptr %[[OVERFLOW_NEXT]], ptr %[[OVERFLOW_P]], align 8
// LLVMCIR: %[[VA_ARG:.+]] = load %struct.Big, ptr %[[OVERFLOW]], align 8
// LLVMCIR: store %struct.Big %[[VA_ARG]], ptr %{{.*}}, align 8
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %{{.*}}, ptr align 8 %{{.*}}, i64 24, i1 false)

struct RevMixed {
  long a;
  double b;
};

struct RevMixed varargs_aggregate_mixed_pair_rev(int count, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, count);
  struct RevMixed res = __builtin_va_arg(args, struct RevMixed);
  __builtin_va_end(args);
  return res;
}

// The mirror of the mixed pair, with INTEGER in the low eightbyte and SSE in
// the high one, so each half has to be read from the cursor for its own class
// and stored to the member for that class.  Reading the SSE half from
// gp_offset would still produce a well-formed sequence, so each load is bound
// to the cursor it came from.
// CIR-LABEL: cir.func {{.*}} @varargs_aggregate_mixed_pair_rev(
// CIR:   %[[GP_OFFSET_P:.+]] = cir.get_member %{{.+}}[0] {name = "gp_offset"} : !cir.ptr<!rec___va_list_tag> -> !cir.ptr<!u32i>
// CIR:   %[[GP_OFFSET:.+]] = cir.load %[[GP_OFFSET_P]] : !cir.ptr<!u32i>, !u32i
// CIR:   %[[GP_LIMIT:.+]] = cir.const #cir.int<40> : !u32i
// CIR:   %[[FITS_GP:.+]] = cir.cmp le %[[GP_OFFSET]], %[[GP_LIMIT]] : !u32i
// CIR:   %[[FP_OFFSET_P:.+]] = cir.get_member %{{.+}}[1] {name = "fp_offset"} : !cir.ptr<!rec___va_list_tag> -> !cir.ptr<!u32i>
// CIR:   %[[FP_OFFSET:.+]] = cir.load %[[FP_OFFSET_P]] : !cir.ptr<!u32i>, !u32i
// CIR:   %[[FP_LIMIT:.+]] = cir.const #cir.int<160> : !u32i
// CIR:   %[[FITS_FP:.+]] = cir.cmp le %[[FP_OFFSET]], %[[FP_LIMIT]] : !u32i
// CIR:   %[[REG_TMP:.+]] = cir.alloca "vaarg.reg" {{.*}} : !cir.ptr<!rec_anon_struct{{[0-9]*}}>
// CIR:     %[[REG_SAVE_B:.+]] = cir.cast bitcast %{{.+}} : !cir.ptr<!void> -> !cir.ptr<!u8i>
// CIR:     %[[LO_ADDR:.+]] = cir.ptr_stride %[[REG_SAVE_B]], %[[GP_OFFSET]] : (!cir.ptr<!u8i>, !u32i) -> !cir.ptr<!u8i>
// CIR:     %[[LO_ADDR_V:.+]] = cir.cast bitcast %[[LO_ADDR]] : !cir.ptr<!u8i> -> !cir.ptr<!s64i>
// CIR:     %[[LO_VAL:.+]] = cir.load %[[LO_ADDR_V]] : !cir.ptr<!s64i>, !s64i
// CIR:     %[[TMP_LO:.+]] = cir.get_member %[[REG_TMP]][0] {{.*}} -> !cir.ptr<!s64i>
// CIR:     cir.store %[[LO_VAL]], %[[TMP_LO]] : !s64i, !cir.ptr<!s64i>
// CIR:     %[[HI_ADDR:.+]] = cir.ptr_stride %[[REG_SAVE_B]], %[[FP_OFFSET]] : (!cir.ptr<!u8i>, !u32i) -> !cir.ptr<!u8i>
// CIR:     %[[HI_ADDR_V:.+]] = cir.cast bitcast %[[HI_ADDR]] : !cir.ptr<!u8i> -> !cir.ptr<!cir.double>
// CIR:     %[[HI_VAL:.+]] = cir.load %[[HI_ADDR_V]] : !cir.ptr<!cir.double>, !cir.double
// CIR:     %[[TMP_HI:.+]] = cir.get_member %[[REG_TMP]][1] {{.*}} -> !cir.ptr<!cir.double>
// CIR:     cir.store %[[HI_VAL]], %[[TMP_HI]] : !cir.double, !cir.ptr<!cir.double>
// CIR:     %[[REG_TMP_B:.+]] = cir.cast bitcast %[[REG_TMP]] : !cir.ptr<!rec_anon_struct{{[0-9]*}}> -> !cir.ptr<!u8i>
// CIR:     cir.yield %[[REG_TMP_B]] : !cir.ptr<!u8i>

// LLVM-LABEL: define dso_local { i64, double } @varargs_aggregate_mixed_pair_rev(i32 noundef %{{.*}}, ...)
// LLVM:   %[[GP_OFFSET:.+]] = load i32, ptr %{{.+}}, align {{[0-9]+}}
// LLVM:   icmp ule i32 %[[GP_OFFSET]], 40
// LLVM:   %[[FP_OFFSET:.+]] = load i32, ptr %{{.+}}, align {{[0-9]+}}
// LLVM:   icmp ule i32 %[[FP_OFFSET]], 160
// LLVM:   %[[RSA:.+]] = load ptr, ptr %{{.+}}, align {{[0-9]+}}
// Both halves index the same save area, each by its own cursor.
// LLVMCIR: %[[GP64:.+]] = zext i32 %[[GP_OFFSET]] to i64
// LLVMCIR: %[[LO_ADDR:.+]] = getelementptr i8, ptr %[[RSA]], i64 %[[GP64]]
// LLVMCIR: %[[LO_VAL:.+]] = load i64, ptr %[[LO_ADDR]], align {{[0-9]+}}
// LLVMCIR: store i64 %[[LO_VAL]], ptr %{{.+}}, align {{[0-9]+}}
// LLVMCIR: %[[FP64:.+]] = zext i32 %[[FP_OFFSET]] to i64
// LLVMCIR: %[[HI_ADDR:.+]] = getelementptr i8, ptr %[[RSA]], i64 %[[FP64]]
// LLVMCIR: %[[HI_VAL:.+]] = load double, ptr %[[HI_ADDR]], align {{[0-9]+}}
// LLVMCIR: store double %[[HI_VAL]], ptr %{{.+}}, align {{[0-9]+}}
// OGCG:    %[[LO_ADDR:.+]] = getelementptr i8, ptr %[[RSA]], i32 %[[GP_OFFSET]]
// OGCG:    %[[HI_ADDR:.+]] = getelementptr i8, ptr %[[RSA]], i32 %[[FP_OFFSET]]
// OGCG:    %[[LO_VAL:.+]] = load i64, ptr %[[LO_ADDR]], align {{[0-9]+}}
// OGCG:    store i64 %[[LO_VAL]], ptr %{{.+}}, align {{[0-9]+}}
// OGCG:    %[[HI_VAL:.+]] = load double, ptr %[[HI_ADDR]], align {{[0-9]+}}
// OGCG:    store double %[[HI_VAL]], ptr %{{.+}}, align {{[0-9]+}}
// LLVM:   add i32 %[[GP_OFFSET]], 8
// LLVM:   add i32 %[[FP_OFFSET]], 16

struct __attribute__((aligned(16))) OverAligned {
  long a;
  long b;
};

struct OverAligned varargs_aggregate_overaligned(int count, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, count);
  struct OverAligned res = __builtin_va_arg(args, struct OverAligned);
  __builtin_va_end(args);
  return res;
}

// An alignment attribute raises the record above what its two long members
// imply, and only the record layout carries that.  Both arms have to honor it:
// the register arm copies through a 16-byte-aligned temp, and the overflow arm
// rounds the cursor up to 16 before reading.
// CIR-LABEL: cir.func {{.*}} @varargs_aggregate_overaligned(
// CIR:   %[[GP_OFFSET:.+]] = cir.load %{{.+}} : !cir.ptr<!u32i>, !u32i
// CIR:   %[[GP_LIMIT:.+]] = cir.const #cir.int<32> : !u32i
// CIR:   %[[FITS:.+]] = cir.cmp le %[[GP_OFFSET]], %[[GP_LIMIT]] : !u32i
// CIR:   %[[TEMP:.+]] = cir.alloca "vaarg.reg" align(16) : !cir.ptr<!rec_OverAligned>
// CIR:   %[[ADDR:.+]] = cir.ternary(%[[FITS]], true {
// CIR:     %[[REG_ADDR:.+]] = cir.ptr_stride %{{.+}}, %[[GP_OFFSET]] : (!cir.ptr<!u8i>, !u32i) -> !cir.ptr<!u8i>
// CIR:     %[[REG_ADDR_V:.+]] = cir.cast bitcast %[[REG_ADDR]] : !cir.ptr<!u8i> -> !cir.ptr<!rec_OverAligned>
// CIR:     %[[VAL:.+]] = cir.load align(8) %[[REG_ADDR_V]] : !cir.ptr<!rec_OverAligned>, !rec_OverAligned
// CIR:     cir.store %[[VAL]], %[[TEMP]] : !rec_OverAligned, !cir.ptr<!rec_OverAligned>
// CIR:     %[[TEMP_B:.+]] = cir.cast bitcast %[[TEMP]] : !cir.ptr<!rec_OverAligned> -> !cir.ptr<!u8i>
// CIR:     cir.yield %[[TEMP_B]] : !cir.ptr<!u8i>
// CIR:     %[[MASK:.+]] = cir.const #cir.int<18446744073709551600> : !u64i
// CIR:     %[[ROUNDED:.+]] = cir.and %{{.+}}, %[[MASK]] : !u64i
// CIR:     %[[ALIGNED:.+]] = cir.cast int_to_ptr %[[ROUNDED]] : !u64i -> !cir.ptr<!u8i>
// CIR:     cir.yield %[[ALIGNED]] : !cir.ptr<!u8i>

// LLVM-LABEL: define dso_local { i64, i64 } @varargs_aggregate_overaligned(i32 noundef %{{.*}}, ...)
// LLVM:   %[[GP_OFFSET:.+]] = load i32, ptr %{{.+}}, align {{[0-9]+}}
// LLVM:   icmp ule i32 %[[GP_OFFSET]], 32
// The register slot is only 8-aligned, so the copy reads at 8 and stores
// into a temp carrying the record's declared 16.  The temp is bound at its
// alloca, so that an under-aligned one cannot satisfy the store below.
// LLVMCIR: %[[TEMP:.+]] = alloca %struct.OverAligned, align 16
// LLVMCIR: %[[VAL:.+]] = load %struct.OverAligned, ptr %{{.+}}, align 8
// LLVMCIR: store %struct.OverAligned %[[VAL]], ptr %[[TEMP]], align 8
// OGCG:    call void @llvm.memcpy.p0.p0.i64(ptr align 16 %[[TEMP:.+]], ptr align 8 %{{.+}}, i64 16, i1 false)
// LLVMCIR: %[[ALIGNED:.+]] = inttoptr i64 %{{.+}} to ptr
// OGCG:    %[[ALIGNED:.+]] = call ptr @llvm.ptrmask.p0.i64(ptr %{{.+}}, i64 -16)
// LLVM:   %[[MEM_NEXT:.+]] = getelementptr i8, ptr %[[ALIGNED]], i{{32|64}} 16
// LLVMCIR: %[[ADDR:.+]] = phi ptr [ %[[ALIGNED]], %{{.+}} ], [ %[[TEMP]], %{{.+}} ]
// OGCG:    %[[ADDR:.+]] = phi ptr [ %[[TEMP]], %{{.+}} ], [ %[[ALIGNED]], %{{.+}} ]

// A record can require more alignment than the types of its members imply,
// and the attribute that raises it need not sit on the record: it can sit on a
// field, or on a member record whose own storage is only byte-aligned.  None
// of those raise the alignment of any type, so the fetch takes the alignment
// from the record layout.  Each of these reads from the overflow area, since
// 32 bytes is too large for registers.

union OverAlignedUnion {
  char buf[32];
} __attribute__((aligned(32)));

long varargs_overaligned_union(int count, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, count);
  union OverAlignedUnion res = __builtin_va_arg(args, union OverAlignedUnion);
  __builtin_va_end(args);
  return res.buf[0];
}

// LLVM-LABEL: define dso_local {{.*}} @varargs_overaligned_union(
// LLVMCIR:   and i64 %{{.+}}, -32
// OGCG:      call ptr @llvm.ptrmask.p0.i64(ptr %{{.+}}, i64 -32)

struct FieldOverAligned {
  __attribute__((aligned(32))) char c;
  char rest[31];
};

long varargs_field_overaligned(int count, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, count);
  struct FieldOverAligned res = __builtin_va_arg(args, struct FieldOverAligned);
  __builtin_va_end(args);
  return res.c;
}

// LLVM-LABEL: define dso_local {{.*}} @varargs_field_overaligned(
// LLVMCIR:   and i64 %{{.+}}, -32
// OGCG:      call ptr @llvm.ptrmask.p0.i64(ptr %{{.+}}, i64 -32)

struct HoldsOverAlignedUnion {
  union OverAlignedUnion u;
};

long varargs_holds_overaligned_union(int count, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, count);
  struct HoldsOverAlignedUnion res =
      __builtin_va_arg(args, struct HoldsOverAlignedUnion);
  __builtin_va_end(args);
  return res.u.buf[0];
}

// LLVM-LABEL: define dso_local {{.*}} @varargs_holds_overaligned_union(
// LLVMCIR:   and i64 %{{.+}}, -32
// OGCG:      call ptr @llvm.ptrmask.p0.i64(ptr %{{.+}}, i64 -32)

// Packing lowers the alignment the members imply, and an attribute can still
// raise the record above it.
struct __attribute__((packed, aligned(32))) PackedOverAligned {
  char c;
  int i;
};

long varargs_packed_overaligned(int count, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, count);
  struct PackedOverAligned res =
      __builtin_va_arg(args, struct PackedOverAligned);
  __builtin_va_end(args);
  return res.i;
}

// LLVM-LABEL: define dso_local {{.*}} @varargs_packed_overaligned(
// LLVMCIR:   and i64 %{{.+}}, -32
// OGCG:      call ptr @llvm.ptrmask.p0.i64(ptr %{{.+}}, i64 -32)
