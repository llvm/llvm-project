// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t.ll %s

struct Empty {};

// No field with data lies in the low eightbyte, so the register carries bytes
// 8 through 15 and the classification reports that byte offset.
struct EmptyLow {
  Empty e;
  long x;
};

struct EmptyLowSse {
  Empty e;
  double d;
};

// One eightbyte holds the only field with data and the rest of the record
// carries none, so the record is wider than the register that carries it.
// Reading the record straight from the slot would also read the next one.
struct TailPad {
  long x;
  Empty e;
};

EmptyLow varargs_empty_low(int count, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, count);
  EmptyLow res = __builtin_va_arg(args, EmptyLow);
  __builtin_va_end(args);
  return res;
}

// CIR-LABEL: cir.func {{.*}} @_Z17varargs_empty_lowiz(
// CIR:   %[[GP_P:.+]] = cir.get_member %{{.+}}[0] {name = "gp_offset"} : !cir.ptr<!rec___va_list_tag> -> !cir.ptr<!u32i>
// CIR:   %[[GP:.+]] = cir.load %[[GP_P]] : !cir.ptr<!u32i>, !u32i
// CIR:   %[[LIMIT:.+]] = cir.const #cir.int<40> : !u32i
// CIR:   %[[FITS:.+]] = cir.cmp le %[[GP]], %[[LIMIT]] : !u32i
// CIR:   %[[TEMP:.+]] = cir.alloca "vaarg.reg" align(8) : !cir.ptr<!rec_EmptyLow>
// CIR:   %[[ADDR:.+]] = cir.ternary(%[[FITS]], true {
// CIR:     %[[RSA_B:.+]] = cir.cast bitcast %{{.+}} : !cir.ptr<!void> -> !cir.ptr<!u8i>
// CIR:     %[[SLOT:.+]] = cir.ptr_stride %[[RSA_B]], %[[GP]] : (!cir.ptr<!u8i>, !u32i) -> !cir.ptr<!u8i>
// CIR:     %[[SLOT_I:.+]] = cir.cast bitcast %[[SLOT]] : !cir.ptr<!u8i> -> !cir.ptr<!s64i>
// CIR:     %[[VAL:.+]] = cir.load align(8) %[[SLOT_I]] : !cir.ptr<!s64i>, !s64i
// CIR:     %[[TEMP_B:.+]] = cir.cast bitcast %[[TEMP]] : !cir.ptr<!rec_EmptyLow> -> !cir.ptr<!u8i>
// CIR:     %[[OFF:.+]] = cir.const #cir.int<8> : !s32i
// CIR:     %[[DST:.+]] = cir.ptr_stride %[[TEMP_B]], %[[OFF]] : (!cir.ptr<!u8i>, !s32i) -> !cir.ptr<!u8i>
// CIR:     %[[DST_I:.+]] = cir.cast bitcast %[[DST]] : !cir.ptr<!u8i> -> !cir.ptr<!s64i>
// CIR:     cir.store %[[VAL]], %[[DST_I]] : !s64i, !cir.ptr<!s64i>
// CIR:     %[[YIELDED:.+]] = cir.cast bitcast %[[TEMP]] : !cir.ptr<!rec_EmptyLow> -> !cir.ptr<!u8i>
// CIR:     cir.yield %[[YIELDED]] : !cir.ptr<!u8i>
// CIR:   %[[RESULT_P:.+]] = cir.cast bitcast %[[ADDR]] : !cir.ptr<!u8i> -> !cir.ptr<!rec_EmptyLow>
// CIR:   cir.load %[[RESULT_P]] : !cir.ptr<!rec_EmptyLow>, !rec_EmptyLow

// LLVM-LABEL: define dso_local i64 @_Z17varargs_empty_lowiz(i32 noundef %{{.*}}, ...)
// LLVM:   %[[GP_P:.+]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr %{{.+}}, i32 0, i32 0
// LLVM:   %[[GP:.+]] = load i32, ptr %[[GP_P]], align {{4|16}}
// LLVM:   %[[FITS:.+]] = icmp ule i32 %[[GP]], 40
// LLVM:   %[[RSA:.+]] = load ptr, ptr %{{.+}}, align {{8|16}}
// LLVMCIR: %[[GP64:.+]] = zext i32 %[[GP]] to i64
// LLVMCIR: %[[SLOT:.+]] = getelementptr i8, ptr %[[RSA]], i64 %[[GP64]]
// OGCG:    %[[SLOT:.+]] = getelementptr i8, ptr %[[RSA]], i32 %[[GP]]
// LLVM:   %[[VAL:.+]] = load i64, ptr %[[SLOT]], align 8
// LLVM:   %[[DST:.+]] = getelementptr i8, ptr %[[TEMP:.+]], i{{32|64}} 8
// LLVM:   store i64 %[[VAL]], ptr %[[DST]], align 8
// LLVM:   %[[BUMPED:.+]] = add i32 %[[GP]], 8
// LLVM:   store i32 %[[BUMPED]], ptr %[[GP_P]], align {{4|16}}

// The overflow area holds the whole record laid out normally, so the memory
// path reads it from its base and the byte offset does not apply.
// LLVM:   %[[OVERFLOW_P:.+]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr %{{.+}}, i32 0, i32 2
// LLVM:   %[[OVERFLOW:.+]] = load ptr, ptr %[[OVERFLOW_P]], align 8
// LLVM:   %[[NEXT:.+]] = getelementptr i8, ptr %[[OVERFLOW]], i{{32|64}} 16
// LLVM:   store ptr %[[NEXT]], ptr %[[OVERFLOW_P]], align 8

// LLVMCIR: %[[ADDR:.+]] = phi ptr [ %[[OVERFLOW]], %{{.+}} ], [ %[[TEMP]], %{{.+}} ]
// OGCG:    %[[ADDR:.+]] = phi ptr [ %[[TEMP]], %{{.+}} ], [ %[[OVERFLOW]], %{{.+}} ]
// LLVMCIR: load %struct.EmptyLow, ptr %[[ADDR]], align 8
// OGCG:    call void @llvm.memcpy.p0.p0.i64(ptr align 8 %{{.+}}, ptr align 8 %[[ADDR]], i64 16, i1 false)

double varargs_empty_low_sse(int count, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, count);
  EmptyLowSse res = __builtin_va_arg(args, EmptyLowSse);
  __builtin_va_end(args);
  return res.d;
}

// The same offset applies when the carrying register is SSE, so the fetch
// gates on fp_offset and still lands the value at byte 8.
// CIR-LABEL: cir.func {{.*}} @_Z21varargs_empty_low_sseiz(
// CIR:   %[[FP_P:.+]] = cir.get_member %{{.+}}[1] {name = "fp_offset"} : !cir.ptr<!rec___va_list_tag> -> !cir.ptr<!u32i>
// CIR:   %[[FP:.+]] = cir.load %[[FP_P]] : !cir.ptr<!u32i>, !u32i
// CIR:   %[[LIMIT:.+]] = cir.const #cir.int<160> : !u32i
// CIR:   %[[FITS:.+]] = cir.cmp le %[[FP]], %[[LIMIT]] : !u32i
// CIR:   %[[TEMP:.+]] = cir.alloca "vaarg.reg" align(8) : !cir.ptr<!rec_EmptyLowSse>
// CIR:   %[[ADDR:.+]] = cir.ternary(%[[FITS]], true {
// CIR:     %[[SLOT:.+]] = cir.ptr_stride %{{.+}}, %[[FP]] : (!cir.ptr<!u8i>, !u32i) -> !cir.ptr<!u8i>
// CIR:     %[[SLOT_D:.+]] = cir.cast bitcast %[[SLOT]] : !cir.ptr<!u8i> -> !cir.ptr<!cir.double>
// CIR:     %[[VAL:.+]] = cir.load align(8) %[[SLOT_D]] : !cir.ptr<!cir.double>, !cir.double
// CIR:     %[[TEMP_B:.+]] = cir.cast bitcast %[[TEMP]] : !cir.ptr<!rec_EmptyLowSse> -> !cir.ptr<!u8i>
// CIR:     %[[OFF:.+]] = cir.const #cir.int<8> : !s32i
// CIR:     %[[DST:.+]] = cir.ptr_stride %[[TEMP_B]], %[[OFF]] : (!cir.ptr<!u8i>, !s32i) -> !cir.ptr<!u8i>
// CIR:     %[[DST_D:.+]] = cir.cast bitcast %[[DST]] : !cir.ptr<!u8i> -> !cir.ptr<!cir.double>
// CIR:     cir.store %[[VAL]], %[[DST_D]] : !cir.double, !cir.ptr<!cir.double>
// CIR:     %[[YIELDED:.+]] = cir.cast bitcast %[[TEMP]] : !cir.ptr<!rec_EmptyLowSse> -> !cir.ptr<!u8i>
// An SSE slot is 16 bytes, so the cursor advances by 16 rather than by 8.
// CIR:     %[[STEP:.+]] = cir.const #cir.int<16> : !u32i
// CIR:     cir.store %{{.+}}, %[[FP_P]] : !u32i, !cir.ptr<!u32i>
// CIR:     cir.yield %[[YIELDED]] : !cir.ptr<!u8i>

// LLVM-LABEL: define dso_local noundef double @_Z21varargs_empty_low_sseiz(i32 noundef %{{.*}}, ...)
// LLVM:   %[[FP_P:.+]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr %{{.+}}, i32 0, i32 1
// LLVM:   %[[FP:.+]] = load i32, ptr %[[FP_P]], align {{4|8|16}}
// LLVM:   %[[FITS:.+]] = icmp ule i32 %[[FP]], 160
// LLVM:   %[[VAL:.+]] = load double, ptr %{{.+}}, align 8
// LLVM:   %[[DST:.+]] = getelementptr i8, ptr %[[TEMP:.+]], i{{32|64}} 8
// LLVM:   store double %[[VAL]], ptr %[[DST]], align 8
// LLVM:   %[[BUMPED:.+]] = add i32 %[[FP]], 16
// LLVM:   store i32 %[[BUMPED]], ptr %[[FP_P]], align {{4|8|16}}
// LLVMCIR: %[[ADDR:.+]] = phi ptr [ %{{.+}}, %{{.+}} ], [ %[[TEMP]], %{{.+}} ]
// OGCG:    %[[ADDR:.+]] = phi ptr [ %[[TEMP]], %{{.+}} ], [ %{{.+}}, %{{.+}} ]
// LLVMCIR: load %struct.EmptyLowSse, ptr %[[ADDR]], align 8
// OGCG:    call void @llvm.memcpy.p0.p0.i64(ptr align 8 %{{.+}}, ptr align 8 %[[ADDR]], i64 16, i1 false)

long varargs_tail_pad(int count, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, count);
  TailPad res = __builtin_va_arg(args, TailPad);
  __builtin_va_end(args);
  return res.x;
}

// With no byte offset the value lands at the temp's base.
// CIR-LABEL: cir.func {{.*}} @_Z16varargs_tail_padiz(
// CIR:   %[[GP:.+]] = cir.load %{{.+}} : !cir.ptr<!u32i>, !u32i
// CIR:   %[[LIMIT:.+]] = cir.const #cir.int<40> : !u32i
// CIR:   %[[FITS:.+]] = cir.cmp le %[[GP]], %[[LIMIT]] : !u32i
// CIR:   %[[TEMP:.+]] = cir.alloca "vaarg.reg" align(8) : !cir.ptr<!rec_TailPad>
// CIR:   %[[ADDR:.+]] = cir.ternary(%[[FITS]], true {
// CIR:     %[[SLOT:.+]] = cir.ptr_stride %{{.+}}, %[[GP]] : (!cir.ptr<!u8i>, !u32i) -> !cir.ptr<!u8i>
// CIR:     %[[SLOT_I:.+]] = cir.cast bitcast %[[SLOT]] : !cir.ptr<!u8i> -> !cir.ptr<!s64i>
// CIR:     %[[VAL:.+]] = cir.load align(8) %[[SLOT_I]] : !cir.ptr<!s64i>, !s64i
// CIR:     %[[TEMP_I:.+]] = cir.cast bitcast %[[TEMP]] : !cir.ptr<!rec_TailPad> -> !cir.ptr<!s64i>
// CIR:     cir.store %[[VAL]], %[[TEMP_I]] : !s64i, !cir.ptr<!s64i>
// CIR:     %[[YIELDED:.+]] = cir.cast bitcast %[[TEMP]] : !cir.ptr<!rec_TailPad> -> !cir.ptr<!u8i>
// CIR:     cir.yield %[[YIELDED]] : !cir.ptr<!u8i>
// CIR:   %[[RESULT_P:.+]] = cir.cast bitcast %[[ADDR]] : !cir.ptr<!u8i> -> !cir.ptr<!rec_TailPad>
// CIR:   cir.load %[[RESULT_P]] : !cir.ptr<!rec_TailPad>, !rec_TailPad

// LLVM-LABEL: define dso_local noundef i64 @_Z16varargs_tail_padiz(i32 noundef %{{.*}}, ...)
// LLVM:   %[[GP:.+]] = load i32, ptr %{{.+}}, align {{4|16}}
// LLVM:   %[[FITS:.+]] = icmp ule i32 %[[GP]], 40
// LLVM:   %[[VAL:.+]] = load i64, ptr %{{.+}}, align 8
// The load is of the eightbyte the register holds, never of the whole record,
// so no byte of the neighbouring slot reaches the result.
// LLVM-NOT: load %struct.TailPad, ptr %[[RSA:.+]]
// LLVMCIR: store i64 %[[VAL]], ptr %[[TEMP:.+]], align 8
// OGCG:    %[[DST:.+]] = getelementptr i8, ptr %[[TEMP:.+]], i32 0
// OGCG:    store i64 %[[VAL]], ptr %[[DST]], align 8
// LLVMCIR: %[[ADDR:.+]] = phi ptr [ %{{.+}}, %{{.+}} ], [ %[[TEMP]], %{{.+}} ]
// OGCG:    %[[ADDR:.+]] = phi ptr [ %[[TEMP]], %{{.+}} ], [ %{{.+}}, %{{.+}} ]
// LLVMCIR: load %struct.TailPad, ptr %[[ADDR]], align 8
// OGCG:    call void @llvm.memcpy.p0.p0.i64(ptr align 8 %{{.+}}, ptr align 8 %[[ADDR]], i64 16, i1 false)
