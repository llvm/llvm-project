// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefixes=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=LLVM,OGCG

typedef float v4f __attribute__((ext_vector_type(4)));
typedef float v8f __attribute__((ext_vector_type(8)));

v4f take_16(int count, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, count);
  v4f res = __builtin_va_arg(args, v4f);
  __builtin_va_end(args);
  return res;
}

// A vector filling one 16-byte slot is fetched from the vector register area,
// at the vector cursor, and the overflow arm rounds up to the vector's own
// 16-byte alignment.
// CIR-LABEL: cir.func {{.*}} @take_16(
// CIR:         %[[FP_OFFSET_P:.+]] = cir.get_member %{{.+}}[1] {name = "fp_offset"} : !cir.ptr<!rec___va_list_tag> -> !cir.ptr<!u32i>
// CIR:         %[[FP_OFFSET:.+]] = cir.load %[[FP_OFFSET_P]] : !cir.ptr<!u32i>, !u32i
// CIR:         %[[FP_LIMIT:.+]] = cir.const #cir.int<160> : !u32i
// CIR:         %[[FITS:.+]] = cir.cmp le %[[FP_OFFSET]], %[[FP_LIMIT]] : !u32i
// CIR:         %[[ADDR:.+]] = cir.ternary(%[[FITS]], true {
// CIR:           %[[REG_SAVE_B:.+]] = cir.cast bitcast %{{.+}} : !cir.ptr<!void> -> !cir.ptr<!u8i>
// CIR:           %[[REG_ADDR:.+]] = cir.ptr_stride %[[REG_SAVE_B]], %[[FP_OFFSET]] : (!cir.ptr<!u8i>, !u32i) -> !cir.ptr<!u8i>
// CIR:           cir.yield %[[REG_ADDR]] : !cir.ptr<!u8i>
// The overflow arm rounds the cursor up before reading, and both the read and
// the advance start from the rounded pointer.
// CIR:           %[[OVERFLOW_B:.+]] = cir.cast bitcast %{{.+}} : !cir.ptr<!void> -> !cir.ptr<!u8i>
// CIR:           %[[AS_INT:.+]] = cir.cast ptr_to_int %[[OVERFLOW_B]] : !cir.ptr<!u8i> -> !u64i
// CIR:           %[[BUMP:.+]] = cir.const #cir.int<15> : !u64i
// CIR:           %[[BUMPED:.+]] = cir.add nuw %[[AS_INT]], %[[BUMP]] : !u64i
// CIR:           %[[MASK:.+]] = cir.const #cir.int<18446744073709551600> : !u64i
// CIR:           %[[ROUNDED:.+]] = cir.and %[[BUMPED]], %[[MASK]] : !u64i
// CIR:           %[[ALIGNED:.+]] = cir.cast int_to_ptr %[[ROUNDED]] : !u64i -> !cir.ptr<!u8i>
// CIR:           %[[STRIDE:.+]] = cir.const #cir.int<16> : !s32i
// CIR:           %[[MEM_NEXT:.+]] = cir.ptr_stride %[[ALIGNED]], %[[STRIDE]] : (!cir.ptr<!u8i>, !s32i) -> !cir.ptr<!u8i>
// CIR:           cir.store %[[MEM_NEXT]], %{{.+}} : !cir.ptr<!u8i>, !cir.ptr<!cir.ptr<!u8i>>
// CIR:           cir.yield %[[ALIGNED]] : !cir.ptr<!u8i>
// CIR:         %[[RESULT_P:.+]] = cir.cast bitcast %[[ADDR]] : !cir.ptr<!u8i> -> !cir.ptr<!cir.vector<4 x !cir.float>>
// CIR:         cir.load %[[RESULT_P]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>

// LLVM-LABEL: define dso_local <4 x float> @take_16(i32 noundef %{{.*}}, ...)
// LLVM:         %[[FP_OFFSET:.+]] = load i32, ptr %{{.*}}, align {{[0-9]+}}
// LLVM:         icmp ule i32 %[[FP_OFFSET]], 160
// LLVM:         %[[RSA:.+]] = load ptr, ptr %{{.+}}, align {{[0-9]+}}
// LLVMCIR:      %[[FP64:.+]] = zext i32 %[[FP_OFFSET]] to i64
// LLVMCIR:      %[[REG_ADDR:.+]] = getelementptr i8, ptr %[[RSA]], i64 %[[FP64]]
// OGCG:         %[[REG_ADDR:.+]] = getelementptr i8, ptr %[[RSA]], i32 %[[FP_OFFSET]]
// LLVM:         add i32 %[[FP_OFFSET]], 16
// LLVMCIR:      %[[ALIGNED:.+]] = inttoptr i64 %{{.+}} to ptr
// OGCG:         %[[ALIGNED:.+]] = call ptr @llvm.ptrmask.p0.i64(ptr %{{.+}}, i64 -16)
// LLVM:         %[[MEM_NEXT:.+]] = getelementptr i8, ptr %[[ALIGNED]], i{{32|64}} 16
// LLVM:         store ptr %[[MEM_NEXT]], ptr %{{.+}}, align 8
// LLVMCIR:      %[[ADDR:.+]] = phi ptr [ %[[ALIGNED]], %{{.+}} ], [ %[[REG_ADDR]], %{{.+}} ]
// OGCG:         %[[ADDR:.+]] = phi ptr [ %[[REG_ADDR]], %{{.+}} ], [ %[[ALIGNED]], %{{.+}} ]
// LLVM:         load <4 x float>, ptr %[[ADDR]], align 16

v8f take_32(int count, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, count);
  v8f res = __builtin_va_arg(args, v8f);
  __builtin_va_end(args);
  return res;
}

// A vector too wide for one slot travels in memory, whatever vector registers
// the target has, so the fetch has no register arm at all and the cursor is
// rounded up to the vector's own 32-byte alignment.
// CIR-LABEL: cir.func {{.*}} @take_32(
// CIR-NOT:     fp_offset
// CIR:         %[[OVERFLOW_P:.+]] = cir.get_member %{{.+}}[2] {name = "overflow_arg_area"}
// CIR:         %[[OVERFLOW:.+]] = cir.load %[[OVERFLOW_P]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR:         %[[OVERFLOW_B:.+]] = cir.cast bitcast %[[OVERFLOW]] : !cir.ptr<!void> -> !cir.ptr<!u8i>
// CIR:         %[[AS_INT:.+]] = cir.cast ptr_to_int %[[OVERFLOW_B]] : !cir.ptr<!u8i> -> !u64i
// CIR:         %[[BUMPED:.+]] = cir.add nuw %[[AS_INT]], %{{.+}} : !u64i
// CIR:         %[[MASK:.+]] = cir.const #cir.int<18446744073709551584> : !u64i
// CIR:         %[[ROUNDED:.+]] = cir.and %[[BUMPED]], %[[MASK]] : !u64i
// CIR:         %[[ALIGNED:.+]] = cir.cast int_to_ptr %[[ROUNDED]] : !u64i -> !cir.ptr<!u8i>
// CIR:         %[[STRIDE:.+]] = cir.const #cir.int<32> : !s32i
// CIR:         %[[MEM_NEXT:.+]] = cir.ptr_stride %[[ALIGNED]], %[[STRIDE]] : (!cir.ptr<!u8i>, !s32i) -> !cir.ptr<!u8i>
// CIR:         cir.store %[[MEM_NEXT]], %{{.+}} : !cir.ptr<!u8i>, !cir.ptr<!cir.ptr<!u8i>>
// CIR:         %[[RESULT_P:.+]] = cir.cast bitcast %[[ALIGNED]] : !cir.ptr<!u8i> -> !cir.ptr<!cir.vector<8 x !cir.float>>
// CIR:         cir.load %[[RESULT_P]] : !cir.ptr<!cir.vector<8 x !cir.float>>, !cir.vector<8 x !cir.float>
// CIR-NOT:     fp_offset
// CIR:       cir.va_end

// LLVM-LABEL: define dso_local <8 x float> @take_32(i32 noundef %{{.*}}, ...)
// LLVM-NOT:     icmp ule i32 %{{.*}}, 160
// LLVMCIR:      %[[ALIGNED:.+]] = inttoptr i64 %{{.+}} to ptr
// OGCG:         %[[ALIGNED:.+]] = call ptr @llvm.ptrmask.p0.i64(ptr %{{.+}}, i64 -32)
// LLVM:         %[[MEM_NEXT:.+]] = getelementptr i8, ptr %[[ALIGNED]], i{{32|64}} 32
// LLVM:         store ptr %[[MEM_NEXT]], ptr %{{.+}}, align 8
// LLVM:         load <8 x float>, ptr %[[ALIGNED]], align 32
// LLVM:       call void @llvm.va_end.p0(
