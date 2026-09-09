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

// A vector filling one 16-byte slot is fetched from the vector register area.
// CIR-LABEL: cir.func {{.*}} @take_16(
// CIR:         %[[FP_OFFSET_P:.+]] = cir.get_member %{{.+}}[1] {name = "fp_offset"} : !cir.ptr<!rec___va_list_tag> -> !cir.ptr<!u32i>
// CIR:         %[[FP_OFFSET:.+]] = cir.load %[[FP_OFFSET_P]] : !cir.ptr<!u32i>, !u32i
// CIR:         %[[FP_LIMIT:.+]] = cir.const #cir.int<160> : !u32i
// CIR:         cir.cmp le %[[FP_OFFSET]], %[[FP_LIMIT]] : !u32i

// LLVM-LABEL: define dso_local <4 x float> @take_16(i32 noundef %{{.*}}, ...)
// LLVM:         %[[FP_OFFSET:.+]] = load i32, ptr %{{.*}}, align {{[0-9]+}}
// LLVM:         icmp ule i32 %[[FP_OFFSET]], 160
// LLVM:         add i32 %[[FP_OFFSET]], 16

v8f take_32(int count, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, count);
  v8f res = __builtin_va_arg(args, v8f);
  __builtin_va_end(args);
  return res;
}

// A vector too wide for one slot travels in memory, whatever vector registers
// the target has, so there is no register arm to bound and the cursor is
// rounded up to the vector's own 32-byte alignment.
// CIR-LABEL: cir.func {{.*}} @take_32(
// CIR-NOT:     fp_offset
// CIR:         %[[OVERFLOW_P:.+]] = cir.get_member %{{.+}}[2] {name = "overflow_arg_area"}
// CIR:         %[[MASK:.+]] = cir.const #cir.int<18446744073709551584> : !u64i
// CIR:         cir.and %{{.+}}, %[[MASK]] : !u64i
// CIR:         %[[STRIDE:.+]] = cir.const #cir.int<32> : !s32i
// CIR-NOT:     fp_offset
// CIR:       cir.va_end

// LLVM-LABEL: define dso_local <8 x float> @take_32(i32 noundef %{{.*}}, ...)
// LLVM-NOT:     icmp ule i32 %{{.*}}, 160
// LLVMCIR:      and i64 %{{.+}}, -32
// OGCG:         call ptr @llvm.ptrmask.p0.i64(ptr %{{.+}}, i64 -32)
// LLVM:         getelementptr i8, ptr %{{.+}}, i{{32|64}} 32
// LLVM:       call void @llvm.va_end.p0(
