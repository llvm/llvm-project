// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM,OGCG --input-file=%t.ll %s

__int128 varargs_int128(int count, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, count);
  __int128 res = __builtin_va_arg(args, __int128);
  __builtin_va_end(args);
  return res;
}

// __int128 spans two INTEGER eightbytes, so it needs two integer registers
// rather than one, which lowers the gate and doubles the bump.
// CIR-LABEL: cir.func {{.*}} @varargs_int128(
// CIR:   %[[GP_OFFSET_P:.+]] = cir.get_member %{{.+}}[0] {name = "gp_offset"} : !cir.ptr<!rec___va_list_tag> -> !cir.ptr<!u32i>
// CIR:   %[[GP_OFFSET:.+]] = cir.load %[[GP_OFFSET_P]] : !cir.ptr<!u32i>, !u32i
// CIR:   %[[GP_LIMIT:.+]] = cir.const #cir.int<32> : !u32i
// CIR:   %[[FITS_GP:.+]] = cir.cmp le %[[GP_OFFSET]], %[[GP_LIMIT]] : !u32i
// CIR:   %[[VA_ARG:.+]] = cir.ternary(%[[FITS_GP]], true {
// CIR:     %[[REG_SAVE:.+]] = cir.load %{{.+}}
// CIR:     %[[REG_SAVE_B:.+]] = cir.cast bitcast %[[REG_SAVE]] : !cir.ptr<!void> -> !cir.ptr<!u8i>
// CIR:     %[[REG_ADDR:.+]] = cir.ptr_stride %[[REG_SAVE_B]], %[[GP_OFFSET]] : (!cir.ptr<!u8i>, !u32i) -> !cir.ptr<!u8i>
// CIR:     %[[REG_TMP:.+]] = cir.alloca "vaarg.reg" {{.*}} : !cir.ptr<!s128i>
// CIR:     %[[REG_ADDR_V:.+]] = cir.cast bitcast %[[REG_ADDR]] : !cir.ptr<!u8i> -> !cir.ptr<!s128i>
// CIR:     %[[REG_VAL:.+]] = cir.load align(8) %[[REG_ADDR_V]] : !cir.ptr<!s128i>, !s128i
// CIR:     cir.store %[[REG_VAL]], %[[REG_TMP]] : !s128i, !cir.ptr<!s128i>
// CIR:     %[[REG_TMP_B:.+]] = cir.cast bitcast %[[REG_TMP]] : !cir.ptr<!s128i> -> !cir.ptr<!u8i>
// CIR:     %[[GP_BUMP:.+]] = cir.const #cir.int<16> : !u32i
// CIR:     %[[GP_NEXT:.+]] = cir.add %[[GP_OFFSET]], %[[GP_BUMP]] : !u32i
// CIR:     cir.store %[[GP_NEXT]], %[[GP_OFFSET_P]] : !u32i, !cir.ptr<!u32i>
// CIR:     cir.yield %[[REG_TMP_B]] : !cir.ptr<!u8i>
// CIR:   }, false {
// CIR:     %[[OVERFLOW:.+]] = cir.load %{{.+}}
// CIR:     cir.yield %{{.+}} : !cir.ptr<!u8i>
// CIR:   }) : (!cir.bool) -> !cir.ptr<!u8i>
// CIR:   %[[VA_ARG_B:.+]] = cir.cast bitcast %[[VA_ARG]] : !cir.ptr<!u8i> -> !cir.ptr<!s128i>
// CIR:   %[[VA_ARG_V:.+]] = cir.load %[[VA_ARG_B]] : !cir.ptr<!s128i>, !s128i

// LLVM-LABEL: define dso_local i128 @varargs_int128(i32 noundef %{{.*}}, ...)
// LLVM:   %[[GP_OFFSET_P:.+]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr %{{.*}}, i32 0, i32 0
// LLVM:   %[[GP_OFFSET:.+]] = load i32, ptr %[[GP_OFFSET_P]], align {{[0-9]+}}
// LLVM:   %[[FITS_GP:.+]] = icmp ule i32 %[[GP_OFFSET]], 32
// LLVM:   br i1 %[[FITS_GP]], label %[[REG_BB:.+]], label %[[MEM_BB:.+]]
// LLVM: [[REG_BB]]:
// LLVM:   %[[REG_SAVE:.+]] = load ptr, ptr %{{.*}}, align {{[0-9]+}}
// LLVM:   %[[REG_ADDR:.+]] = getelementptr i8, ptr %[[REG_SAVE]], i{{32|64}} %{{.*}}
// A GP register slot is only 8-byte aligned, so the value is copied to a
// 16-byte-aligned temp before it is read as a whole.
// LLVMCIR: %[[VAL:.+]] = load i128, ptr %[[REG_ADDR]], align 8
// LLVMCIR: store i128 %[[VAL]], ptr %[[REG_TMP:.+]], align 16
// OGCG:    call void @llvm.memcpy.p0.p0.i64(ptr align 16 %[[REG_TMP:.+]], ptr align 8 %[[REG_ADDR]], i64 16, i1 false)
// LLVM:   %[[GP_NEXT:.+]] = add i32 %[[GP_OFFSET]], 16
// LLVM:   store i32 %[[GP_NEXT]], ptr %[[GP_OFFSET_P]], align {{[0-9]+}}
// LLVM:   br label %[[END_BB:.+]]
// LLVM: [[MEM_BB]]:
// LLVM:   br label %[[END_BB]]
// LLVM: [[END_BB]]:
// LLVMCIR: %[[ADDR:.+]] = phi ptr [ %{{.*}}, %[[MEM_BB]] ], [ %[[REG_TMP]], %[[REG_BB]] ]
// OGCG:    %[[ADDR:.+]] = phi ptr [ %[[REG_TMP]], %[[REG_BB]] ], [ %{{.*}}, %[[MEM_BB]] ]
// LLVM:   %[[VA_ARG:.+]] = load i128, ptr %[[ADDR]], align 16
// LLVM:   store i128 %[[VA_ARG]], ptr %{{.*}}, align 16
