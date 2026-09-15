// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM,OGCG --input-file=%t.ll %s

long double varargs_long_double(int count, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, count);
  long double res = __builtin_va_arg(args, long double);
  __builtin_va_end(args);
  return res;
}

// x87 long double is always MEMORY class: no register-save-area branch.
// CIR-LABEL: cir.func {{.*}} @varargs_long_double(
// CIR-SAME: -> !cir.long_double<!cir.f80>
// CIR:   cir.va_start %{{.+}} : !cir.ptr<!rec___va_list_tag>
// CIR:   %[[VA_PTR:.+]] = cir.cast array_to_ptrdecay %{{.+}} : !cir.ptr<!cir.array<!rec___va_list_tag x 1>> -> !cir.ptr<!rec___va_list_tag>
// CIR:   %[[OVERFLOW_P:.+]] = cir.get_member %[[VA_PTR]][2] {name = "overflow_arg_area"} : !cir.ptr<!rec___va_list_tag> -> !cir.ptr<!cir.ptr<!void>>
// CIR:   %[[OVERFLOW:.+]] = cir.load %[[OVERFLOW_P]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR:   %[[OVERFLOW_B:.+]] = cir.cast bitcast %[[OVERFLOW]] : !cir.ptr<!void> -> !cir.ptr<!u8i>
// CIR:   %[[AS_INT:.+]] = cir.cast ptr_to_int %[[OVERFLOW_B]] : !cir.ptr<!u8i> -> !u64i
// CIR:   %[[BUMP:.+]] = cir.const #cir.int<15> : !u64i
// CIR:   %[[BUMPED:.+]] = cir.add nuw %[[AS_INT]], %[[BUMP]] : !u64i
// CIR:   %[[MASK:.+]] = cir.const #cir.int<18446744073709551600> : !u64i
// CIR:   %[[ROUNDED:.+]] = cir.and %[[BUMPED]], %[[MASK]] : !u64i
// CIR:   %[[ALIGNED:.+]] = cir.cast int_to_ptr %[[ROUNDED]] : !u64i -> !cir.ptr<!u8i>
// CIR:   %[[STRIDE:.+]] = cir.const #cir.int<16> : !s32i
// CIR:   %[[OVERFLOW_NEXT:.+]] = cir.ptr_stride %[[ALIGNED]], %[[STRIDE]] : (!cir.ptr<!u8i>, !s32i) -> !cir.ptr<!u8i>
// CIR:   cir.store %[[OVERFLOW_NEXT]], %{{.+}} : !cir.ptr<!u8i>, !cir.ptr<!cir.ptr<!u8i>>
// CIR:   %[[VA_ARG_B:.+]] = cir.cast bitcast %[[ALIGNED]] : !cir.ptr<!u8i> -> !cir.ptr<!cir.long_double<!cir.f80>>
// CIR:   %[[VA_ARG_V:.+]] = cir.load %[[VA_ARG_B]] : !cir.ptr<!cir.long_double<!cir.f80>>, !cir.long_double<!cir.f80>

// LLVM-LABEL: define dso_local x86_fp80 @varargs_long_double(i32 noundef %{{.*}}, ...)
// LLVM:   call void @llvm.va_start.p0(ptr %{{.*}})
// LLVM:   %[[OVERFLOW_P:.+]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr %{{.*}}, i32 0, i32 2
// LLVM:   %[[OVERFLOW:.+]] = load ptr, ptr %[[OVERFLOW_P]], align 8
// The overflow cursor is rounded up to 16 before the read, spelled as plain
// arithmetic on one side and as a pointer mask on the other.
// LLVMCIR: %[[AS_INT:.+]] = ptrtoint ptr %[[OVERFLOW]] to i64
// LLVMCIR: %[[BUMPED:.+]] = add nuw i64 %[[AS_INT]], 15
// LLVMCIR: %[[ROUNDED:.+]] = and i64 %[[BUMPED]], -16
// LLVMCIR: %[[ALIGNED:.+]] = inttoptr i64 %[[ROUNDED]] to ptr
// OGCG:    %[[UNALIGNED:.+]] = getelementptr inbounds i8, ptr %[[OVERFLOW]], i32 15
// OGCG:    %[[ALIGNED:.+]] = call ptr @llvm.ptrmask.p0.i64(ptr %[[UNALIGNED]], i64 -16)
// LLVM:   %[[OVERFLOW_NEXT:.+]] = getelementptr i8, ptr %[[ALIGNED]], i{{32|64}} 16
// LLVM:   store ptr %[[OVERFLOW_NEXT]], ptr %[[OVERFLOW_P]], align 8
// LLVM:   %[[VA_ARG:.+]] = load x86_fp80, ptr %[[ALIGNED]], align 16
// LLVM:   store x86_fp80 %[[VA_ARG]], ptr %{{.*}}, align 16
