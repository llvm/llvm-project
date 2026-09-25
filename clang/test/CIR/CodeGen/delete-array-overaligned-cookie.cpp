// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -mconstructor-aliases -emit-cir -mmlir -mlir-print-ir-before=cir-cxxabi-lowering %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --input-file=%t-before.cir -check-prefix=CIR,CIR-BEFORE %s
// RUN: FileCheck --input-file=%t.cir --check-prefix=CIR,CIR-AFTER %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -mconstructor-aliases -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -mconstructor-aliases -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll --check-prefix=LLVM %s

typedef decltype(sizeof(0)) size_t;
namespace std { enum class align_val_t : size_t {}; }

struct alignas(64) OverAligned {
  int *p;
  OverAligned() : p(nullptr) {}
  ~OverAligned();
};

void test_delete_array(OverAligned *a) {
  delete[] a;
}
// CIR-LABEL: cir.func {{.*}}@_Z17test_delete_arrayP11OverAligned(
// CIR-BEFORE:     cir.delete_array %{{.*}} : !cir.ptr<!rec_OverAligned> {delete_fn = @_ZdaPvmSt11align_val_t, delete_params = #cir.usual_delete_params<size = true, alignment = 64>, element_align = 64 : i64, element_dtor = @_ZN11OverAlignedD1Ev}

// CIR-AFTER: %[[A:.*]] = cir.alloca "a" align(8) init : !cir.ptr<!cir.ptr<!rec_OverAligned>>
// CIR-AFTER: %[[LOAD_A:.*]] = cir.load align(8) %[[A]] : !cir.ptr<!cir.ptr<!rec_OverAligned>>, !cir.ptr<!rec_OverAligned>
// CIR-AFTER:   %[[CAST_TO_BYTES:.*]] = cir.cast bitcast %[[LOAD_A]] : !cir.ptr<!rec_OverAligned> -> !cir.ptr<!u8i>
// CIR-AFTER:   %[[COOKIE_OFFSET:.*]] = cir.const #cir.int<-64> : !s64i
// CIR-AFTER:   %[[BEFORE_COOKIE:.*]] = cir.ptr_stride %[[CAST_TO_BYTES]], %[[COOKIE_OFFSET]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR-AFTER:   %[[COUNT_OFFSET:.*]] = cir.const #cir.int<56> : !s64i
// CIR-AFTER:   %[[AFTER_COUNT:.*]] = cir.ptr_stride %[[BEFORE_COOKIE]], %[[COUNT_OFFSET]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR-AFTER:   %[[TO_SIZE:.*]] = cir.cast bitcast %[[AFTER_COUNT]] : !cir.ptr<!u8i> -> !cir.ptr<!u64i>
// CIR-AFTER:   %[[LOAD_VAL:.*]] = cir.load align(8) %[[TO_SIZE]] : !cir.ptr<!u64i>, !u64i
// CIR-AFTER:   } cleanup normal {
// CIR-AFTER:     %[[ELT_SIZE:.*]] = cir.const #cir.int<64> : !u64i
// CIR-AFTER:     %[[ELT_OFFSET:.*]] = cir.mul %[[ELT_SIZE]], %[[LOAD_VAL]] : !u64i
// CIR-AFTER:     %[[COOKIE_SIZE:.*]] = cir.const #cir.int<64> : !u64i
// CIR-AFTER:     %[[TOTAL_SIZE:.*]] = cir.add %[[ELT_OFFSET]], %[[COOKIE_SIZE]] : !u64i
// CIR-AFTER:     %[[ALIGN:.*]] = cir.const #cir.int<64> : !u64i
// CIR-AFTER:  cir.call @_ZdaPvmSt11align_val_t(%{{.*}}, %[[TOTAL_SIZE]], %[[ALIGN]]) nothrow : (!cir.ptr<!void>, !u64i, !u64i) -> ()

// LLVM-LABEL: define {{.*}}@_Z17test_delete_arrayP11OverAligned(
// LLVM: %[[A:.*]] = alloca ptr, align 8
// LLVM: %[[LOAD_A:.*]] = load ptr, ptr %[[A]], align 8
// LLVM: %[[BEFORE_COOKIE:.*]] = getelementptr {{.*}}i8, ptr %[[LOAD_A]], i64 -64
// LLVM: %[[AFTER_COUNT:.*]] = getelementptr {{.*}}i8, ptr %[[BEFORE_COOKIE]], i64 56
// LLVM: %[[LOAD_VAL:.*]] = load i64, ptr %[[AFTER_COUNT]], align 8
// LLVM: %[[ELT_OFFSET:.*]] = mul i64 64, %[[LOAD_VAL]]
// LLVM: %[[TOTAL_SIZE:.*]] = add i64 %[[ELT_OFFSET]], 64
// LLVM: call void @_ZdaPvmSt11align_val_t(ptr {{.*}}%{{.*}}, i64 {{.*}}%[[TOTAL_SIZE]], i64 {{.*}}64) 


struct alignas(16) NotAlignedNewButNeedCookie {
  float x, y, z, a;
  ~NotAlignedNewButNeedCookie();
};

void test_not_aligned_new_but_need_cookie(NotAlignedNewButNeedCookie *a) {
  delete [] a;
}
// CIR-LABEL: cir.func {{.*}}@_Z36test_not_aligned_new_but_need_cookieP26NotAlignedNewButNeedCookie(
// CIR-BEFORE: cir.delete_array %{{.*}} : !cir.ptr<!rec_NotAlignedNewButNeedCookie> {delete_fn = @_ZdaPvm, delete_params = #cir.usual_delete_params<size = true>, element_align = 16 : i64, element_dtor = @_ZN26NotAlignedNewButNeedCookieD1Ev}
// CIR-AFTER: %[[A:.*]] = cir.alloca "a" align(8) init : !cir.ptr<!cir.ptr<!rec_NotAlignedNewButNeedCookie>>

// CIR-AFTER: %[[LOAD_A:.*]] = cir.load align(8) %[[A]] : !cir.ptr<!cir.ptr<!rec_NotAlignedNewButNeedCookie>>, !cir.ptr<!rec_NotAlignedNewButNeedCookie>
// CIR-AFTER: %[[CAST_TO_BYTES:.*]] = cir.cast bitcast %[[LOAD_A]] : !cir.ptr<!rec_NotAlignedNewButNeedCookie> -> !cir.ptr<!u8i>
// CIR-AFTER: %[[COOKIE_OFFSET:.*]] = cir.const #cir.int<-16> : !s64i
// CIR-AFTER: %[[BEFORE_COOKIE:.*]] = cir.ptr_stride %[[CAST_TO_BYTES]], %[[COOKIE_OFFSET]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR-AFTER: %[[COUNT_OFFSET:.*]] = cir.const #cir.int<8> : !s64i
// CIR-AFTER: %[[AFTER_COUNT:.*]] = cir.ptr_stride %[[BEFORE_COOKIE]], %[[COUNT_OFFSET]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR-AFTER: %[[TO_SIZE:.*]] = cir.cast bitcast %[[AFTER_COUNT]] : !cir.ptr<!u8i> -> !cir.ptr<!u64i>
// CIR-AFTER: %[[LOAD_VAL:.*]] = cir.load align(8) %[[TO_SIZE]] : !cir.ptr<!u64i>, !u64i
// CIR-AFTER: } cleanup normal {
// CIR-AFTER:   %[[ELT_SIZE:.*]] = cir.const #cir.int<16> : !u64i
// CIR-AFTER:   %[[ELT_OFFSET:.*]] = cir.mul %[[ELT_SIZE]], %[[LOAD_VAL]] : !u64i
// CIR-AFTER:   %[[COOKIE_SIZE:.*]] = cir.const #cir.int<16> : !u64i
// CIR-AFTER:   %[[TOTAL_SIZE:.*]] = cir.add %[[ELT_OFFSET]], %[[COOKIE_SIZE]] : !u64i
// CIR-AFTER:   cir.call @_ZdaPvm(%{{.*}}, %[[TOTAL_SIZE]]) nothrow : (!cir.ptr<!void>, !u64i) -> ()

// LLVM-LABEL: define {{.*}}@_Z36test_not_aligned_new_but_need_cookieP26NotAlignedNewButNeedCookie(
// LLVM: %[[A:.*]] = alloca ptr, align 8
// LLVM: %[[LOAD_A:.*]] = load ptr, ptr %[[A]], align 8
// LLVM: %[[BEFORE_COOKIE:.*]] = getelementptr {{.*}}i8, ptr %[[LOAD_A]], i64 -16
// LLVM: %[[AFTER_COUNT:.*]] = getelementptr {{.*}}i8, ptr %[[BEFORE_COOKIE]], i64 8
// LLVM: %[[LOAD_VAL:.*]] = load i64, ptr %[[AFTER_COUNT]], align 8
// LLVM: %[[ELT_OFFEST:.*]] = mul i64 16, %[[LOAD_VAL]]
// LLVM: %[[TOTAL_SIZE:.*]] = add i64 %[[ELT_OFFSET]], 16
// LLVM: call void @_ZdaPvm(ptr {{.*}}%{{.*}}, i64 {{.*}}%[[TOTAL_SIZE]])
