// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// Tests for array new expressions where the array size is not implicitly
// converted to size_t by Sema (pre-C++14 behavior). The CIR codegen must
// handle the signed/width conversion itself.

typedef __typeof__(sizeof(int)) size_t;

// Sized non-allocating (placement) new.
void *operator new[](size_t, void *p) noexcept { return p; }

struct S {
  int x;
  S();
};

// Signed int array size with multi-byte element (typeSizeMultiplier != 1).
// The sign extension is done and the multiply overflow catches negative values.
void t_new_signed_size(int n) {
  auto p = new double[n];
}

// CIR-LABEL: cir.func {{.*}} @_Z17t_new_signed_sizei
// CIR:    %[[N:.*]] = cir.load{{.*}} %[[ARG_ALLOCA:.*]]
// CIR:    %[[N_SEXT:.*]] = cir.cast integral %[[N]] : !s32i -> !s64i
// CIR:    %[[N_SIZE_T:.*]] = cir.cast integral %[[N_SEXT]] : !s64i -> !u64i
// CIR:    %[[ELEMENT_SIZE:.*]] = cir.const #cir.int<8> : !u64i
// CIR:    %[[RESULT:.*]], %[[OVERFLOW:.*]] = cir.mul.overflow %[[N_SIZE_T]], %[[ELEMENT_SIZE]] : !u64i -> !u64i
// CIR:    %[[ALL_ONES:.*]] = cir.const #cir.int<18446744073709551615> : !u64i
// CIR:    %[[ALLOC_SIZE:.*]] = cir.select if %[[OVERFLOW]] then %[[ALL_ONES]] else %[[RESULT]]

// LLVM-LABEL: define{{.*}} void @_Z17t_new_signed_sizei
// LLVM:    %[[N:.*]] = load i32, ptr %{{.+}}
// LLVM:    %[[N_SIZE_T:.*]] = sext i32 %[[N]] to i64
// LLVM:    %[[MUL_OVERFLOW:.*]] = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %[[N_SIZE_T]], i64 8)

// OGCG-LABEL: define{{.*}} void @_Z17t_new_signed_sizei
// OGCG:    %[[N:.*]] = load i32, ptr %{{.+}}
// OGCG:    %[[N_SIZE_T:.*]] = sext i32 %[[N]] to i64
// OGCG:    %[[MUL_OVERFLOW:.*]] = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %[[N_SIZE_T]], i64 8)

// Signed int array size with single-byte element (typeSizeMultiplier == 1).
// A signed comparison catches negative values directly.
void t_new_signed_size_char(int n) {
  auto p = new char[n];
}

// CIR-LABEL: cir.func {{.*}} @_Z22t_new_signed_size_chari
// CIR:    %[[N:.*]] = cir.load{{.*}} %[[ARG_ALLOCA:.*]]
// CIR:    %[[N_SEXT:.*]] = cir.cast integral %[[N]] : !s32i -> !s64i
// CIR:    %[[ZERO:.*]] = cir.const #cir.int<0> : !s64i
// CIR:    %[[IS_NEG:.*]] = cir.cmp lt %[[N_SEXT]], %[[ZERO]] : !s64i
// CIR:    %[[N_SIZE_T:.*]] = cir.cast integral %[[N_SEXT]] : !s64i -> !u64i
// CIR:    %[[ALL_ONES:.*]] = cir.const #cir.int<18446744073709551615> : !u64i
// CIR:    %[[ALLOC_SIZE:.*]] = cir.select if %[[IS_NEG]] then %[[ALL_ONES]] else %[[N_SIZE_T]]

// LLVM-LABEL: define{{.*}} void @_Z22t_new_signed_size_chari
// LLVM:    %[[N:.*]] = load i32, ptr %{{.+}}
// LLVM:    %[[N_SIZE_T:.*]] = sext i32 %[[N]] to i64
// LLVM:    %[[IS_NEG:.*]] = icmp slt i64 %[[N_SIZE_T]], 0

// OGCG-LABEL: define{{.*}} void @_Z22t_new_signed_size_chari
// OGCG:    %[[N:.*]] = load i32, ptr %{{.+}}
// OGCG:    %[[N_SIZE_T:.*]] = sext i32 %[[N]] to i64
// OGCG:    %[[IS_NEG:.*]] = icmp slt i64 %[[N_SIZE_T]], 0
