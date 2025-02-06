// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -I%S/../Inputs -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

#include "std-cxx.h"

void t_new_constant_size() {
  auto p = new double[16];
}

// LLVM: @_Z19t_new_constant_sizev()
// LLVM:   %[[ALLOCA:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[ADDR:.*]] = call ptr @_Znam(i64 128)
// LLVM:   store ptr %[[ADDR]], ptr %[[ALLOCA]], align 8

void t_new_multidim_constant_size() {
  auto p = new double[2][3][4];
}

// LLVM: @_Z28t_new_multidim_constant_sizev()
// LLVM:   %[[ALLOCA:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[ADDR:.*]] = call ptr @_Znam(i64 192)
// LLVM:   store ptr %[[ADDR]], ptr %[[ALLOCA]], align 8

class C {
  public:
    ~C();
};

void t_constant_size_nontrivial() {
  auto p = new C[3];
}

// Note: The below differs from the IR emitted by clang without -fclangir in
//       several respects. (1) The alloca here has an extra "i64 1"
//       (2) The operator new call is missing "noalias noundef nonnull" on
//       the call and "noundef" on the argument, (3) The getelementptr is
//       missing "inbounds"

// LLVM: @_Z26t_constant_size_nontrivialv()
// LLVM:   %[[ALLOCA:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[COOKIE_PTR:.*]] = call ptr @_Znam(i64 11)
// LLVM:   store i64 3, ptr %[[COOKIE_PTR]], align 8
// LLVM:   %[[ALLOCATED_PTR:.*]] = getelementptr i8, ptr %[[COOKIE_PTR]], i64 8
// LLVM:   store ptr %[[ALLOCATED_PTR]], ptr %[[ALLOCA]], align 8

class D {
  public:
    int x;
    ~D();
};

void t_constant_size_nontrivial2() {
  auto p = new D[3];
}

// LLVM: @_Z27t_constant_size_nontrivial2v()
// LLVM:   %[[ALLOCA:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[COOKIE_PTR:.*]] = call ptr @_Znam(i64 20)
// LLVM:   store i64 3, ptr %[[COOKIE_PTR]], align 8
// LLVM:   %[[ALLOCATED_PTR:.*]] = getelementptr i8, ptr %[[COOKIE_PTR]], i64 8
// LLVM:   store ptr %[[ALLOCATED_PTR]], ptr %[[ALLOCA]], align 8

void t_constant_size_memset_init() {
  auto p = new int[16] {};
}

// LLVM: @_Z27t_constant_size_memset_initv()
// LLVM:   %[[ALLOCA:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[ADDR:.*]] = call ptr @_Znam(i64 64)
// LLVM:   call void @llvm.memset.p0.i64(ptr %[[ADDR]], i8 0, i64 64, i1 false)
// LLVM:   store ptr %[[ADDR]], ptr %[[ALLOCA]], align 8

void t_constant_size_partial_init() {
  auto p = new int[16] { 1, 2, 3 };
}

// LLVM: @_Z28t_constant_size_partial_initv()
// LLVM:   %[[ALLOCA:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[ADDR:.*]] = call ptr @_Znam(i64 64)
// LLVM:   store i32 1, ptr %[[ADDR]], align 4
// LLVM:   %[[ELEM_1_PTR:.*]] = getelementptr i32, ptr %[[ADDR]], i64 1
// LLVM:   store i32 2, ptr %[[ELEM_1_PTR]], align 4
// LLVM:   %[[ELEM_2_PTR:.*]] = getelementptr i32, ptr %[[ELEM_1_PTR]], i64 1
// LLVM:   store i32 3, ptr %[[ELEM_2_PTR]], align 4
// LLVM:   %[[ELEM_3_PTR:.*]] = getelementptr i32, ptr %[[ELEM_2_PTR]], i64 1
// LLVM:   call void @llvm.memset.p0.i64(ptr %[[ELEM_3_PTR]], i8 0, i64 52, i1 false)
// LLVM:   store ptr %[[ADDR]], ptr %[[ALLOCA]], align 8

void t_new_var_size(size_t n) {
  auto p = new char[n];
}

// LLVM:  @_Z14t_new_var_sizem
// LLVM:    %[[N:.*]] = load i64, ptr %[[ARG_ALLOCA:.*]]
// LLVM:    %[[ADDR:.*]] = call ptr @_Znam(i64 %[[N]])

void t_new_var_size2(int n) {
  auto p = new char[n];
}

// LLVM:  @_Z15t_new_var_size2i
// LLVM:    %[[N:.*]] = load i32, ptr %[[ARG_ALLOCA:.*]]
// LLVM:    %[[N_SIZE_T:.*]] = sext i32 %[[N]] to i64
// LLVM:    %[[ADDR:.*]] = call ptr @_Znam(i64 %[[N_SIZE_T]])

void t_new_var_size3(size_t n) {
  auto p = new double[n];
}

// LLVM:  @_Z15t_new_var_size3m
// LLVM:    %[[N:.*]] = load i64, ptr %[[ARG_ALLOCA:.*]]
// LLVM:    %[[RESULT_PAIR:.*]] = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %[[N]], i64 8)
// LLVM:    %[[RESULT:.*]] = extractvalue { i64, i1 } %[[RESULT_PAIR]], 0
// LLVM:    %[[OVERFLOW:.*]] = extractvalue { i64, i1 } %[[RESULT_PAIR]], 1
// LLVM:    %[[ALLOC_SIZE:.*]] = select i1 %[[OVERFLOW]], i64 -1, i64 %[[RESULT]]
// LLVM:    %[[ADDR:.*]] = call ptr @_Znam(i64 %[[ALLOC_SIZE]])

void t_new_var_size4(int n) {
  auto p = new double[n];
}

// LLVM:  @_Z15t_new_var_size4i
// LLVM:    %[[N:.*]] = load i32, ptr %[[ARG_ALLOCA:.*]]
// LLVM:    %[[N_SIZE_T:.*]] = sext i32 %[[N]] to i64
// LLVM:    %[[RESULT_PAIR:.*]] = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %[[N_SIZE_T]], i64 8)
// LLVM:    %[[RESULT:.*]] = extractvalue { i64, i1 } %[[RESULT_PAIR]], 0
// LLVM:    %[[OVERFLOW:.*]] = extractvalue { i64, i1 } %[[RESULT_PAIR]], 1
// LLVM:    %[[ALLOC_SIZE:.*]] = select i1 %[[OVERFLOW]], i64 -1, i64 %[[RESULT]]
// LLVM:    %[[ADDR:.*]] = call ptr @_Znam(i64 %[[ALLOC_SIZE]])

void t_new_var_size5(int n) {
  auto p = new double[n][2][3];
}

// NUM_ELEMENTS is not used in this case because cookies aren't required

// LLVM:  @_Z15t_new_var_size5i
// LLVM:    %[[N:.*]] = load i32, ptr %[[ARG_ALLOCA:.*]]
// LLVM:    %[[N_SIZE_T:.*]] = sext i32 %[[N]] to i64
// LLVM:    %[[RESULT_PAIR:.*]] = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %[[N_SIZE_T]], i64 48)
// LLVM:    %[[RESULT:.*]] = extractvalue { i64, i1 } %[[RESULT_PAIR]], 0
// LLVM:    %[[OVERFLOW:.*]] = extractvalue { i64, i1 } %[[RESULT_PAIR]], 1
// LLVM:    %[[NUM_ELEMENTS:.*]] = mul i64 %[[N_SIZE_T]], 6
// LLVM:    %[[ALLOC_SIZE:.*]] = select i1 %[[OVERFLOW]], i64 -1, i64 %[[RESULT]]
// LLVM:    %[[ADDR:.*]] = call ptr @_Znam(i64 %[[ALLOC_SIZE]])

void t_new_var_size6(int n) {
  auto p = new double[n] { 1, 2, 3 };
}

// LLVM:  @_Z15t_new_var_size6i
// LLVM:    %[[N:.*]] = load i32, ptr %[[ARG_ALLOCA:.*]]
// LLVM:    %[[N_SIZE_T:.*]] = sext i32 %[[N]] to i64
// LLVM:    %[[LT_MIN_SIZE:.*]] = icmp ult i64 %[[N_SIZE_T]], 3
// LLVM:    %[[RESULT_PAIR:.*]] = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %[[N_SIZE_T]], i64 8)
// LLVM:    %[[RESULT:.*]] = extractvalue { i64, i1 } %[[RESULT_PAIR]], 0
// LLVM:    %[[OVERFLOW:.*]] = extractvalue { i64, i1 } %[[RESULT_PAIR]], 1
// LLVM:    %[[ANY_OVERFLOW:.*]] = or i1 %[[LT_MIN_SIZE]], %[[OVERFLOW]]
// LLVM:    %[[ALLOC_SIZE:.*]] = select i1 %[[ANY_OVERFLOW]], i64 -1, i64 %[[RESULT]]
// LLVM:    %[[ADDR:.*]] = call ptr @_Znam(i64 %[[ALLOC_SIZE]])

void t_new_var_size7(__int128 n) {
  auto p = new double[n];
}

// LLVM:  @_Z15t_new_var_size7n
// LLVM:    %[[N:.*]] = load i128, ptr %[[ARG_ALLOCA:.*]]
// LLVM:    %[[N_SIZE_T:.*]] = trunc i128 %[[N]] to i64
// LLVM:    %[[RESULT_PAIR:.*]] = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %[[N_SIZE_T]], i64 8)
// LLVM:    %[[RESULT:.*]] = extractvalue { i64, i1 } %[[RESULT_PAIR]], 0
// LLVM:    %[[OVERFLOW:.*]] = extractvalue { i64, i1 } %[[RESULT_PAIR]], 1
// LLVM:    %[[ALLOC_SIZE:.*]] = select i1 %[[OVERFLOW]], i64 -1, i64 %[[RESULT]]
// LLVM:    %[[ADDR:.*]] = call ptr @_Znam(i64 %[[ALLOC_SIZE]])

void t_new_var_size_nontrivial(size_t n) {
  auto p = new D[n];
}

// LLVM:  @_Z25t_new_var_size_nontrivialm
// LLVM:    %[[N:.*]] = load i64, ptr %[[ARG_ALLOCA:.*]]
// LLVM:    %[[RESULT_PAIR:.*]] = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %[[N]], i64 4)
// LLVM:    %[[SIZE_WITHOUT_COOKIE:.*]] = extractvalue { i64, i1 } %[[RESULT_PAIR]], 0
// LLVM:    %[[OVERFLOW:.*]] = extractvalue { i64, i1 } %[[RESULT_PAIR]], 1
// LLVM:    %[[RESULT_PAIR2:.*]] = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %[[SIZE_WITHOUT_COOKIE]], i64 8)
// LLVM:    %[[SIZE:.*]] = extractvalue { i64, i1 } %[[RESULT_PAIR2]], 0
// LLVM:    %[[OVERFLOW2:.*]] = extractvalue { i64, i1 } %[[RESULT_PAIR2]], 1
// LLVM:    %[[ANY_OVERFLOW:.*]] = or i1 %[[OVERFLOW]], %[[OVERFLOW2]]
// LLVM:    %[[ALLOC_SIZE:.*]] = select i1 %[[ANY_OVERFLOW]], i64 -1, i64 %[[SIZE]]
// LLVM:    %[[ADDR:.*]] = call ptr @_Znam(i64 %[[ALLOC_SIZE]])
