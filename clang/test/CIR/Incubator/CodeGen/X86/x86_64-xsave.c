// RUN: %clang_cc1 %s -DTEST_XSAVE -O0 -triple=x86_64-unknown-linux -target-feature +xsave -fclangir -emit-cir -o %t.cir -Wall -Wno-unused-but-set-variable -Werror
// RUN: FileCheck --check-prefix=CIR-XSAVE --input-file=%t.cir %s
// RUN: %clang_cc1 %s -DTEST_XSAVE -O0 -triple=x86_64-unknown-linux -target-feature +xsave -fclangir -emit-llvm -o %t.ll -Wall -Wno-unused-but-set-variable -Werror
// RUN: FileCheck --check-prefix=LLVM-XSAVE --input-file=%t.ll %s
// RUN: %clang_cc1 %s -DTEST_XSAVE -O0 -triple=x86_64-unknown-linux -target-feature +xsave -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Wno-unused-but-set-variable -Werror
// RUN: FileCheck --check-prefix=CIR-XSAVE --input-file=%t.cir %s
// RUN: %clang_cc1 %s -DTEST_XSAVE -O0 -triple=x86_64-unknown-linux -target-feature +xsave -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Wno-unused-but-set-variable -Werror
// RUN: FileCheck --check-prefix=LLVM-XSAVE --input-file=%t.ll %s

// RUN: %clang_cc1 %s -DTEST_XGETBV -O0 -triple=x86_64-unknown-linux -target-feature +xsave -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Wno-unused-but-set-variable -Werror
// RUN: FileCheck --check-prefix=CIR-XGETBV --input-file=%t.cir %s
// RUN: %clang_cc1 %s -DTEST_XGETBV -O0 -triple=x86_64-unknown-linux -target-feature +xsave -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Wno-unused-but-set-variable -Werror
// RUN: FileCheck --check-prefix=LLVM-XGETBV --input-file=%t.ll %s
// RUN: %clang_cc1 %s -DTEST_XSETBV -O0 -triple=x86_64-unknown-linux -target-feature +xsave -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Wno-unused-but-set-variable -Werror
// RUN: FileCheck --check-prefix=CIR-XSETBV --input-file=%t.cir %s
// RUN: %clang_cc1 %s -DTEST_XSETBV -O0 -triple=x86_64-unknown-linux -target-feature +xsave -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Wno-unused-but-set-variable -Werror
// RUN: FileCheck --check-prefix=LLVM-XSETBV --input-file=%t.ll %s

// RUN: %clang_cc1 %s -DTEST_XSAVEOPT -O0 -triple=x86_64-unknown-linux -target-feature +xsave -target-feature +xsaveopt -fclangir -emit-cir -o %t.cir -Wall -Wno-unused-but-set-variable -Werror
// RUN: FileCheck --check-prefix=CIR-XSAVEOPT --input-file=%t.cir %s
// RUN: %clang_cc1 %s -DTEST_XSAVEOPT -O0 -triple=x86_64-unknown-linux -target-feature +xsave -target-feature +xsaveopt -fclangir -emit-llvm -o %t.ll -Wall -Wno-unused-but-set-variable -Werror
// RUN: FileCheck --check-prefix=LLVM-XSAVEOPT --input-file=%t.ll %s
// RUN: %clang_cc1 %s -DTEST_XSAVEOPT -O0 -triple=x86_64-unknown-linux -target-feature +xsave -target-feature +xsaveopt -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Wno-unused-but-set-variable -Werror
// RUN: FileCheck --check-prefix=CIR-XSAVEOPT --input-file=%t.cir %s
// RUN: %clang_cc1 %s -DTEST_XSAVEOPT -O0 -triple=x86_64-unknown-linux -target-feature +xsave -target-feature +xsaveopt -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Wno-unused-but-set-variable -Werror
// RUN: FileCheck --check-prefix=LLVM-XSAVEOPT --input-file=%t.ll %s

// RUN: %clang_cc1 %s -DTEST_XSAVEC -O0 -triple=x86_64-unknown-linux -target-feature +xsave -target-feature +xsavec -fclangir -emit-cir -o %t.cir -Wall -Wno-unused-but-set-variable -Werror
// RUN: FileCheck --check-prefix=CIR-XSAVEC --input-file=%t.cir %s
// RUN: %clang_cc1 %s -DTEST_XSAVEC -O0 -triple=x86_64-unknown-linux -target-feature +xsave -target-feature +xsavec -fclangir -emit-llvm -o %t.ll -Wall -Wno-unused-but-set-variable -Werror
// RUN: FileCheck --check-prefix=LLVM-XSAVEC --input-file=%t.ll %s
// RUN: %clang_cc1 %s -DTEST_XSAVEC -O0 -triple=x86_64-unknown-linux -target-feature +xsave -target-feature +xsavec -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Wno-unused-but-set-variable -Werror
// RUN: FileCheck --check-prefix=CIR-XSAVEC --input-file=%t.cir %s
// RUN: %clang_cc1 %s -DTEST_XSAVEC -O0 -triple=x86_64-unknown-linux -target-feature +xsave -target-feature +xsavec -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Wno-unused-but-set-variable -Werror
// RUN: FileCheck --check-prefix=LLVM-XSAVEC --input-file=%t.ll %s

// RUN: %clang_cc1 %s -DTEST_XSAVES -O0 -triple=x86_64-unknown-linux -target-feature +xsave -target-feature +xsaves -fclangir -emit-cir -o %t.cir -Wall -Wno-unused-but-set-variable -Werror
// RUN: FileCheck --check-prefix=CIR-XSAVES --input-file=%t.cir %s
// RUN: %clang_cc1 %s -DTEST_XSAVES -O0 -triple=x86_64-unknown-linux -target-feature +xsave -target-feature +xsaves -fclangir -emit-llvm -o %t.ll -Wall -Wno-unused-but-set-variable -Werror
// RUN: FileCheck --check-prefix=LLVM-XSAVES --input-file=%t.ll %s
// RUN: %clang_cc1 %s -DTEST_XSAVES -O0 -triple=x86_64-unknown-linux -target-feature +xsave -target-feature +xsaves -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Wno-unused-but-set-variable -Werror
// RUN: FileCheck --check-prefix=CIR-XSAVES --input-file=%t.cir %s
// RUN: %clang_cc1 %s -DTEST_XSAVES -O0 -triple=x86_64-unknown-linux -target-feature +xsave -target-feature +xsaves -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Wno-unused-but-set-variable -Werror
// RUN: FileCheck --check-prefix=LLVM-XSAVES --input-file=%t.ll %s

// This test mimics clang/test/CodeGen/X86/x86_64-xsave.c, which eventually
// CIR shall be able to support fully.

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H
#include <x86intrin.h>


void test(void) {
  unsigned long long tmp_ULLi;
  unsigned int       tmp_Ui;
  void*              tmp_vp;
  tmp_ULLi = 0; tmp_Ui = 0; tmp_vp = 0;

#ifdef TEST_XSAVE
// CIR-XSAVE: [[tmp_vp_1:%.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR-XSAVE: [[tmp_ULLi_1:%.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!u64i>, !u64i
// CIR-XSAVE: [[high64_1:%.*]] = cir.shift(right, [[tmp_ULLi_1]] : !u64i, %{{.*}} : !u64i) -> !u64i
// CIR-XSAVE: [[high32_1:%.*]] = cir.cast integral [[high64_1]] : !u64i -> !s32i
// CIR-XSAVE: [[low32_1:%.*]] = cir.cast integral [[tmp_ULLi_1]] : !u64i -> !s32i
// CIR-XSAVE: %{{.*}} = cir.llvm.intrinsic "x86.xsave" [[tmp_vp_1]], [[high32_1]], [[low32_1]] : (!cir.ptr<!void>, !s32i, !s32i) -> !void

// LLVM-XSAVE: [[tmp_vp_1:%.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM-XSAVE: [[tmp_ULLi_1:%.*]] = load i64, ptr %{{.*}}, align 8
// LLVM-XSAVE: [[high64_1:%.*]] = lshr i64 [[tmp_ULLi_1]], 32
// LLVM-XSAVE: [[high32_1:%.*]] = trunc i64 [[high64_1]] to i32
// LLVM-XSAVE: [[low32_1:%.*]] = trunc i64 [[tmp_ULLi_1]] to i32
// LLVM-XSAVE: call void @llvm.x86.xsave(ptr [[tmp_vp_1]], i32 [[high32_1]], i32 [[low32_1]])
  (void)__builtin_ia32_xsave(tmp_vp, tmp_ULLi);


// CIR-XSAVE: [[tmp_vp_2:%.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR-XSAVE: [[tmp_ULLi_2:%.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!u64i>, !u64i
// CIR-XSAVE: [[high64_2:%.*]] = cir.shift(right, [[tmp_ULLi_2]] : !u64i, %{{.*}} : !u64i) -> !u64i
// CIR-XSAVE: [[high32_2:%.*]] = cir.cast integral [[high64_2]] : !u64i -> !s32i
// CIR-XSAVE: [[low32_2:%.*]] = cir.cast integral [[tmp_ULLi_2]] : !u64i -> !s32i
// CIR-XSAVE: %{{.*}} = cir.llvm.intrinsic "x86.xsave64" [[tmp_vp_2]], [[high32_2]], [[low32_2]] : (!cir.ptr<!void>, !s32i, !s32i) -> !void

// LLVM-XSAVE: [[tmp_vp_2:%.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM-XSAVE: [[tmp_ULLi_2:%.*]] = load i64, ptr %{{.*}}, align 8
// LLVM-XSAVE: [[high64_2:%.*]] = lshr i64 [[tmp_ULLi_2]], 32
// LLVM-XSAVE: [[high32_2:%.*]] = trunc i64 [[high64_2]] to i32
// LLVM-XSAVE: [[low32_2:%.*]] = trunc i64 [[tmp_ULLi_2]] to i32
// LLVM-XSAVE: call void @llvm.x86.xsave64(ptr [[tmp_vp_2]], i32 [[high32_2]], i32 [[low32_2]])
  (void)__builtin_ia32_xsave64(tmp_vp, tmp_ULLi);


// CIR-XSAVE: [[tmp_vp_3:%.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR-XSAVE: [[tmp_ULLi_3:%.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!u64i>, !u64i
// CIR-XSAVE: [[high64_3:%.*]] = cir.shift(right, [[tmp_ULLi_3]] : !u64i, %{{.*}} : !u64i) -> !u64i
// CIR-XSAVE: [[high32_3:%.*]] = cir.cast integral [[high64_3]] : !u64i -> !s32i
// CIR-XSAVE: [[low32_3:%.*]] = cir.cast integral [[tmp_ULLi_3]] : !u64i -> !s32i
// CIR-XSAVE: %{{.*}} = cir.llvm.intrinsic "x86.xrstor" [[tmp_vp_3]], [[high32_3]], [[low32_3]] : (!cir.ptr<!void>, !s32i, !s32i) -> !void

// LLVM-XSAVE: [[tmp_vp_3:%.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM-XSAVE: [[tmp_ULLi_3:%.*]] = load i64, ptr %{{.*}}, align 8
// LLVM-XSAVE: [[high64_3:%.*]] = lshr i64 [[tmp_ULLi_3]], 32
// LLVM-XSAVE: [[high32_3:%.*]] = trunc i64 [[high64_3]] to i32
// LLVM-XSAVE: [[low32_3:%.*]] = trunc i64 [[tmp_ULLi_3]] to i32
// LLVM-XSAVE: call void @llvm.x86.xrstor(ptr [[tmp_vp_3]], i32 [[high32_3]], i32 [[low32_3]])
  (void)__builtin_ia32_xrstor(tmp_vp, tmp_ULLi);


// CIR-XSAVE: [[tmp_vp_4:%.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR-XSAVE: [[tmp_ULLi_4:%.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!u64i>, !u64i
// CIR-XSAVE: [[high64_4:%.*]] = cir.shift(right, [[tmp_ULLi_4]] : !u64i, %{{.*}} : !u64i) -> !u64i
// CIR-XSAVE: [[high32_4:%.*]] = cir.cast integral [[high64_4]] : !u64i -> !s32i
// CIR-XSAVE: [[low32_4:%.*]] = cir.cast integral [[tmp_ULLi_4]] : !u64i -> !s32i
// CIR-XSAVE: %{{.*}} = cir.llvm.intrinsic "x86.xrstor64" [[tmp_vp_4]], [[high32_4]], [[low32_4]] : (!cir.ptr<!void>, !s32i, !s32i) -> !void

// LLVM-XSAVE: [[tmp_vp_4:%.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM-XSAVE: [[tmp_ULLi_4:%.*]] = load i64, ptr %{{.*}}, align 8
// LLVM-XSAVE: [[high64_4:%.*]] = lshr i64 [[tmp_ULLi_4]], 32
// LLVM-XSAVE: [[high32_4:%.*]] = trunc i64 [[high64_4]] to i32
// LLVM-XSAVE: [[low32_4:%.*]] = trunc i64 [[tmp_ULLi_4]] to i32
// LLVM-XSAVE: call void @llvm.x86.xrstor64(ptr [[tmp_vp_4]], i32 [[high32_4]], i32 [[low32_4]])
  (void)__builtin_ia32_xrstor64(tmp_vp, tmp_ULLi);
  
  
// CIR-XSAVE: {{%.*}} = cir.llvm.intrinsic "x86.xsave" {{%.*}} : (!cir.ptr<!void>, !s32i, !s32i) -> !void
// LLVM-XSAVE: call void @llvm.x86.xsave 
  (void)_xsave(tmp_vp, tmp_ULLi);

// CIR-XSAVE: {{%.*}} = cir.llvm.intrinsic "x86.xsave64" {{%.*}} : (!cir.ptr<!void>, !s32i, !s32i) -> !void
// LLVM-XSAVE: call void @llvm.x86.xsave64
  (void)_xsave64(tmp_vp, tmp_ULLi);

// CIR-XSAVE: {{%.*}} = cir.llvm.intrinsic "x86.xrstor" {{%.*}} : (!cir.ptr<!void>, !s32i, !s32i) -> !void
// LLVM-XSAVE: call void @llvm.x86.xrstor
  (void)_xrstor(tmp_vp, tmp_ULLi);

// CIR-XSAVE: {{%.*}} = cir.llvm.intrinsic "x86.xrstor64" {{%.*}} : (!cir.ptr<!void>, !s32i, !s32i) -> !void
// LLVM-XSAVE: call void @llvm.x86.xrstor64
  (void)_xrstor64(tmp_vp, tmp_ULLi);
#endif

#ifdef TEST_XSAVEOPT
// CIR-XSAVEOPT: [[tmp_vp_1:%.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR-XSAVEOPT: [[tmp_ULLi_1:%.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!u64i>, !u64i
// CIR-XSAVEOPT: [[high64_1:%.*]] = cir.shift(right, [[tmp_ULLi_1]] : !u64i, %{{.*}} : !u64i) -> !u64i
// CIR-XSAVEOPT: [[high32_1:%.*]] = cir.cast integral [[high64_1]] : !u64i -> !s32i
// CIR-XSAVEOPT: [[low32_1:%.*]] = cir.cast integral [[tmp_ULLi_1]] : !u64i -> !s32i
// CIR-XSAVEOPT: %{{.*}} = cir.llvm.intrinsic "x86.xsaveopt" [[tmp_vp_1]], [[high32_1]], [[low32_1]] : (!cir.ptr<!void>, !s32i, !s32i) -> !void

// LLVM-XSAVEOPT: [[tmp_vp_1:%.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM-XSAVEOPT: [[tmp_ULLi_1:%.*]] = load i64, ptr %{{.*}}, align 8
// LLVM-XSAVEOPT: [[high64_1:%.*]] = lshr i64 [[tmp_ULLi_1]], 32
// LLVM-XSAVEOPT: [[high32_1:%.*]] = trunc i64 [[high64_1]] to i32
// LLVM-XSAVEOPT: [[low32_1:%.*]] = trunc i64 [[tmp_ULLi_1]] to i32
// LLVM-XSAVEOPT: call void @llvm.x86.xsaveopt(ptr [[tmp_vp_1]], i32 [[high32_1]], i32 [[low32_1]])
  (void)__builtin_ia32_xsaveopt(tmp_vp, tmp_ULLi);

// CIR-XSAVEOPT: [[tmp_vp_2:%.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR-XSAVEOPT: [[tmp_ULLi_2:%.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!u64i>, !u64i
// CIR-XSAVEOPT: [[high64_2:%.*]] = cir.shift(right, [[tmp_ULLi_2]] : !u64i, %{{.*}} : !u64i) -> !u64i
// CIR-XSAVEOPT: [[high32_2:%.*]] = cir.cast integral [[high64_2]] : !u64i -> !s32i
// CIR-XSAVEOPT: [[low32_2:%.*]] = cir.cast integral [[tmp_ULLi_2]] : !u64i -> !s32i
// CIR-XSAVEOPT: %{{.*}} = cir.llvm.intrinsic "x86.xsaveopt64" [[tmp_vp_2]], [[high32_2]], [[low32_2]] : (!cir.ptr<!void>, !s32i, !s32i) -> !void

// LLVM-XSAVEOPT: [[tmp_vp_2:%.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM-XSAVEOPT: [[tmp_ULLi_2:%.*]] = load i64, ptr %{{.*}}, align 8
// LLVM-XSAVEOPT: [[high64_2:%.*]] = lshr i64 [[tmp_ULLi_2]], 32
// LLVM-XSAVEOPT: [[high32_2:%.*]] = trunc i64 [[high64_2]] to i32
// LLVM-XSAVEOPT: [[low32_2:%.*]] = trunc i64 [[tmp_ULLi_2]] to i32
// LLVM-XSAVEOPT: call void @llvm.x86.xsaveopt64(ptr [[tmp_vp_2]], i32 [[high32_2]], i32 [[low32_2]])
  (void)__builtin_ia32_xsaveopt64(tmp_vp, tmp_ULLi);

// CIR-XSAVEOPT: {{%.*}} = cir.llvm.intrinsic "x86.xsaveopt" {{%.*}} : (!cir.ptr<!void>, !s32i, !s32i) -> !void 
// LLVM-XSAVEOPT: call void @llvm.x86.xsaveopt
  (void)_xsaveopt(tmp_vp, tmp_ULLi);
  
// CIR-XSAVEOPT: {{%.*}} = cir.llvm.intrinsic "x86.xsaveopt64" {{%.*}} : (!cir.ptr<!void>, !s32i, !s32i) -> !void 
// LLVM-XSAVEOPT: call void @llvm.x86.xsaveopt64
  (void)_xsaveopt64(tmp_vp, tmp_ULLi);
#endif

#ifdef TEST_XSAVEC
// CIR-XSAVEC: [[tmp_vp_1:%.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR-XSAVEC: [[tmp_ULLi_1:%.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!u64i>, !u64i
// CIR-XSAVEC: [[high64_1:%.*]] = cir.shift(right, [[tmp_ULLi_1]] : !u64i, %{{.*}} : !u64i) -> !u64i
// CIR-XSAVEC: [[high32_1:%.*]] = cir.cast integral [[high64_1]] : !u64i -> !s32i
// CIR-XSAVEC: [[low32_1:%.*]] = cir.cast integral [[tmp_ULLi_1]] : !u64i -> !s32i
// CIR-XSAVEC: %{{.*}} = cir.llvm.intrinsic "x86.xsavec" [[tmp_vp_1]], [[high32_1]], [[low32_1]] : (!cir.ptr<!void>, !s32i, !s32i) -> !void

// LLVM-XSAVEC: [[tmp_vp_1:%.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM-XSAVEC: [[tmp_ULLi_1:%.*]] = load i64, ptr %{{.*}}, align 8
// LLVM-XSAVEC: [[high64_1:%.*]] = lshr i64 [[tmp_ULLi_1]], 32
// LLVM-XSAVEC: [[high32_1:%.*]] = trunc i64 [[high64_1]] to i32
// LLVM-XSAVEC: [[low32_1:%.*]] = trunc i64 [[tmp_ULLi_1]] to i32
// LLVM-XSAVEC: call void @llvm.x86.xsavec(ptr [[tmp_vp_1]], i32 [[high32_1]], i32 [[low32_1]])
  (void)__builtin_ia32_xsavec(tmp_vp, tmp_ULLi);


// CIR-XSAVEC: [[tmp_vp_2:%.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR-XSAVEC: [[tmp_ULLi_2:%.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!u64i>, !u64i
// CIR-XSAVEC: [[high64_2:%.*]] = cir.shift(right, [[tmp_ULLi_2]] : !u64i, %{{.*}} : !u64i) -> !u64i
// CIR-XSAVEC: [[high32_2:%.*]] = cir.cast integral [[high64_2]] : !u64i -> !s32i
// CIR-XSAVEC: [[low32_2:%.*]] = cir.cast integral [[tmp_ULLi_2]] : !u64i -> !s32i
// CIR-XSAVEC: %{{.*}} = cir.llvm.intrinsic "x86.xsavec64" [[tmp_vp_2]], [[high32_2]], [[low32_2]] : (!cir.ptr<!void>, !s32i, !s32i) -> !void

// LLVM-XSAVEC: [[tmp_vp_2:%.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM-XSAVEC: [[tmp_ULLi_2:%.*]] = load i64, ptr %{{.*}}, align 8
// LLVM-XSAVEC: [[high64_2:%.*]] = lshr i64 [[tmp_ULLi_2]], 32
// LLVM-XSAVEC: [[high32_2:%.*]] = trunc i64 [[high64_2]] to i32
// LLVM-XSAVEC: [[low32_2:%.*]] = trunc i64 [[tmp_ULLi_2]] to i32
// LLVM-XSAVEC: call void @llvm.x86.xsavec64(ptr [[tmp_vp_2]], i32 [[high32_2]], i32 [[low32_2]])
  (void)__builtin_ia32_xsavec64(tmp_vp, tmp_ULLi);
  
// CIR-XSAVEC: {{%.*}} = cir.llvm.intrinsic "x86.xsavec" {{%.*}} : (!cir.ptr<!void>, !s32i, !s32i) -> !void 
// LLVM-XSAVEC: call void @llvm.x86.xsavec
  (void)_xsavec(tmp_vp, tmp_ULLi);
  
// CIR-XSAVEC: {{%.*}} = cir.llvm.intrinsic "x86.xsavec64" {{%.*}} : (!cir.ptr<!void>, !s32i, !s32i) -> !void 
// LLVM-XSAVEC: call void @llvm.x86.xsavec64
  (void)_xsavec64(tmp_vp, tmp_ULLi);
#endif

#ifdef TEST_XSAVES
// CIR-XSAVES: [[tmp_vp_1:%.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR-XSAVES: [[tmp_ULLi_1:%.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!u64i>, !u64i
// CIR-XSAVES: [[high64_1:%.*]] = cir.shift(right, [[tmp_ULLi_1]] : !u64i, %{{.*}} : !u64i) -> !u64i
// CIR-XSAVES: [[high32_1:%.*]] = cir.cast integral [[high64_1]] : !u64i -> !s32i
// CIR-XSAVES: [[low32_1:%.*]] = cir.cast integral [[tmp_ULLi_1]] : !u64i -> !s32i
// CIR-XSAVES: %{{.*}} = cir.llvm.intrinsic "x86.xsaves" [[tmp_vp_1]], [[high32_1]], [[low32_1]] : (!cir.ptr<!void>, !s32i, !s32i) -> !void

// LLVM-XSAVES: [[tmp_vp_1:%.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM-XSAVES: [[tmp_ULLi_1:%.*]] = load i64, ptr %{{.*}}, align 8
// LLVM-XSAVES: [[high64_1:%.*]] = lshr i64 [[tmp_ULLi_1]], 32
// LLVM-XSAVES: [[high32_1:%.*]] = trunc i64 [[high64_1]] to i32
// LLVM-XSAVES: [[low32_1:%.*]] = trunc i64 [[tmp_ULLi_1]] to i32
// LLVM-XSAVES: call void @llvm.x86.xsaves(ptr [[tmp_vp_1]], i32 [[high32_1]], i32 [[low32_1]])
  (void)__builtin_ia32_xsaves(tmp_vp, tmp_ULLi);


// CIR-XSAVES: [[tmp_vp_2:%.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR-XSAVES: [[tmp_ULLi_2:%.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!u64i>, !u64i
// CIR-XSAVES: [[high64_2:%.*]] = cir.shift(right, [[tmp_ULLi_2]] : !u64i, %{{.*}} : !u64i) -> !u64i
// CIR-XSAVES: [[high32_2:%.*]] = cir.cast integral [[high64_2]] : !u64i -> !s32i
// CIR-XSAVES: [[low32_2:%.*]] = cir.cast integral [[tmp_ULLi_2]] : !u64i -> !s32i
// CIR-XSAVES: %{{.*}} = cir.llvm.intrinsic "x86.xsaves64" [[tmp_vp_2]], [[high32_2]], [[low32_2]] : (!cir.ptr<!void>, !s32i, !s32i) -> !void

// LLVM-XSAVES: [[tmp_vp_2:%.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM-XSAVES: [[tmp_ULLi_2:%.*]] = load i64, ptr %{{.*}}, align 8
// LLVM-XSAVES: [[high64_2:%.*]] = lshr i64 [[tmp_ULLi_2]], 32
// LLVM-XSAVES: [[high32_2:%.*]] = trunc i64 [[high64_2]] to i32
// LLVM-XSAVES: [[low32_2:%.*]] = trunc i64 [[tmp_ULLi_2]] to i32
// LLVM-XSAVES: call void @llvm.x86.xsaves64(ptr [[tmp_vp_2]], i32 [[high32_2]], i32 [[low32_2]])
  (void)__builtin_ia32_xsaves64(tmp_vp, tmp_ULLi);


// CIR-XSAVES: [[tmp_vp_3:%.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR-XSAVES: [[tmp_ULLi_3:%.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!u64i>, !u64i
// CIR-XSAVES: [[high64_3:%.*]] = cir.shift(right, [[tmp_ULLi_3]] : !u64i, %{{.*}} : !u64i) -> !u64i
// CIR-XSAVES: [[high32_3:%.*]] = cir.cast integral [[high64_3]] : !u64i -> !s32i
// CIR-XSAVES: [[low32_3:%.*]] = cir.cast integral [[tmp_ULLi_3]] : !u64i -> !s32i
// CIR-XSAVES: %{{.*}} = cir.llvm.intrinsic "x86.xrstors" [[tmp_vp_3]], [[high32_3]], [[low32_3]] : (!cir.ptr<!void>, !s32i, !s32i) -> !void

// LLVM-XSAVES: [[tmp_vp_3:%.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM-XSAVES: [[tmp_ULLi_3:%.*]] = load i64, ptr %{{.*}}, align 8
// LLVM-XSAVES: [[high64_3:%.*]] = lshr i64 [[tmp_ULLi_3]], 32
// LLVM-XSAVES: [[high32_3:%.*]] = trunc i64 [[high64_3]] to i32
// LLVM-XSAVES: [[low32_3:%.*]] = trunc i64 [[tmp_ULLi_3]] to i32
// LLVM-XSAVES: call void @llvm.x86.xrstors(ptr [[tmp_vp_3]], i32 [[high32_3]], i32 [[low32_3]])
  (void)__builtin_ia32_xrstors(tmp_vp, tmp_ULLi);


// CIR-XSAVES: [[tmp_vp_4:%.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR-XSAVES: [[tmp_ULLi_4:%.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!u64i>, !u64i
// CIR-XSAVES: [[high64_4:%.*]] = cir.shift(right, [[tmp_ULLi_4]] : !u64i, %{{.*}} : !u64i) -> !u64i
// CIR-XSAVES: [[high32_4:%.*]] = cir.cast integral [[high64_4]] : !u64i -> !s32i
// CIR-XSAVES: [[low32_4:%.*]] = cir.cast integral [[tmp_ULLi_4]] : !u64i -> !s32i
// CIR-XSAVES: %{{.*}} = cir.llvm.intrinsic "x86.xrstors64" [[tmp_vp_4]], [[high32_4]], [[low32_4]] : (!cir.ptr<!void>, !s32i, !s32i) -> !void

// LLVM-XSAVES: [[tmp_vp_4:%.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM-XSAVES: [[tmp_ULLi_4:%.*]] = load i64, ptr %{{.*}}, align 8
// LLVM-XSAVES: [[high64_4:%.*]] = lshr i64 [[tmp_ULLi_4]], 32
// LLVM-XSAVES: [[high32_4:%.*]] = trunc i64 [[high64_4]] to i32
// LLVM-XSAVES: [[low32_4:%.*]] = trunc i64 [[tmp_ULLi_4]] to i32
// LLVM-XSAVES: call void @llvm.x86.xrstors64(ptr [[tmp_vp_4]], i32 [[high32_4]], i32 [[low32_4]])
  (void)__builtin_ia32_xrstors64(tmp_vp, tmp_ULLi);
  
  
// CIR-XSAVES: {{%.*}} = cir.llvm.intrinsic "x86.xsaves" {{%.*}} : (!cir.ptr<!void>, !s32i, !s32i) -> !void
// LLVM-XSAVES: call void @llvm.x86.xsaves
  (void)_xsaves(tmp_vp, tmp_ULLi);

// CIR-XSAVES: {{%.*}} = cir.llvm.intrinsic "x86.xsaves64" {{%.*}} : (!cir.ptr<!void>, !s32i, !s32i) -> !void
// LLVM-XSAVES: call void @llvm.x86.xsaves64
  (void)_xsaves64(tmp_vp, tmp_ULLi);

// CIR-XSAVES: {{%.*}} = cir.llvm.intrinsic "x86.xrstors" {{%.*}} : (!cir.ptr<!void>, !s32i, !s32i) -> !void
// LLVM-XSAVES: call void @llvm.x86.xrstors
  (void)_xrstors(tmp_vp, tmp_ULLi);

// CIR-XSAVES: {{%.*}} = cir.llvm.intrinsic "x86.xrstors64" {{%.*}} : (!cir.ptr<!void>, !s32i, !s32i) -> !void
// LLVM-XSAVES: call void @llvm.x86.xrstors64
  (void)_xrstors64(tmp_vp, tmp_ULLi);
#endif

#ifdef TEST_XGETBV

// CIR-XGETBV: [[tmp_Ui:%.*]] =  cir.load align(4) %{{.*}} : !cir.ptr<!u32i>, !u32i
// CIR-XGETBV: {{%.*}} = cir.llvm.intrinsic "x86.xgetbv" [[tmp_Ui]] : (!u32i) -> !u64i

// LLVM-XGETBV: [[tmp_Ui:%.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-XGETBV: call i64 @llvm.x86.xgetbv(i32 [[tmp_Ui]])
  tmp_ULLi = __builtin_ia32_xgetbv(tmp_Ui);
  
// CIR-XGETBV: {{%.*}} = cir.llvm.intrinsic "x86.xgetbv" {{%.*}} : (!u32i) -> !u64i
// LLVM-XGETBV: call i64 @llvm.x86.xgetbv
  tmp_ULLi = _xgetbv(tmp_Ui);
#endif

#ifdef TEST_XSETBV
// CIR-XSETBV: [[tmp_Ui_1:%.*]] = cir.load align(4) %{{.*}} : !cir.ptr<!u32i>, !u32i
// CIR-XSETBV: [[tmp_ULLi_1:%.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!u64i>, !u64i
// CIR-XSETBV: [[high64_1:%.*]] = cir.shift(right, [[tmp_ULLi_1]] : !u64i, %{{.*}} : !u64i) -> !u64i
// CIR-XSETBV: [[high32_1:%.*]] = cir.cast integral [[high64_1]] : !u64i -> !s32i
// CIR-XSETBV: [[low32_1:%.*]] = cir.cast integral [[tmp_ULLi_1]] : !u64i -> !s32i
// CIR-XSETBV: %{{.*}} = cir.llvm.intrinsic "x86.xsetbv" [[tmp_Ui_1]], [[high32_1]], [[low32_1]] : (!u32i, !s32i, !s32i) -> !void

// LLVM-XSETBV: [[tmp_Ui_1:%.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-XSETBV: [[tmp_ULLi_1:%.*]] = load i64, ptr %{{.*}}, align 8
// LLVM-XSETBV: [[high64_1:%.*]] = lshr i64 [[tmp_ULLi_1]], 32
// LLVM-XSETBV: [[high32_1:%.*]] = trunc i64 [[high64_1]] to i32
// LLVM-XSETBV: [[low32_1:%.*]] = trunc i64 [[tmp_ULLi_1]] to i32
// LLVM-XSETBV: call void @llvm.x86.xsetbv(i32 [[tmp_Ui_1]], i32 [[high32_1]], i32 [[low32_1]])
  (void)__builtin_ia32_xsetbv(tmp_Ui, tmp_ULLi);

// CIR-XSETBV: {{%.*}} = cir.llvm.intrinsic "x86.xsetbv" {{%.*}} : (!u32i, !s32i, !s32i) -> !void
// LLVM-XSETBV: call void @llvm.x86.xsetbv
  (void)_xsetbv(tmp_Ui, tmp_ULLi);
#endif
}
