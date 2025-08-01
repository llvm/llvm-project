// REQUIRES: powerpc-registered-target
// REQUIRES: asserts
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple powerpc-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-PPC
// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -emit-llvm -o - %s | FileCheck %s

#include <stdarg.h>

struct x {
  long a;
  double b;
};

void testva (int n, ...)
{
  va_list ap;

  struct x t = va_arg (ap, struct x);
// CHECK: call void @llvm.memcpy

// CHECK-PPC:  [[ARRAYDECAY:%.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %ap, i32 0, i32 0
// CHECK-PPC-NEXT:  [[GPRPTR:%.+]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr [[ARRAYDECAY]], i32 0, i32 0
// CHECK-PPC-NEXT:  [[GPR:%.+]] = load i8, ptr [[GPRPTR]], align 4
// CHECK-PPC-NEXT:  [[COND:%.+]] = icmp ult i8 [[GPR]], 8
// CHECK-PPC-NEXT:  br i1 [[COND]], label %[[USING_REGS:[a-z_0-9]+]], label %[[USING_OVERFLOW:[a-z_0-9]+]]
//
// CHECK-PPC:[[USING_REGS]]
// CHECK-PPC-NEXT:  [[REGSAVE_AREA_P:%.+]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr [[ARRAYDECAY]], i32 0, i32 4
// CHECK-PPC-NEXT:  [[REGSAVE_AREA:%.+]] = load ptr, ptr [[REGSAVE_AREA_P]], align 4
// CHECK-PPC-NEXT:  [[OFFSET:%.+]] = mul i8 [[GPR]], 4
// CHECK-PPC-NEXT:  [[RAW_REGADDR:%.+]] = getelementptr inbounds i8, ptr [[REGSAVE_AREA]], i8 [[OFFSET]]
// CHECK-PPC-NEXT:  [[USED_GPR:%[0-9]+]] = add i8 [[GPR]], 1
// CHECK-PPC-NEXT:  store i8 [[USED_GPR]], ptr [[GPRPTR]], align 4
// CHECK-PPC-NEXT:  br label %[[CONT:[a-z0-9]+]]
//
// CHECK-PPC:[[USING_OVERFLOW]]
// CHECK-PPC-NEXT:  store i8 8, ptr [[GPRPTR]], align 4
// CHECK-PPC-NEXT:  [[OVERFLOW_AREA_P:%[0-9]+]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr [[ARRAYDECAY]], i32 0, i32 3
// CHECK-PPC-NEXT:  [[OVERFLOW_AREA:%.+]] = load ptr, ptr [[OVERFLOW_AREA_P]], align 4
// CHECK-PPC-NEXT: [[GEP_ALIGN:%[0-9]+]] = getelementptr inbounds i8, ptr %argp.cur, i32 7
// CHECK-PPC-NEXT: %argp.cur.aligned = call ptr @llvm.ptrmask.p0.i32(ptr [[GEP_ALIGN]], i32 -8)
// CHECK-PPC-NEXT:  [[NEW_OVERFLOW_AREA:%[0-9]+]] = getelementptr inbounds i8, ptr %argp.cur.aligned, i32 4
// CHECK-PPC-NEXT:  store ptr [[NEW_OVERFLOW_AREA:%[0-9]+]], ptr [[OVERFLOW_AREA_P]], align 4
// CHECK-PPC-NEXT:  br label %[[CONT]]
//
// CHECK-PPC:[[CONT]]
// CHECK-PPC-NEXT:  [[VAARG_ADDR:%[a-z.0-9]+]] = phi ptr [ [[RAW_REGADDR]], %[[USING_REGS]] ], [ %argp.cur.aligned, %[[USING_OVERFLOW]] ]
// CHECK-PPC-NEXT:  [[AGGR:%[a-z0-9]+]] = load ptr, ptr [[VAARG_ADDR]]
// CHECK-PPC-NEXT:  call void @llvm.memcpy.p0.p0.i32(ptr align 8 %t, ptr align 8 [[AGGR]], i32 16, i1 false)

  int v = va_arg (ap, int);

// CHECK: getelementptr inbounds i8, ptr %{{[a-z.0-9]*}}, i64 4
// CHECK-PPC:       [[ARRAYDECAY:%[a-z0-9]+]] = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %ap, i32 0, i32 0
// CHECK-PPC-NEXT:  [[GPRPTR:%.+]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr [[ARRAYDECAY]], i32 0, i32 0
// CHECK-PPC-NEXT:  [[GPR:%.+]] = load i8, ptr [[GPRPTR]], align 4
// CHECK-PPC-NEXT:  [[COND:%.+]] = icmp ult i8 [[GPR]], 8
// CHECK-PPC-NEXT:  br i1 [[COND]], label %[[USING_REGS:.+]], label %[[USING_OVERFLOW:.+]]{{$}}
//
// CHECK-PPC:[[USING_REGS]]
// CHECK-PPC-NEXT:  [[REGSAVE_AREA_P:%.+]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr [[ARRAYDECAY]], i32 0, i32 4
// CHECK-PPC-NEXT:  [[REGSAVE_AREA:%.+]] = load ptr, ptr [[REGSAVE_AREA_P]], align 4
// CHECK-PPC-NEXT:  [[OFFSET:%.+]] = mul i8 [[GPR]], 4
// CHECK-PPC-NEXT:  [[RAW_REGADDR:%.+]] = getelementptr inbounds i8, ptr [[REGSAVE_AREA]], i8 [[OFFSET]]
// CHECK-PPC-NEXT:  [[USED_GPR:%[0-9]+]] = add i8 [[GPR]], 1
// CHECK-PPC-NEXT:  store i8 [[USED_GPR]], ptr [[GPRPTR]], align 4
// CHECK-PPC-NEXT:  br label %[[CONT:[a-z0-9]+]]
//
// CHECK-PPC:[[USING_OVERFLOW]]
// CHECK-PPC-NEXT:  store i8 8, ptr [[GPRPTR]], align 4
// CHECK-PPC-NEXT:  [[OVERFLOW_AREA_P:%[0-9]+]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr [[ARRAYDECAY]], i32 0, i32 3
// CHECK-PPC-NEXT:  [[OVERFLOW_AREA:%.+]] = load ptr, ptr [[OVERFLOW_AREA_P]], align 4
// CHECK-PPC-NEXT:  [[NEW_OVERFLOW_AREA:%[0-9]+]] = getelementptr inbounds i8, ptr [[OVERFLOW_AREA]], i32 4
// CHECK-PPC-NEXT:  store ptr [[NEW_OVERFLOW_AREA]], ptr [[OVERFLOW_AREA_P]]
// CHECK-PPC-NEXT:  br label %[[CONT]]
//
// CHECK-PPC:[[CONT]]
// CHECK-PPC-NEXT:  [[VAARG_ADDR:%[a-z.0-9]+]] = phi ptr [ [[RAW_REGADDR]], %[[USING_REGS]] ], [ [[OVERFLOW_AREA]], %[[USING_OVERFLOW]] ]
// CHECK-PPC-NEXT:  [[THIRTYFIVE:%[0-9]+]] = load i32, ptr [[VAARG_ADDR]]
// CHECK-PPC-NEXT:  store i32 [[THIRTYFIVE]], ptr %v, align 4

#ifdef __powerpc64__
  __int128_t u = va_arg (ap, __int128_t);
#endif
// CHECK: load i128, ptr %argp.cur3
}
