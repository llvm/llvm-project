// REQUIRES: powerpc-registered-target
// REQUIRES: asserts
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple powerpc-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-PPC

#include <stdarg.h>

struct x {
  long a;
  double b;
};

void testva (int n, ...)
{
  va_list ap;

  struct x t = va_arg (ap, struct x);
// CHECK: bitcast i8* %{{[a-z.0-9]*}} to %struct.x*
// CHECK: bitcast %struct.x* %t to i8*
// CHECK: bitcast %struct.x* %{{[0-9]+}} to i8*
// CHECK: call void @llvm.memcpy

// CHECK-PPC:  [[ARRAYDECAY:%.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0
// CHECK-PPC-NEXT:  [[GPRPTR:%.+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* [[ARRAYDECAY]], i32 0, i32 0
// CHECK-PPC-NEXT:  [[GPR:%.+]] = load i8, i8* [[GPRPTR]], align 4
// CHECK-PPC-NEXT:  [[COND:%.+]] = icmp ult i8 [[GPR]], 8
// CHECK-PPC-NEXT:  br i1 [[COND]], label %[[USING_REGS:[a-z_0-9]+]], label %[[USING_OVERFLOW:[a-z_0-9]+]]
//
// CHECK-PPC:[[USING_REGS]]
// CHECK-PPC-NEXT:  [[REGSAVE_AREA_P:%.+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* [[ARRAYDECAY]], i32 0, i32 4
// CHECK-PPC-NEXT:  [[REGSAVE_AREA:%.+]] = load i8*, i8** [[REGSAVE_AREA_P]], align 4
// CHECK-PPC-NEXT:  [[OFFSET:%.+]] = mul i8 [[GPR]], 4
// CHECK-PPC-NEXT:  [[RAW_REGADDR:%.+]] = getelementptr inbounds i8, i8* [[REGSAVE_AREA]], i8 [[OFFSET]]
// CHECK-PPC-NEXT:  [[REGADDR:%.+]] = bitcast i8* [[RAW_REGADDR]] to %struct.x**
// CHECK-PPC-NEXT:  [[USED_GPR:%[0-9]+]] = add i8 [[GPR]], 1
// CHECK-PPC-NEXT:  store i8 [[USED_GPR]], i8* [[GPRPTR]], align 4
// CHECK-PPC-NEXT:  br label %[[CONT:[a-z0-9]+]]
//
// CHECK-PPC:[[USING_OVERFLOW]]
// CHECK-PPC-NEXT:  store i8 8, i8* [[GPRPTR]], align 4
// CHECK-PPC-NEXT:  [[OVERFLOW_AREA_P:%[0-9]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* [[ARRAYDECAY]], i32 0, i32 3
// CHECK-PPC-NEXT:  [[OVERFLOW_AREA:%.+]] = load i8*, i8** [[OVERFLOW_AREA_P]], align 4
// CHECK-PPC-NEXT:  %{{[0-9]+}} =  ptrtoint i8* %argp.cur to i32
// CHECK-PPC-NEXT:  %{{[0-9]+}} = add i32 %{{[0-9]+}}, 7
// CHECK-PPC-NEXT:  %{{[0-9]+}} = and i32 %{{[0-9]+}}, -8
// CHECK-PPC-NEXT:  %argp.cur.aligned = inttoptr i32 %{{[0-9]+}} to i8*
// CHECK-PPC-NEXT:  [[MEMADDR:%.+]] = bitcast i8* %argp.cur.aligned to %struct.x**
// CHECK-PPC-NEXT:  [[NEW_OVERFLOW_AREA:%[0-9]+]] = getelementptr inbounds i8, i8* %argp.cur.aligned, i32 4
// CHECK-PPC-NEXT:  store i8* [[NEW_OVERFLOW_AREA:%[0-9]+]], i8** [[OVERFLOW_AREA_P]], align 4
// CHECK-PPC-NEXT:  br label %[[CONT]]
//
// CHECK-PPC:[[CONT]]
// CHECK-PPC-NEXT:  [[VAARG_ADDR:%[a-z.0-9]+]] = phi %struct.x** [ [[REGADDR]], %[[USING_REGS]] ], [ [[MEMADDR]], %[[USING_OVERFLOW]] ]
// CHECK-PPC-NEXT:  [[AGGR:%[a-z0-9]+]] = load %struct.x*, %struct.x** [[VAARG_ADDR]]
// CHECK-PPC-NEXT:  [[DEST:%[0-9]+]] = bitcast %struct.x* %t to i8*
// CHECK-PPC-NEXT:  [[SRC:%.+]] = bitcast %struct.x* [[AGGR]] to i8*
// CHECK-PPC-NEXT:  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 [[DEST]], i8* align 8 [[SRC]], i32 16, i1 false)

  int v = va_arg (ap, int);
  
// CHECK: getelementptr inbounds i8, i8* %{{[a-z.0-9]*}}, i64 4
// CHECK: bitcast i8* %{{[0-9]+}} to i32*
// CHECK-PPC:       [[ARRAYDECAY:%[a-z0-9]+]] = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0
// CHECK-PPC-NEXT:  [[GPRPTR:%.+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* [[ARRAYDECAY]], i32 0, i32 0
// CHECK-PPC-NEXT:  [[GPR:%.+]] = load i8, i8* [[GPRPTR]], align 4
// CHECK-PPC-NEXT:  [[COND:%.+]] = icmp ult i8 [[GPR]], 8
// CHECK-PPC-NEXT:  br i1 [[COND]], label %[[USING_REGS:.+]], label %[[USING_OVERFLOW:.+]]{{$}}
//
// CHECK-PPC:[[USING_REGS]]
// CHECK-PPC-NEXT:  [[REGSAVE_AREA_P:%.+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* [[ARRAYDECAY]], i32 0, i32 4
// CHECK-PPC-NEXT:  [[REGSAVE_AREA:%.+]] = load i8*, i8** [[REGSAVE_AREA_P]], align 4
// CHECK-PPC-NEXT:  [[OFFSET:%.+]] = mul i8 [[GPR]], 4
// CHECK-PPC-NEXT:  [[RAW_REGADDR:%.+]] = getelementptr inbounds i8, i8* [[REGSAVE_AREA]], i8 [[OFFSET]]
// CHECK-PPC-NEXT:  [[REGADDR:%.+]] = bitcast i8* [[RAW_REGADDR]] to i32*
// CHECK-PPC-NEXT:  [[USED_GPR:%[0-9]+]] = add i8 [[GPR]], 1
// CHECK-PPC-NEXT:  store i8 [[USED_GPR]], i8* [[GPRPTR]], align 4
// CHECK-PPC-NEXT:  br label %[[CONT:[a-z0-9]+]]
//
// CHECK-PPC:[[USING_OVERFLOW]]
// CHECK-PPC-NEXT:  store i8 8, i8* [[GPRPTR]], align 4
// CHECK-PPC-NEXT:  [[OVERFLOW_AREA_P:%[0-9]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* [[ARRAYDECAY]], i32 0, i32 3
// CHECK-PPC-NEXT:  [[OVERFLOW_AREA:%.+]] = load i8*, i8** [[OVERFLOW_AREA_P]], align 4
// CHECK-PPC-NEXT:  [[MEMADDR:%.+]] = bitcast i8* [[OVERFLOW_AREA]] to i32*
// CHECK-PPC-NEXT:  [[NEW_OVERFLOW_AREA:%[0-9]+]] = getelementptr inbounds i8, i8* [[OVERFLOW_AREA]], i32 4
// CHECK-PPC-NEXT:  store i8* [[NEW_OVERFLOW_AREA]], i8** [[OVERFLOW_AREA_P]]
// CHECK-PPC-NEXT:  br label %[[CONT]]
//
// CHECK-PPC:[[CONT]]
// CHECK-PPC-NEXT:  [[VAARG_ADDR:%[a-z.0-9]+]] = phi i32* [ [[REGADDR]], %[[USING_REGS]] ], [ [[MEMADDR]], %[[USING_OVERFLOW]] ]
// CHECK-PPC-NEXT:  [[THIRTYFIVE:%[0-9]+]] = load i32, i32* [[VAARG_ADDR]]
// CHECK-PPC-NEXT:  store i32 [[THIRTYFIVE]], i32* %v, align 4

#ifdef __powerpc64__
  __int128_t u = va_arg (ap, __int128_t);
#endif
// CHECK: bitcast i8* %{{[a-z.0-9]+}} to i128*
// CHECK-NEXT: load i128, i128* %{{[0-9]+}}
}
