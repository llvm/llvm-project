// RUN: %clang_cc1 -triple aarch64-none-elf -target-abi aapcs-soft -target-feature -fp-armv8 -target-feature -neon -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -triple aarch64-none-elf -target-abi aapcs-soft -target-feature -fp-armv8 -target-feature -neon -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=LLVM,OGCG

// Under the soft-float ABI there are no vector registers, so floating-point
// values are read through __gr_top and __gr_offs.
double va_double(__builtin_va_list args) {
  return __builtin_va_arg(args, double);
}

// LLVM-LABEL: define {{.*}} double @va_double(
// LLVM:   %[[GR_OFFS_P:.*]] = getelementptr inbounds nuw %struct.__va_list, ptr %[[ARGS:.*]], i32 0, i32 3
// LLVM:   %[[GR_OFFS:.*]] = load i32, ptr %[[GR_OFFS_P]], align 8
// LLVM:   %[[NEW_OFFS:.*]] = add i32 %[[GR_OFFS]], 8
// LLVM:   store i32 %[[NEW_OFFS]], ptr %[[GR_OFFS_P]], align 8
// LLVM:   %[[GR_TOP_P:.*]] = getelementptr inbounds nuw %struct.__va_list, ptr %[[ARGS]], i32 0, i32 1
// LLVM:   %[[GR_TOP:.*]] = load ptr, ptr %[[GR_TOP_P]], align 8
// LLVMCIR: %[[REG_ADDR:.*]] = getelementptr i8, ptr %[[GR_TOP]], i64 %{{.*}}
// OGCG:  %[[REG_ADDR:.*]] = getelementptr inbounds i8, ptr %[[GR_TOP]], i32 %[[GR_OFFS]]
// LLVM:   %[[STACK:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM:   getelementptr {{.*}}i8, ptr %[[STACK]], i64 8
// LLVMCIR: %[[ADDR:.*]] = phi ptr [ %[[STACK]], %{{.*}} ], [ %[[REG_ADDR]], %{{.*}} ]
// OGCG:  %[[ADDR:.*]] = phi ptr [ %[[REG_ADDR]], %{{.*}} ], [ %[[STACK]], %{{.*}} ]
// LLVM:   %{{.*}} = load double, ptr %[[ADDR]], align 8

// A record of two doubles is not an HFA here.  It is contiguous in two general
// registers.
struct TwoDoubles {
  double a, b;
};

struct TwoDoubles va_two_doubles(__builtin_va_list args) {
  return __builtin_va_arg(args, struct TwoDoubles);
}

// LLVM-LABEL: define {{.*}} [2 x i64] @va_two_doubles(
// LLVM:   %[[GR_OFFS_P:.*]] = getelementptr inbounds nuw %struct.__va_list, ptr %[[ARGS:.*]], i32 0, i32 3
// LLVM:   %[[GR_OFFS:.*]] = load i32, ptr %[[GR_OFFS_P]], align 8
// LLVM:   %[[NEW_OFFS:.*]] = add i32 %[[GR_OFFS]], 16
// LLVM:   store i32 %[[NEW_OFFS]], ptr %[[GR_OFFS_P]], align 8
// LLVM:   %[[GR_TOP:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVMCIR: %[[REG_ADDR:.*]] = getelementptr i8, ptr %[[GR_TOP]], i64 %{{.*}}
// OGCG:  %[[REG_ADDR:.*]] = getelementptr inbounds i8, ptr %[[GR_TOP]], i32 %[[GR_OFFS]]
// LLVM:   %[[STACK:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM:   getelementptr {{.*}}i8, ptr %[[STACK]], i64 16
// LLVMCIR: %[[ADDR:.*]] = phi ptr [ %[[STACK]], %{{.*}} ], [ %[[REG_ADDR]], %{{.*}} ]
// OGCG:  %[[ADDR:.*]] = phi ptr [ %[[REG_ADDR]], %{{.*}} ], [ %[[STACK]], %{{.*}} ]
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %{{.*}}, ptr align 8 %[[ADDR]], i64 16, i1 false)
