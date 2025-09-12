// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir --check-prefix=CIR %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

long double i = 0;
long double t2(long double i2) {
    return i2 + i ;
}

// CIR: cir.global external @i = #cir.fp<0.000000e+00> : !cir.long_double<!cir.f128> {alignment = 16 : i64} loc({{.*}})
// CIR-LABEL:   cir.func dso_local @t2(%arg0: !cir.long_double<!cir.f128> loc({{.*}})) -> !cir.long_double<!cir.f128>
// CIR-NEXT:    %[[#I2:]] = cir.alloca !cir.long_double<!cir.f128>, !cir.ptr<!cir.long_double<!cir.f128>>, ["i2", init] {alignment = 16 : i64}
// CIR-NEXT:    %[[#RETVAL:]] = cir.alloca !cir.long_double<!cir.f128>, !cir.ptr<!cir.long_double<!cir.f128>>, ["__retval"] {alignment = 16 : i64}
// CIR-NEXT:    cir.store %arg0, %[[#I2]] : !cir.long_double<!cir.f128>, !cir.ptr<!cir.long_double<!cir.f128>>
// CIR-NEXT:    %[[#I2_LOAD:]] = cir.load{{.*}} %[[#I2]] : !cir.ptr<!cir.long_double<!cir.f128>>, !cir.long_double<!cir.f128>
// CIR-NEXT:    %[[#I:]] = cir.get_global @i : !cir.ptr<!cir.long_double<!cir.f128>>
// CIR-NEXT:    %[[#I_LOAD:]] = cir.load{{.*}} %[[#I]] : !cir.ptr<!cir.long_double<!cir.f128>>, !cir.long_double<!cir.f128>
// CIR-NEXT:    %[[#ADD:]] = cir.binop(add, %[[#I2_LOAD]], %[[#I_LOAD]]) : !cir.long_double<!cir.f128>
// CIR-NEXT:    cir.store %[[#ADD]], %[[#RETVAL]] : !cir.long_double<!cir.f128>, !cir.ptr<!cir.long_double<!cir.f128>>
// CIR-NEXT:    %[[#RETVAL_LOAD:]] = cir.load{{.*}} %[[#RETVAL]] : !cir.ptr<!cir.long_double<!cir.f128>>, !cir.long_double<!cir.f128>
// CIR-NEXT:    cir.return %[[#RETVAL_LOAD]] : !cir.long_double<!cir.f128>

//LLVM:         @i = global fp128 0xL00000000000000000000000000000000, align 16
//LLVM-LABEL:   define dso_local fp128 @t2(fp128 noundef %i2)
//LLVM-NEXT :   entry:
//LLVM-NEXT :   %[[#I2_ADDR:]]= alloca fp128, align 16
//LLVM-NEXT :   store fp128 %i2, ptr %[[#I2_ADDR]], align 16
//LLVM-NEXT :   %[[#I2_LOAD:]] = load fp128, ptr %[[#I2_ADDR]], align 16
//LLVM-NEXT :   %[[#I_LOAD:]] = load fp128, ptr @i, align 16
//LLVM-NEXT :   %[[#RETVAL:]] = fadd fp128 %[[#I2_LOAD]], %[[#I_LOAD]]
//LLVM-NEXT :   ret fp128 %[[#RETVAL]]
