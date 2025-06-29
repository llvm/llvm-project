// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

constexpr extern int cx_var = __builtin_is_constant_evaluated();

// CIR: cir.global {{.*}} @cx_var = #cir.int<1> : !s32i
// LLVM: @cx_var = {{.*}} i32 1
// OGCG: @cx_var = {{.*}} i32 1

constexpr extern float cx_var_single = __builtin_huge_valf();

// CIR: cir.global {{.*}} @cx_var_single = #cir.fp<0x7F800000> : !cir.float
// LLVM: @cx_var_single = {{.*}} float 0x7FF0000000000000
// OGCG: @cx_var_single = {{.*}} float 0x7FF0000000000000

constexpr extern long double cx_var_ld = __builtin_huge_vall();

// CIR: cir.global {{.*}} @cx_var_ld = #cir.fp<0x7FFF8000000000000000> : !cir.long_double<!cir.f80>
// LLVM: @cx_var_ld = {{.*}} x86_fp80 0xK7FFF8000000000000000
// OGCG: @cx_var_ld = {{.*}} x86_fp80 0xK7FFF8000000000000000

int is_constant_evaluated() {
  return __builtin_is_constant_evaluated();
}

// CIR: cir.func{{.*}} @_Z21is_constant_evaluatedv() -> !s32i
// CIR: %[[ZERO:.+]] = cir.const #cir.int<0>

// LLVM: define {{.*}}i32 @_Z21is_constant_evaluatedv()
// LLVM: %[[MEM:.+]] = alloca i32
// LLVM: store i32 0, ptr %[[MEM]]
// LLVM: %[[RETVAL:.+]] = load i32, ptr %[[MEM]]
// LLVM: ret i32 %[[RETVAL]]
// LLVM: }

// OGCG: define {{.*}}i32 @_Z21is_constant_evaluatedv()
// OGCG: ret i32 0
// OGCG: }

long double constant_fp_builtin_ld() {
  return __builtin_fabsl(-0.1L);
}

// CIR: cir.func{{.*}} @_Z22constant_fp_builtin_ldv() -> !cir.long_double<!cir.f80>
// CIR: %[[PONE:.+]] = cir.const #cir.fp<1.000000e-01> : !cir.long_double<!cir.f80>

// LLVM: define {{.*}}x86_fp80 @_Z22constant_fp_builtin_ldv()
// LLVM: %[[MEM:.+]] = alloca x86_fp80
// LLVM: store x86_fp80 0xK3FFBCCCCCCCCCCCCCCCD, ptr %[[MEM]]
// LLVM: %[[RETVAL:.+]] = load x86_fp80, ptr %[[MEM]]
// LLVM: ret x86_fp80 %[[RETVAL]]
// LLVM: }

// OGCG: define {{.*}}x86_fp80 @_Z22constant_fp_builtin_ldv()
// OGCG: ret x86_fp80 0xK3FFBCCCCCCCCCCCCCCCD
// OGCG: }

float constant_fp_builtin_single() {
  return __builtin_fabsf(-0.1f);
}

// CIR: cir.func{{.*}} @_Z26constant_fp_builtin_singlev() -> !cir.float
// CIR: %[[PONE:.+]] = cir.const #cir.fp<1.000000e-01> : !cir.float

// LLVM: define {{.*}}float @_Z26constant_fp_builtin_singlev()
// LLVM: %[[MEM:.+]] = alloca float
// LLVM: store float 0x3FB99999A0000000, ptr %[[MEM]]
// LLVM: %[[RETVAL:.+]] = load float, ptr %[[MEM]]
// LLVM: ret float %[[RETVAL]]
// LLVM: }

// OGCG: define {{.*}}float @_Z26constant_fp_builtin_singlev()
// OGCG: ret float 0x3FB99999A0000000
// OGCG: }

void library_builtins() {
  __builtin_printf(nullptr);
  __builtin_abort();
}

// CIR: cir.func{{.*}} @_Z16library_builtinsv() {
// CIR: %[[NULL:.+]] = cir.const #cir.ptr<null> : !cir.ptr<!s8i>
// CIR: cir.call @printf(%[[NULL]]) : (!cir.ptr<!s8i>) -> !s32i
// CIR: cir.call @abort() : () -> ()

// LLVM: define{{.*}} void @_Z16library_builtinsv()
// LLVM: call i32 (ptr, ...) @printf(ptr null)
// LLVM: call void @abort()

// OGCG: define{{.*}} void @_Z16library_builtinsv()
// OGCG: call i32 (ptr, ...) @printf(ptr noundef null)
// OGCG: call void @abort()

void assume(bool arg) {
  __builtin_assume(arg);
}

// CIR: cir.func{{.*}} @_Z6assumeb
// CIR:   cir.assume %{{.+}} : !cir.bool
// CIR: }

// LLVM: define {{.*}}void @_Z6assumeb
// LLVM:   call void @llvm.assume(i1 %{{.+}})
// LLVM: }

// OGCG: define {{.*}}void @_Z6assumeb
// OGCG:   call void @llvm.assume(i1 %{{.+}})
// OGCG: }

void expect(int x, int y) {
  __builtin_expect(x, y);
}

// CIR-LABEL: cir.func{{.*}} @_Z6expectii
// CIR:         %[[X:.+]] = cir.load align(4) %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:    %[[X_LONG:.+]] = cir.cast(integral, %[[X]] : !s32i), !s64i
// CIR-NEXT:    %[[Y:.+]] = cir.load align(4) %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:    %[[Y_LONG:.+]] = cir.cast(integral, %[[Y]] : !s32i), !s64i
// CIR-NEXT:    %{{.+}} = cir.expect(%[[X_LONG]], %[[Y_LONG]]) : !s64i
// CIR:       }

// LLVM-LABEL: define{{.*}} void @_Z6expectii
// LLVM:         %[[X:.+]] = load i32, ptr %{{.+}}, align 4
// LLVM-NEXT:    %[[X_LONG:.+]] = sext i32 %[[X]] to i64
// LLVM-NEXT:    %[[Y:.+]] = load i32, ptr %{{.+}}, align 4
// LLVM-NEXT:    %[[Y_LONG:.+]] = sext i32 %[[Y]] to i64
// LLVM-NEXT:    %{{.+}} = call i64 @llvm.expect.i64(i64 %[[X_LONG]], i64 %[[Y_LONG]])
// LLVM:       }

void expect_prob(int x, int y) {
  __builtin_expect_with_probability(x, y, 0.25);
}

// CIR-LABEL: cir.func{{.*}} @_Z11expect_probii
// CIR:         %[[X:.+]] = cir.load align(4) %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:    %[[X_LONG:.+]] = cir.cast(integral, %[[X]] : !s32i), !s64i
// CIR-NEXT:    %[[Y:.+]] = cir.load align(4) %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:    %[[Y_LONG:.+]] = cir.cast(integral, %[[Y]] : !s32i), !s64i
// CIR-NEXT:    %{{.+}} = cir.expect(%[[X_LONG]], %[[Y_LONG]], 2.500000e-01) : !s64i
// CIR:       }

// LLVM:       define{{.*}} void @_Z11expect_probii
// LLVM:         %[[X:.+]] = load i32, ptr %{{.+}}, align 4
// LLVM-NEXT:    %[[X_LONG:.+]] = sext i32 %[[X]] to i64
// LLVM-NEXT:    %[[Y:.+]] = load i32, ptr %{{.+}}, align 4
// LLVM-NEXT:    %[[Y_LONG:.+]] = sext i32 %[[Y]] to i64
// LLVM-NEXT:    %{{.+}} = call i64 @llvm.expect.with.probability.i64(i64 %[[X_LONG]], i64 %[[Y_LONG]], double 2.500000e-01)
// LLVM:       }
