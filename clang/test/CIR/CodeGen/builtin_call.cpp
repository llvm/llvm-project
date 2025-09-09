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
// CIR: cir.call @printf(%[[NULL]]) nothrow : (!cir.ptr<!s8i>) -> !s32i
// CIR: cir.call @abort() nothrow : () -> ()

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

void *assume_aligned(void *ptr) {
  return __builtin_assume_aligned(ptr, 16);
}

// CIR: @_Z14assume_alignedPv
// CIR:   %{{.+}} = cir.assume_aligned %{{.+}} alignment 16 : !cir.ptr<!void>
// CIR: }

// LLVM: @_Z14assume_alignedPv
// LLVM:   call void @llvm.assume(i1 true) [ "align"(ptr %{{.+}}, i64 16) ]
// LLVM: }

// OGCG: @_Z14assume_alignedPv
// OGCG:   call void @llvm.assume(i1 true) [ "align"(ptr %{{.+}}, i64 16) ]
// OGCG: }

void *assume_aligned_misalignment(void *ptr, unsigned misalignment) {
  return __builtin_assume_aligned(ptr, 16, misalignment);
}

// CIR: @_Z27assume_aligned_misalignmentPvj
// CIR:   %{{.+}} = cir.assume_aligned %{{.+}} alignment 16[offset %{{.+}} : !u64i] : !cir.ptr<!void>
// CIR: }

// LLVM: @_Z27assume_aligned_misalignmentPvj
// LLVM:   call void @llvm.assume(i1 true) [ "align"(ptr %{{.+}}, i64 16, i64 %{{.+}}) ]
// LLVM: }

// OGCG: @_Z27assume_aligned_misalignmentPvj
// OGCG:   call void @llvm.assume(i1 true) [ "align"(ptr %{{.+}}, i64 16, i64 %{{.+}}) ]
// OGCG: }

void assume_separate_storage(void *p1, void *p2) {
  __builtin_assume_separate_storage(p1, p2);
}

// CIR: cir.func{{.*}} @_Z23assume_separate_storagePvS_
// CIR:   cir.assume_separate_storage %{{.+}}, %{{.+}} : !cir.ptr<!void>
// CIR: }

// LLVM: define {{.*}}void @_Z23assume_separate_storagePvS_
// LLVM:   call void @llvm.assume(i1 true) [ "separate_storage"(ptr %{{.+}}, ptr %{{.+}}) ]
// LLVM: }

// OGCG: define {{.*}}void @_Z23assume_separate_storagePvS_
// OGCG:   call void @llvm.assume(i1 true) [ "separate_storage"(ptr %{{.+}}, ptr %{{.+}}) ]
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

void unreachable() {
  __builtin_unreachable();
}

// CIR-LABEL: @_Z11unreachablev
// CIR:         cir.unreachable
// CIR:       }

// LLVM-LABEL: @_Z11unreachablev
// LLVM:         unreachable
// LLVM:       }

// OGCG-LABEL: @_Z11unreachablev
// OGCG:         unreachable
// OGCG:       }

void f1();
void unreachable2() {
  __builtin_unreachable();
  f1();
}

// CIR-LABEL: @_Z12unreachable2v
// CIR:         cir.unreachable
// CIR-NEXT:  ^{{.+}}:
// CIR-NEXT:    cir.call @_Z2f1v() : () -> ()
// CIR:       }

// LLVM-LABEL: @_Z12unreachable2v
// LLVM:         unreachable
// LLVM:       {{.+}}:
// LLVM-NEXT:    call void @_Z2f1v()
// LLVM:       }

// OGCG-LABEL: @_Z12unreachable2v
// OGCG:         unreachable

void trap() {
  __builtin_trap();
}

// CIR-LABEL: @_Z4trapv
// CIR:         cir.trap
// CIR:       }

// LLVM-LABEL: @_Z4trapv
// LLVM:         call void @llvm.trap()
// LLVM:       }

// OGCG-LABEL: @_Z4trapv
// OGCG:         call void @llvm.trap()
// OGCG:       }

void trap2() {
  __builtin_trap();
  f1();
}

// CIR-LABEL: @_Z5trap2v
// CIR:         cir.trap
// CIR-NEXT:  ^{{.+}}:
// CIR-NEXT:    cir.call @_Z2f1v() : () -> ()
// CIR:       }

// LLVM-LABEL: @_Z5trap2v
// LLVM:         call void @llvm.trap()
// LLVM-NEXT:    unreachable
// LLVM:       {{.+}}:
// LLVM-NEXT:    call void @_Z2f1v()
// LLVM:       }

// OGCG-LABEL: define{{.*}} void @_Z5trap2v
// OGCG:         call void @llvm.trap()
// OGCG-NEXT:    call void @_Z2f1v()
// OGCG:         ret void
// OGCG:       }

void *test_alloca(unsigned long n) {
  return __builtin_alloca(n);
}

// CIR-LABEL: @_Z11test_allocam(
// CIR:         %{{.+}} = cir.alloca !u8i, !cir.ptr<!u8i>, %{{.+}} : !u64i, ["bi_alloca"]

// LLVM-LABEL: @_Z11test_allocam(
// LLVM:         alloca i8, i64 %{{.+}}

// OGCG-LABEL: @_Z11test_allocam(
// OGCG:         alloca i8, i64 %{{.+}}

bool test_multiple_allocas(unsigned long n) {
  void *a = __builtin_alloca(n);
  void *b = __builtin_alloca(n);
  return a != b;
}

// CIR-LABEL: @_Z21test_multiple_allocasm(
// CIR:         %{{.+}} = cir.alloca !u8i, !cir.ptr<!u8i>, %{{.+}} : !u64i, ["bi_alloca"]
// CIR:         %{{.+}} = cir.alloca !u8i, !cir.ptr<!u8i>, %{{.+}} : !u64i, ["bi_alloca"]

// LLVM-LABEL: @_Z21test_multiple_allocasm(
// LLVM:         alloca i8, i64 %{{.+}}
// LLVM:         alloca i8, i64 %{{.+}}

// OGCG-LABEL: @_Z21test_multiple_allocasm(
// OGCG:         alloca i8, i64 %{{.+}}
// OGCG:         alloca i8, i64 %{{.+}}
