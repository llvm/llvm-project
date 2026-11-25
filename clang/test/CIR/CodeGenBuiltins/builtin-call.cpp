// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=LLVM,OGCG

constexpr extern int cx_var = __builtin_is_constant_evaluated();

// CIR: cir.global {{.*}} @cx_var = #cir.int<1> : !s32i
// LLVM: @cx_var = {{.*}} i32 1

constexpr extern float cx_var_single = __builtin_huge_valf();

// CIR: cir.global {{.*}} @cx_var_single = #cir.fp<0x7F800000> : !cir.float
// LLVM: @cx_var_single = {{.*}} float +inf

constexpr extern long double cx_var_ld = __builtin_huge_vall();

// CIR: cir.global {{.*}} @cx_var_ld = #cir.fp<0x7FFF8000000000000000> : !cir.long_double<!cir.f80>
// LLVM: @cx_var_ld = {{.*}} x86_fp80 +inf

bool is_constant_evaluated() {
  return __builtin_is_constant_evaluated();
}

// CIR: cir.func{{.*}} @_Z21is_constant_evaluatedv() -> (!cir.bool{{.*}})
// CIR: %[[FALSE:.+]] = cir.const #false

// LLVM: define {{.*}}i1 @_Z21is_constant_evaluatedv()
// LLVM: ret i1 false
// LLVM: }

long double constant_fp_builtin_ld() {
  return __builtin_fabsl(-0.1L);
}

// CIR: cir.func{{.*}} @_Z22constant_fp_builtin_ldv() -> (!cir.long_double<!cir.f80>{{.*}})
// CIR: %[[PONE:.+]] = cir.const #cir.fp<1.000000e-01> : !cir.long_double<!cir.f80>

// LLVM: define {{.*}}x86_fp80 @_Z22constant_fp_builtin_ldv()
// LLVM: ret x86_fp80 1.000000e-01
// LLVM: }

float constant_fp_builtin_single() {
  return __builtin_fabsf(-0.1f);
}

// CIR: cir.func{{.*}} @_Z26constant_fp_builtin_singlev() -> (!cir.float{{.*}})
// CIR: %[[PONE:.+]] = cir.const #cir.fp<1.000000e-01> : !cir.float

// LLVM: define {{.*}}float @_Z26constant_fp_builtin_singlev()
// LLVM: ret float 1.000000e-01
// LLVM: }

void library_builtins() {
  __builtin_printf(nullptr);
  __builtin_abort();
}

// CIR: cir.func{{.*}} @_Z16library_builtinsv()
// CIR: %[[NULL:.+]] = cir.const #cir.ptr<null> : !cir.ptr<!s8i>
// CIR: cir.call @printf(%[[NULL]]) nothrow : (!cir.ptr<!s8i> {llvm.noundef}) -> !s32i
// CIR: cir.call @abort() nothrow {noreturn} : () -> ()

// LLVM: define{{.*}} void @_Z16library_builtinsv()
// LLVM: call i32 (ptr, ...) @printf(ptr noundef null)
// LLVM: call void @abort()

void assume(bool arg) {
  __builtin_assume(arg);
}

// CIR: cir.func{{.*}} @_Z6assumeb
// CIR:   cir.assume %{{.+}} : !cir.bool
// CIR: }

// LLVM: define {{.*}}void @_Z6assumeb
// LLVM:   call void @llvm.assume(i1 %{{.+}})
// LLVM: }

void *assume_aligned(void *ptr) {
  return __builtin_assume_aligned(ptr, 16);
}

// CIR: @_Z14assume_alignedPv
// CIR:   cir.assume %{{.+}} align(%{{.+}}, %{{.+}} : !cir.ptr<!void>, !u64i) : !cir.bool
// CIR: }

// LLVM: @_Z14assume_alignedPv
// LLVM:   call void @llvm.assume(i1 true) [ "align"(ptr %{{.+}}, i64 16) ]
// LLVM: }

void *assume_aligned_misalignment(void *ptr, unsigned misalignment) {
  return __builtin_assume_aligned(ptr, 16, misalignment);
}

// CIR: @_Z27assume_aligned_misalignmentPvj
// CIR:   cir.assume %{{.+}} align(%{{.+}}, %{{.+}}, %{{.+}} : !cir.ptr<!void>, !u64i, !u64i) : !cir.bool
// CIR: }

// LLVM: @_Z27assume_aligned_misalignmentPvj
// LLVM:   call void @llvm.assume(i1 true) [ "align"(ptr %{{.+}}, i64 16, i64 %{{.+}}) ]
// LLVM: }

void assume_separate_storage(void *p1, void *p2) {
  __builtin_assume_separate_storage(p1, p2);
}

// CIR: cir.func{{.*}} @_Z23assume_separate_storagePvS_
// CIR:   cir.assume %{{.+}} separate_storage(%{{.+}}, %{{.+}} : !cir.ptr<!void>, !cir.ptr<!void>) : !cir.bool
// CIR: }

// LLVM: define {{.*}}void @_Z23assume_separate_storagePvS_
// LLVM:   call void @llvm.assume(i1 true) [ "separate_storage"(ptr %{{.+}}, ptr %{{.+}}) ]
// LLVM: }

void assume_dereferenceable(void *p, unsigned long n) {
  __builtin_assume_dereferenceable(p, n);
}

// CIR: cir.func{{.*}} @_Z22assume_dereferenceablePvm
// CIR:   cir.assume %{{.+}} dereferenceable(%{{.+}}, %{{.+}} : !cir.ptr<!void>, !u64i) : !cir.bool
// CIR: }

// LLVM: define {{.*}}void @_Z22assume_dereferenceablePvm
// LLVM:   call void @llvm.assume(i1 true) [ "dereferenceable"(ptr %{{.+}}, i64 %{{.+}}) ]
// LLVM: }

void assume_dereferenceable_const(void *p) {
  __builtin_assume_dereferenceable(p, 16);
}

// CIR: cir.func{{.*}} @_Z28assume_dereferenceable_constPv
// CIR:   cir.assume %{{.+}} dereferenceable(%{{.+}}, %{{.+}} : !cir.ptr<!void>, !u64i) : !cir.bool
// CIR: }

// LLVM: define {{.*}}void @_Z28assume_dereferenceable_constPv
// LLVM:   call void @llvm.assume(i1 true) [ "dereferenceable"(ptr %{{.+}}, i64 16) ]
// LLVM: }

void assume_dereferenceable_narrow_size(void *p, unsigned n) {
  __builtin_assume_dereferenceable(p, n);
}

// CIR: cir.func{{.*}} @_Z34assume_dereferenceable_narrow_sizePvj
// CIR:   cir.assume %{{.+}} dereferenceable(%{{.+}}, %{{.+}} : !cir.ptr<!void>, !u64i) : !cir.bool
// CIR: }

// LLVM: define {{.*}}void @_Z34assume_dereferenceable_narrow_sizePvj
// LLVM:   call void @llvm.assume(i1 true) [ "dereferenceable"(ptr %{{.+}}, i64 %{{.+}}) ]
// LLVM: }

void expect(int x, int y) {
  __builtin_expect(x, y);
}

// At -O0, __builtin_expect is a passthrough (no cir.expect / llvm.expect).
// See pred-info-builtins.c for -O2 tests that verify cir.expect emission.
// CIR-LABEL: cir.func{{.*}} @_Z6expectii
// CIR-NOT:     cir.expect
// CIR:       }

// LLVM-LABEL: define{{.*}} void @_Z6expectii
// LLVM-NOT:     call i64 @llvm.expect
// LLVM:       }

void expect_prob(int x, int y) {
  __builtin_expect_with_probability(x, y, 0.25);
}

// CIR-LABEL: cir.func{{.*}} @_Z11expect_probii
// CIR-NOT:     cir.expect
// CIR:       }

// LLVM-LABEL: define{{.*}} void @_Z11expect_probii
// LLVM-NOT:     call i64 @llvm.expect
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

// LLVMCIR-LABEL: @_Z12unreachable2v
// LLVMCIR:         unreachable
// LLVMCIR:       {{.+}}:
// LLVMCIR-NEXT:    call void @_Z2f1v()
// LLVMCIR:       }

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

void trap2() {
  __builtin_trap();
  f1();
}

// CIR-LABEL: @_Z5trap2v
// CIR:         cir.trap
// CIR-NEXT:  ^{{.+}}:
// CIR-NEXT:    cir.call @_Z2f1v() : () -> ()
// CIR:       }

// LLVMCIR-LABEL: @_Z5trap2v
// LLVMCIR:         call void @llvm.trap()
// LLVMCIR-NEXT:    unreachable
// LLVMCIR:       {{.+}}:
// LLVMCIR-NEXT:    call void @_Z2f1v()
// LLVMCIR:       }

// OGCG-LABEL: define{{.*}} void @_Z5trap2v
// OGCG:         call void @llvm.trap()
// OGCG-NEXT:    unreachable
// OGCG-NEXT:  }

void *test_alloca(unsigned long n) {
  return __builtin_alloca(n);
}

// CIR-LABEL: @_Z11test_allocam(
// CIR:         %{{.+}} = cir.alloca "bi_alloca" {{.*}} size(%{{.+}}) : !cir.ptr<!u8i>

// LLVM-LABEL: @_Z11test_allocam(
// LLVM:         alloca i8, i64 %{{.+}}

bool test_multiple_allocas(unsigned long n) {
  void *a = __builtin_alloca(n);
  void *b = __builtin_alloca(n);
  return a != b;
}

// CIR-LABEL: @_Z21test_multiple_allocasm(
// CIR:         %{{.+}} = cir.alloca "bi_alloca" {{.*}} size(%{{.+}}) : !cir.ptr<!u8i>
// CIR:         %{{.+}} = cir.alloca "bi_alloca" {{.*}} size(%{{.+}}) : !cir.ptr<!u8i>

// LLVM-LABEL: @_Z21test_multiple_allocasm(
// LLVM:         alloca i8, i64 %{{.+}}
// LLVM:         alloca i8, i64 %{{.+}}

