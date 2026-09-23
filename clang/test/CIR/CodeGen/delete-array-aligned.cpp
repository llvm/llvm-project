// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -mconstructor-aliases -emit-cir -mmlir -mlir-print-ir-before=cir-cxxabi-lowering %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --input-file=%t-before.cir -check-prefix=CIR,CIR-BEFORE %s
// RUN: FileCheck --input-file=%t.cir --check-prefix=CIR,CIR-AFTER %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -mconstructor-aliases -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -mconstructor-aliases -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll --check-prefix=LLVM %s

typedef decltype(sizeof(0)) size_t;
namespace std { enum class align_val_t : size_t {}; }

struct alignas(128) OverAlignedTy {
  OverAlignedTy();
  void* v;
};
void test_global_aligned(OverAlignedTy *p) { delete[] p; }
// CIR-LABEL: cir.func no_inline dso_local @_Z19test_global_alignedP13OverAlignedTy(
// CIR-BEFORE: cir.delete_array %{{.*}} : !cir.ptr<!rec_OverAlignedTy> {delete_fn = @_ZdaPvSt11align_val_t, delete_params = #cir.usual_delete_params<alignment = 128>}

// CIR-AFTER:      %[[PTR:.*]] = cir.cast bitcast %{{.*}} : !cir.ptr<!rec_OverAlignedTy> -> !cir.ptr<!void>
// CIR-AFTER-NEXT: cir.cleanup.scope {
// CIR-AFTER-NEXT:   cir.yield
// CIR-AFTER-NEXT: } cleanup normal {
// CIR-AFTER-NEXT:   %[[ALIGN:.*]] = cir.const #cir.int<128> : !u64i
// CIR-AFTER-NEXT:   cir.call @_ZdaPvSt11align_val_t(%[[PTR]], %[[ALIGN]]) nothrow : (!cir.ptr<!void>, !u64i) -> ()
// CIR-AFTER-NEXT:   cir.yield
// CIR-AFTER-NEXT: }

// LLVM-LABEL: define dso_local void @_Z19test_global_alignedP13OverAlignedTy(
// LLVM: call void @_ZdaPvSt11align_val_t(ptr {{(noundef )?}}%{{.*}}, i64 {{(noundef )?}}128)

struct alignas(128) OverAlignedTyClassDelete {
  OverAlignedTyClassDelete();
  void operator delete[](void *, std::align_val_t);
  void *v;
};
void test_class_align_only(OverAlignedTyClassDelete *p) { delete[] p; }
// CIR-LABEL: cir.func no_inline dso_local @_Z21test_class_align_onlyP24OverAlignedTyClassDelete(
// CIR-BEFORE: cir.delete_array %{{.*}} : !cir.ptr<!rec_OverAlignedTyClassDelete> {delete_fn = @_ZN24OverAlignedTyClassDeletedaEPvSt11align_val_t, delete_params = #cir.usual_delete_params<alignment = 128>}
// CIR-AFTER: %[[PTR:.*]] = cir.cast bitcast %{{.*}} : !cir.ptr<!rec_OverAlignedTyClassDelete> -> !cir.ptr<!void>
// CIR-AFTER-NEXT: cir.cleanup.scope {
// CIR-AFTER-NEXT:   cir.yield
// CIR-AFTER-NEXT: } cleanup normal {
// CIR-AFTER-NEXT:   %[[ALIGN:.*]] = cir.const #cir.int<128> : !u64i
// CIR-AFTER-NEXT:   cir.call @_ZN24OverAlignedTyClassDeletedaEPvSt11align_val_t(%[[PTR]], %[[ALIGN]]) nothrow : (!cir.ptr<!void>, !u64i) -> ()
// CIR-AFTER-NEXT:   cir.yield
// CIR-AFTER-NEXT: }

// LLVM-LABEL: define dso_local void @_Z21test_class_align_onlyP24OverAlignedTyClassDelete(
// LLVM: call void @_ZN24OverAlignedTyClassDeletedaEPvSt11align_val_t(ptr {{(noundef )?}}%{{.*}}, i64 {{(noundef )?}}128)

struct alignas(128) OverAlignedTyClassDeleteCookie {
  OverAlignedTyClassDeleteCookie();
  void operator delete[](void *, size_t, std::align_val_t);
  void *v;
};
void test_class_size_align(OverAlignedTyClassDeleteCookie *p) { delete[] p; }
// CIR-LABEL: cir.func no_inline dso_local @_Z21test_class_size_alignP30OverAlignedTyClassDeleteCookie(
// CIR-BEFORE: cir.delete_array %{{.*}} : !cir.ptr<!rec_OverAlignedTyClassDeleteCookie> {delete_fn = @_ZN30OverAlignedTyClassDeleteCookiedaEPvmSt11align_val_t, delete_params = #cir.usual_delete_params<size = true, alignment = 128>}
// CIR-AFTER:      %[[ORIG_PTR:.*]] = cir.cast bitcast %{{.*}} : !cir.ptr<!rec_OverAlignedTyClassDeleteCookie> -> !cir.ptr<!u8i>
// CIR-AFTER:      %[[COOKIE_STRIDE:.*]] = cir.ptr_stride %[[ORIG_PTR]], %{{.*}} : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR-AFTER-NEXT: %[[PTR:.*]] = cir.cast bitcast %[[COOKIE_STRIDE]] : !cir.ptr<!u8i> -> !cir.ptr<!void>
// CIR-AFTER:      cir.cleanup.scope {
// CIR-AFTER-NEXT:   cir.yield
// CIR-AFTER-NEXT: } cleanup normal {
// CIR-AFTER:        %[[SIZE:.*]] = cir.add %{{.*}}, %{{.*}} : !u64i
// CIR-AFTER-NEXT:   %[[ALIGN:.*]] = cir.const #cir.int<128> : !u64i
// CIR-AFTER-NEXT:   cir.call @_ZN30OverAlignedTyClassDeleteCookiedaEPvmSt11align_val_t(%[[PTR]], %[[SIZE]], %[[ALIGN]]) nothrow : (!cir.ptr<!void>, !u64i, !u64i) -> ()
// CIR-AFTER-NEXT:   cir.yield
// CIR-AFTER-NEXT: }

// LLVM-LABEL: define dso_local void @_Z21test_class_size_alignP30OverAlignedTyClassDeleteCookie(
// LLVM: call void @_ZN30OverAlignedTyClassDeleteCookiedaEPvmSt11align_val_t(ptr {{(noundef )?}}%{{.*}}, i64 {{(noundef )?}}%{{.*}}, i64 {{(noundef )?}}128)

struct alignas(128) OverAlignedTyClassDeleteDtor {
  OverAlignedTyClassDeleteDtor();
  ~OverAlignedTyClassDeleteDtor();
  void operator delete[](void *, std::align_val_t);
  void *v;
};
void test_class_align_only_dtor(OverAlignedTyClassDeleteDtor *p) { delete[] p; }
// CIR-LABEL: cir.func no_inline dso_local @_Z26test_class_align_only_dtorP28OverAlignedTyClassDeleteDtor(
// CIR-BEFORE: cir.delete_array %{{.*}} : !cir.ptr<!rec_OverAlignedTyClassDeleteDtor> {delete_fn = @_ZN28OverAlignedTyClassDeleteDtordaEPvSt11align_val_t, delete_params = #cir.usual_delete_params<alignment = 128>, element_dtor = @_ZN28OverAlignedTyClassDeleteDtorD1Ev}
// CIR-AFTER:      %[[ORIG_PTR:.*]] = cir.cast bitcast %{{.*}} : !cir.ptr<!rec_OverAlignedTyClassDeleteDtor> -> !cir.ptr<!u8i>
// CIR-AFTER:      %[[COOKIE_STRIDE:.*]] = cir.ptr_stride %[[ORIG_PTR]], %{{.*}} : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR-AFTER-NEXT: %[[PTR:.*]] = cir.cast bitcast %[[COOKIE_STRIDE]] : !cir.ptr<!u8i> -> !cir.ptr<!void>
// CIR-AFTER:      cir.cleanup.scope {
// CIR-AFTER:        cir.if %{{.*}} {
// CIR-AFTER:          cir.do {
// CIR-AFTER:            cir.call @_ZN28OverAlignedTyClassDeleteDtorD1Ev(%{{.*}}) nothrow : (!cir.ptr<!rec_OverAlignedTyClassDeleteDtor>) -> ()
// CIR-AFTER-NEXT:       cir.yield
// CIR-AFTER-NEXT:     } while {
// CIR-AFTER:            cir.condition(%{{.*}})
// CIR-AFTER-NEXT:     }
// CIR-AFTER-NEXT:   }
// CIR-AFTER-NEXT:   cir.yield
// CIR-AFTER-NEXT: } cleanup normal {
// CIR-AFTER-NEXT:   %[[ALIGN:.*]] = cir.const #cir.int<128> : !u64i
// CIR-AFTER-NEXT:   cir.call @_ZN28OverAlignedTyClassDeleteDtordaEPvSt11align_val_t(%[[PTR]], %[[ALIGN]]) nothrow : (!cir.ptr<!void>, !u64i) -> ()
// CIR-AFTER-NEXT:   cir.yield
// CIR-AFTER-NEXT: }

// LLVM-LABEL: define dso_local void @_Z26test_class_align_only_dtorP28OverAlignedTyClassDeleteDtor(
// LLVM: call void @_ZN28OverAlignedTyClassDeleteDtorD1Ev(ptr {{.*}}%{{.*}})
// LLVM: call void @_ZN28OverAlignedTyClassDeleteDtordaEPvSt11align_val_t(ptr {{(noundef )?}}%{{.*}}, i64 {{(noundef )?}}128)

struct alignas(128) OverAlignedTyClassDeleteCookieDtor {
  OverAlignedTyClassDeleteCookieDtor();
  ~OverAlignedTyClassDeleteCookieDtor();
  void operator delete[](void *, size_t, std::align_val_t);
  void *v;
};
void test_class_size_align_dtor(OverAlignedTyClassDeleteCookieDtor *p) {
  delete[] p;
}
// CIR-LABEL: cir.func no_inline dso_local @_Z26test_class_size_align_dtorP34OverAlignedTyClassDeleteCookieDtor(
// CIR-BEFORE: cir.delete_array %{{.*}} : !cir.ptr<!rec_OverAlignedTyClassDeleteCookieDtor> {delete_fn = @_ZN34OverAlignedTyClassDeleteCookieDtordaEPvmSt11align_val_t, delete_params = #cir.usual_delete_params<size = true, alignment = 128>, element_dtor = @_ZN34OverAlignedTyClassDeleteCookieDtorD1Ev}
// CIR-AFTER:      %[[ORIG_PTR:.*]] = cir.cast bitcast %{{.*}} : !cir.ptr<!rec_OverAlignedTyClassDeleteCookieDtor> -> !cir.ptr<!u8i>
// CIR-AFTER:      %[[COOKIE_STRIDE:.*]] = cir.ptr_stride %[[ORIG_PTR]], %{{.*}} : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR-AFTER-NEXT: %[[PTR:.*]] = cir.cast bitcast %[[COOKIE_STRIDE]] : !cir.ptr<!u8i> -> !cir.ptr<!void>
// CIR-AFTER:      cir.cleanup.scope {
// CIR-AFTER:       cir.if %{{.*}} {
// CIR-AFTER:           cir.do {
// CIR-AFTER:            cir.call @_ZN34OverAlignedTyClassDeleteCookieDtorD1Ev(%{{.*}}) nothrow : (!cir.ptr<!rec_OverAlignedTyClassDeleteCookieDtor>) -> ()
// CIR-AFTER-NEXT:       cir.yield
// CIR-AFTER-NEXT:     } while {
// CIR-AFTER:            cir.condition(%{{.*}})
// CIR-AFTER-NEXT:     }
// CIR-AFTER-NEXT:   }
// CIR-AFTER-NEXT:   cir.yield
// CIR-AFTER-NEXT: } cleanup normal {
// CIR-AFTER:        %[[SIZE:.*]] = cir.add %{{.*}}, %{{.*}} : !u64i
// CIR-AFTER-NEXT:   %[[ALIGN:.*]] = cir.const #cir.int<128> : !u64i
// CIR-AFTER-NEXT:   cir.call @_ZN34OverAlignedTyClassDeleteCookieDtordaEPvmSt11align_val_t(%7, %[[SIZE]], %[[ALIGN]]) nothrow : (!cir.ptr<!void>, !u64i, !u64i) -> ()
// CIR-AFTER-NEXT:   cir.yield
// CIR-AFTER-NEXT: }

// LLVM-LABEL: define dso_local void @_Z26test_class_size_align_dtorP34OverAlignedTyClassDeleteCookieDtor(
// LLVM: call void @_ZN34OverAlignedTyClassDeleteCookieDtorD1Ev(ptr {{.*}}%{{.*}})
// LLVM: call void @_ZN34OverAlignedTyClassDeleteCookieDtordaEPvmSt11align_val_t(ptr {{(noundef )?}}%{{.*}}, i64 {{(noundef )?}}%{{.*}}, i64 {{(noundef )?}}128)
