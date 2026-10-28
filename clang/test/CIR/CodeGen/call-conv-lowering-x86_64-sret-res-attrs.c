// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -fno-clangir-call-conv-lowering -menable-no-infs -menable-no-nans \
// RUN:   -emit-cir %s -o %t-before.cir
// RUN: FileCheck --check-prefix=BEFORE --input-file=%t-before.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -menable-no-infs \
// RUN:   -menable-no-nans -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s \
// RUN:   --implicit-check-not=llvm.nofpclass
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -menable-no-infs \
// RUN:   -menable-no-nans -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -menable-no-infs \
// RUN:   -menable-no-nans -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// A return with a floating representation gets llvm.nofpclass under
// -menable-no-infs or -menable-no-nans, and a _Complex __float128 is returned
// indirectly, so the attribute describes a result the sret rewrite removes.
// The BEFORE lines pin that the attribute is there to remove, so the CIR
// checks cannot pass by it never having been attached.
_Complex __float128 ret_cf128(void) { return 1.0Q; }

// BEFORE: cir.func{{.*}} @ret_cf128() -> (!cir.complex<!cir.f128> {llvm.nofpclass = 519 : i64})

// CIR:      cir.func{{.*}} @ret_cf128(%arg0: !cir.ptr<!cir.complex<!cir.f128>>
// CIR-SAME:   llvm.sret = !cir.complex<!cir.f128>

// LLVM: define dso_local void @ret_cf128(ptr dead_on_unwind noalias writable sret({ fp128, fp128 }) align 16 %{{.+}})

_Complex __float128 ret_cf128_decl(void);
void call_cf128(void) { _Complex __float128 x = ret_cf128_decl(); (void)x; }

// BEFORE: cir.call @ret_cf128_decl() : () -> (!cir.complex<!cir.f128> {llvm.nofpclass = 519 : i64})
// BEFORE: cir.func private @ret_cf128_decl() -> (!cir.complex<!cir.f128> {llvm.nofpclass = 519 : i64})

// CIR:      cir.call @ret_cf128_decl(%{{.+}}) : (!cir.ptr<!cir.complex<!cir.f128>>
// CIR-SAME:   llvm.sret = !cir.complex<!cir.f128>
// CIR-SAME:   -> ()

// LLVM: call void @ret_cf128_decl(ptr dead_on_unwind writable sret({ fp128, fp128 }) align 16 %{{.+}})
// LLVM: declare void @ret_cf128_decl(ptr dead_on_unwind writable sret({ fp128, fp128 }) align 16)
