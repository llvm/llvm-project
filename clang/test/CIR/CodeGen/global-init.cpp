// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --input-file=%t-before.cir %s --check-prefix=CIR-BEFORE-LPP
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

struct NeedsCtor {
  NeedsCtor();
};

NeedsCtor needsCtor;

// CIR-BEFORE-LPP: cir.global external @needsCtor = ctor : !rec_NeedsCtor {
// CIR-BEFORE-LPP:   %[[THIS:.*]] = cir.get_global @needsCtor : !cir.ptr<!rec_NeedsCtor>
// CIR-BEFORE-LPP:   cir.call @_ZN9NeedsCtorC1Ev(%[[THIS]]) : (!cir.ptr<!rec_NeedsCtor>) -> ()

// CIR: module @{{.*}} attributes {
// CIR-SAME: cir.global_ctors = [#cir.global_ctor<"_GLOBAL__sub_I_[[FILENAME:.*]]", 65535>]

// CIR: cir.global external @needsCtor = #cir.zero : !rec_NeedsCtor
// CIR: cir.func internal private @__cxx_global_var_init() {
// CIR:   %0 = cir.get_global @needsCtor : !cir.ptr<!rec_NeedsCtor>
// CIR:   cir.call @_ZN9NeedsCtorC1Ev(%0) : (!cir.ptr<!rec_NeedsCtor>) -> ()

// CIR: cir.func private @_GLOBAL__sub_I_[[FILENAME:.*]]() {
// CIR:   cir.call @__cxx_global_var_init() : () -> ()
// CIR:   cir.return
// CIR: }

// LLVM: @needsCtor = global %struct.NeedsCtor zeroinitializer, align 1
// LLVM: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_[[FILENAME:.*]], ptr null }]
// LLVM: declare void @_ZN9NeedsCtorC1Ev(ptr)

// LLVM: define internal void @__cxx_global_var_init()
// LLVM:   call void @_ZN9NeedsCtorC1Ev(ptr @needsCtor)

// LLVM: define void @_GLOBAL__sub_I_[[FILENAME]]()
// LLVM:   call void @__cxx_global_var_init()

// OGCG: @needsCtor = global %struct.NeedsCtor zeroinitializer, align 1
// OGCG: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_[[FILENAME:.*]], ptr null }]

// OGCG: define internal void @__cxx_global_var_init() {{.*}} section ".text.startup" {
// OGCG:   call void @_ZN9NeedsCtorC1Ev(ptr noundef nonnull align 1 dereferenceable(1) @needsCtor)

// OGCG: define internal void @_GLOBAL__sub_I_[[FILENAME]]() {{.*}} section ".text.startup" {
// OGCG:   call void @__cxx_global_var_init()
