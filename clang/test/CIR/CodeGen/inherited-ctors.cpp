// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct Base {
  Base(int i);
  Base(float, ...);
};

struct Derived : Base {
  using Base::Base;
};

struct VirtDerived : virtual Base {
  using Base::Base;
};

struct VirtualDelegatingCtor : VirtDerived {
  VirtualDelegatingCtor(int x) : Base(x), VirtDerived(x){}
};

void emitDelegateCallArgs() {
  // ONLY PassPrototypeArgs
  Derived canEmitDelegateCallArgs{1};
}

void cannotEmitDelegateCallArgs() {
  // Inside of the PassPrototypeArgs && !canEmitDelegateCallArgs
  Derived cannotEmitDelgateCallArgs{1.1f,2,3.0};
}
void fallsthrough() {
  // !PassPrototypeArgs
  VirtualDelegatingCtor noInheritingCtorHasParams{1};
}


// LLVM and OGCG check labels are identical other than the 1 difference called out (and ordering).
// CIR-LABEL: cir.func private @_ZN4BaseC2Ei(!cir.ptr<!rec_Base>{{.*}}, !s32i{{.*}}) special_member<#cir.cxx_ctor<!rec_Base, custom>>
// LLVM-LABEL: declare void @_ZN4BaseC2Ei(ptr {{.*}}, i32 {{.*}})
//
//
// CIR-LABEL: cir.func no_inline comdat linkonce_odr @_ZN7DerivedCI24BaseEi(%{{.*}}: !cir.ptr<!rec_Derived>{{.*}}, %{{.*}}: !s32i{{.*}}) special_member<#cir.cxx_ctor<!rec_Derived, custom>>
// CIR: %[[THIS_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_Derived>, !cir.ptr<!cir.ptr<!rec_Derived>>, ["this", init]
// CIR: %[[INT_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["", init]
// CIR: %[[THIS_LOAD:.*]] = cir.load %[[THIS_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_Derived>>, !cir.ptr<!rec_Derived>
// CIR: %[[BASE_ADDR:.*]] = cir.base_class_addr %[[THIS_LOAD]] : !cir.ptr<!rec_Derived> nonnull [0] -> !cir.ptr<!rec_Base>
// CIR: %[[INT:.*]] = cir.load align(4) %[[INT_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.call @_ZN4BaseC2Ei(%[[BASE_ADDR]], %[[INT]]) : (!cir.ptr<!rec_Base>{{.*}}, !s32i{{.*}}) -> () 
//
// LLVM-LABEL: define linkonce_odr void @_ZN7DerivedCI24BaseEi(ptr {{.*}}, i32 {{.*}})
// LLVM: %[[THIS_ALLOCA:.*]] = alloca ptr
// LLVM: %[[INT_ALLOCA:.*]] = alloca i32
// LLVM: %[[BASE_ADDR:.*]] = load ptr, ptr %[[THIS_ALLOCA]]
// LLVM: %[[INT:.*]] = load i32, ptr %[[INT_ALLOCA]]
// LLVM: call void @_ZN4BaseC2Ei(ptr {{.*}}%[[BASE_ADDR]], i32 {{.*}}[[INT]])
//
//
// CIR-LABEL: cir.func no_inline dso_local @_Z20emitDelegateCallArgsv()
// CIR: cir.call @_ZN7DerivedCI14BaseEi(%{{.*}}, %{{.*}}) : (!cir.ptr<!rec_Derived>{{.*}}, !s32i{{.*}}) -> ()
// LLVM-LABEL: define dso_local void @_Z20emitDelegateCallArgsv()
// LLVM: call void @_ZN7DerivedCI14BaseEi(ptr {{.*}}, i32 {{.*}}1)
//
// CIR-LABEL: cir.func private @_ZN4BaseC2Efz(!cir.ptr<!rec_Base>{{.*}}, !cir.float{{.*}}, ...) special_member<#cir.cxx_ctor<!rec_Base, custom>>
// LLVM-LABEL: declare void @_ZN4BaseC2Efz(ptr {{.*}}, float {{.*}}, ...)
//
// CIR-LABEL: cir.func no_inline dso_local @_Z26cannotEmitDelegateCallArgsv()
// CIR: %[[TMP_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_Derived>, !cir.ptr<!cir.ptr<!rec_Derived>>, ["tmp", init]
// CIR: %[[FP_1_1:.*]] = cir.const #cir.fp<1.1{{.*}}> : !cir.float
// CIR: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
// CIR: %[[THREE:.*]] = cir.const #cir.fp<3.0{{.*}}> : !cir.double
// CIR: %[[LOAD_DERIVED:.*]] = cir.load %[[TMP_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_Derived>>, !cir.ptr<!rec_Derived>
// CIR: %[[BASE_ADDR:.*]] = cir.base_class_addr %[[LOAD_DERIVED]] : !cir.ptr<!rec_Derived> nonnull [0] -> !cir.ptr<!rec_Base>
// CIR: cir.call @_ZN4BaseC2Efz(%[[BASE_ADDR]], %[[FP_1_1]], %[[TWO]], %[[THREE]]) : (!cir.ptr<!rec_Base>{{.*}}, !cir.float{{.*}}, !s32i{{.*}}, !cir.double{{.*}}) -> ()
//
// LLVM-LABEL: define dso_local void @_Z26cannotEmitDelegateCallArgsv()
// LLVM: %[[TMP_ALLOCA:.*]] = alloca ptr
// LLVM: %[[TMP_LOAD:.*]] = load ptr, ptr %[[TMP_ALLOCA]]
// LLVM: call void (ptr, float, ...) @_ZN4BaseC2Efz(ptr {{.*}}%[[TMP_LOAD]], float {{.*}}0x3FF19999A{{.*}}, i32 {{.*}}2, double {{.*}}3.000000e+00)
//
// CIR-LABEL: cir.func no_inline comdat linkonce_odr @_ZN11VirtDerivedCI24BaseEi(%{{.*}}: !cir.ptr<!rec_VirtDerived> {{.*}}, %{{.*}}: !cir.ptr<!cir.ptr<!void>>{{.*}}) special_member<#cir.cxx_ctor<!rec_VirtDerived, custom>>
// CIR: %[[THIS_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_VirtDerived>, !cir.ptr<!cir.ptr<!rec_VirtDerived>>, ["this", init] {alignment = 8 : i64}
// CIR: %[[VTT_ALLOCA:.]] = cir.alloca !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!cir.ptr<!cir.ptr<!void>>>, ["vtt", init] {alignment = 8 : i64}
// CIR: %[[THIS:.*]] = cir.load %[[THIS_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_VirtDerived>>, !cir.ptr<!rec_VirtDerived>
// CIR: %[[VTT:.*]] = cir.load align(8) %[[VTT_ALLOCA]] : !cir.ptr<!cir.ptr<!cir.ptr<!void>>>, !cir.ptr<!cir.ptr<!void>>
// CIR: %[[VTT_ADDR:.*]] = cir.vtt.address_point %[[VTT]] : !cir.ptr<!cir.ptr<!void>>, offset = 0 -> !cir.ptr<!cir.ptr<!void>>
// CIR: %[[VTT_ADDR_CAST:.*]] = cir.cast bitcast %[[VTT_ADDR]] : !cir.ptr<!cir.ptr<!void>> -> !cir.ptr<!cir.vptr>
// CIR: %[[VTT_ADDR_LOAD:.*]] = cir.load align(8) %[[VTT_ADDR_CAST]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR: %[[VPTR:.*]] = cir.vtable.get_vptr %[[THIS]] : !cir.ptr<!rec_VirtDerived> -> !cir.ptr<!cir.vptr>
// CIR: cir.store align(8) %[[VTT_ADDR_LOAD]], %[[VPTR]] : !cir.vptr, !cir.ptr<!cir.vptr>

// LLVM-LABEL: define linkonce_odr void @_ZN11VirtDerivedCI24BaseEi(ptr {{.*}}, ptr {{.*}})
// LLVM: %[[THIS_ALLOCA:.*]] = alloca ptr
// LLVM: %[[VTT_ALLOCA:.*]] = alloca ptr
// LLVM: %[[THIS:.*]] = load ptr, ptr %[[THIS_ALLOCA]]
// LLVM: %[[VTT:.*]] = load ptr, ptr %[[VTT_ALLOCA]]
// LLVM: %[[VTT_ADDR_LOAD:.*]] = load ptr, ptr %[[VTT]]
// LLVM: store ptr %[[VTT_ADDR_LOAD]], ptr %[[THIS]]
//
//
// CIR-LABEL: cir.func no_inline comdat linkonce_odr @_ZN21VirtualDelegatingCtorC1Ei(%{{.*}}: !cir.ptr<!rec_VirtualDelegatingCtor> {{.*}}, %{{.*}}: !s32i {{.*}}) special_member<#cir.cxx_ctor<!rec_VirtualDelegatingCtor, custom>>
// CIR: %[[THIS_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_VirtualDelegatingCtor>, !cir.ptr<!cir.ptr<!rec_VirtualDelegatingCtor>>, ["this", init] {alignment = 8 : i64}
// CIR: %[[X_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init] {alignment = 4 : i64}
// CIR: %[[THIS_LOAD:.*]] = cir.load %[[THIS_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_VirtualDelegatingCtor>>, !cir.ptr<!rec_VirtualDelegatingCtor>
// CIR: %[[BASE_ADDR:.*]] = cir.base_class_addr %[[THIS_LOAD]] : !cir.ptr<!rec_VirtualDelegatingCtor> nonnull [0] -> !cir.ptr<!rec_Base>
// CIR: %[[X_LOAD:.*]] = cir.load align(4) %[[X_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.call @_ZN4BaseC2Ei(%[[BASE_ADDR]], %[[X_LOAD]]) : (!cir.ptr<!rec_Base> {{.*}}, !s32i {{{.*}}) -> ()
//
// LLVM-LABEL: define linkonce_odr void @_ZN21VirtualDelegatingCtorC1Ei(ptr {{.*}}, i32 {{.*}})
// LLVM: %[[THIS_ALLOCA:.*]] = alloca ptr
// LLVM: %[[X_ALLOCA:.*]] = alloca i32
// LLVM: %[[THIS_LOAD:.*]] = load ptr, ptr %[[THIS_ALLOCA]]
// LLVM: %[[X_LOAD:.*]] = load i32, ptr %[[X_ALLOCA]]
// LLVM: call void @_ZN4BaseC2Ei(ptr {{.*}}%[[THIS_LOAD]], i32 {{.*}}%[[X_LOAD]])

// Note: Due to an innocuous bug in LLVM-IR codegen, this line is different.
// LLVM-IR codegen emits this with 3 arguments, despite the 3rd not being used
// in the body, and not being included in the declaration/definition of this
// function. CIR cannot reproduce this, as we have a verifier that checks that
// the arg counts match.
// CIR: %[[BASE_ADDR:.*]] = cir.base_class_addr %[[THIS_LOAD]] : !cir.ptr<!rec_VirtualDelegatingCtor> nonnull [0] -> !cir.ptr<!rec_VirtDerived>
// CIR: %[[ADDR_PT:.*]] = cir.vtt.address_point @_ZTT21VirtualDelegatingCtor, offset = 1 -> !cir.ptr<!cir.ptr<!void>>
// CIR: cir.call @_ZN11VirtDerivedCI24BaseEi(%[[BASE_ADDR]], %[[ADDR_PT]]) : (!cir.ptr<!rec_VirtDerived>{{.*}}, !cir.ptr<!cir.ptr<!void>>{{.*}}) -> ()
// CIR: %[[ADDR_PT:.*]] = cir.vtable.address_point(@_ZTV21VirtualDelegatingCtor, address_point = <index = 0, offset = 3>) : !cir.vptr
// CIR: %[[VPTR:.*]] = cir.vtable.get_vptr %[[THIS_LOAD]] : !cir.ptr<!rec_VirtualDelegatingCtor> -> !cir.ptr<!cir.vptr>
// CIR: cir.store align(8) %[[ADDR_PT]], %[[VPTR]] : !cir.vptr, !cir.ptr<!cir.vptr>

// LLVM: call void @_ZN11VirtDerivedCI24BaseEi(ptr {{.*}}%[[THIS_LOAD]], ptr {{.*}}(i8, ptr @_ZTT21VirtualDelegatingCtor, i64 8))
// LLVM: store ptr getelementptr inbounds nuw (i8, ptr @_ZTV21VirtualDelegatingCtor, i64 24), ptr %[[THIS_LOAD]]
//

// CIR-LABEL: cir.func no_inline dso_local @_Z12fallsthroughv()
// CIR: cir.call @_ZN21VirtualDelegatingCtorC1Ei(%{{.*}}, %{{.*}}) : (!cir.ptr<!rec_VirtualDelegatingCtor> {{.*}}, !s32i {{.*}}) -> ()
// LLVM-LABEL: define dso_local void @_Z12fallsthroughv()
// LLVM: call void @_ZN21VirtualDelegatingCtorC1Ei(ptr {{.*}}, i32 {{.*}}1)


// OGCG-LABEL: define dso_local void @_Z20emitDelegateCallArgsv()
// OGCG: call void @_ZN7DerivedCI14BaseEi(ptr {{.*}}, i32 {{.*}}1)
//
// OGCG-LABEL: define linkonce_odr void @_ZN7DerivedCI14BaseEi(ptr {{.*}}, i32 {{.*}}) 
// OGCG: call void @_ZN7DerivedCI24BaseEi(ptr {{.*}}, i32 {{.*}})
//
// OGCG-LABEL: define dso_local void @_Z26cannotEmitDelegateCallArgsv()
// OGCG: %[[TMP_ALLOCA:.*]] = alloca ptr
// OGCG: %[[TMP_LOAD:.*]] = load ptr, ptr %[[TMP_ALLOCA]]
// OGCG: call void (ptr, float, ...) @_ZN4BaseC2Efz(ptr {{.*}}%[[TMP_LOAD]], float {{.*}}0x3FF19999A{{.*}}, i32 {{.*}}2, double {{.*}}3.000000e+00)
// 
// OGCG-LABEL: declare void @_ZN4BaseC2Efz(ptr {{.*}}, float {{.*}}, ...)
//
// OGCG-LABEL: define dso_local void @_Z12fallsthroughv()
// OGCG: call void @_ZN21VirtualDelegatingCtorC1Ei(ptr {{.*}}, i32 {{.*}}1)
//
// OGCG-LABEL: define linkonce_odr void @_ZN21VirtualDelegatingCtorC1Ei(ptr {{.*}}, i32 {{.*}})
// OGCG: %[[THIS_ALLOCA:.*]] = alloca ptr
// OGCG: %[[X_ALLOCA:.*]] = alloca i32
// OGCG: %[[THIS_LOAD:.*]] = load ptr, ptr %[[THIS_ALLOCA]]
// OGCG: %[[X_LOAD:.*]] = load i32, ptr %[[X_ALLOCA]]
// OGCG: call void @_ZN4BaseC2Ei(ptr {{.*}}%[[THIS_LOAD]], i32 {{.*}}%[[X_LOAD]])
// Note: see the note above for the CIR/LLVM-IR difference here.
// OGCG: %[[X_LOAD:.*]] = load i32, ptr %[[X_ALLOCA]]
// OGCG: call void @_ZN11VirtDerivedCI24BaseEi(ptr {{.*}}%[[THIS_LOAD]], ptr {{.*}}(i8, ptr @_ZTT21VirtualDelegatingCtor, i64 8), i32{{.*}}%[[X_LOAD]])
// OGCG: store ptr getelementptr inbounds inrange(-24, 0) ({ [3 x ptr] }, ptr @_ZTV21VirtualDelegatingCtor, i32 0, i32 0, i32 3), ptr %[[THIS_LOAD]]
//
// OGCG-LABEL: define linkonce_odr void @_ZN7DerivedCI24BaseEi(ptr {{.*}}, i32 {{.*}})
// OGCG: %[[THIS_ALLOCA:.*]] = alloca ptr
// OGCG: %[[INT_ALLOCA:.*]] = alloca i32
// OGCG: %[[BASE_ADDR:.*]] = load ptr, ptr %[[THIS_ALLOCA]]
// OGCG: %[[INT:.*]] = load i32, ptr %[[INT_ALLOCA]]
// OGCG: call void @_ZN4BaseC2Ei(ptr {{.*}}%[[BASE_ADDR]], i32 {{.*}}[[INT]])
//
// OGCG-LABEL: declare void @_ZN4BaseC2Ei(ptr {{.*}}, i32 {{.*}})

// OGCG-LABEL: define linkonce_odr void @_ZN11VirtDerivedCI24BaseEi(ptr {{.*}}, ptr {{.*}})
// OGCG: %[[THIS_ALLOCA:.*]] = alloca ptr
// OGCG: %[[VTT_ALLOCA:.*]] = alloca ptr
// OGCG: %[[THIS:.*]] = load ptr, ptr %[[THIS_ALLOCA]]
// OGCG: %[[VTT:.*]] = load ptr, ptr %[[VTT_ALLOCA]]
// OGCG: %[[VTT_ADDR_LOAD:.*]] = load ptr, ptr %[[VTT]]
// OGCG: store ptr %[[VTT_ADDR_LOAD]], ptr %[[THIS]]
