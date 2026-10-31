// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t.ll %s

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

struct VBaseInheriting : VirtDerived {
  using VirtDerived::VirtDerived;
};

struct IndirectVBaseInheriting : VBaseInheriting {
  using VBaseInheriting::VBaseInheriting;
};

void inheritedFromVBase() {
  // VirtDerived's and VBaseInheriting's inheriting constructors take only
  // 'this' and the VTT when called to construct a base subobject.
  VBaseInheriting delegating{1};
  VBaseInheriting inlined{1.1f, 2, 3.0};
  IndirectVBaseInheriting indirect{1};
}


// CIR-LABEL: cir.func no_inline dso_local @_Z20emitDelegateCallArgsv()
// CIR: cir.call @_ZN7DerivedCI14BaseEi(%{{.*}}, %{{.*}}) : (!cir.ptr<!rec_Derived>{{.*}}, !s32i{{.*}}) -> ()
// LLVM-LABEL: define dso_local void @_Z20emitDelegateCallArgsv()
// LLVM: call void @_ZN7DerivedCI14BaseEi(ptr {{.*}}, i32 {{.*}}1)
//
// LLVM-LABEL: define linkonce_odr void @_ZN7DerivedCI14BaseEi(ptr {{.*}}, i32 {{.*}})
// LLVM: call void @_ZN7DerivedCI24BaseEi(ptr {{.*}}, i32 {{.*}})
//
// CIR-LABEL: cir.func no_inline dso_local @_Z26cannotEmitDelegateCallArgsv()
// CIR: %[[TMP_ALLOCA:.*]] = cir.alloca "tmp" {{.*}} init : !cir.ptr<!cir.ptr<!rec_Derived>>
// CIR: %[[FP_1_1:.*]] = cir.const #cir.fp<1.1{{.*}}> : !cir.float
// CIR: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
// CIR: %[[THREE:.*]] = cir.const #cir.fp<3.0{{.*}}> : !cir.double
// CIR: %[[LOAD_DERIVED:.*]] = cir.load %[[TMP_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_Derived>>, !cir.ptr<!rec_Derived>
// CIR: %[[BASE_ADDR:.*]] = cir.base_class_addr nonnull %[[LOAD_DERIVED]] [0] : !cir.ptr<!rec_Derived> -> !cir.ptr<!rec_Base>
// CIR: cir.call @_ZN4BaseC2Efz(%[[BASE_ADDR]], %[[FP_1_1]], %[[TWO]], %[[THREE]]) : (!cir.ptr<!rec_Base>{{.*}}, !cir.float{{.*}}, !s32i{{.*}}, !cir.double{{.*}}) -> ()
//
// LLVM-LABEL: define dso_local void @_Z26cannotEmitDelegateCallArgsv()
// LLVM: %[[TMP_ALLOCA:.*]] = alloca ptr
// LLVM: %[[TMP_LOAD:.*]] = load ptr, ptr %[[TMP_ALLOCA]]
// LLVM: call void (ptr, float, ...) @_ZN4BaseC2Efz(ptr {{.*}}%[[TMP_LOAD]], float {{.*}}1.100000e+00, i32 {{.*}}2, double {{.*}}3.000000e+00)
//
// CIR-LABEL: cir.func private @_ZN4BaseC2Efz(!cir.ptr<!rec_Base>{{.*}}, !cir.float{{.*}}, ...) func_info<#cir.cxx_ctor<!rec_Base, custom>>
// LLVM-LABEL: declare void @_ZN4BaseC2Efz(ptr {{.*}}, float {{.*}}, ...)
//
// CIR-LABEL: cir.func no_inline dso_local @_Z12fallsthroughv()
// CIR: cir.call @_ZN21VirtualDelegatingCtorC1Ei(%{{.*}}, %{{.*}}) : (!cir.ptr<!rec_VirtualDelegatingCtor> {{.*}}, !s32i {{.*}}) -> ()
// LLVM-LABEL: define dso_local void @_Z12fallsthroughv()
// LLVM: call void @_ZN21VirtualDelegatingCtorC1Ei(ptr {{.*}}, i32 {{.*}}1)
//
//
// CIR-LABEL: cir.func no_inline comdat alignment(2) linkonce_odr @_ZN21VirtualDelegatingCtorC1Ei(%{{.*}}: !cir.ptr<!rec_VirtualDelegatingCtor> {{.*}}, %{{.*}}: !s32i {{.*}}) func_info<#cir.cxx_ctor<!rec_VirtualDelegatingCtor, custom>>
// CIR: %[[THIS_ALLOCA:.*]] = cir.alloca "this" align(8) init : !cir.ptr<!cir.ptr<!rec_VirtualDelegatingCtor>>
// CIR: %[[X_ALLOCA:.*]] = cir.alloca "x" align(4) init : !cir.ptr<!s32i>
// CIR: %[[THIS_LOAD:.*]] = cir.load %[[THIS_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_VirtualDelegatingCtor>>, !cir.ptr<!rec_VirtualDelegatingCtor>
// CIR: %[[BASE_ADDR:.*]] = cir.base_class_addr nonnull %[[THIS_LOAD]] [0] : !cir.ptr<!rec_VirtualDelegatingCtor> -> !cir.ptr<!rec_Base>
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
// CIR: %[[BASE_ADDR:.*]] = cir.base_class_addr nonnull %[[THIS_LOAD]] [0] : !cir.ptr<!rec_VirtualDelegatingCtor> -> !cir.ptr<!rec_VirtDerived>
// CIR: %[[ADDR_PT:.*]] = cir.vtt.address_point @_ZTT21VirtualDelegatingCtor, offset = 1 -> !cir.ptr<!cir.ptr<!void>>
// CIR: cir.call @_ZN11VirtDerivedCI24BaseEi(%[[BASE_ADDR]], %[[ADDR_PT]]) : (!cir.ptr<!rec_VirtDerived>{{.*}}, !cir.ptr<!cir.ptr<!void>>{{.*}}) -> ()
// CIR: %[[ADDR_PT:.*]] = cir.vtable.address_point(@_ZTV21VirtualDelegatingCtor, address_point = <index = 0, offset = 3>) : !cir.vptr
// CIR: %[[VPTR:.*]] = cir.vtable.get_vptr %[[THIS_LOAD]] : !cir.ptr<!rec_VirtualDelegatingCtor> -> !cir.ptr<!cir.vptr>
// CIR: cir.store align(8) %[[ADDR_PT]], %[[VPTR]] : !cir.vptr, !cir.ptr<!cir.vptr>

// LLVMCIR: call void @_ZN11VirtDerivedCI24BaseEi(ptr {{.*}}%[[THIS_LOAD]], ptr {{.*}}(i8, ptr @_ZTT21VirtualDelegatingCtor, i64 8))
// LLVMCIR: store ptr getelementptr inbounds nuw (i8, ptr @_ZTV21VirtualDelegatingCtor, i64 24), ptr %[[THIS_LOAD]]
// OGCG: %[[X_LOAD:.*]] = load i32, ptr %[[X_ALLOCA]]
// OGCG: call void @_ZN11VirtDerivedCI24BaseEi(ptr {{.*}}%[[THIS_LOAD]], ptr {{.*}}(i8, ptr @_ZTT21VirtualDelegatingCtor, i64 8), i32{{.*}}%[[X_LOAD]])
// OGCG: store ptr getelementptr inbounds inrange(-24, 0) (i8, ptr @_ZTV21VirtualDelegatingCtor, i64 24), ptr %[[THIS_LOAD]]
//
// CIR-LABEL: cir.func no_inline dso_local @_Z18inheritedFromVBasev()
// CIR: %[[DELEGATING:.*]] = cir.alloca "delegating" align(8) init : !cir.ptr<!rec_VBaseInheriting>
// CIR: %[[INLINED:.*]] = cir.alloca "inlined" align(8) init : !cir.ptr<!rec_VBaseInheriting>
// CIR: %[[TMP_ALLOCA:.*]] = cir.alloca "tmp" align(8) init : !cir.ptr<!cir.ptr<!rec_VBaseInheriting>>
// CIR: %[[INDIRECT:.*]] = cir.alloca "indirect" align(8) init : !cir.ptr<!rec_IndirectVBaseInheriting>
// CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR: cir.call @_ZN15VBaseInheritingCI14BaseEi(%[[DELEGATING]], %[[ONE]]) : (!cir.ptr<!rec_VBaseInheriting> {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !s32i {llvm.noundef}) -> ()
// CIR: %[[FP_1_1:.*]] = cir.const #cir.fp<1.100000e+00> : !cir.float
// CIR: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
// CIR: %[[THREE:.*]] = cir.const #cir.fp<3.000000e+00> : !cir.double
// CIR: cir.store align(8) %[[INLINED]], %[[TMP_ALLOCA]] : !cir.ptr<!rec_VBaseInheriting>, !cir.ptr<!cir.ptr<!rec_VBaseInheriting>>
// CIR: %[[TMP_LOAD:.*]] = cir.load %[[TMP_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_VBaseInheriting>>, !cir.ptr<!rec_VBaseInheriting>
// CIR: %[[BASE_ADDR:.*]] = cir.base_class_addr nonnull %[[TMP_LOAD]] [0] : !cir.ptr<!rec_VBaseInheriting> -> !cir.ptr<!rec_Base>
// CIR: cir.call @_ZN4BaseC2Efz(%[[BASE_ADDR]], %[[FP_1_1]], %[[TWO]], %[[THREE]]) : (!cir.ptr<!rec_Base> {llvm.align = 1 : i64, llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, !cir.float {llvm.noundef}, !s32i {llvm.noundef}, !cir.double {llvm.noundef}) -> ()
// CIR: %[[BASE_ADDR:.*]] = cir.base_class_addr nonnull %[[TMP_LOAD]] [0] : !cir.ptr<!rec_VBaseInheriting> -> !cir.ptr<!rec_VirtDerived>
// CIR: %[[VTT_ADDR:.*]] = cir.vtt.address_point @_ZTT15VBaseInheriting, offset = 1 -> !cir.ptr<!cir.ptr<!void>>
// CIR: cir.call @_ZN11VirtDerivedCI24BaseEfz(%[[BASE_ADDR]], %[[VTT_ADDR]]) : (!cir.ptr<!rec_VirtDerived> {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !cir.ptr<!cir.ptr<!void>> {llvm.noundef}) -> ()
// CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR: cir.call @_ZN23IndirectVBaseInheritingCI14BaseEi(%[[INDIRECT]], %[[ONE]]) : (!cir.ptr<!rec_IndirectVBaseInheriting> {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !s32i {llvm.noundef}) -> ()
//
// CIR-LABEL: cir.func no_inline comdat alignment(2) linkonce_odr @_ZN15VBaseInheritingCI14BaseEi(%{{[a-z0-9]+}}: !cir.ptr<!rec_VBaseInheriting> {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef} loc({{[^)]+}}), %{{[a-z0-9]+}}: !s32i {llvm.noundef} loc({{[^)]+}})) func_info<#cir.cxx_ctor<!rec_VBaseInheriting, custom>>
// CIR: %[[THIS_ALLOCA:.*]] = cir.alloca "this" align(8) init : !cir.ptr<!cir.ptr<!rec_VBaseInheriting>>
// CIR: %[[INT_ALLOCA:.*]] = cir.alloca "" align(4) init : !cir.ptr<!s32i>
// CIR: %[[THIS_LOAD:.*]] = cir.load %[[THIS_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_VBaseInheriting>>, !cir.ptr<!rec_VBaseInheriting>
// CIR: %[[BASE_ADDR:.*]] = cir.base_class_addr nonnull %[[THIS_LOAD]] [0] : !cir.ptr<!rec_VBaseInheriting> -> !cir.ptr<!rec_Base>
// CIR: %[[INT:.*]] = cir.load align(4) %[[INT_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.call @_ZN4BaseC2Ei(%[[BASE_ADDR]], %[[INT]]) : (!cir.ptr<!rec_Base> {llvm.align = 1 : i64, llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, !s32i {llvm.noundef}) -> ()
// CIR: %[[BASE_ADDR:.*]] = cir.base_class_addr nonnull %[[THIS_LOAD]] [0] : !cir.ptr<!rec_VBaseInheriting> -> !cir.ptr<!rec_VirtDerived>
// CIR: %[[VTT_ADDR:.*]] = cir.vtt.address_point @_ZTT15VBaseInheriting, offset = 1 -> !cir.ptr<!cir.ptr<!void>>
// CIR: cir.call @_ZN11VirtDerivedCI24BaseEi(%[[BASE_ADDR]], %[[VTT_ADDR]]) : (!cir.ptr<!rec_VirtDerived> {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !cir.ptr<!cir.ptr<!void>> {llvm.noundef}) -> ()
//
// CIR-LABEL: cir.func no_inline comdat alignment(2) linkonce_odr @_ZN11VirtDerivedCI24BaseEfz(%{{[a-z0-9]+}}: !cir.ptr<!rec_VirtDerived> {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef} loc({{[^)]+}}), %{{[a-z0-9]+}}: !cir.ptr<!cir.ptr<!void>> {llvm.noundef} loc({{[^)]+}})) func_info<#cir.cxx_ctor<!rec_VirtDerived, custom>>
// CIR: %[[THIS_ALLOCA:.*]] = cir.alloca "this" align(8) init : !cir.ptr<!cir.ptr<!rec_VirtDerived>>
// CIR: %[[VTT_ALLOCA:.*]] = cir.alloca "vtt" align(8) init : !cir.ptr<!cir.ptr<!cir.ptr<!void>>>
// CIR: %[[THIS:.*]] = cir.load %[[THIS_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_VirtDerived>>, !cir.ptr<!rec_VirtDerived>
// CIR-NEXT: %[[VTT:.*]] = cir.load align(8) %[[VTT_ALLOCA]] : !cir.ptr<!cir.ptr<!cir.ptr<!void>>>, !cir.ptr<!cir.ptr<!void>>
// CIR-NEXT: %[[VTT_ADDR:.*]] = cir.vtt.address_point %[[VTT]] : !cir.ptr<!cir.ptr<!void>>, offset = 0 -> !cir.ptr<!cir.ptr<!void>>
// CIR-NEXT: %[[VTT_ADDR_CAST:.*]] = cir.cast bitcast %[[VTT_ADDR]] : !cir.ptr<!cir.ptr<!void>> -> !cir.ptr<!cir.vptr>
// CIR-NEXT: %[[VTT_ADDR_LOAD:.*]] = cir.load align(8) %[[VTT_ADDR_CAST]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR-NEXT: %[[VPTR:.*]] = cir.vtable.get_vptr %[[THIS]] : !cir.ptr<!rec_VirtDerived> -> !cir.ptr<!cir.vptr>
// CIR-NEXT: cir.store align(8) %[[VTT_ADDR_LOAD]], %[[VPTR]] : !cir.vptr, !cir.ptr<!cir.vptr>
// CIR-NEXT: cir.return
//
// CIR-LABEL: cir.func no_inline comdat alignment(2) linkonce_odr @_ZN23IndirectVBaseInheritingCI14BaseEi(%{{[a-z0-9]+}}: !cir.ptr<!rec_IndirectVBaseInheriting> {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef} loc({{[^)]+}}), %{{[a-z0-9]+}}: !s32i {llvm.noundef} loc({{[^)]+}})) func_info<#cir.cxx_ctor<!rec_IndirectVBaseInheriting, custom>>
// CIR: %[[THIS_ALLOCA:.*]] = cir.alloca "this" align(8) init : !cir.ptr<!cir.ptr<!rec_IndirectVBaseInheriting>>
// CIR: %[[INT_ALLOCA:.*]] = cir.alloca "" align(4) init : !cir.ptr<!s32i>
// CIR: %[[THIS_LOAD:.*]] = cir.load %[[THIS_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_IndirectVBaseInheriting>>, !cir.ptr<!rec_IndirectVBaseInheriting>
// CIR: %[[BASE_ADDR:.*]] = cir.base_class_addr nonnull %[[THIS_LOAD]] [0] : !cir.ptr<!rec_IndirectVBaseInheriting> -> !cir.ptr<!rec_Base>
// CIR: %[[INT:.*]] = cir.load align(4) %[[INT_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.call @_ZN4BaseC2Ei(%[[BASE_ADDR]], %[[INT]]) : (!cir.ptr<!rec_Base> {llvm.align = 1 : i64, llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, !s32i {llvm.noundef}) -> ()
// CIR: %[[BASE_ADDR:.*]] = cir.base_class_addr nonnull %[[THIS_LOAD]] [0] : !cir.ptr<!rec_IndirectVBaseInheriting> -> !cir.ptr<!rec_VBaseInheriting>
// CIR: %[[VTT_ADDR:.*]] = cir.vtt.address_point @_ZTT23IndirectVBaseInheriting, offset = 1 -> !cir.ptr<!cir.ptr<!void>>
// CIR: cir.call @_ZN15VBaseInheritingCI24BaseEi(%[[BASE_ADDR]], %[[VTT_ADDR]]) : (!cir.ptr<!rec_VBaseInheriting> {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !cir.ptr<!cir.ptr<!void>> {llvm.noundef}) -> ()
//
// LLVM-LABEL: define dso_local void @_Z18inheritedFromVBasev()
// LLVM: %[[DELEGATING:.*]] = alloca %struct.VBaseInheriting
// LLVM: %[[INLINED:.*]] = alloca %struct.VBaseInheriting
// LLVM: %[[TMP_ALLOCA:.*]] = alloca ptr
// LLVM: %[[INDIRECT:.*]] = alloca %struct.IndirectVBaseInheriting
// LLVM: call void @_ZN15VBaseInheritingCI14BaseEi(ptr noundef nonnull align 8 dereferenceable(8) %[[DELEGATING]], i32 noundef 1)
// LLVM: store ptr %[[INLINED]], ptr %[[TMP_ALLOCA]]
// LLVM: %[[TMP_LOAD:.*]] = load ptr, ptr %[[TMP_ALLOCA]]
// LLVM: call void (ptr, float, ...) @_ZN4BaseC2Efz(ptr noundef nonnull align 1 dereferenceable(1) %[[TMP_LOAD]], float noundef 1.100000e+00, i32 noundef 2, double noundef 3.000000e+00)
// LLVM: call void @_ZN11VirtDerivedCI24BaseEfz(ptr noundef nonnull align 8 dereferenceable(8) %[[TMP_LOAD]], ptr noundef getelementptr inbounds nuw (i8, ptr @_ZTT15VBaseInheriting, i64 8))
// LLVM: call void @_ZN23IndirectVBaseInheritingCI14BaseEi(ptr noundef nonnull align 8 dereferenceable(8) %[[INDIRECT]], i32 noundef 1)
//
// LLVM-LABEL: define linkonce_odr void @_ZN15VBaseInheritingCI14BaseEi(ptr noundef nonnull align 8 dereferenceable(8) %{{[a-z0-9]+}}, i32 noundef %{{[a-z0-9]+}})
// LLVM: %[[THIS_ALLOCA:.*]] = alloca ptr
// LLVM: %[[INT_ALLOCA:.*]] = alloca i32
// LLVM: %[[THIS_LOAD:.*]] = load ptr, ptr %[[THIS_ALLOCA]]
// LLVM: %[[INT:.*]] = load i32, ptr %[[INT_ALLOCA]]
// LLVM: call void @_ZN4BaseC2Ei(ptr noundef nonnull align 1 dereferenceable(1) %[[THIS_LOAD]], i32 noundef %[[INT]])
// LLVM: call void @_ZN11VirtDerivedCI24BaseEi(ptr noundef nonnull align 8 dereferenceable(8) %[[THIS_LOAD]], ptr noundef getelementptr inbounds nuw (i8, ptr @_ZTT15VBaseInheriting, i64 8))
//
// LLVM-LABEL: define linkonce_odr void @_ZN11VirtDerivedCI24BaseEfz(ptr noundef nonnull align 8 dereferenceable(8) %{{[a-z0-9]+}}, ptr noundef %{{[a-z0-9]+}})
// LLVM: %[[THIS_ALLOCA:.*]] = alloca ptr
// LLVM: %[[VTT_ALLOCA:.*]] = alloca ptr
// LLVM: %[[THIS:.*]] = load ptr, ptr %[[THIS_ALLOCA]]
// LLVM-NEXT: %[[VTT:.*]] = load ptr, ptr %[[VTT_ALLOCA]]
// LLVM-NEXT: %[[VTT_ADDR_LOAD:.*]] = load ptr, ptr %[[VTT]]
// LLVM-NEXT: store ptr %[[VTT_ADDR_LOAD]], ptr %[[THIS]]
// LLVM-NEXT: ret void
//
// LLVM-LABEL: define linkonce_odr void @_ZN23IndirectVBaseInheritingCI14BaseEi(ptr noundef nonnull align 8 dereferenceable(8) %{{[a-z0-9]+}}, i32 noundef %{{[a-z0-9]+}})
// LLVM: %[[THIS_ALLOCA:.*]] = alloca ptr
// LLVM: %[[INT_ALLOCA:.*]] = alloca i32
// LLVM: %[[THIS_LOAD:.*]] = load ptr, ptr %[[THIS_ALLOCA]]
// LLVM: %[[INT:.*]] = load i32, ptr %[[INT_ALLOCA]]
// LLVM: call void @_ZN4BaseC2Ei(ptr noundef nonnull align 1 dereferenceable(1) %[[THIS_LOAD]], i32 noundef %[[INT]])
// LLVM: call void @_ZN15VBaseInheritingCI24BaseEi(ptr noundef nonnull align 8 dereferenceable(8) %[[THIS_LOAD]], ptr noundef getelementptr inbounds nuw (i8, ptr @_ZTT23IndirectVBaseInheriting, i64 8))
//

// CIR-LABEL: cir.func no_inline comdat alignment(2) linkonce_odr @_ZN7DerivedCI24BaseEi(%{{.*}}: !cir.ptr<!rec_Derived>{{.*}}, %{{.*}}: !s32i{{.*}}) func_info<#cir.cxx_ctor<!rec_Derived, custom>>
// CIR: %[[THIS_ALLOCA:.*]] = cir.alloca "this" {{.*}} init : !cir.ptr<!cir.ptr<!rec_Derived>>
// CIR: %[[INT_ALLOCA:.*]] = cir.alloca "" {{.*}} init : !cir.ptr<!s32i>
// CIR: %[[THIS_LOAD:.*]] = cir.load %[[THIS_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_Derived>>, !cir.ptr<!rec_Derived>
// CIR: %[[BASE_ADDR:.*]] = cir.base_class_addr nonnull %[[THIS_LOAD]] [0] : !cir.ptr<!rec_Derived> -> !cir.ptr<!rec_Base>
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
// CIR-LABEL: cir.func private @_ZN4BaseC2Ei(!cir.ptr<!rec_Base>{{.*}}, !s32i{{.*}}) func_info<#cir.cxx_ctor<!rec_Base, custom>>
// LLVM-LABEL: declare void @_ZN4BaseC2Ei(ptr {{.*}}, i32 {{.*}})
//
//
// CIR-LABEL: cir.func no_inline comdat alignment(2) linkonce_odr @_ZN11VirtDerivedCI24BaseEi(%{{.*}}: !cir.ptr<!rec_VirtDerived> {{.*}}, %{{.*}}: !cir.ptr<!cir.ptr<!void>>{{.*}}) func_info<#cir.cxx_ctor<!rec_VirtDerived, custom>>
// CIR: %[[THIS_ALLOCA:.*]] = cir.alloca "this" align(8) init : !cir.ptr<!cir.ptr<!rec_VirtDerived>>
// CIR: %[[VTT_ALLOCA:.]] = cir.alloca "vtt" align(8) init : !cir.ptr<!cir.ptr<!cir.ptr<!void>>>
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

// CIR-LABEL: cir.func no_inline comdat alignment(2) linkonce_odr @_ZN15VBaseInheritingCI24BaseEi(%{{[a-z0-9]+}}: !cir.ptr<!rec_VBaseInheriting> {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef} loc({{[^)]+}}), %{{[a-z0-9]+}}: !cir.ptr<!cir.ptr<!void>> {llvm.noundef} loc({{[^)]+}})) func_info<#cir.cxx_ctor<!rec_VBaseInheriting, custom>>
// CIR: %[[THIS_ALLOCA:.*]] = cir.alloca "this" align(8) init : !cir.ptr<!cir.ptr<!rec_VBaseInheriting>>
// CIR: %[[VTT_ALLOCA:.*]] = cir.alloca "vtt" align(8) init : !cir.ptr<!cir.ptr<!cir.ptr<!void>>>
// CIR: %[[THIS:.*]] = cir.load %[[THIS_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_VBaseInheriting>>, !cir.ptr<!rec_VBaseInheriting>
// CIR-NEXT: %[[VTT:.*]] = cir.load align(8) %[[VTT_ALLOCA]] : !cir.ptr<!cir.ptr<!cir.ptr<!void>>>, !cir.ptr<!cir.ptr<!void>>
// CIR-NEXT: %[[BASE_ADDR:.*]] = cir.base_class_addr nonnull %[[THIS]] [0] : !cir.ptr<!rec_VBaseInheriting> -> !cir.ptr<!rec_VirtDerived>
// CIR-NEXT: %[[SUB_VTT:.*]] = cir.vtt.address_point %[[VTT]] : !cir.ptr<!cir.ptr<!void>>, offset = 1 -> !cir.ptr<!cir.ptr<!void>>
// CIR-NEXT: cir.call @_ZN11VirtDerivedCI24BaseEi(%[[BASE_ADDR]], %[[SUB_VTT]]) : (!cir.ptr<!rec_VirtDerived> {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !cir.ptr<!cir.ptr<!void>> {llvm.noundef}) -> ()

// LLVM-LABEL: define linkonce_odr void @_ZN15VBaseInheritingCI24BaseEi(ptr noundef nonnull align 8 dereferenceable(8) %{{[a-z0-9]+}}, ptr noundef %{{[a-z0-9]+}})
// LLVM: %[[THIS_ALLOCA:.*]] = alloca ptr
// LLVM: %[[THIS:.*]] = load ptr, ptr %[[THIS_ALLOCA]]
// LLVM: call void @_ZN11VirtDerivedCI24BaseEi(ptr noundef nonnull align 8 dereferenceable(8) %[[THIS]], ptr noundef %{{[a-z0-9]+}})
