// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s --check-prefix=CIR-BEFORE-LPP
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM-BOTH,LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM-BOTH,OGCG

int get_i();
struct CtorDtor {
  constexpr CtorDtor(int i) : i(i){}
  ~CtorDtor(){}
    int i;
};
// Wrappers & Aliases: 
// CIR-LABEL: cir.func comdat linkonce_odr private hidden @_ZTW12maybe_inited() -> !cir.ptr<!s32i> {
// CIR: %[[GET_INIT_FUNC:.*]] = cir.get_global @_ZTH12maybe_inited : !cir.ptr<!cir.func<()>>
// CIR: %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!cir.func<()>>
// CIR: %[[IS_VALID:.*]] = cir.cmp ne %[[GET_INIT_FUNC]], %[[NULL]] : !cir.ptr<!cir.func<()>>
// CIR: cir.if %[[IS_VALID]] {
// CIR: cir.call @_ZTH12maybe_inited() : () -> ()
// CIR: }
// CIR: %[[GET_GLOB:.*]] = cir.get_global thread_local @maybe_inited : !cir.ptr<!s32i>
// CIR: cir.return %[[GET_GLOB]] : !cir.ptr<!s32i>

// CIR-LABEL: cir.func comdat weak_odr private hidden @_ZTW21definitely_inited_dyn() -> !cir.ptr<!s32i> {
// CIR: cir.call @_ZTH21definitely_inited_dyn() : () -> ()
// CIR: %[[GET_GLOB:.*]] = cir.get_global thread_local @definitely_inited_dyn : !cir.ptr<!s32i>
// CIR: cir.return %[[GET_GLOB]] : !cir.ptr<!s32i>

// CIR: cir.func @_ZTH21definitely_inited_dyn() alias(@__tls_init)

// CIR: cir.func comdat weak_odr private hidden @_ZTW17definitely_inited() -> !cir.ptr<!s32i> {
// CIR:   %[[GET_GLOB:.*]] = cir.get_global thread_local @definitely_inited : !cir.ptr<!s32i>
// CIR:   cir.return %[[GET_GLOB]] : !cir.ptr<!s32i>

// CIR: cir.func @_ZTH17tls_int_self_init() alias(@__tls_init)

// CIR-LABEL: cir.func comdat weak_odr private hidden @_ZTW17tls_int_self_init() -> !cir.ptr<!s32i> {
// CIR: cir.call @_ZTH17tls_int_self_init() : () -> ()
// CIR: %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_int_self_init : !cir.ptr<!s32i>
// CIR: cir.return %[[GET_GLOB]] : !cir.ptr<!s32i>

// CIR-LABEL: cir.func comdat weak_odr private hidden @_ZTW11tls_int_ref() -> !cir.ptr<!cir.ptr<!s32i>> {
// CIR: cir.call @_ZTH11tls_int_ref() : () -> ()
// CIR: %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_int_ref : !cir.ptr<!cir.ptr<!s32i>>
// CIR: cir.return %0 : !cir.ptr<!cir.ptr<!s32i>>

// CIR: cir.func @_ZTH11tls_int_ref() alias(@__tls_init)

// CIR-LABEL: cir.func comdat weak_odr private hidden @_ZTW11tls_int_dyn() -> !cir.ptr<!s32i> {
// CIR: cir.call @_ZTH11tls_int_dyn() : () -> ()
// CIR: %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_int_dyn : !cir.ptr<!s32i>
// CIR: cir.return %[[GET_GLOB]] : !cir.ptr<!s32i>

// CIR: cir.func @_ZTH11tls_int_dyn() alias(@__tls_init)

// Full init of all variables (func names below).
// CIR-LABEL: cir.func internal private @__tls_init() {
// CIR: cir.return

// CIR-LABEL: cir.func comdat weak_odr private hidden @_ZTW7tls_int() -> !cir.ptr<!s32i> {
// CIR: %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_int : !cir.ptr<!s32i>
// CIR: cir.return %[[GET_GLOB]]

// LLVM-BOTH-DAG: @tls_int = thread_local global i32 5, align 4
// LLVM-BOTH-DAG: @tls_int_dyn = thread_local global i32 0, align 4
// LLVM-BOTH-DAG: @tls_int_ref = thread_local global ptr null, align 8
// LLVM-BOTH-DAG: @tls_int_self_init = thread_local global i32 0, align 4
// LLVM-BOTH-DAG: @definitely_inited = thread_local global i32 5, align 4
// LLVM-BOTH-DAG: @definitely_inited_dyn = thread_local global i32 0, align 4
// LLVM-BOTH-DAG: @maybe_inited = external thread_local global i32, align 4
//
// LLVM-BOTH-DAG: @_ZTH21definitely_inited_dyn = alias void (), ptr @__tls_init
// LLVM-BOTH-DAG: @_ZTH17tls_int_self_init = alias void (), ptr @__tls_init
// LLVM-BOTH-DAG: @_ZTH11tls_int_ref = alias void (), ptr @__tls_init
// LLVM-BOTH-DAG: @_ZTH11tls_int_dyn = alias void (), ptr @__tls_init

// Wrappers: 
// LLVM: define linkonce_odr hidden ptr @_ZTW12maybe_inited() {
// LLVM:   %[[HAS_INIT_FUNC:.*]] = icmp ne ptr @_ZTH12maybe_inited, null
// LLVM:   br i1 %[[HAS_INIT_FUNC]]
// LLVM:   call void @_ZTH12maybe_inited()
// LLVM:   %[[GET_GLOB:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @maybe_inited)
// LLVM:   ret ptr %[[GET_GLOB]]
//
// LLVM: define weak_odr hidden ptr @_ZTW21definitely_inited_dyn() {
// LLVM:   call void @_ZTH21definitely_inited_dyn()
// LLVM:   %[[GET_GLOB:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @definitely_inited_dyn)
// LLVM:   ret ptr %[[GET_GLOB]]
//
// LLVM: define weak_odr hidden ptr @_ZTW17definitely_inited() {
// LLVM:   %[[GET_GLOB:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @definitely_inited)
// LLVM:   ret ptr %[[GET_GLOB]]
// LLVM: }
//
// LLVM: define weak_odr hidden ptr @_ZTW17tls_int_self_init() {
// LLVM:   call void @_ZTH17tls_int_self_init()
// LLVM:   %[[GET_GLOB:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @tls_int_self_init)
// LLVM:   ret ptr %[[GET_GLOB]]
//
// LLVM: define weak_odr hidden ptr @_ZTW11tls_int_ref() {
// LLVM:   call void @_ZTH11tls_int_ref()
// LLVM:   %[[GET_GLOB:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @tls_int_ref)
// LLVM:   ret ptr %[[GET_GLOB]]
//
// LLVM: define weak_odr hidden ptr @_ZTW11tls_int_dyn() {
// LLVM:   call void @_ZTH11tls_int_dyn()
// LLVM:   %[[GET_GLOB:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @tls_int_dyn)
// LLVM:   ret ptr %[[GET_GLOB]]

// LLVM: define internal void @__tls_init() {
// LLVM:   ret void

// LLVM: define weak_odr hidden ptr @_ZTW7tls_int() {
// LLVM:   %[[GET_GLOB:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @tls_int)
// LLVM:   ret ptr %[[GET_GLOB]]
// LLVM: }


thread_local int tls_int = 5;
// CIR-BEFORE-LPP: cir.global external tls_dyn dyn_tls_refs = <"_ZTW7tls_int", "_ZTH7tls_int"> @tls_int = #cir.int<5> : !s32i
// CIR: cir.global external tls_dyn dyn_tls_refs = <"_ZTW7tls_int", "_ZTH7tls_int"> @tls_int = #cir.int<5> : !s32i

thread_local int tls_int_dyn = get_i();
// CIR-BEFORE-LPP: cir.global external tls_dyn dyn_tls_refs = <"_ZTW11tls_int_dyn", "_ZTH11tls_int_dyn"> @tls_int_dyn = ctor : !s32i {
// CIR-BEFORE-LPP:   %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_int_dyn : !cir.ptr<!s32i>
// CIR-BEFORE-LPP:   %[[CALL:.*]] = cir.call @_Z5get_iv() : () -> (!s32i {llvm.noundef})
// CIR-BEFORE-LPP:   cir.store {{.*}}%[[CALL]], %[[GET_GLOB]] : !s32i, !cir.ptr<!s32i>
// CIR-BEFORE-LPP: }
// CIR: cir.global external tls_dyn dyn_tls_refs = <"_ZTW11tls_int_dyn", "_ZTH11tls_int_dyn"> @tls_int_dyn = #cir.int<0> : !s32i 
// CIR: cir.func internal private @[[TLS_INT_DYN_INIT:.*]]() {
// CIR:   %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_int_dyn : !cir.ptr<!s32i>
// CIR:   %[[CALL:.*]] = cir.call @_Z5get_iv() : () -> (!s32i {llvm.noundef})
// CIR:   cir.store {{.*}}%[[CALL]], %[[GET_GLOB]] : !s32i, !cir.ptr<!s32i>
// CIR:   cir.return
// LLVM: define internal void @[[TLS_INT_DYN_INIT:.*]]() {
// OGCG: define internal void @[[TLS_INT_DYN_INIT:.*]]()
// OGCG:   %[[CALL:.*]] = call noundef i32 @_Z5get_iv()
// LLVM-BOTH:   %[[GET_GLOB:.*]] = call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@tls_int_dyn)
// LLVM:   %[[CALL:.*]] = call noundef i32 @_Z5get_iv()
// LLVM-BOTH:   store i32 %[[CALL]], ptr %[[GET_GLOB]], align 4
// LLVM-BOTH:   ret void

thread_local int &tls_int_ref = tls_int_dyn;
// CIR-BEFORE-LPP: cir.global external tls_dyn dyn_tls_refs = <"_ZTW11tls_int_ref", "_ZTH11tls_int_ref"> @tls_int_ref = ctor : !cir.ptr<!s32i> {
// CIR-BEFORE-LPP:   %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_int_ref : !cir.ptr<!cir.ptr<!s32i>>
// CIR-BEFORE-LPP:   %[[GET_OTHER:.*]] = cir.get_global thread_local @tls_int_dyn : !cir.ptr<!s32i>
// CIR-BEFORE-LPP:   cir.store {{.*}}%[[GET_OTHER]], %[[GET_GLOB]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR-BEFORE-LPP: }
// CIR: cir.global external tls_dyn dyn_tls_refs = <"_ZTW11tls_int_ref", "_ZTH11tls_int_ref"> @tls_int_ref = #cir.ptr<null> : !cir.ptr<!s32i>
// CIR: cir.func internal private @[[TLS_INT_REF_INIT:.*]]() {
// CIR:   %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_int_ref : !cir.ptr<!cir.ptr<!s32i>>
// CIR:   %[[GET_REF:.*]] = cir.call @_ZTW11tls_int_dyn() : () -> !cir.ptr<!s32i>
// CIR:   cir.store {{.*}}%[[GET_REF]], %[[GET_GLOB]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR:   cir.return
// LLVM: define internal void @[[TLS_INT_REF_INIT:.*]]() {
// OGCG: define internal void @[[TLS_INT_REF_INIT:.*]]()
// OGCG:   %[[GET_REF:.*]] = call ptr @_ZTW11tls_int_dyn()
// LLVM-BOTH:   %[[GET_GLOB:.*]] = call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@tls_int_ref)
// LLVM:   %[[GET_REF:.*]] = call ptr @_ZTW11tls_int_dyn()
// LLVM-BOTH:   store ptr %[[GET_REF]], ptr %[[GET_GLOB]], align 8
// LLVM-BOTH:   ret void

// OGCG: define weak_odr hidden noundef ptr @_ZTW11tls_int_dyn() {{.*}} comdat {
// OGCG:   call void @_ZTH11tls_int_dyn()
// OGCG:   %[[GET_GLOB:.*]] = call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@tls_int_dyn)
// OGCG:   ret ptr %[[GET_GLOB]]

thread_local int tls_int_self_init = tls_int_self_init + get_i();
// CIR-BEFORE-LPP:  cir.global external tls_dyn dyn_tls_refs = <"_ZTW17tls_int_self_init", "_ZTH17tls_int_self_init"> @tls_int_self_init = ctor : !s32i {
// CIR-BEFORE-LPP:    %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_int_self_init : !cir.ptr<!s32i>
// CIR-BEFORE-LPP:    %[[GET_SELF:.*]] = cir.get_global thread_local @tls_int_self_init : !cir.ptr<!s32i>
// CIR-BEFORE-LPP:    %[[LOAD_SELF:.*]] = cir.load {{.*}}%[[GET_SELF]] : !cir.ptr<!s32i>, !s32i
// CIR-BEFORE-LPP:    %[[CALL:.*]] = cir.call @_Z5get_iv() : () -> (!s32i {llvm.noundef})
// CIR-BEFORE-LPP:    %[[ADD:.*]] = cir.add nsw %[[LOAD_SELF]], %[[CALL]] : !s32i
// CIR-BEFORE-LPP:    cir.store {{.*}}%[[ADD]], %[[GET_GLOB]] : !s32i, !cir.ptr<!s32i>
// CIR-BEFORE-LPP:  }
// CIR: cir.global external tls_dyn dyn_tls_refs = <"_ZTW17tls_int_self_init", "_ZTH17tls_int_self_init"> @tls_int_self_init = #cir.int<0> : !s32i
// CIR: cir.func internal private @[[TLS_INT_SELF_REF_INIT:.*]]() {
// CIR:   %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_int_self_init : !cir.ptr<!s32i>
// CIR:   %[[GET_SELF_FROM_WRAPPER:.*]] = cir.call @_ZTW17tls_int_self_init() : () -> !cir.ptr<!s32i>
// CIR:   %[[SELF_LOAD:.*]] = cir.load {{.*}}%[[GET_SELF_FROM_WRAPPER]] : !cir.ptr<!s32i>, !s32i
// CIR:   %[[CALL:.*]] = cir.call @_Z5get_iv() : () -> (!s32i {llvm.noundef})
// CIR:   %[[ADD:.*]] = cir.add nsw %[[SELF_LOAD]], %[[CALL]] : !s32i
// CIR:   cir.store{{.*}} %[[ADD]], %[[GET_GLOB]] : !s32i, !cir.ptr<!s32i>
// CIR:   cir.return
// LLVM: define internal void @[[TLS_INT_SELF_REF_INIT:.*]]() {
// OGCG: define internal void @[[TLS_INT_SELF_REF_INIT:.*]]()
// LLVM:   %[[GET_GLOB:.*]] = call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@tls_int_self_init)
// LLVM-BOTH:   %[[GET_SELF_FROM_WRAPPER:.*]] = call ptr @_ZTW17tls_int_self_init()
// LLVM-BOTH:   %[[SELF_LOAD:.*]] = load i32, ptr %[[GET_SELF_FROM_WRAPPER]], align 4
// LLVM-BOTH:   %[[CALL:.*]] = call noundef i32 @_Z5get_iv()
// LLVM-BOTH:   %[[ADD:.*]] = add nsw i32 %[[SELF_LOAD]], %[[CALL]]
// OGCG:   %[[GET_GLOB:.*]] = call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@tls_int_self_init)
// LLVM-BOTH:   store i32 %[[ADD]], ptr %[[GET_GLOB]], align 4
// LLVM-BOTH:   ret void
//
// OGCG: define weak_odr hidden noundef ptr @_ZTW17tls_int_self_init() {{.*}} comdat {
// OGCG:   call void @_ZTH17tls_int_self_init()
// OGCG:   %[[GET_GLOB:.*]] = call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@tls_int_self_init)
// OGCG:   ret ptr %[[GET_GLOB]]

extern thread_local int definitely_inited = 5;
// CIR-BEFORE-LPP: cir.global external tls_dyn dyn_tls_refs = <"_ZTW17definitely_inited", "_ZTH17definitely_inited"> @definitely_inited = #cir.int<5> : !s32i
// CIR: cir.global external tls_dyn dyn_tls_refs = <"_ZTW17definitely_inited", "_ZTH17definitely_inited"> @definitely_inited = #cir.int<5> : !s32i

extern thread_local int definitely_inited_dyn = get_i();
// CIR-BEFORE-LPP: cir.global external tls_dyn dyn_tls_refs = <"_ZTW21definitely_inited_dyn", "_ZTH21definitely_inited_dyn"> @definitely_inited_dyn = ctor : !s32i {
// CIR-BEFORE-LPP:   %[[GET_GLOB:.*]] = cir.get_global thread_local @definitely_inited_dyn : !cir.ptr<!s32i>
// CIR-BEFORE-LPP:   %[[CALL:.*]] = cir.call @_Z5get_iv() : () -> (!s32i {llvm.noundef})
// CIR-BEFORE-LPP:   cir.store {{.*}}%[[CALL]], %[[GET_GLOB]] : !s32i, !cir.ptr<!s32i>
// CIR-BEFORE-LPP: }
// CIR: cir.global external tls_dyn dyn_tls_refs = <"_ZTW21definitely_inited_dyn", "_ZTH21definitely_inited_dyn"> @definitely_inited_dyn = #cir.int<0> : !s32i
// CIR: cir.func internal private @[[DEF_INITED_DYN:.*]]() {
// CIR:   %[[GET_GLOB:.*]] = cir.get_global thread_local @definitely_inited_dyn : !cir.ptr<!s32i>
// CIR:   %[[CALL:.*]] = cir.call @_Z5get_iv() : () -> (!s32i {llvm.noundef})
// CIR:   cir.store align(4) %[[CALL]], %[[GET_GLOB]] : !s32i, !cir.ptr<!s32i>
// CIR:   cir.return
// LLVM: define internal void @[[DEF_INITED_DYN:.*]]() {
// OGCG: define internal void @[[DEF_INITED_DYN:.*]]()
// LLVM:   %[[GET_GLOB:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @definitely_inited_dyn)
// LLVM-BOTH:   %[[CALL:.*]] = call noundef i32 @_Z5get_iv()
// OGCG:   %[[GET_GLOB:.*]] = call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@definitely_inited_dyn)
// LLVM-BOTH:   store i32 %[[CALL]], ptr %[[GET_GLOB]], align 4
// LLVM-BOTH:   ret void

extern thread_local int maybe_inited;
// CIR-BEFORE-LPP: cir.global "private" external tls_dyn dyn_tls_refs = <"_ZTW12maybe_inited", "_ZTH12maybe_inited"> @maybe_inited : !s32i
// CIR: cir.global "private" external tls_dyn dyn_tls_refs = <"_ZTW12maybe_inited", "_ZTH12maybe_inited"> @maybe_inited : !s32i

void uses() {
  auto a = tls_int;
// CIR-BEFORE-LPP: cir.get_global thread_local @tls_int : !cir.ptr<!s32i>
// CIR: cir.call @_ZTW7tls_int() : () -> !cir.ptr<!s32i>
// Note: CIR is currently ALWAYS using the wrapper here even though it doesn't
// need to, however this is a 'no-op' anyway, so we'd expect this to be
// optimized away.
// LLVM: call ptr @_ZTW7tls_int()
// OGCG: call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@tls_int)
  auto b = tls_int_dyn;
// CIR-BEFORE-LPP: cir.get_global thread_local @tls_int_dyn : !cir.ptr<!s32i>
// CIR: cir.call @_ZTW11tls_int_dyn() : () -> !cir.ptr<!s32i>
// LLVM-BOTH: call ptr @_ZTW11tls_int_dyn()
  auto c = tls_int_ref;
// CIR-BEFORE-LPP: cir.get_global thread_local @tls_int_ref : !cir.ptr<!cir.ptr<!s32i>>
// CIR: cir.call @_ZTW11tls_int_ref() : () -> !cir.ptr<!cir.ptr<!s32i>>
// LLVM-BOTH: call ptr @_ZTW11tls_int_ref()
  auto d = tls_int_self_init;
// CIR-BEFORE-LPP: cir.get_global thread_local @tls_int_self_init : !cir.ptr<!s32i>
// CIR: cir.call @_ZTW17tls_int_self_init() : () -> !cir.ptr<!s32i>
// LLVM-BOTH: call ptr @_ZTW17tls_int_self_init()
  auto e = maybe_inited;
// CIR-BEFORE-LPP: cir.get_global thread_local @maybe_inited : !cir.ptr<!s32i>
// CIR: cir.call @_ZTW12maybe_inited() : () -> !cir.ptr<!s32i>
// LLVM-BOTH: call ptr @_ZTW12maybe_inited()
  auto f = definitely_inited;
// CIR-BEFORE-LPP: cir.get_global thread_local @definitely_inited : !cir.ptr<!s32i>
// CIR: cir.call @_ZTW17definitely_inited() : () -> !cir.ptr<!s32i>
// Note: CIR is currently ALWAYS using the wrapper here even though it doesn't
// need to, however this is a 'no-op' anyway, so we'd expect this to be
// optimized away.
// LLVM: call ptr @_ZTW17definitely_inited()
// OGCG: call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@definitely_inited)
  auto g = definitely_inited_dyn;
// CIR-BEFORE-LPP: cir.get_global thread_local @definitely_inited_dyn : !cir.ptr<!s32i>
// CIR: cir.call @_ZTW21definitely_inited_dyn() : () -> !cir.ptr<!s32i>
// LLVM-BOTH: call ptr @_ZTW21definitely_inited_dyn()
}

// OGCG Wrappers: For some reason this puts them at the end, otherwise they are
// basically identical (return val has a noundef?). Note some are above because
// they are referenced up there.
// Also: these have 'comdat' but the above LLVM versions don't, because we
// haven't yet lowered comdat on functions.
// OGCG: define weak_odr hidden noundef ptr @_ZTW11tls_int_ref() {{.*}} comdat {
// OGCG:   call void @_ZTH11tls_int_ref()
// OGCG:   %[[GET_GLOB:.*]] = call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@tls_int_ref)
// OGCG:   %[[GET_PTR:.*]] = load ptr, ptr %[[GET_GLOB]]
// OGCG:   ret ptr %[[GET_PTR]]
//
// OGCG: define linkonce_odr hidden noundef ptr @_ZTW12maybe_inited() {{.*}} comdat {
// OGCG:   %[[HAS_INIT_FUNC:.*]] = icmp ne ptr @_ZTH12maybe_inited, null
// OGCG:   br i1 %[[HAS_INIT_FUNC]]
// OGCG:   call void @_ZTH12maybe_inited()
// OGCG:   %[[GET_GLOB:.*]] = call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@maybe_inited)
// OGCG:   ret ptr %[[GET_GLOB]]
//
// OGCG: define weak_odr hidden noundef ptr @_ZTW21definitely_inited_dyn() {{.*}} comdat {
// OGCG:   call void @_ZTH21definitely_inited_dyn()
// OGCG:   %[[GET_GLOB:.*]] = call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@definitely_inited_dyn)
// OGCG:   ret ptr %[[GET_GLOB]]
//
// The init function here happens in the middle for some reason?  
// OGCG: define internal void @__tls_init()
// OGCG:   %[[GET_GUARD:.*]] = load i8, ptr @__tls_guard, align 1
// OGCG:   %[[IS_UNINIT:.*]] = icmp eq i8 %[[GET_GUARD]], 0
// OGCG:   br i1 %[[IS_UNINIT]]
//
// OGCG:   store i8 1, ptr @__tls_guard, align 1
// OGCG:   call void @[[TLS_INT_DYN_INIT]]()
// OGCG:   call void @[[TLS_INT_REF_INIT]]()
// OGCG:   call void @[[TLS_INT_SELF_REF_INIT]]()
// OGCG:   call void @[[DEF_INITED_DYN]]()
// OGCG:   br label 
// OGCG:   ret void
//
// OGCG: define weak_odr hidden noundef ptr @_ZTW7tls_int() {{.*}} comdat {
// OGCG:   %[[GET_GLOB:.*]] = call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@tls_int)
// OGCG:   ret ptr %[[GET_GLOB]]
// OGCG: }
//
// OGCG: define weak_odr hidden noundef ptr @_ZTW17definitely_inited() {{.*}} comdat {
// OGCG:   %[[GET_GLOB:.*]] = call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@definitely_inited)
// OGCG:   ret ptr %[[GET_GLOB]]
// OGCG: }
