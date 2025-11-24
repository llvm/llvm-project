// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s -check-prefix=CIR --input-file=%t.cir
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck %s -check-prefix=LLVM --input-file=%t-cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck %s -check-prefix=OGCG --input-file=%t.ll

struct HasDtor {
  ~HasDtor();
};
struct S {
  static inline HasDtor hd;
};

// CIR: cir.global linkonce_odr comdat @_ZN1S2hdE = #cir.zero : !rec_HasDtor

// CIR: cir.func internal private @__cxx_global_var_init() {
// CIR:   %[[HD:.*]] = cir.get_global @_ZN1S2hdE : !cir.ptr<!rec_HasDtor>
// CIR:   %[[DTOR:.*]] = cir.get_global @_ZN7HasDtorD1Ev : !cir.ptr<!cir.func<(!cir.ptr<!rec_HasDtor>)>>
// CIR:   %[[DTOR_CAST:.*]] = cir.cast bitcast %[[DTOR]] : !cir.ptr<!cir.func<(!cir.ptr<!rec_HasDtor>)>> -> !cir.ptr<!cir.func<(!cir.ptr<!void>)>>
// CIR:   %[[HD_CAST:.*]] = cir.cast bitcast %[[HD]] : !cir.ptr<!rec_HasDtor> -> !cir.ptr<!void>
// CIR:   %[[HANDLE:.*]] = cir.get_global @__dso_handle : !cir.ptr<i8>
// CIR:   cir.call @__cxa_atexit(%[[DTOR_CAST]], %[[HD_CAST]], %[[HANDLE]])

// LLVM: @_ZN1S2hdE = linkonce_odr global %struct.HasDtor zeroinitializer, comdat
// LLVM: @_ZN5Outer5Inner2hdE = linkonce_odr global %struct.HasDtor zeroinitializer, comdat

// LLVM: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_static_members.cpp, ptr null }]
// LLVM: define internal void @__cxx_global_var_init()
// LLVM:   call void @__cxa_atexit(ptr @_ZN7HasDtorD1Ev, ptr @_ZN1S2hdE, ptr @__dso_handle)

// FIXME(cir): OGCG has a guard variable for this case that we don't generate in CIR.
//             This is needed because the variable linkonce_odr linkage.

// OGCG: @_ZN1S2hdE = linkonce_odr global %struct.HasDtor zeroinitializer, comdat
// OGCG: @_ZGVN1S2hdE = linkonce_odr global i64 0, comdat($_ZN1S2hdE)
// OGCG: @_ZN5Outer5Inner2hdE = linkonce_odr global %struct.HasDtor zeroinitializer, comdat
// OGCG: @_ZGVN5Outer5Inner2hdE = linkonce_odr global i64 0, comdat($_ZN5Outer5Inner2hdE)
// OGCG: @llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [
// OGCG-SAME:      { i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init, ptr @_ZN1S2hdE },
// OGCG-SAME:      { i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init.1, ptr @_ZN5Outer5Inner2hdE }]

// OGCG: define internal void @__cxx_global_var_init() {{.*}} section ".text.startup" comdat($_ZN1S2hdE) {
// OGCG:   %[[GUARD:.*]] = load atomic i8, ptr @_ZGVN1S2hdE acquire
// OGCG:   %[[UNINIT:.*]] = icmp eq i8 %[[GUARD]], 0
// OGCG:   br i1 %[[UNINIT]], label %[[INIT_CHECK:.*]], label %[[INIT_END:.*]]
// OGCG: [[INIT_CHECK:.*]]:
// OGCG:   %[[GUARD_ACQUIRE:.*]] = call i32 @__cxa_guard_acquire(ptr @_ZGVN1S2hdE)
// OGCG:   %[[TOBOOL:.*]] = icmp ne i32 %[[GUARD_ACQUIRE]], 0
// OGCG:   br i1 %[[TOBOOL]], label %[[INIT:.*]], label %[[INIT_END]]
// OGCG: [[INIT:.*]]:
// OGCG:   %[[ATEXIT:.*]] = call i32 @__cxa_atexit(ptr @_ZN7HasDtorD1Ev, ptr @_ZN1S2hdE, ptr @__dso_handle)
// OGCG:   call void @__cxa_guard_release(ptr @_ZGVN1S2hdE)
// OGCG:   br label %[[INIT_END]]
// OGCG: [[INIT_END]]:

struct Outer {
  struct Inner {
    static inline HasDtor hd;
  };
};

// CIR: cir.global linkonce_odr comdat @_ZN5Outer5Inner2hdE = #cir.zero : !rec_HasDtor
// CIR: cir.func internal private @__cxx_global_var_init.1()
// CIR:   %[[HD:.*]] = cir.get_global @_ZN5Outer5Inner2hdE : !cir.ptr<!rec_HasDtor>
// CIR:   %[[DTOR:.*]] = cir.get_global @_ZN7HasDtorD1Ev : !cir.ptr<!cir.func<(!cir.ptr<!rec_HasDtor>)>>
// CIR:   %[[DTOR_CAST:.*]] = cir.cast bitcast %[[DTOR]] : !cir.ptr<!cir.func<(!cir.ptr<!rec_HasDtor>)>> -> !cir.ptr<!cir.func<(!cir.ptr<!void>)>>
// CIR:   %[[HD_CAST:.*]] = cir.cast bitcast %[[HD]] : !cir.ptr<!rec_HasDtor> -> !cir.ptr<!void>
// CIR:   %[[HANDLE:.*]] = cir.get_global @__dso_handle : !cir.ptr<i8>
// CIR:   cir.call @__cxa_atexit(%[[DTOR_CAST]], %[[HD_CAST]], %[[HANDLE]]) : (!cir.ptr<!cir.func<(!cir.ptr<!void>)>>, !cir.ptr<!void>, !cir.ptr<i8>) -> ()

// LLVM: define internal void @__cxx_global_var_init.1()
// LLVM:   call void @__cxa_atexit(ptr @_ZN7HasDtorD1Ev, ptr @_ZN5Outer5Inner2hdE, ptr @__dso_handle)

// OGCG: define internal void @__cxx_global_var_init.1() {{.*}} section ".text.startup" comdat($_ZN5Outer5Inner2hdE) {
// OGCG:   %[[GUARD:.*]] = load atomic i8, ptr @_ZGVN5Outer5Inner2hdE acquire
// OGCG:   %[[UNINIT:.*]] = icmp eq i8 %[[GUARD]], 0
// OGCG:   br i1 %[[UNINIT]], label %[[INIT_CHECK:.*]], label %[[INIT_END:.*]]
// OGCG: [[INIT_CHECK:.*]]:
// OGCG:   %[[GUARD_ACQUIRE:.*]] = call i32 @__cxa_guard_acquire(ptr @_ZGVN5Outer5Inner2hdE)
// OGCG:   %[[TOBOOL:.*]] = icmp ne i32 %[[GUARD_ACQUIRE]], 0
// OGCG:   br i1 %[[TOBOOL]], label %[[INIT:.*]], label %[[INIT_END]]
// OGCG: [[INIT:.*]]:
// OGCG:   %[[ATEXIT:.*]] = call i32 @__cxa_atexit(ptr @_ZN7HasDtorD1Ev, ptr @_ZN5Outer5Inner2hdE, ptr @__dso_handle)
// OGCG:   call void @__cxa_guard_release(ptr @_ZGVN5Outer5Inner2hdE)
// OGCG:   br label %[[INIT_END]]
// OGCG: [[INIT_END]]:


// CIR: cir.func private @_GLOBAL__sub_I_static_members.cpp()
// CIR:   cir.call @__cxx_global_var_init()

// LLVM: define void @_GLOBAL__sub_I_static_members.cpp()
// LLVM:   call void @__cxx_global_var_init()
