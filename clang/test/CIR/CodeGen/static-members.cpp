// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s -check-prefix=CIR --input-file=%t.cir
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck %s -check-prefix=LLVM --input-file=%t-cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck %s -check-prefix=LLVM --input-file=%t.ll

struct HasDtor {
  ~HasDtor();
};
struct S {
  static inline HasDtor hd;
};

// CIR: module @
// CIR-SAME: cir.global_ctors = [#cir.global_ctor<"__cxx_global_var_init", 65535, @_ZN1S2hdE>, #cir.global_ctor<"__cxx_global_var_init.1", 65535, @_ZN5Outer5Inner2hdE>, #cir.global_ctor<"__cxx_global_var_init.2", 65535, @_ZN9RefMember3refE>, #cir.global_ctor<"__cxx_global_var_init.3", 65535, @_ZN13NonThreadSafeIiE1fE>]

// Guard variables.
// CIR-DAG: cir.global "private" linkonce_odr comdat("_ZN1S2hdE") @_ZGVN1S2hdE = #cir.int<0> : !s64i
// LLVM-DAG: @_ZGVN1S2hdE = linkonce_odr global i64 0, comdat($_ZN1S2hdE), align 8
// CIR-DAG: cir.global "private" linkonce_odr comdat("_ZN5Outer5Inner2hdE") @_ZGVN5Outer5Inner2hdE = #cir.int<0> : !s64i
// LLVM-DAG: @_ZGVN5Outer5Inner2hdE = linkonce_odr global i64 0, comdat($_ZN5Outer5Inner2hdE), align 8
// CIR-DAG: cir.global "private" linkonce_odr comdat("_ZN9RefMember3refE") @_ZGVN9RefMember3refE = #cir.int<0> : !s64i
// LLVM-DAG: @_ZGVN9RefMember3refE = linkonce_odr global i64 0, comdat($_ZN9RefMember3refE), align 8
// CIR-DAG: cir.global "private" linkonce_odr comdat("_ZN13NonThreadSafeIiE1fE") @_ZGVN13NonThreadSafeIiE1fE = #cir.int<0> : !s64i
// LLVM-DAG: @_ZGVN13NonThreadSafeIiE1fE = linkonce_odr global i64 0, comdat($_ZN13NonThreadSafeIiE1fE), align 8

// LLVM-DAG: @_ZN1S2hdE = linkonce_odr global %struct.HasDtor zeroinitializer, comdat, align 1
// LLVM-DAG: @_ZN5Outer5Inner2hdE = linkonce_odr global %struct.HasDtor zeroinitializer, comdat, align 1
// LLVM-DAG: @_ZN9RefMember3refE = linkonce_odr global ptr null, comdat, align 8
// LLVM-DAG: @_ZN13NonThreadSafeIiE1fE = linkonce_odr global i32 0, comdat, align 4

// The COMDAT keys must be kept in `llvm.used` so the linker doesn't
// garbage-collect them (and, with them, their `llvm.global_ctors` entries).
// LLVM-DAG: @llvm.used = appending global [4 x ptr] [ptr @_ZN1S2hdE, ptr @_ZN5Outer5Inner2hdE, ptr @_ZN9RefMember3refE, ptr @_ZN13NonThreadSafeIiE1fE], section "llvm.metadata"
// LLVM-DAG: @llvm.global_ctors = appending global [4 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init, ptr @_ZN1S2hdE }, { i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init.1, ptr @_ZN5Outer5Inner2hdE }, { i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init.2, ptr @_ZN9RefMember3refE }, { i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init.3, ptr @_ZN13NonThreadSafeIiE1fE }]


// CIR: cir.global linkonce_odr comdat dynamic_init_guard<"_ZGVN1S2hdE"> @_ZN1S2hdE = #cir.zero : !rec_HasDtor align(1) ast(#cir.var.decl.ast) dynamic_init_info<local = false, tls = none, is_inline = true, tsk = undeclared>
// CIR-LABEL: cir.func internal private @__cxx_global_var_init() {
// CIR: %[[GET_GUARD:.*]] = cir.get_global @_ZGVN1S2hdE : !cir.ptr<!s64i>
// CIR: %[[TO_CHAR:.*]] = cir.cast bitcast %[[GET_GUARD]] : !cir.ptr<!s64i> -> !cir.ptr<!s8i>
// CIR: %[[LOAD_GUARD:.*]] = cir.load align(8) syncscope(system) atomic(acquire) %[[TO_CHAR]] : !cir.ptr<!s8i>, !s8i
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s8i
// CIR: %[[CMP:.*]] = cir.cmp eq %[[LOAD_GUARD]], %[[ZERO]] : !s8i
// CIR: cir.if %[[CMP]] {
// CIR:   %[[ACQUIRE_GUARD:.*]] = cir.call @__cxa_guard_acquire(%[[GET_GUARD]]) : (!cir.ptr<!s64i>) -> !s32i
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   %[[CMP:.*]] = cir.cmp ne %[[ACQUIRE_GUARD]], %[[ZERO]] : !s32i
// CIR:   cir.if %[[CMP]] {
// CIR:     %[[GET_HD:.*]] = cir.get_global @_ZN1S2hdE : !cir.ptr<!rec_HasDtor>
// CIR:     %[[GET_DTOR:.*]] = cir.get_global @_ZN7HasDtorD1Ev : !cir.ptr<!cir.func<(!cir.ptr<!rec_HasDtor>)>>
// CIR:     %[[CAST_DTOR:.*]] = cir.cast bitcast %[[GET_DTOR]] : !cir.ptr<!cir.func<(!cir.ptr<!rec_HasDtor>)>> -> !cir.ptr<!cir.func<(!cir.ptr<!void>)>>
// CIR:     %[[CAST_HD:.*]] = cir.cast bitcast %[[GET_HD]] : !cir.ptr<!rec_HasDtor> -> !cir.ptr<!void>
// CIR:     %[[DSO_HANDLE:.*]] = cir.get_global @__dso_handle : !cir.ptr<!u8i>
// CIR:     %[[AT_EXIT:.*]] = cir.call @__cxa_atexit(%[[CAST_DTOR]], %[[CAST_HD]], %[[DSO_HANDLE]]) : (!cir.ptr<!cir.func<(!cir.ptr<!void>)>>, !cir.ptr<!void>, !cir.ptr<!u8i>) -> !s32i
// CIR:     cir.call @__cxa_guard_release(%[[GET_GUARD]]) : (!cir.ptr<!s64i>) -> ()
// CIR:   }
// CIR: }
// CIR: cir.return

// LLVM-LABEL: define internal void @__cxx_global_var_init()
// LLVM: %[[LOAD_GUARD:.*]] = load atomic i8, ptr @_ZGVN1S2hdE acquire, align 8
// LLVM: %[[CMP:.*]] = icmp eq i8 %[[LOAD_GUARD]], 0
// LLVM: br i1 %[[CMP]], label %[[UNINIT:.*]], label %[[RET:.*]]

// LLVM: [[UNINIT]]:
// LLVM:   %[[ACQUIRE_GUARD:.*]] = call i32 @__cxa_guard_acquire(ptr @_ZGVN1S2hdE)
// LLVM:   %[[CMP:.*]] = icmp ne i32 %[[ACQUIRE_GUARD]], 0
// CIR leaves an extra 'block' for this target, but both go to 'ret' via only
// empty blocks.
// LLVM:   br i1 %[[CMP]], label %[[DO_INIT:.*]], label %{{.*}}

// LLVM: [[DO_INIT]]:
// LLVM:   %[[AT_EXIT:.*]] = call i32 @__cxa_atexit(ptr @_ZN7HasDtorD1Ev, ptr @_ZN1S2hdE, ptr @__dso_handle)
// LLVM:   call void @__cxa_guard_release(ptr @_ZGVN1S2hdE)
// CIR leaves an extra 'block' for this target, but both go to 'ret' via only
// empty blocks.
// LLVM:   br label %{{.*}}

// LLVM: [[RET]]:
// LLVM:   ret void

struct Outer {
  struct Inner {
    static inline HasDtor hd;
  };
};
// CIR: cir.global linkonce_odr comdat dynamic_init_guard<"_ZGVN5Outer5Inner2hdE"> @_ZN5Outer5Inner2hdE = #cir.zero : !rec_HasDtor align(1) ast(#cir.var.decl.ast) dynamic_init_info<local = false, tls = none, is_inline = true, tsk = undeclared>
// CIR-LABEL: cir.func internal private @__cxx_global_var_init.1() {
// CIR: %[[GET_GUARD:.*]] = cir.get_global @_ZGVN5Outer5Inner2hdE : !cir.ptr<!s64i>
// CIR: %[[TO_CHAR:.*]] = cir.cast bitcast %[[GET_GUARD]] : !cir.ptr<!s64i> -> !cir.ptr<!s8i>
// CIR: %[[LOAD_GUARD:.*]] = cir.load align(8) syncscope(system) atomic(acquire) %[[TO_CHAR]] : !cir.ptr<!s8i>, !s8i
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s8i
// CIR: %[[CMP:.*]] = cir.cmp eq %[[LOAD_GUARD]], %[[ZERO]] : !s8i
// CIR: cir.if %[[CMP]] {
// CIR:   %[[ACQUIRE_GUARD:.*]] = cir.call @__cxa_guard_acquire(%[[GET_GUARD]]) : (!cir.ptr<!s64i>) -> !s32i
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   %[[CMP:.*]] = cir.cmp ne %[[ACQUIRE_GUARD]], %[[ZERO]] : !s32i
// CIR:   cir.if %[[CMP]] {
// CIR:     %[[GET_HD:.*]] = cir.get_global @_ZN5Outer5Inner2hdE : !cir.ptr<!rec_HasDtor>
// CIR:     %[[GET_DTOR:.*]] = cir.get_global @_ZN7HasDtorD1Ev : !cir.ptr<!cir.func<(!cir.ptr<!rec_HasDtor>)>>
// CIR:     %[[CAST_DTOR:.*]] = cir.cast bitcast %[[GET_DTOR]] : !cir.ptr<!cir.func<(!cir.ptr<!rec_HasDtor>)>> -> !cir.ptr<!cir.func<(!cir.ptr<!void>)>>
// CIR:     %[[CAST_HD:.*]] = cir.cast bitcast %[[GET_HD]] : !cir.ptr<!rec_HasDtor> -> !cir.ptr<!void>
// CIR:     %[[DSO_HANDLE:.*]] = cir.get_global @__dso_handle : !cir.ptr<!u8i>
// CIR:     %[[AT_EXIT:.*]] = cir.call @__cxa_atexit(%[[CAST_DTOR]], %[[CAST_HD]], %[[DSO_HANDLE]]) : (!cir.ptr<!cir.func<(!cir.ptr<!void>)>>, !cir.ptr<!void>, !cir.ptr<!u8i>) -> !s32i
// CIR:     cir.call @__cxa_guard_release(%[[GET_GUARD]]) : (!cir.ptr<!s64i>) -> ()
// CIR:   }
// CIR: }
// CIR: cir.return

// LLVM-LABEL: define internal void @__cxx_global_var_init.1()
// LLVM: %[[LOAD_GUARD:.*]] = load atomic i8, ptr @_ZGVN5Outer5Inner2hdE acquire, align 8
// LLVM: %[[CMP:.*]] = icmp eq i8 %[[LOAD_GUARD]], 0
// LLVM: br i1 %[[CMP]], label %[[UNINIT:.*]], label %[[RET:.*]]

// LLVM: [[UNINIT]]:
// LLVM:   %[[ACQUIRE_GUARD:.*]] = call i32 @__cxa_guard_acquire(ptr @_ZGVN5Outer5Inner2hdE)
// LLVM:   %[[CMP:.*]] = icmp ne i32 %[[ACQUIRE_GUARD]], 0
// CIR leaves an extra 'block' for this target, but both go to 'ret' via only
// empty blocks.
// LLVM:   br i1 %[[CMP]], label %[[DO_INIT:.*]], label %{{.*}}

// LLVM: [[DO_INIT]]:
// LLVM:   %[[AT_EXIT:.*]] = call i32 @__cxa_atexit(ptr @_ZN7HasDtorD1Ev, ptr @_ZN5Outer5Inner2hdE, ptr @__dso_handle)
// LLVM:   call void @__cxa_guard_release(ptr @_ZGVN5Outer5Inner2hdE)
// CIR leaves an extra 'block' for this target, but both go to 'ret' via only
// empty blocks.
// LLVM:   br label %{{.*}}

// LLVM: [[RET]]:
// LLVM:   ret void

int &get_ref();
struct RefMember {
  static inline int &ref = get_ref();
};

int useRefMember() {
  return RefMember::ref;
}

// CIR: cir.global linkonce_odr comdat dynamic_init_guard<"_ZGVN9RefMember3refE"> @_ZN9RefMember3refE = #cir.ptr<null> : !cir.ptr<!s32i> align(8) ast(#cir.var.decl.ast) dynamic_init_info<local = false, tls = none, is_inline = true, tsk = undeclared>
// CIR-LABEL: cir.func internal private @__cxx_global_var_init.2() {
// CIR: %[[GET_GUARD:.*]] = cir.get_global @_ZGVN9RefMember3refE : !cir.ptr<!s64i>
// CIR: %[[TO_CHAR:.*]] = cir.cast bitcast %[[GET_GUARD]] : !cir.ptr<!s64i> -> !cir.ptr<!s8i>
// CIR: %[[LOAD_GUARD:.*]] = cir.load align(8) syncscope(system) atomic(acquire) %[[TO_CHAR]] : !cir.ptr<!s8i>, !s8i
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s8i
// CIR: %[[CMP:.*]] = cir.cmp eq %[[LOAD_GUARD]], %[[ZERO]] : !s8i
// CIR: cir.if %[[CMP]] {
// CIR:   %[[ACQUIRE_GUARD:.*]] = cir.call @__cxa_guard_acquire(%[[GET_GUARD]]) : (!cir.ptr<!s64i>) -> !s32i
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   %[[CMP:.*]] = cir.cmp ne %[[ACQUIRE_GUARD]], %[[ZERO]] : !s32i
// CIR:   cir.if %[[CMP]] {
// Note the lack of a `static_local` attribute here: unlike a true function-
// local static, this get_global must not be marked static_local even though
// the global it refers to carries a dynamic_init_guard.
// CIR:     %[[GET_REF:.*]] = cir.get_global @_ZN9RefMember3refE : !cir.ptr<!cir.ptr<!s32i>>
// CIR:     %[[CALL:.*]] = cir.call @_Z7get_refv() : () -> (!cir.ptr<!s32i> {{.*}})
// CIR:     cir.store align(8) %[[CALL]], %[[GET_REF]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR:     cir.call @__cxa_guard_release(%[[GET_GUARD]]) : (!cir.ptr<!s64i>) -> ()
// CIR:   }
// CIR: }
// CIR: cir.return

// LLVM-LABEL: define internal void @__cxx_global_var_init.2()
// LLVM: %[[LOAD_GUARD:.*]] = load atomic i8, ptr @_ZGVN9RefMember3refE acquire, align 8
// LLVM: %[[CMP:.*]] = icmp eq i8 %[[LOAD_GUARD]], 0
// LLVM: br i1 %[[CMP]], label %[[UNINIT:.*]], label %[[RET:.*]]

// LLVM: [[UNINIT]]:
// LLVM:   %[[ACQUIRE_GUARD:.*]] = call i32 @__cxa_guard_acquire(ptr @_ZGVN9RefMember3refE)
// LLVM:   %[[CMP:.*]] = icmp ne i32 %[[ACQUIRE_GUARD]], 0
// CIR leaves an extra 'block' for this target, but both go to 'ret' via only
// empty blocks.
// LLVM:   br i1 %[[CMP]], label %[[DO_INIT:.*]], label %{{.*}}

// LLVM: [[DO_INIT]]:
// LLVM:   %[[CALL:.*]] = call {{.*}}ptr @_Z7get_refv()
// LLVM:   store ptr %[[CALL]], ptr @_ZN9RefMember3refE
// LLVM:   call void @__cxa_guard_release(ptr @_ZGVN9RefMember3refE)
// CIR leaves an extra 'block' for this target, but both go to 'ret' via only
// empty blocks.
// LLVM:   br label %{{.*}}

// LLVM: [[RET]]:
// LLVM:   ret void

// Not thread-safe example(because not inline), so doesn't have guard/release.
int get_i();
template <typename T> struct NonThreadSafe {
  static T f;
};

template <typename T> T NonThreadSafe<T>::f = get_i();

int useNonThreadSafe() {
  return NonThreadSafe<int>::f;
}

// CIR: cir.global linkonce_odr comdat dynamic_init_guard<"_ZGVN13NonThreadSafeIiE1fE"> @_ZN13NonThreadSafeIiE1fE = #cir.int<0> : !s32i align(4) ast(#cir.var.decl.ast) dynamic_init_info<local = false, tls = none, is_inline = false, tsk = implicit_instantiation>

// CIR-LABEL: cir.func internal private @__cxx_global_var_init.3() {
// CIR:   %[[GET_GUARD:.*]] = cir.get_global @_ZGVN13NonThreadSafeIiE1fE : !cir.ptr<!s64i>
// CIR:   %[[TO_CHAR:.*]] = cir.cast bitcast %[[GET_GUARD]] : !cir.ptr<!s64i> -> !cir.ptr<!s8i>
// CIR:   %[[LOAD_GUARD:.*]] = cir.load align(8) %[[TO_CHAR]] : !cir.ptr<!s8i>, !s8i
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s8i
// CIR:   %[[CMP:.*]] = cir.cmp eq %[[LOAD_GUARD]], %[[ZERO]] : !s8i
// CIR:   cir.if %[[CMP]] {
// CIR:     %[[GUARD_BYTE:.*]] = cir.cast bitcast %[[GET_GUARD]] : !cir.ptr<!s64i> -> !cir.ptr<!s8i>
// CIR:     %[[ONE:.*]] = cir.const #cir.int<1> : !s8i
// CIR:     cir.store %[[ONE]], %[[GUARD_BYTE]] : !s8i, !cir.ptr<!s8i>
// CIR:     %[[GET_F:.*]] = cir.get_global @_ZN13NonThreadSafeIiE1fE : !cir.ptr<!s32i>
// CIR:     %[[CALL:.*]] = cir.call @_Z5get_iv() : () -> (!s32i {llvm.noundef})
// CIR:     cir.store align(4) %[[CALL]], %[[GET_F]] : !s32i, !cir.ptr<!s32i>
// CIR:   }
// CIR:   cir.return
// CIR: }

// LLVM-LABEL: define internal void @__cxx_global_var_init.3()
// LLVM: %[[LOAD_GUARD:.*]] = load i8, ptr @_ZGVN13NonThreadSafeIiE1fE, align 8
// LLVM: %[[CMP:.*]] = icmp eq i8 %[[LOAD_GUARD]], 0
// LLVM: br i1 %[[CMP]], label %[[UNINIT:.*]], label %[[RET2:.*]]

// LLVM: [[UNINIT]]:
// LLVM-NOT: call {{.*}}@__cxa_guard_acquire
// LLVM:   store i{{.*}} 1, ptr @_ZGVN13NonThreadSafeIiE1fE
// LLVM:   %[[CALL:.*]] = call noundef i32 @_Z5get_iv()
// LLVM:   store i32 %[[CALL]], ptr @_ZN13NonThreadSafeIiE1fE
// LLVM-NOT: call {{.*}}@__cxa_guard_release
// LLVM:   br label %{{.*}}

// LLVM: [[RET2]]:
// LLVM:   ret void

