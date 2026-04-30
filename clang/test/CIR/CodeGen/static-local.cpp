// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s --check-prefix=CIR-BEFORE-LPP,CIR-BOTH
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR,CIR-BOTH
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM,LLVM-CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM,OGCG

// Guard Variables:
// CIR-DAG: cir.global "private" internal dso_local @_ZGVZ17referenced_insidevE12static_local = #cir.int<0> : !s64i
// LLVM-DAG: @_ZGVZ17referenced_insidevE12static_local = internal global i64 0
// CIR-DAG: cir.global "private" internal dso_local @_ZGVZ14test_ctor_dtorvE9ctor_dtor = #cir.int<0> : !s64i
// LLVM-DAG: @_ZGVZ14test_ctor_dtorvE9ctor_dtor = internal global i64 0
// CIR-DAG: cir.global "private" internal dso_local @_ZGVZ9test_dtorvE4dtor = #cir.int<0> : !s64i
// LLVM-DAG: @_ZGVZ9test_dtorvE4dtor = internal global i64 0
// CIR-DAG: cir.global "private" internal dso_local @_ZGVZ8self_refiE12magic_static = #cir.int<0> : !s64i
// LLVM-DAG: @_ZGVZ8self_refiE12magic_static = internal global i64 0
// CIR-DAG: cir.global "private" internal dso_local @_ZGVZN8InMember8mem_funcEiiiiE12magic_static = #cir.int<0> : !s64i
// LLVM-DAG: @_ZGVZN8InMember8mem_funcEiiiiE12magic_static = internal global i64 0
// CIR-DAG: cir.global "private" internal dso_local @_ZGVZ10multi_refsiiiiiiiE17refs_magic_static = #cir.int<0> : !s64i
// LLVM-DAG: @_ZGVZ10multi_refsiiiiiiiE17refs_magic_static = internal global i64 0
// CIR-DAG: cir.global "private" internal dso_local @_ZGVZ10multi_refsiiiiiiiE12magic_static = #cir.int<0> : !s64i
// LLVM-DAG: @_ZGVZ10multi_refsiiiiiiiE12magic_static = internal global i64 0
// CIR-DAG: cir.global "private" internal dso_local @_ZGVZ29references_param_and_previousiE17refs_magic_static = #cir.int<0> : !s64i
// LLVM-DAG: @_ZGVZ29references_param_and_previousiE17refs_magic_static = internal global i64 0
// CIR-DAG: cir.global "private" internal dso_local @_ZGVZ29references_param_and_previousiE12magic_static = #cir.int<0> : !s64i
// LLVM-DAG: @_ZGVZ29references_param_and_previousiE12magic_static = internal global i64 0
// CIR-DAG: cir.global "private" internal dso_local @_ZGVZ1fvE1a = #cir.int<0> : !s64i
// LLVM-DAG: @_ZGVZ1fvE1a = internal global i64 0
// CIR-DAG: cir.global "private" linkonce_odr comdat @_ZGVZ10getInlineAvE1a = #cir.int<0> : !s64i
// LLVM-DAG: @_ZGVZ10getInlineAvE1a = linkonce_odr global i64 0, comdat
// CIR-DAG: cir.global "private" internal dso_local @_ZGVZ8ref_initvE1y = #cir.int<0> : !s64i
// LLVM-DAG: @_ZGVZ8ref_initvE1y = internal global i64 0
// CIR-DAG: cir.global "private" internal dso_local @_ZGVZ23array_static_local_dtorvE2sm = #cir.int<0> : !s64i
// LLVM-DAG: @_ZGVZ23array_static_local_dtorvE2sm = internal global i64 0

// CIR-BOTH-DAG: cir.global linkonce_odr comdat static_local_guard<"_ZGVZ10getInlineAvE1a"> @_ZZ10getInlineAvE1a = #cir.zero : !rec_A
// LLVM-DAG: @_ZZ10getInlineAvE1a = linkonce_odr global %class.A zeroinitializer, comdat
// CIR-BOTH-DAG: cir.global "private" internal dso_local @_ZZ23referenced_inside_constvE12static_local = #cir.int<42> : !s32i
// LLVM-DAG: @_ZZ23referenced_inside_constvE12static_local = internal global i32 42
// CIR-BOTH-DAG: cir.global "private" internal dso_local static_local_guard<"_ZGVZ17referenced_insidevE12static_local"> @_ZZ17referenced_insidevE12static_local = #cir.int<0> : !s32i
// LLVM-DAG: @_ZZ17referenced_insidevE12static_local = internal global i32 0
// CIR-BOTH-DAG: cir.global "private" internal dso_local static_local_guard<"_ZGVZ14test_ctor_dtorvE9ctor_dtor"> @_ZZ14test_ctor_dtorvE9ctor_dtor = #cir.zero : !rec_HasCtorDtor
// LLVM-DAG: @_ZZ14test_ctor_dtorvE9ctor_dtor = internal global %struct.HasCtorDtor zeroinitializer
// CIR-BOTH-DAG: cir.global "private" internal dso_local static_local_guard<"_ZGVZ9test_dtorvE4dtor"> @_ZZ9test_dtorvE4dtor = #cir.zero : !rec_HasDtor
// LLVM-DAG: @_ZZ9test_dtorvE4dtor = internal global %struct.HasDtor zeroinitializer
// CIR-BOTH-DAG: cir.global "private" internal dso_local static_local_guard<"_ZGVZ8self_refiE12magic_static"> @_ZZ8self_refiE12magic_static = #cir.int<0> : !s32i
// LLVM-DAG: @_ZZ8self_refiE12magic_static = internal global i32 0
// CIR-BOTH-DAG: cir.global "private" internal dso_local static_local_guard<"_ZGVZN8InMember8mem_funcEiiiiE12magic_static"> @_ZZN8InMember8mem_funcEiiiiE12magic_static = #cir.int<0> : !s32i
// LLVM-DAG: @_ZZN8InMember8mem_funcEiiiiE12magic_static = internal global i32 0
// CIR-BOTH-DAG: cir.global "private" internal dso_local static_local_guard<"_ZGVZ10multi_refsiiiiiiiE17refs_magic_static"> @_ZZ10multi_refsiiiiiiiE17refs_magic_static = #cir.zero : !rec_A
// LLVM-DAG: @_ZZ10multi_refsiiiiiiiE17refs_magic_static = internal global %class.A zeroinitializer
// CIR-BOTH-DAG: cir.global "private" internal dso_local static_local_guard<"_ZGVZ10multi_refsiiiiiiiE12magic_static"> @_ZZ10multi_refsiiiiiiiE12magic_static = #cir.zero : !rec_A
// LLVM-DAG: @_ZZ10multi_refsiiiiiiiE12magic_static = internal global %class.A zeroinitializer
// CIR-BOTH-DAG: cir.global "private" internal dso_local static_local_guard<"_ZGVZ29references_param_and_previousiE17refs_magic_static"> @_ZZ29references_param_and_previousiE17refs_magic_static = #cir.int<0> : !s32i
// LLVM-DAG: @_ZZ29references_param_and_previousiE17refs_magic_static = internal global i32 0
// CIR-BOTH-DAG: cir.global "private" internal dso_local static_local_guard<"_ZGVZ29references_param_and_previousiE12magic_static"> @_ZZ29references_param_and_previousiE12magic_static = #cir.int<0> : !s32i
// LLVM-DAG: @_ZZ29references_param_and_previousiE12magic_static = internal global i32 0
// CIR-BOTH-DAG: cir.global "private" internal dso_local static_local_guard<"_ZGVZ1fvE1a"> @_ZZ1fvE1a = #cir.zero : !rec_A
// LLVM-DAG: @_ZZ1fvE1a = internal global %class.A zeroinitializer
// CIR-BOTH-DAG: cir.global "private" internal dso_local static_local_guard<"_ZGVZ8ref_initvE1y"> @_ZZ8ref_initvE1y = #cir.ptr<null> : !cir.ptr<!s32i>
// LLVM-DAG: @_ZZ8ref_initvE1y = internal global ptr null
// CIR-BOTH-DAG: cir.global "private" internal dso_local static_local_guard<"_ZGVZ23array_static_local_dtorvE2sm"> @_ZZ23array_static_local_dtorvE2sm = #cir.zero : !cir.array<!rec_HasCtorDtor x 2>
// LLVM-DAG: @_ZZ23array_static_local_dtorvE2sm = internal global [2 x %struct.HasCtorDtor] zeroinitializer
// CIR-BOTH-DAG: cir.global "private" internal dso_local @_ZZ15use_static_declvE1p = #cir.global_view<@_ZZ15use_static_declvE1x> : !cir.ptr<!s32i>
// LLVM-DAG: @_ZZ15use_static_declvE1p = internal global ptr @_ZZ15use_static_declvE1x
// CIR-BOTH-DAG: cir.global "private" internal dso_local @_ZZ15use_static_declvE1x = #cir.int<42> : !s32i
// LLVM-DAG: @_ZZ15use_static_declvE1x = internal global i32 42



void use_static_decl() {
  static int x = 42;
  static int *p = &x;
// CIR-BOTH-LABEL: cir.func no_inline dso_local @_Z15use_static_declv()
// CIR-BOTH: cir.get_global @_ZZ15use_static_declvE1x : !cir.ptr<!s32i>
// CIR-BOTH: cir.get_global @_ZZ15use_static_declvE1p : !cir.ptr<!cir.ptr<!s32i>>
// LLVM-LABEL: define dso_local void @_Z15use_static_declv()
// LLVM:   ret void
}

class A {
public:
  A();
  A(int);
  int var;
};

void use(A*);
void f() {
  static A a;
  use(&a);
// CIR-BOTH-LABEL: cir.func no_inline dso_local @_Z1fv()
// CIR-BOTH: %[[GET_GLOB:.*]] = cir.get_global static_local @_ZZ1fvE1a : !cir.ptr<!rec_A>
//
// CIR-BEFORE-LPP:   cir.local_init static_local @_ZZ1fvE1a ctor {
//
// CIR: %[[GET_GUARD:.*]] = cir.get_global @_ZGVZ1fvE1a : !cir.ptr<!s64i>
// CIR: %[[GUARD_BYTE_PTR:.*]] = cir.cast bitcast %[[GET_GUARD]] : !cir.ptr<!s64i> -> !cir.ptr<!s8i>
// CIR: %[[GUARD_LOAD:.*]] = cir.load {{.*}}syncscope(system) atomic(acquire) %[[GUARD_BYTE_PTR]] : !cir.ptr<!s8i>, !s8i
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s8i
// CIR: %[[IS_UNINIT:.*]] = cir.cmp eq %[[GUARD_LOAD]], %[[ZERO]] : !s8i
// CIR: cir.if %[[IS_UNINIT]] {
// CIR:   %[[ACQUIRE:.*]] = cir.call @__cxa_guard_acquire(%[[GET_GUARD]]) : (!cir.ptr<!s64i>) -> !s32i
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   %[[IS_UNINIT:.*]] = cir.cmp ne %[[ACQUIRE]], %[[ZERO]] : !s32i
// CIR:   cir.if %[[IS_UNINIT]] {
//
// CIR-BOTH:     %[[GET_GLOB_INIT:.*]] = cir.get_global static_local @_ZZ1fvE1a : !cir.ptr<!rec_A>
// CIR-BOTH:     cir.call @_ZN1AC1Ev(%[[GET_GLOB_INIT]]) : (!cir.ptr<!rec_A>{{.*}}) -> ()
//
// CIR-BEFORE-LPP:     cir.yield
// CIR-BEFORE-LPP:   }
//
// CIR:     cir.call @__cxa_guard_release(%[[GET_GUARD]]) : (!cir.ptr<!s64i>) -> ()
// CIR:   }
// CIR: }
//
// CIR-BOTH:   cir.call @_Z3useP1A(%[[GET_GLOB]]) : (!cir.ptr<!rec_A> {llvm.noundef}) -> ()
// CIR-BOTH:   cir.return

// LLVM-LABEL: define dso_local void @_Z1fv()
// LLVM: %[[GET_GUARD:.*]] = load atomic i8, ptr @_ZGVZ1fvE1a acquire
// LLVM: %[[IS_UNINIT:.*]] = icmp eq i8 %[[GET_GUARD]], 0
// LLVM: br i1 %[[IS_UNINIT]]
//
// LLVM:  %[[ACQUIRE:.*]] = call i32 @__cxa_guard_acquire(ptr @_ZGVZ1fvE1a)
// LLVM:  %[[IS_UNINIT:.*]] = icmp ne i32 %[[ACQUIRE]], 0
// LLVM:  br i1 %[[IS_UNINIT]]
//
// LLVM:  call void @_ZN1AC1Ev(ptr {{.*}}@_ZZ1fvE1a)
// LLVM:  call void @__cxa_guard_release(ptr @_ZGVZ1fvE1a)
//
// LLVM:  call void @_Z3useP1A(ptr noundef @_ZZ1fvE1a)
// LLVM:  ret void
}

// Static local in an inline function: the variable and guard both get
// linkonce_odr linkage and their own COMDAT groups.
void use(const A *);
inline const A &getInlineA() {
  static A a;
  return a;
}

void call_inline() {
  use(&getInlineA());
}

// CIR-BOTH-LABEL: cir.func no_inline comdat linkonce_odr @_Z10getInlineAv() 
// CIR-BOTH:  %[[GET_MS:.*]] = cir.get_global static_local @_ZZ10getInlineAvE1a : !cir.ptr<!rec_A>
//
// CIR-BEFORE-LPP:  cir.local_init static_local @_ZZ10getInlineAvE1a ctor {
//
// CIR: %[[GET_GUARD:.*]] = cir.get_global @_ZGVZ10getInlineAvE1a : !cir.ptr<!s64i>
// CIR: %[[GUARD_BYTE_PTR:.*]] = cir.cast bitcast %[[GET_GUARD]] : !cir.ptr<!s64i> -> !cir.ptr<!s8i>
// CIR: %[[GUARD_LOAD:.*]] = cir.load{{.*}} syncscope(system) atomic(acquire) %[[GUARD_BYTE_PTR]] : !cir.ptr<!s8i>, !s8i
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s8i
// CIR: %[[IS_UNINIT:.*]] = cir.cmp eq %[[GUARD_LOAD]], %[[ZERO]] : !s8i
// CIR: cir.if %[[IS_UNINIT]] {
// CIR:   %[[ACQUIRE:.*]] = cir.call @__cxa_guard_acquire(%[[GET_GUARD]]) : (!cir.ptr<!s64i>) -> !s32i
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   %[[IS_UNINIT:.*]] = cir.cmp ne %[[ACQUIRE]], %[[ZERO]] : !s32i
// CIR:   cir.if %[[IS_UNINIT]] {
//
// CIR-BOTH:    %[[GET_MS_INIT:.*]] = cir.get_global static_local @_ZZ10getInlineAvE1a : !cir.ptr<!rec_A>
// CIR-BOTH:    cir.call @_ZN1AC1Ev(%[[GET_MS_INIT]]) : (!cir.ptr<!rec_A> {{.*}}) -> ()
// CIR-BEFORE-LPP:    cir.yield
// CIR-BEFORE-LPP:  }
//
// CIR:     cir.call @__cxa_guard_release(%[[GET_GUARD]]) : (!cir.ptr<!s64i>) -> ()
// CIR:   }
// CIR: }
//
// LLVM-LABEL: define {{.*}}@_Z10getInlineAv()
// LLVM: %[[GET_GUARD:.*]] = load atomic i8, ptr @_ZGVZ10getInlineAvE1a acquire
// LLVM: %[[IS_UNINIT:.*]] = icmp eq i8 %[[GET_GUARD]], 0
// LLVM: br i1 %[[IS_UNINIT]]
//
// LLVM:  %[[ACQUIRE:.*]] = call i32 @__cxa_guard_acquire(ptr @_ZGVZ10getInlineAvE1a)
// LLVM:  %[[IS_UNINIT:.*]] = icmp ne i32 %[[ACQUIRE]], 0
// LLVM:  br i1 %[[IS_UNINIT]]
//
// LLVM:  call void @_ZN1AC1Ev(ptr {{.*}}@_ZZ10getInlineAvE1a)
// LLVM:  call void @__cxa_guard_release(ptr @_ZGVZ10getInlineAvE1a)

int bar();

void references_param_and_previous(int param) {
  static int magic_static = param + bar();
  static int refs_magic_static = magic_static;

// CIR-BOTH-LABEL: cir.func no_inline dso_local @_Z29references_param_and_previousi
// CIR-BOTH:    %[[PARAM_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["param", init]
// CIR-BOTH:    %[[GET_MAG_STATIC:.*]] = cir.get_global static_local @_ZZ29references_param_and_previousiE12magic_static : !cir.ptr<!s32i>
//
// CIR-BEFORE-LPP:    cir.local_init static_local @_ZZ29references_param_and_previousiE12magic_static ctor {
//
// CIR: %[[GET_GUARD:.*]] = cir.get_global @_ZGVZ29references_param_and_previousiE12magic_static : !cir.ptr<!s64i>
// CIR: %[[GUARD_BYTE_PTR:.*]] = cir.cast bitcast %[[GET_GUARD]] : !cir.ptr<!s64i> -> !cir.ptr<!s8i>
// CIR: %[[GUARD_LOAD:.*]] = cir.load{{.*}} syncscope(system) atomic(acquire) %[[GUARD_BYTE_PTR]] : !cir.ptr<!s8i>, !s8i
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s8i
// CIR: %[[IS_UNINIT:.*]] = cir.cmp eq %[[GUARD_LOAD]], %[[ZERO]] : !s8i
// CIR: cir.if %[[IS_UNINIT]] {
// CIR:   %[[ACQUIRE:.*]] = cir.call @__cxa_guard_acquire(%[[GET_GUARD]]) : (!cir.ptr<!s64i>) -> !s32i
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   %[[IS_UNINIT:.*]] = cir.cmp ne %[[ACQUIRE]], %[[ZERO]] : !s32i
// CIR:   cir.if %[[IS_UNINIT]] {
//
// CIR-BOTH:      %[[GET_MAG_STATIC_INIT:.*]] = cir.get_global static_local @_ZZ29references_param_and_previousiE12magic_static : !cir.ptr<!s32i>
// CIR-BOTH:      %[[LOAD_PARAM:.*]] = cir.load {{.*}}%[[PARAM_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR-BOTH:      %[[CALL_BAR:.*]] = cir.call @_Z3barv() : () -> (!s32i {llvm.noundef})
// CIR-BOTH:      %[[ADD:.*]] = cir.add nsw %[[LOAD_PARAM]], %[[CALL_BAR]] : !s32i
// CIR-BOTH:      cir.store {{.*}} %[[ADD]], %[[GET_MAG_STATIC_INIT]] : !s32i, !cir.ptr<!s32i>
//
// CIR-BEFORE-LPP:      cir.yield
// CIR-BEFORE-LPP:    }
//
// CIR:     cir.call @__cxa_guard_release(%[[GET_GUARD]]) : (!cir.ptr<!s64i>) -> ()
// CIR:   }
// CIR: }
//
// CIR-BOTH:    %[[GET_MS_REF:.*]] = cir.get_global static_local @_ZZ29references_param_and_previousiE17refs_magic_static : !cir.ptr<!s32i>
//
// CIR-BEFORE-LPP:    cir.local_init static_local @_ZZ29references_param_and_previousiE17refs_magic_static ctor {
//
// CIR: %[[GET_GUARD:.*]] = cir.get_global @_ZGVZ29references_param_and_previousiE17refs_magic_static : !cir.ptr<!s64i>
// CIR: %[[GUARD_BYTE_PTR:.*]] = cir.cast bitcast %[[GET_GUARD]] : !cir.ptr<!s64i> -> !cir.ptr<!s8i>
// CIR: %[[GUARD_LOAD:.*]] = cir.load{{.*}} syncscope(system) atomic(acquire) %[[GUARD_BYTE_PTR]] : !cir.ptr<!s8i>, !s8i
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s8i
// CIR: %[[IS_UNINIT:.*]] = cir.cmp eq %[[GUARD_LOAD]], %[[ZERO]] : !s8i
// CIR: cir.if %[[IS_UNINIT]] {
// CIR:   %[[ACQUIRE]] = cir.call @__cxa_guard_acquire(%8) : (!cir.ptr<!s64i>) -> !s32i
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   %[[IS_UNINIT:.*]] = cir.cmp ne %[[ACQUIRE]], %[[ZERO]] : !s32i
// CIR:   cir.if %[[IS_UNINIT]] {
//
// CIR-BOTH:      %[[GET_MS_REF_INIT:.*]] = cir.get_global static_local @_ZZ29references_param_and_previousiE17refs_magic_static : !cir.ptr<!s32i>
// CIR-BOTH:      %[[LOAD_MAG_STATIC:.*]] = cir.load {{.*}}%[[GET_MAG_STATIC]] : !cir.ptr<!s32i>, !s32i
// CIR-BOTH:      cir.store {{.*}}%[[LOAD_MAG_STATIC]], %[[GET_MS_REF_INIT]] : !s32i, !cir.ptr<!s32i>
//
// CIR-BEFORE-LPP:      cir.yield
// CIR-BEFORE-LPP:    }
//
// CIR:     cir.call @__cxa_guard_release(%[[GET_GUARD]]) : (!cir.ptr<!s64i>) -> ()
// CIR:   }
// CIR: }
// CIR-BOTH:    cir.return

// LLVM: define dso_local void @_Z29references_param_and_previousi(
// LLVM:   %[[PARAM_ALLOCA:.*]] = alloca i32
// LLVM:   %[[GET_GUARD:.*]] = load atomic i8, ptr @_ZGVZ29references_param_and_previousiE12magic_static acquire
// LLVM:   %[[IS_UNINIT:.*]] = icmp eq i8 %[[GET_GUARD]], 0
// LLVM:   br i1 %[[IS_UNINIT]]
//
// LLVM:   %[[ACQUIRE:.*]] = call i32 @__cxa_guard_acquire(ptr @_ZGVZ29references_param_and_previousiE12magic_static)
// LLVM:   %[[IS_UNINIT:.*]] = icmp ne i32 %[[ACQUIRE]], 0
// LLVM:   br i1 %[[IS_UNINIT]]

// LLVM:   %[[PARAM_LOAD:.*]] = load i32, ptr %[[PARAM_ALLOCA]]
// LLVM:   %[[CALL_BAR:.*]] = call noundef i32 @_Z3barv()
// LLVM:   %[[ADD:.*]] = add nsw i32 %[[PARAM_LOAD]], %[[CALL_BAR]]
// LLVM:   store i32 %[[ADD]], ptr @_ZZ29references_param_and_previousiE12magic_static
// LLVM:   call void @__cxa_guard_release(ptr @_ZGVZ29references_param_and_previousiE12magic_static)
//
// LLVM:   %[[GET_GUARD:.*]] = load atomic i8, ptr @_ZGVZ29references_param_and_previousiE17refs_magic_static acquire
// LLVM:   %[[IS_UNINIT:.*]] = icmp eq i8 %[[GET_GUARD]], 0
// LLVM:   br i1 %[[IS_UNINIT]]
//
// LLVM:   %[[ACQUIRE:.*]] = call i32 @__cxa_guard_acquire(ptr @_ZGVZ29references_param_and_previousiE17refs_magic_static)
// LLVM:   %[[IS_UNINIT:.*]] = icmp ne i32 %[[ACQUIRE]], 0
// LLVM:   br i1 %[[IS_UNINIT]]
//
// LLVM:   %[[LOAD_MAG_STATIC:.*]] = load i32, ptr @_ZZ29references_param_and_previousiE12magic_static
// LLVM:   store i32 %[[LOAD_MAG_STATIC]], ptr @_ZZ29references_param_and_previousiE17refs_magic_static
// LLVM:   call void @__cxa_guard_release(ptr @_ZGVZ29references_param_and_previousiE17refs_magic_static)
//
// LLVM:   ret void
}

void multi_refs(int one, int two, int, int three, int, int four, int) {
  static A magic_static = one + three + four + bar();
  static A refs_magic_static = magic_static;
// CIR-BOTH-LABEL: cir.func no_inline dso_local @_Z10multi_refsiiiiiii(
// CIR-BOTH:   %[[ONE_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["one", init]
// CIR-BOTH:   %[[THREE_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["three", init]
// CIR-BOTH:   %[[FOUR_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["four", init]
// CIR-BOTH:   %[[GET_MS:.*]] = cir.get_global static_local @_ZZ10multi_refsiiiiiiiE12magic_static : !cir.ptr<!rec_A>
// CIR-BEFORE-LPP:   cir.local_init static_local @_ZZ10multi_refsiiiiiiiE12magic_static ctor {
//
// CIR: %[[GET_GUARD:.*]] = cir.get_global @_ZGVZ10multi_refsiiiiiiiE12magic_static : !cir.ptr<!s64i>
// CIR: %[[GUARD_BYTE_PTR:.*]] = cir.cast bitcast %[[GET_GUARD]] : !cir.ptr<!s64i> -> !cir.ptr<!s8i>
// CIR: %[[GUARD_LOAD:.*]] = cir.load{{.*}} syncscope(system) atomic(acquire) %[[GUARD_BYTE_PTR]] : !cir.ptr<!s8i>, !s8i
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s8i
// CIR: %[[IS_UNINIT:.*]] = cir.cmp eq %[[GUARD_LOAD]], %[[ZERO]] : !s8i
// CIR: cir.if %[[IS_UNINIT]] {
// CIR:   %[[ACQUIRE:.*]] = cir.call @__cxa_guard_acquire(%[[GET_GUARD]]) : (!cir.ptr<!s64i>) -> !s32i
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   %[[IS_UNINIT:.*]] = cir.cmp ne %[[ACQUIRE]], %[[ZERO]] : !s32i
// CIR:   cir.if %[[IS_UNINIT]] {
//
// CIR-BOTH:     %[[GET_MS_INIT:.*]] = cir.get_global static_local @_ZZ10multi_refsiiiiiiiE12magic_static : !cir.ptr<!rec_A>
// CIR-BOTH:     %[[ONE_LOAD:.*]] = cir.load {{.*}} %[[ONE_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR-BOTH:     %[[THREE_LOAD:.*]] = cir.load {{.*}} %[[THREE_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR-BOTH:     %[[ADD1:.*]] = cir.add nsw %[[ONE_LOAD]], %[[THREE_LOAD]] : !s32i
// CIR-BOTH:     %[[FOUR_LOAD:.*]] = cir.load {{.*}} %[[FOUR_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR-BOTH:     %[[ADD2:.*]] = cir.add nsw %[[ADD1]], %[[FOUR_LOAD]] : !s32i
// CIR-BOTH:     %[[CALL_BAR:.*]] = cir.call @_Z3barv() : () -> (!s32i {llvm.noundef})
// CIR-BOTH:     %[[ADD3:.*]] = cir.add nsw %[[ADD2]], %[[CALL_BAR]] : !s32i
// CIR-BOTH:     cir.call @_ZN1AC1Ei(%[[GET_MS_INIT]], %[[ADD3]]) : (!cir.ptr<!rec_A>{{.*}}) -> ()
//
// CIR-BEFORE-LPP:     cir.yield
// CIR-BEFORE-LPP:   }
//
// CIR:     cir.call @__cxa_guard_release(%[[GET_GUARD]]) : (!cir.ptr<!s64i>) -> ()
// CIR:   }
// CIR: }
//
// CIR-BOTH:   %[[GET_REF_MS:.*]] = cir.get_global static_local @_ZZ10multi_refsiiiiiiiE17refs_magic_static : !cir.ptr<!rec_A>
// CIR-BEFORE-LPP:   cir.local_init static_local @_ZZ10multi_refsiiiiiiiE17refs_magic_static ctor {
//
// CIR: %[[GET_GUARD:.*]] = cir.get_global @_ZGVZ10multi_refsiiiiiiiE17refs_magic_static : !cir.ptr<!s64i>
// CIR: %[[GUARD_BYTE_PTR:.*]] = cir.cast bitcast %[[GET_GUARD]] : !cir.ptr<!s64i> -> !cir.ptr<!s8i>
// CIR: %[[GUARD_LOAD:.*]] = cir.load{{.*}} syncscope(system) atomic(acquire) %[[GUARD_BYTE_PTR]] : !cir.ptr<!s8i>, !s8i
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s8i
// CIR: %[[IS_UNINIT:.*]] = cir.cmp eq %[[GUARD_LOAD]], %[[ZERO]] : !s8i
// CIR: cir.if %[[IS_UNINIT]] {
// CIR:   %[[ACQUIRE:.*]] = cir.call @__cxa_guard_acquire(%[[GET_GUARD]]) : (!cir.ptr<!s64i>) -> !s32i
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   %[[IS_UNINIT:.*]] = cir.cmp ne %[[ACQUIRE]], %[[ZERO]] : !s32i
// CIR:   cir.if %[[IS_UNINIT]] {
//
// CIR-BOTH:     %[[GET_REF_MS_INIT:.*]] = cir.get_global static_local @_ZZ10multi_refsiiiiiiiE17refs_magic_static : !cir.ptr<!rec_A>
// CIR-BOTH:     cir.copy %[[GET_MS]] to %[[GET_REF_MS_INIT]] : !cir.ptr<!rec_A>
// CIR-BEFORE-LPP:     cir.yield
// CIR-BEFORE-LPP:   }
//
// CIR:     cir.call @__cxa_guard_release(%[[GET_GUARD]]) : (!cir.ptr<!s64i>) -> ()
// CIR:   }
// CIR: }
//
// CIR-BOTH:   cir.return
//
// LLVM-LABEL: define dso_local void @_Z10multi_refsiiiiiii(
// LLVM:  %[[ONE_ALLOCA:.*]] = alloca i32
// LLVM:  alloca i32
// LLVM:  alloca i32
// LLVM:  %[[THREE_ALLOCA:.*]] = alloca i32
// LLVM:  alloca i32
// LLVM:  %[[FOUR_ALLOCA:.*]] = alloca i32
// LLVM:  %[[GET_GUARD:.*]] = load atomic i8, ptr @_ZGVZ10multi_refsiiiiiiiE12magic_static acquire
// LLVM:  %[[IS_UNINIT:.*]] = icmp eq i8 %[[GET_GUARD]], 0
// LLVM:  br i1 %[[IS_UNINIT]]
//
// LLVM:  %[[ACQUIRE:.*]] = call i32 @__cxa_guard_acquire(ptr @_ZGVZ10multi_refsiiiiiiiE12magic_static)
// LLVM:  %[[IS_UNINIT:.*]] = icmp ne i32 %[[ACQUIRE]], 0

// LLVM:  br i1 %[[IS_UNINIT]]
//
// LLVM:  %[[ONE_LOAD:.*]] = load i32, ptr %[[ONE_ALLOCA]]
// LLVM:  %[[THREE_LOAD:.*]] = load i32, ptr %[[THREE_ALLOCA]]
// LLVM:  %[[ADD1:.*]] = add nsw i32 %[[ONE_LOAD]], %[[THREE_LOAD]]
// LLVM:  %[[FOUR_LOAD:.*]] = load i32, ptr %[[FOUR_ALLOCA]]
// LLVM:  %[[ADD2:.*]] = add nsw i32 %[[ADD1]], %[[FOUR_LOAD]]
// LLVM:  %[[CALL_BAR:.*]] = call noundef i32 @_Z3barv()
// LLVM:  %[[ADD3:.*]] = add nsw i32 %[[ADD2]], %[[CALL_BAR]]
// LLVM:  call void @_ZN1AC1Ei(ptr {{.*}}@_ZZ10multi_refsiiiiiiiE12magic_static, i32 {{.*}}%[[ADD3]])
// LLVM:  call void @__cxa_guard_release(ptr @_ZGVZ10multi_refsiiiiiiiE12magic_static)
//
// LLVM:  %[[GET_GUARD:.*]] = load atomic i8, ptr @_ZGVZ10multi_refsiiiiiiiE17refs_magic_static acquire
// LLVM:  %[[IS_UNINIT:.*]] = icmp eq i8 %[[GET_GUARD]], 0
// LLVM:  br i1 %[[IS_UNINIT]]
//
// LLVM:  %[[ACQUIRE:.*]] = call i32 @__cxa_guard_acquire(ptr @_ZGVZ10multi_refsiiiiiiiE17refs_magic_static)
// LLVM:  %[[IS_UNINIT:.*]] = icmp ne i32 %[[ACQUIRE]], 0
// LLVM:  br i1 %[[IS_UNINIT]]
//
// LLVM:  call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}@_ZZ10multi_refsiiiiiiiE17refs_magic_static, ptr {{.*}}@_ZZ10multi_refsiiiiiiiE12magic_static, i64 4, i1 false)
// LLVM:  call void @__cxa_guard_release(ptr @_ZGVZ10multi_refsiiiiiiiE17refs_magic_static)
//
// LLVM:  ret void
}

struct InMember {
  int mem_get_int();
  void mem_func(int one, int two, int, int three);
};

void InMember::mem_func(int one, int two, int, int three) {
  int some_local = mem_get_int();
  static int magic_static = three + mem_get_int() + one + some_local;
// CIR-BOTH-LABEL:  cir.func no_inline dso_local @_ZN8InMember8mem_funcEiiii(
// CIR-BOTH:    %[[THIS_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_InMember>, !cir.ptr<!cir.ptr<!rec_InMember>>, ["this", init]
// CIR-BOTH:    %[[ONE_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["one", init]
// CIR-BOTH:    %[[THREE_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["three", init]
// CIR-BOTH:    %[[LOCAL_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["some_local", init]
// CIR-BOTH:    %[[THIS_LOAD:.*]] = cir.load %[[THIS_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_InMember>>, !cir.ptr<!rec_InMember>
// CIR-BOTH:    %[[GET_INT:.*]] = cir.call @_ZN8InMember11mem_get_intEv(%[[THIS_LOAD]]) : (!cir.ptr<!rec_InMember>{{.*}}) -> (!s32i {llvm.noundef})
// CIR-BOTH:    cir.store{{.*}} %[[GET_INT]], %[[LOCAL_ALLOCA]] : !s32i, !cir.ptr<!s32i>
// CIR-BOTH:    %[[GET_MS:.*]] = cir.get_global static_local @_ZZN8InMember8mem_funcEiiiiE12magic_static : !cir.ptr<!s32i>
//
// CIR-BEFORE-LPP:    cir.local_init static_local @_ZZN8InMember8mem_funcEiiiiE12magic_static ctor {
//
// CIR: %[[GET_GUARD:.*]] = cir.get_global @_ZGVZN8InMember8mem_funcEiiiiE12magic_static : !cir.ptr<!s64i>
// CIR: %[[GUARD_BYTE_PTR:.*]] = cir.cast bitcast %[[GET_GUARD]] : !cir.ptr<!s64i> -> !cir.ptr<!s8i>
// CIR: %[[GUARD_LOAD:.*]] = cir.load{{.*}} syncscope(system) atomic(acquire) %[[GUARD_BYTE_PTR]] : !cir.ptr<!s8i>, !s8i
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s8i
// CIR: %[[IS_UNINIT:.*]] = cir.cmp eq %[[GUARD_LOAD]], %[[ZERO]] : !s8i
// CIR: cir.if %[[IS_UNINIT]] {
// CIR:   %[[ACQUIRE:.*]] = cir.call @__cxa_guard_acquire(%[[GET_GUARD]]) : (!cir.ptr<!s64i>) -> !s32i
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   %[[IS_UNINIT:.*]] = cir.cmp ne %[[ACQUIRE]], %[[ZERO]] : !s32i
// CIR:   cir.if %[[IS_UNINIT]] {
//
// CIR-BOTH:      %[[GET_MS_INIT:.*]] = cir.get_global static_local @_ZZN8InMember8mem_funcEiiiiE12magic_static : !cir.ptr<!s32i>
// CIR-BOTH:      %[[LOAD_THREE:.*]] = cir.load{{.*}} %[[THREE_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR-BOTH:      %[[GET_INT:.*]] = cir.call @_ZN8InMember11mem_get_intEv(%[[THIS_LOAD]]) : (!cir.ptr<!rec_InMember>{{.*}}) -> (!s32i {llvm.noundef})
// CIR-BOTH:      %[[ADD1:.*]] = cir.add nsw %[[LOAD_THREE]], %[[GET_INT]] : !s32i
// CIR-BOTH:      %[[LOAD_ONE:.*]] = cir.load{{.*}} %[[ONE_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR-BOTH:      %[[ADD2:.*]] = cir.add nsw %[[ADD1]], %[[LOAD_ONE]] : !s32i
// CIR-BOTH:      %[[LOAD_LOCAL:.*]] = cir.load{{.*}} %[[LOCAL_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR-BOTH:      %[[ADD3:.*]] = cir.add nsw %[[ADD2]], %[[LOAD_LOCAL]] : !s32i
// CIR-BOTH:      cir.store{{.*}} %[[ADD3]], %[[GET_MS_INIT]] : !s32i, !cir.ptr<!s32i>

// CIR-BEFORE-LPP:      cir.yield
// CIR-BEFORE-LPP:    }

// CIR:     cir.call @__cxa_guard_release(%[[GET_GUARD]]) : (!cir.ptr<!s64i>) -> ()
// CIR:   }
// CIR: }
// CIR-BOTH: cir.return
//
// LLVM-LABEL: define dso_local void @_ZN8InMember8mem_funcEiiii(
// LLVM:   %[[THIS_ALLOCA:.*]] = alloca ptr
// LLVM:   %[[ONE_ALLOCA:.*]] = alloca i32
// LLVM:   alloca i32
// LLVM:   alloca i32
// LLVM:   %[[THREE_ALLOCA:.*]] = alloca i32
// LLVM:   %[[LOCAL_ALLOCA:.*]] = alloca i32
// LLVM:   %[[THIS_LOAD:.*]] = load ptr, ptr %[[THIS_ALLOCA]]
// LLVM:   %[[GET_INT:.*]] = call noundef i32 @_ZN8InMember11mem_get_intEv(ptr {{.*}}%[[THIS_LOAD]])
// LLVM:   store i32 %[[GET_INT]], ptr %[[LOCAL_ALLOCA]]
// LLVM:   %[[GET_GUARD:.*]] = load atomic i8, ptr @_ZGVZN8InMember8mem_funcEiiiiE12magic_static acquire
// LLVM:   %[[IS_UNINIT:.*]] = icmp eq i8 %[[GET_GUARD]], 0
// LLVM:   br i1 %[[IS_UNINIT]]
//
// LLVM:   %[[ACQUIRE:.*]] = call i32 @__cxa_guard_acquire(ptr @_ZGVZN8InMember8mem_funcEiiiiE12magic_static)
// LLVM:   %[[IS_UNINIT:.*]] = icmp ne i32 %[[ACQUIRE]], 0
// LLVM:   br i1 %[[IS_UNINIT]]
//
// LLVM:   %[[THREE_LOAD:.*]] = load i32, ptr %[[THREE_ALLOCA]]
// LLVM:   %[[GET_INT:.*]] = call noundef i32 @_ZN8InMember11mem_get_intEv(ptr {{.*}}%[[THIS_LOAD]])
// LLVM:   %[[ADD1:.*]] = add nsw i32 %[[THREE_LOAD]], %[[GET_INT]]
// LLVM:   %[[ONE_LOAD:.*]] = load i32, ptr %[[ONE_ALLOCA]]
// LLVM:   %[[ADD2:.*]] = add nsw i32 %[[ADD1]], %[[ONE_LOAD]]
// LLVM:   %[[LOCAL_LOAD:.*]] = load i32, ptr %[[LOCAL_ALLOCA]]
// LLVM:   %[[ADD3:.*]] = add nsw i32 %[[ADD2]], %[[LOCAL_LOAD]]
// LLVM:   store i32 %[[ADD3]], ptr @_ZZN8InMember8mem_funcEiiiiE12magic_static
// LLVM:   call void @__cxa_guard_release(ptr @_ZGVZN8InMember8mem_funcEiiiiE12magic_static)
//
// LLVM:   ret void
}

void self_ref(int one) {
  static int magic_static = magic_static + one;
// CIR-BOTH-LABEL:  cir.func no_inline dso_local @_Z8self_refi(
// CIR-BOTH:    %[[ONE_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["one", init]
// CIR-BOTH:    %[[GET_MS:.*]] = cir.get_global static_local @_ZZ8self_refiE12magic_static : !cir.ptr<!s32i>
//
// CIR-BEFORE-LPP:    cir.local_init static_local @_ZZ8self_refiE12magic_static ctor {
//
// CIR: %[[GET_GUARD:.*]] = cir.get_global @_ZGVZ8self_refiE12magic_static : !cir.ptr<!s64i>
// CIR: %[[GUARD_BYTE_PTR:.*]] = cir.cast bitcast %[[GET_GUARD]] : !cir.ptr<!s64i> -> !cir.ptr<!s8i>
// CIR: %[[GUARD_LOAD:.*]] = cir.load{{.*}} syncscope(system) atomic(acquire) %[[GUARD_BYTE_PTR]] : !cir.ptr<!s8i>, !s8i
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s8i
// CIR: %[[IS_UNINIT:.*]] = cir.cmp eq %[[GUARD_LOAD]], %[[ZERO]] : !s8i
// CIR: cir.if %[[IS_UNINIT]] {
// CIR:   %[[ACQUIRE:.*]] = cir.call @__cxa_guard_acquire(%[[GET_GUARD]]) : (!cir.ptr<!s64i>) -> !s32i
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   %[[IS_UNINIT:.*]] = cir.cmp ne %[[ACQUIRE]], %[[ZERO]] : !s32i
// CIR:   cir.if %[[IS_UNINIT]] {
//
// CIR-BOTH:      %[[GET_MS_INIT:.*]] = cir.get_global static_local @_ZZ8self_refiE12magic_static : !cir.ptr<!s32i>
// CIR-BOTH:      %[[GET_MS_LOAD:.*]] = cir.load {{.*}}%[[GET_MS]] : !cir.ptr<!s32i>, !s32i
// CIR-BOTH:      %[[ONE_LOAD:.*]] = cir.load {{.*}}%[[ONE_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR-BOTH:      %[[ADD:.*]] = cir.add nsw %[[GET_MS_LOAD]], %[[ONE_LOAD]] : !s32i
// CIR-BOTH:      cir.store {{.*}}%[[ADD]], %[[GET_MS_INIT]] : !s32i, !cir.ptr<!s32i>
//
// CIR-BEFORE-LPP:      cir.yield
// CIR-BEFORE-LPP:    }
//
// CIR:     cir.call @__cxa_guard_release(%[[GET_GUARD]]) : (!cir.ptr<!s64i>) -> ()
// CIR:   }
// CIR: }
//
// CIR-BOTH:    cir.return
//
// LLVM-LABEL: define dso_local void @_Z8self_refi(
// LLVM:  %[[ONE_ALLOCA:.*]] = alloca i32
// LLVM:  %[[GET_GUARD:.*]] = load atomic i8, ptr @_ZGVZ8self_refiE12magic_static acquire
// LLVM:  %[[IS_UNINIT:.*]] = icmp eq i8 %[[GET_GUARD]], 0
// LLVM:  br i1 %[[IS_UNINIT]]
//
// LLVM:  %[[ACQUIRE:.*]] = call i32 @__cxa_guard_acquire(ptr @_ZGVZ8self_refiE12magic_static)
// LLVM:  %[[IS_UNINIT:.*]] = icmp ne i32 %[[ACQUIRE]], 0
// LLVM:  br i1 %[[IS_UNINIT]]
//
// LLVM:  %[[GET_MS:.*]] = load i32, ptr @_ZZ8self_refiE12magic_static
// LLVM:  %[[ONE_LOAD:.*]] = load i32, ptr %[[ONE_ALLOCA]]
// LLVM:  %[[ADD:.*]] = add nsw i32 %[[GET_MS]], %[[ONE_LOAD]]
// LLVM:  store i32 %[[ADD]], ptr @_ZZ8self_refiE12magic_static
// LLVM:  call void @__cxa_guard_release(ptr @_ZGVZ8self_refiE12magic_static)
//
// LLVM:  ret void
}

struct HasDtor {
 ~HasDtor();
};

void test_dtor() {
  static HasDtor dtor;
// CIR-BOTH-LABEL: cir.func no_inline dso_local @_Z9test_dtorv()
// CIR-BOTH: %[[GET_MS:.*]] = cir.get_global static_local @_ZZ9test_dtorvE4dtor : !cir.ptr<!rec_HasDtor>
//
// CIR-BEFORE-LPP: cir.local_init static_local @_ZZ9test_dtorvE4dtor dtor {
//
// CIR: %[[GET_GUARD:.*]] = cir.get_global @_ZGVZ9test_dtorvE4dtor : !cir.ptr<!s64i>
// CIR: %[[GUARD_BYTE_PTR:.*]] = cir.cast bitcast %[[GET_GUARD]] : !cir.ptr<!s64i> -> !cir.ptr<!s8i>
// CIR: %[[GUARD_LOAD:.*]] = cir.load {{.*}}syncscope(system) atomic(acquire) %[[GUARD_BYTE_PTR]] : !cir.ptr<!s8i>, !s8i
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s8i
// CIR: %[[IS_UNINIT:.*]] = cir.cmp eq %[[GUARD_LOAD]], %[[ZERO]] : !s8i
// CIR: cir.if %[[IS_UNINIT]] {
// CIR:   %[[ACQUIRE:.*]] = cir.call @__cxa_guard_acquire(%[[GET_GUARD]]) : (!cir.ptr<!s64i>) -> !s32i
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   %[[IS_UNINIT:.*]] = cir.cmp ne %[[ACQUIRE]], %[[ZERO]] : !s32i
// CIR:   cir.if %[[IS_UNINIT]] {
//
// CIR-BEFORE-LPP:      %[[GET_MS_INIT:.*]] = cir.get_global static_local @_ZZ9test_dtorvE4dtor : !cir.ptr<!rec_HasDtor>
// CIR-BEFORE-LPP:      cir.call @_ZN7HasDtorD1Ev(%[[GET_MS_INIT]]) : (!cir.ptr<!rec_HasDtor>) -> ()
// CIR-BEFORE-LPP:      cir.yield
// CIR-BEFORE-LPP:    }
//
// CIR:    %[[GET_MS_DEL:.*]] = cir.get_global static_local @_ZZ9test_dtorvE4dtor : !cir.ptr<!rec_HasDtor>
// CIR:    %[[GET_DTOR:.*]] = cir.get_global @_ZN7HasDtorD1Ev : !cir.ptr<!cir.func<(!cir.ptr<!rec_HasDtor>)>>
// CIR:    %[[DTOR_DECAY:.*]] = cir.cast bitcast %[[GET_DTOR]] : !cir.ptr<!cir.func<(!cir.ptr<!rec_HasDtor>)>> -> !cir.ptr<!cir.func<(!cir.ptr<!void>)>>
// CIR:    %[[MS_DECAY:.*]] = cir.cast bitcast %[[GET_MS_DEL]] : !cir.ptr<!rec_HasDtor> -> !cir.ptr<!void>
// CIR:    %[[DSO_HANDLE:.*]] = cir.get_global @__dso_handle : !cir.ptr<i8>
// CIR:    cir.call @__cxa_atexit(%[[DTOR_DECAY]], %[[MS_DECAY]], %[[DSO_HANDLE]]) : (!cir.ptr<!cir.func<(!cir.ptr<!void>)>>, !cir.ptr<!void>, !cir.ptr<i8>) -> ()
// CIR:    cir.call @__cxa_guard_release(%[[GET_GUARD]]) : (!cir.ptr<!s64i>) -> ()

// CIR:   }
// CIR: }
// CIR-BOTH: cir.return
//
// LLVM-LABEL: define dso_local void @_Z9test_dtorv()
// LLVM: %[[GET_GUARD:.*]] = load atomic i8, ptr @_ZGVZ9test_dtorvE4dtor acquire
// LLVM: %[[IS_UNINIT:.*]] = icmp eq i8 %[[GET_GUARD]], 0
// LLVM: br i1 %[[IS_UNINIT]]
//
// LLVM: %[[ACQUIRE:.*]] = call i32 @__cxa_guard_acquire(ptr @_ZGVZ9test_dtorvE4dtor)
// LLVM: %[[IS_UNINIT:.*]] = icmp ne i32 %[[ACQUIRE]], 0
// LLVM: br i1 %[[IS_UNINIT]]
//
// LLVM: call {{.*}}@__cxa_atexit(ptr @_ZN7HasDtorD1Ev, ptr @_ZZ9test_dtorvE4dtor, ptr @__dso_handle)
// LLVM: call void @__cxa_guard_release(ptr @_ZGVZ9test_dtorvE4dtor)
// LLVM: ret void
}

struct HasCtorDtor {
 HasCtorDtor();
 ~HasCtorDtor();
};

void test_ctor_dtor() {
  static HasCtorDtor ctor_dtor;
// CIR-BOTH-LABEL: cir.func no_inline dso_local @_Z14test_ctor_dtorv()
// CIR-BOTH:   %[[GET_MS:.*]] = cir.get_global static_local @_ZZ14test_ctor_dtorvE9ctor_dtor : !cir.ptr<!rec_HasCtorDtor>
// CIR-BEFORE-LPP:   cir.local_init static_local @_ZZ14test_ctor_dtorvE9ctor_dtor ctor {
//
// CIR: %[[GET_GUARD:.*]] = cir.get_global @_ZGVZ14test_ctor_dtorvE9ctor_dtor : !cir.ptr<!s64i>
// CIR: %[[GUARD_BYTE_PTR:.*]] = cir.cast bitcast %[[GET_GUARD]] : !cir.ptr<!s64i> -> !cir.ptr<!s8i>
// CIR: %[[GUARD_LOAD:.*]] = cir.load {{.*}}syncscope(system) atomic(acquire) %[[GUARD_BYTE_PTR]] : !cir.ptr<!s8i>, !s8i
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s8i
// CIR: %[[IS_UNINIT:.*]] = cir.cmp eq %[[GUARD_LOAD]], %[[ZERO]] : !s8i
// CIR: cir.if %[[IS_UNINIT]] {
// CIR:   %[[ACQUIRE:.*]] = cir.call @__cxa_guard_acquire(%[[GET_GUARD]]) : (!cir.ptr<!s64i>) -> !s32i
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   %[[IS_UNINIT:.*]] = cir.cmp ne %[[ACQUIRE]], %[[ZERO]] : !s32i
// CIR:   cir.if %[[IS_UNINIT]] {
//
// CIR-BOTH:     %[[GET_MS_INIT:.*]] = cir.get_global static_local @_ZZ14test_ctor_dtorvE9ctor_dtor : !cir.ptr<!rec_HasCtorDtor>
// CIR-BOTH:     cir.call @_ZN11HasCtorDtorC1Ev(%[[GET_MS_INIT]]) : (!cir.ptr<!rec_HasCtorDtor> {{.*}}) -> ()
//
//
// CIR-BEFORE-LPP:     cir.yield
// CIR-BEFORE-LPP:   } dtor {
// CIR-BEFORE-LPP:     %[[GET_MS_INIT:.*]] = cir.get_global static_local @_ZZ14test_ctor_dtorvE9ctor_dtor : !cir.ptr<!rec_HasCtorDtor>
// CIR-BEFORE-LPP:     cir.call @_ZN11HasCtorDtorD1Ev(%[[GET_MS_INIT]]) : (!cir.ptr<!rec_HasCtorDtor>) -> ()
// CIR-BEFORE-LPP:     cir.yield
// CIR-BEFORE-LPP:   }
//
// CIR:    %[[GET_MS_DEL:.*]] = cir.get_global static_local @_ZZ14test_ctor_dtorvE9ctor_dtor : !cir.ptr<!rec_HasCtorDtor>
// CIR:    %[[GET_DTOR:.*]] = cir.get_global @_ZN11HasCtorDtorD1Ev : !cir.ptr<!cir.func<(!cir.ptr<!rec_HasCtorDtor>)>>
// CIR:    %[[DTOR_DECAY:.*]] = cir.cast bitcast %[[GET_DTOR]] : !cir.ptr<!cir.func<(!cir.ptr<!rec_HasCtorDtor>)>> -> !cir.ptr<!cir.func<(!cir.ptr<!void>)>>
// CIR:    %[[MS_DECAY:.*]] = cir.cast bitcast %[[GET_MS_DEL]] : !cir.ptr<!rec_HasCtorDtor> -> !cir.ptr<!void>
// CIR:    %[[DSO_HANDLE:.*]] = cir.get_global @__dso_handle : !cir.ptr<i8>
// CIR:    cir.call @__cxa_atexit(%[[DTOR_DECAY]], %[[MS_DECAY]], %[[DSO_HANDLE]]) : (!cir.ptr<!cir.func<(!cir.ptr<!void>)>>, !cir.ptr<!void>, !cir.ptr<i8>) -> ()
// CIR:    cir.call @__cxa_guard_release(%[[GET_GUARD]]) : (!cir.ptr<!s64i>) -> ()
// CIR:   }
// CIR: }
//
// CIR-BOTH:   cir.return
//
// LLVM-LABEL: define dso_local void @_Z14test_ctor_dtorv()
// LLVM: %[[GET_GUARD:.*]] = load atomic i8, ptr @_ZGVZ14test_ctor_dtorvE9ctor_dtor acquire
// LLVM: %[[IS_UNINIT:.*]] = icmp eq i8 %[[GET_GUARD]], 0
// LLVM: br i1 %[[IS_UNINIT]]
//
// LLVM: %[[ACQUIRE:.*]] = call i32 @__cxa_guard_acquire(ptr @_ZGVZ14test_ctor_dtorvE9ctor_dtor)
// LLVM: %[[IS_UNINIT:.*]] = icmp ne i32 %[[ACQUIRE]], 0
// LLVM: br i1 %[[IS_UNINIT]]
//
// LLVM:   call void @_ZN11HasCtorDtorC1Ev(ptr {{.*}}@_ZZ14test_ctor_dtorvE9ctor_dtor)
// LLVM:   call {{.*}}@__cxa_atexit(ptr @_ZN11HasCtorDtorD1Ev, ptr @_ZZ14test_ctor_dtorvE9ctor_dtor, ptr @__dso_handle)
// LLVM:   call void @__cxa_guard_release(ptr @_ZGVZ14test_ctor_dtorvE9ctor_dtor)
//
// LLVM:   ret void
}

int referenced_inside() {
  static int static_local = bar();
  auto lam = []() { return static_local; };
  return lam();
// CIR-BOTH-LABEL: cir.func no_inline lambda internal private dso_local @_ZZ17referenced_insidevENK3$_0clEv(
// CIR-BOTH:   %[[THIS_ALLOCA:.*]] = cir.alloca !cir.ptr<!{{.*}}>, !cir.ptr<!cir.ptr<!{{.*}}>>, ["this", init]
// CIR-BOTH:   %[[RET_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR-BOTH:   %[[LOAD_THIS:.*]] = cir.load %[[THIS_ALLOCA]] : !cir.ptr<!cir.ptr<!{{.*}}>>, !cir.ptr<!{{.*}}>
// CIR-BOTH:   %[[GET_SL:.*]] = cir.get_global static_local @_ZZ17referenced_insidevE12static_local : !cir.ptr<!s32i>
// CIR-BOTH:   %[[LOAD_SL:.*]] = cir.load {{.*}}%[[GET_SL]] : !cir.ptr<!s32i>, !s32i
// CIR-BOTH:   cir.store %[[LOAD_SL]], %[[RET_ALLOCA]] : !s32i, !cir.ptr<!s32i>
// CIR-BOTH:   %[[LOAD_RET:.*]] = cir.load %[[RET_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR-BOTH:   cir.return %[[LOAD_RET]] : !s32i
// CIR-BOTH: }
//
// CIR-BOTH-LABEL: cir.func no_inline dso_local @_Z17referenced_insidev()
// CIR-BOTH:   %[[RET_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR-BOTH:   %[[LAMBDA_ALLOCA:.*]] = cir.alloca !{{.*}}, !cir.ptr<!{{.*}}>, ["lam"]
// CIR-BOTH:   %[[GET_SL:.*]] = cir.get_global static_local @_ZZ17referenced_insidevE12static_local : !cir.ptr<!s32i>
// CIR-BEFORE-LLP:   cir.local_init static_local @_ZZ17referenced_insidevE12static_local ctor {
//
// CIR: %[[GET_GUARD:.*]] = cir.get_global @_ZGVZ17referenced_insidevE12static_local : !cir.ptr<!s64i>
// CIR: %[[GUARD_BYTE_PTR:.*]] = cir.cast bitcast %[[GET_GUARD]] : !cir.ptr<!s64i> -> !cir.ptr<!s8i>
// CIR: %[[GUARD_LOAD:.*]] = cir.load {{.*}}syncscope(system) atomic(acquire) %[[GUARD_BYTE_PTR]] : !cir.ptr<!s8i>, !s8i
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s8i
// CIR: %[[IS_UNINIT:.*]] = cir.cmp eq %[[GUARD_LOAD]], %[[ZERO]] : !s8i
// CIR: cir.if %[[IS_UNINIT]] {
// CIR:   %[[ACQUIRE:.*]] = cir.call @__cxa_guard_acquire(%[[GET_GUARD]]) : (!cir.ptr<!s64i>) -> !s32i
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   %[[IS_UNINIT:.*]] = cir.cmp ne %[[ACQUIRE]], %[[ZERO]] : !s32i
// CIR:   cir.if %[[IS_UNINIT]] {
//
// CIR-BOTH:     %[[GET_SL_INIT:.*]] = cir.get_global static_local @_ZZ17referenced_insidevE12static_local : !cir.ptr<!s32i>
// CIR-BOTH:     %[[CALL:.*]] = cir.call @_Z3barv() : () -> (!s32i {llvm.noundef})
// CIR-BOTH:     cir.store {{.*}} %[[CALL]], %[[GET_SL_INIT]] : !s32i, !cir.ptr<!s32i>
//
// CIR-BEFORE-LLP:     cir.yield
// CIR-BEFORE-LLP:   }
//
// CIR:   }
// CIR: }
//
// CIR-BOTH:   %[[CTOR:.*]] = cir.call @_ZZ17referenced_insidevENK3$_0clEv(%[[LAMBDA_ALLOCA]])
// CIR-BOTH:   cir.store %[[CTOR]], %[[RET_ALLOCA]] : !s32i, !cir.ptr<!s32i>
// CIR-BOTH:   %[[RET_LOAD:.*]] = cir.load %[[RET_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR-BOTH:   cir.return %[[RET_LOAD]] : !s32i
// CIR-BOTH: }
//

// LLVM-CIR-LABEL: define internal noundef i32 @"_ZZ17referenced_insidevENK3$_0clEv"(
// LLVM-CIR:   load i32, ptr @_ZZ17referenced_insidevE12static_local
//
// LLVM-CIR-LABEL: define dso_local noundef i32 @_Z17referenced_insidev()
// LLVM-CIR: %[[GET_GUARD:.*]] = load atomic i8, ptr @_ZGVZ17referenced_insidevE12static_local acquire
// LLVM-CIR: %[[IS_UNINIT:.*]] = icmp eq i8 %[[GET_GUARD]], 0
// LLVM-CIR: br i1 %[[IS_UNINIT]]
//
// LLVM-CIR: %[[ACQUIRE:.*]] = call i32 @__cxa_guard_acquire(ptr @_ZGVZ17referenced_insidevE12static_local)
// LLVM-CIR: %[[IS_UNINIT:.*]] = icmp ne i32 %[[ACQUIRE]], 0
// LLVM-CIR: br i1 %[[IS_UNINIT]]
//
// LLVM-CIR: %[[CALL:.*]] = call noundef i32 @_Z3barv()
// LLVM-CIR: store i32 %[[CALL]], ptr @_ZZ17referenced_insidevE12static_local
// LLVM-CIR: call void @__cxa_guard_release(ptr @_ZGVZ17referenced_insidevE12static_local)
//
// OGCG-LABEL: define dso_local noundef i32 @_Z17referenced_insidev()
// OGCG: %[[GET_GUARD:.*]] = load atomic i8, ptr @_ZGVZ17referenced_insidevE12static_local acquire
// OGCG: %[[IS_UNINIT:.*]] = icmp eq i8 %[[GET_GUARD]], 0
// OGCG: br i1 %[[IS_UNINIT]]
//
// OGCG: %[[ACQUIRE:.*]] = call i32 @__cxa_guard_acquire(ptr @_ZGVZ17referenced_insidevE12static_local)
// OGCG: %[[IS_UNINIT:.*]] = icmp ne i32 %[[ACQUIRE]], 0
// OGCG: br i1 %[[IS_UNINIT]]
//
// OGCG: %[[CALL:.*]] = call noundef i32 @_Z3barv()
// OGCG: store i32 %[[CALL]], ptr @_ZZ17referenced_insidevE12static_local
// OGCG: call void @__cxa_guard_release(ptr @_ZGVZ17referenced_insidevE12static_local)
//
// OGCG-LABEL: define internal noundef i32 @"_ZZ17referenced_insidevENK3$_0clEv"(
// OGCG:   load i32, ptr @_ZZ17referenced_insidevE12static_local
}

int referenced_inside_const() {
  static int static_local = 42;
  auto lam = []() { return static_local; };
  return lam();
// CIR-BOTH-LABEL: cir.func no_inline lambda internal private dso_local @_ZZ23referenced_inside_constvENK3$_0clEv(
// CIR-BOTH:   %[[THIS_ALLOCA:.*]] = cir.alloca !cir.ptr<!{{.*}}>, !cir.ptr<!cir.ptr<!{{.*}}>>, ["this", init]
// CIR-BOTH:   %[[RET_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR-BOTH:   %[[LOAD_THIS:.*]] = cir.load %[[THIS_ALLOCA]] : !cir.ptr<!cir.ptr<!{{.*}}>>, !cir.ptr<!{{.*}}>
// CIR-BOTH:   %[[GET_SL:.*]] = cir.get_global @_ZZ23referenced_inside_constvE12static_local : !cir.ptr<!s32i>
// CIR-BOTH:   %[[LOAD_SL:.*]] = cir.load {{.*}}%[[GET_SL]] : !cir.ptr<!s32i>, !s32i
// CIR-BOTH:   cir.store %[[LOAD_SL]], %[[RET_ALLOCA]] : !s32i, !cir.ptr<!s32i>
// CIR-BOTH:   %[[LOAD_RET:.*]] = cir.load %[[RET_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR-BOTH:   cir.return %[[LOAD_RET]] : !s32i
// CIR-BOTH: }
//
// CIR-BOTH-LABEL: cir.func no_inline dso_local @_Z23referenced_inside_constv()
// CIR-BOTH:   %[[RET_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR-BOTH:   %[[LAMBDA_ALLOCA:.*]] = cir.alloca !{{.*}}, !cir.ptr<!{{.*}}>, ["lam"]
// CIR-BOTH:   %[[GET_SL:.*]] = cir.get_global @_ZZ23referenced_inside_constvE12static_local : !cir.ptr<!s32i>
// CIR-BOTH-NOT: static_local
// CIR-BOTH-NOT: atomic 
// CIR-BOTH:   %[[CTOR:.*]] = cir.call @_ZZ23referenced_inside_constvENK3$_0clEv(%[[LAMBDA_ALLOCA]])
// CIR-BOTH:   cir.store %[[CTOR]], %[[RET_ALLOCA]] : !s32i, !cir.ptr<!s32i>
// CIR-BOTH:   %[[RET_LOAD:.*]] = cir.load %[[RET_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR-BOTH:   cir.return %[[RET_LOAD]] : !s32i
// CIR-BOTH: }

// LLVM-CIR-LABEL: define internal noundef i32 @"_ZZ23referenced_inside_constvENK3$_0clEv"(
// LLVM-CIR:   load i32, ptr @_ZZ23referenced_inside_constvE12static_local
//
// LLVM-CIR-LABEL: define dso_local noundef i32 @_Z23referenced_inside_constv()
// LLVM-CIR-NOT: atomic
// LLVM-CIR: call noundef i32 @"_ZZ23referenced_inside_constvENK3$_0clEv"(ptr 
//
// OGCG-LABEL: define dso_local noundef i32 @_Z23referenced_inside_constv()
// OGCG-NOT: atomic
// OGCG: call noundef i32 @"_ZZ23referenced_inside_constvENK3$_0clEv"(ptr 
//
// OGCG-LABEL: define internal noundef i32 @"_ZZ23referenced_inside_constvENK3$_0clEv"(
// OGCG:   load i32, ptr @_ZZ23referenced_inside_constvE12static_local
}

// Reference-typed static local with a non-constant initializer. The
// cir.get_global emitted inside the initializer region for the reference
// must carry the static_local marker so that it matches the
// static_local_guard attribute on the corresponding cir.global. Otherwise
// the cir.get_global verifier rejects the IR with
// "static_local attribute mismatch".
int g = 5;
int &source();
int &ref_init() {
  static int &y = source();
  return y;
// CIR-BOTH-LABEL: cir.func no_inline dso_local @_Z8ref_initv()
// CIR-BOTH:    %[[GET_REF:.*]] = cir.get_global static_local @_ZZ8ref_initvE1y : !cir.ptr<!cir.ptr<!s32i>>
//
// CIR-BEFORE-LPP:    cir.local_init static_local @_ZZ8ref_initvE1y ctor {
// CIR-BEFORE-LPP:      %[[GET_REF_INIT:.*]] = cir.get_global static_local @_ZZ8ref_initvE1y : !cir.ptr<!cir.ptr<!s32i>>
// CIR-BEFORE-LPP:      %[[CALL_SOURCE:.*]] = cir.call @_Z6sourcev() : () -> (!cir.ptr<!s32i>{{.*}})
// CIR-BEFORE-LPP:      cir.store {{.*}}%[[CALL_SOURCE]], %[[GET_REF_INIT]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR-BEFORE-LPP:      cir.yield
// CIR-BEFORE-LPP:    }
//
// CIR: %[[GET_GUARD:.*]] = cir.get_global @_ZGVZ8ref_initvE1y : !cir.ptr<!s64i>
// CIR: cir.if
// CIR:   cir.call @__cxa_guard_acquire(%[[GET_GUARD]])
// CIR:   cir.if
// CIR:     %[[GET_REF_INIT2:.*]] = cir.get_global static_local @_ZZ8ref_initvE1y : !cir.ptr<!cir.ptr<!s32i>>
// CIR:     %[[CALL_SOURCE2:.*]] = cir.call @_Z6sourcev() : () -> (!cir.ptr<!s32i>{{.*}})
// CIR:     cir.store {{.*}}%[[CALL_SOURCE2]], %[[GET_REF_INIT2]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR:     cir.call @__cxa_guard_release(%[[GET_GUARD]])
// CIR:   }
// CIR: }
// CIR-BOTH:    cir.return
//
// LLVM-LABEL: define dso_local {{.*}} @_Z8ref_initv()
// LLVM:  %[[GET_GUARD:.*]] = load atomic i8, ptr @_ZGVZ8ref_initvE1y acquire
// LLVM:  call i32 @__cxa_guard_acquire(ptr @_ZGVZ8ref_initvE1y)
// LLVM:  %[[CALL_SOURCE:.*]] = call {{.*}}ptr @_Z6sourcev()
// LLVM:  store ptr %[[CALL_SOURCE]], ptr @_ZZ8ref_initvE1y
// LLVM:  call void @__cxa_guard_release(ptr @_ZGVZ8ref_initvE1y)
}

// Static local array with a non-trivial destructor. The cir.get_global
// emitted inside the dtor region for the array must carry the static_local
// marker so that it matches the static_local_guard attribute on the
// corresponding cir.global. Otherwise the cir.get_global verifier rejects
// the IR with "static_local attribute mismatch".
void array_static_local_dtor() {
  static HasCtorDtor sm[2];
// CIR-BOTH-LABEL: cir.func no_inline dso_local @_Z23array_static_local_dtorv()
// CIR-BOTH:    %[[GET_ARR:.*]] = cir.get_global static_local @_ZZ23array_static_local_dtorvE2sm : !cir.ptr<!cir.array<!rec_HasCtorDtor x 2>>
//
// CIR-BEFORE-LPP:    cir.local_init static_local @_ZZ23array_static_local_dtorvE2sm ctor {
// CIR-BEFORE-LPP:    } dtor {
// CIR-BEFORE-LPP:      %[[GET_ARR_DTOR:.*]] = cir.get_global static_local @_ZZ23array_static_local_dtorvE2sm : !cir.ptr<!cir.array<!rec_HasCtorDtor x 2>>
// CIR-BEFORE-LPP:      cir.array.dtor %[[GET_ARR_DTOR]]
// CIR-BEFORE-LPP:        cir.call @_ZN11HasCtorDtorD1Ev
// CIR-BEFORE-LPP:      cir.yield
// CIR-BEFORE-LPP:    }
//
// CIR: %[[GET_GUARD:.*]] = cir.get_global @_ZGVZ23array_static_local_dtorvE2sm : !cir.ptr<!s64i>
// CIR: cir.if
// CIR:   cir.call @__cxa_guard_acquire(%[[GET_GUARD]])
// CIR:   cir.if
// CIR:     %[[GET_ARR_INIT:.*]] = cir.get_global static_local @_ZZ23array_static_local_dtorvE2sm : !cir.ptr<!cir.array<!rec_HasCtorDtor x 2>>
// CIR:     cir.call @__cxa_atexit
// CIR:     cir.call @__cxa_guard_release(%[[GET_GUARD]])
// CIR:   }
// CIR: }
// CIR-BOTH:    cir.return
}
