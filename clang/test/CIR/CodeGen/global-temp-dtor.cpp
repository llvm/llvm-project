// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --input-file=%t-before.cir %s --check-prefixes=CIR-BEFORE
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=LLVM,OGCG

// Exercises lifetime-extended reference temporaries with non-trivial
// destructors where the extending declaration has static or thread storage
// duration, for both non-array and array temporary types.

struct NonTrivial {
  ~NonTrivial();
  int x;
};

const NonTrivial &static_ref = NonTrivial();
thread_local const NonTrivial &thread_ref = NonTrivial();

typedef NonTrivial NonTrivialArr[2];

const NonTrivialArr &static_arr_ref = NonTrivialArr{};
thread_local const NonTrivialArr &thread_arr_ref = NonTrivialArr{};

//===----------------------------------------------------------------------===//
// CIR dialect (post-LoweringPrepare): each temporary is emitted as a private
// internal cir.global, with its destructor registered through __cxa_atexit
// or __cxa_thread_atexit. Arrays go through a generated
// __cxx_global_array_dtor helper. The runtime helpers themselves are
// declared exactly once.
//===----------------------------------------------------------------------===//

// CIR-BEFORE: cir.global external @static_ref = ctor : !cir.ptr<!rec_NonTrivial> {
// CIR-BEFORE:   %[[STATIC_REF:.*]] = cir.get_global @static_ref
// CIR-BEFORE:   %[[REF_TEMP:.*]] = cir.get_global @_ZGR10static_ref_
// CIR-BEFORE:   %[[ZERO:.*]] = cir.const #cir.zero : !rec_NonTrivial
// CIR-BEFORE:   cir.store{{.*}} %[[ZERO]], %[[REF_TEMP]]
// CIR-BEFORE:   cir.store{{.*}} %[[REF_TEMP]], %[[STATIC_REF]]
// CIR-BEFORE: }

// CIR-BEFORE: cir.global "private" internal @_ZGR10static_ref_ = #cir.zero : !rec_NonTrivial dtor {
// CIR-BEFORE:   %[[REF_TEMP:.*]] = cir.get_global @_ZGR10static_ref_
// CIR-BEFORE:   cir.call @_ZN10NonTrivialD1Ev(%[[REF_TEMP]])
// CIR-BEFORE: }

// CIR: cir.global "private" internal @_ZGR10static_ref_ = #cir.zero : !rec_NonTrivial
// CIR: cir.func internal private @__cxx_global_var_init.1()
// CIR:   cir.get_global @_ZGR10static_ref_
// CIR:   cir.get_global @_ZN10NonTrivialD1Ev
// CIR:   cir.get_global @__dso_handle
// CIR:   cir.call @__cxa_atexit

// CIR-BEFORE: cir.global external tls_dyn dyn_tls_refs = <"_ZTW10thread_ref", "_ZTH10thread_ref"> @thread_ref = ctor : !cir.ptr<!rec_NonTrivial> {
// CIR-BEFORE:   %[[THREAD_REF:.*]] = cir.get_global thread_local @thread_ref
// CIR-BEFORE:   %[[REF_TEMP:.*]] = cir.get_global @_ZGR10thread_ref_
// CIR-BEFORE:   %[[ZERO:.*]] = cir.const #cir.zero : !rec_NonTrivial
// CIR-BEFORE:   cir.store{{.*}} %[[ZERO]], %[[REF_TEMP]]
// CIR-BEFORE:   cir.store{{.*}} %[[REF_TEMP]], %[[THREAD_REF]]
// CIR-BEFORE: }

// CIR-BEFORE: cir.global "private" internal tls_dyn @_ZGR10thread_ref_ = #cir.zero : !rec_NonTrivial dtor {
// CIR-BEFORE:   %[[REF_TEMP:.*]] = cir.get_global @_ZGR10thread_ref_
// CIR-BEFORE:   cir.call @_ZN10NonTrivialD1Ev(%[[REF_TEMP]])
// CIR-BEFORE: }

// CIR: cir.global "private" internal tls_dyn @_ZGR10thread_ref_ = #cir.zero : !rec_NonTrivial
// CIR: cir.func internal private @__cxx_global_var_init.3()
// CIR:   cir.get_global @_ZGR10thread_ref_
// CIR:   cir.get_global @_ZN10NonTrivialD1Ev
// CIR:   cir.get_global @__dso_handle
// CIR:   cir.call @__cxa_thread_atexit

// CIR-BEFORE: cir.global external @static_arr_ref = ctor : !cir.ptr<!cir.array<!rec_NonTrivial x 2>> {
// CIR-BEFORE:   %[[ARRAY_INIT_TEMP:.*]] = cir.alloca {{.*}}"arrayinit.temp"
// CIR-BEFORE:   %[[STATIC_ARR_REF:.*]] = cir.get_global @static_arr_ref
// CIR-BEFORE:   %[[STATIC_ARR_REF_TEMP:.*]] = cir.get_global @_ZGR14static_arr_ref_
// CIR-BEFORE:   %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[STATIC_ARR_REF_TEMP]]
// CIR-BEFORE:   cir.store{{.*}} %[[DECAY]], %[[ARRAY_INIT_TEMP]]
// CIR-BEFORE:   %[[TWO:.*]] = cir.const #cir.int<2> : !s64i
// CIR-BEFORE:   %[[NEXT:.*]] = cir.ptr_stride %[[DECAY]], %[[TWO]] : (!cir.ptr<!rec_NonTrivial>, !s64i) -> !cir.ptr<!rec_NonTrivial>
// CIR-BEFORE:   cir.do {
// CIR-BEFORE:   } while {
// CIR-BEFORE:     cir.condition
// CIR-BEFORE:   }
// CIR-BEFORE:   cir.store{{.*}} %[[STATIC_ARR_REF_TEMP]], %[[STATIC_ARR_REF]] : !cir.ptr<!cir.array<!rec_NonTrivial x 2>>, !cir.ptr<!cir.ptr<!cir.array<!rec_NonTrivial x 2>>>
// CIR-BEFORE: }

// CIR-BEFORE: cir.global "private" internal @_ZGR14static_arr_ref_ = #cir.zero : !cir.array<!rec_NonTrivial x 2> dtor {
// CIR-BEFORE:   %[[STATIC_ARR_REF_TEMP:.*]] = cir.get_global @_ZGR14static_arr_ref_
// CIR-BEFORE:   cir.array.dtor %[[STATIC_ARR_REF_TEMP]] : !cir.ptr<!cir.array<!rec_NonTrivial x 2>> {
// CIR-BEFORE:   ^bb0(%[[ELEMENT:.*]]: !cir.ptr<!rec_NonTrivial>):
// CIR-BEFORE:     cir.call @_ZN10NonTrivialD1Ev(%[[ELEMENT]])
// CIR-BEFORE:   }
// CIR-BEFORE: }

// CIR: cir.global "private" internal @_ZGR14static_arr_ref_ = #cir.zero : !cir.array<!rec_NonTrivial x 2>
// CIR: cir.func internal private @__cxx_global_array_dtor(
// CIR:   cir.call @_ZN10NonTrivialD1Ev

// CIR: cir.func internal private @__cxx_global_var_init.5()
// CIR:   cir.get_global @_ZGR14static_arr_ref_
// CIR:   cir.get_global @__cxx_global_array_dtor
// CIR:   cir.call @__cxa_atexit

// CIR-BEFORE: cir.global external tls_dyn dyn_tls_refs = <"_ZTW14thread_arr_ref", "_ZTH14thread_arr_ref"> @thread_arr_ref = ctor : !cir.ptr<!cir.array<!rec_NonTrivial x 2>> {
// CIR-BEFORE:   %[[ARRAY_INIT_TEMP:.*]] = cir.alloca {{.*}}"arrayinit.temp"
// CIR-BEFORE:   %[[THREAD_ARR_REF:.*]] = cir.get_global thread_local @thread_arr_ref
// CIR-BEFORE:   %[[THREAD_ARR_REF_TEMP:.*]] = cir.get_global @_ZGR14thread_arr_ref_
// CIR-BEFORE:   %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[THREAD_ARR_REF_TEMP]]
// CIR-BEFORE:   cir.store{{.*}} %[[DECAY]], %[[ARRAY_INIT_TEMP]]
// CIR-BEFORE:   %[[TWO:.*]] = cir.const #cir.int<2> : !s64i
// CIR-BEFORE:   %[[NEXT:.*]] = cir.ptr_stride %[[DECAY]], %[[TWO]] : (!cir.ptr<!rec_NonTrivial>, !s64i) -> !cir.ptr<!rec_NonTrivial>
// CIR-BEFORE:   cir.do {
// CIR-BEFORE:   } while {
// CIR-BEFORE:     cir.condition
// CIR-BEFORE:   }
// CIR-BEFORE:   cir.store{{.*}} %[[THREAD_ARR_REF_TEMP]], %[[THREAD_ARR_REF]] : !cir.ptr<!cir.array<!rec_NonTrivial x 2>>, !cir.ptr<!cir.ptr<!cir.array<!rec_NonTrivial x 2>>>
// CIR-BEFORE: }

// CIR-BEFORE: cir.global "private" internal tls_dyn @_ZGR14thread_arr_ref_ = #cir.zero : !cir.array<!rec_NonTrivial x 2> dtor
// CIR-BEFORE:   %[[THREAD_ARR_REF_TEMP:.*]] = cir.get_global @_ZGR14thread_arr_ref_
// CIR-BEFORE:   cir.array.dtor %[[THREAD_ARR_REF_TEMP]] : !cir.ptr<!cir.array<!rec_NonTrivial x 2>> {
// CIR-BEFORE:   ^bb0(%[[ELEMENT:.*]]: !cir.ptr<!rec_NonTrivial>):
// CIR-BEFORE:     cir.call @_ZN10NonTrivialD1Ev(%[[ELEMENT]])
// CIR-BEFORE:   }

// CIR: cir.global "private" internal tls_dyn @_ZGR14thread_arr_ref_ = #cir.zero : !cir.array<!rec_NonTrivial x 2>
// CIR: cir.func internal private @__cxx_global_array_dtor.1(
// CIR:   cir.call @_ZN10NonTrivialD1Ev

// CIR: cir.func internal private @__cxx_global_var_init.7()
// CIR:   cir.get_global @_ZGR14thread_arr_ref_
// CIR:   cir.get_global @__cxx_global_array_dtor.1
// CIR:   cir.call @__cxa_thread_atexit

//===----------------------------------------------------------------------===//
// LLVM IR: the shared declarations are identical between both pipelines.
//===----------------------------------------------------------------------===//

// LLVM-DAG: @static_ref = global ptr null
// LLVM-DAG: @_ZGR10static_ref_ = internal global %struct.NonTrivial zeroinitializer
// LLVM-DAG: @thread_ref = thread_local global ptr null
// LLVM-DAG: @_ZGR10thread_ref_ = internal thread_local global %struct.NonTrivial zeroinitializer
// LLVM-DAG: @static_arr_ref = global ptr null
// LLVM-DAG: @_ZGR14static_arr_ref_ = internal global [2 x %struct.NonTrivial] zeroinitializer
// LLVM-DAG: @thread_arr_ref = thread_local global ptr null
// LLVM-DAG: @_ZGR14thread_arr_ref_ = internal thread_local global [2 x %struct.NonTrivial] zeroinitializer

//===----------------------------------------------------------------------===//
// LLVM IR: function-body shape diverges between the pipelines. OGCG emits a
// single __cxx_global_var_init per reference variable that initializes the
// temporary, registers the cleanup and stores into the reference. CIR-lowered
// IR splits the binding code and the cleanup registration across two
// __cxx_global_var_init functions per reference.
//===----------------------------------------------------------------------===//

// Static, non-array.

// LLVMCIR-LABEL: define internal void @__cxx_global_var_init()
// LLVMCIR:         store %struct.NonTrivial zeroinitializer, ptr @_ZGR10static_ref_
// LLVMCIR:         store ptr @_ZGR10static_ref_, ptr @static_ref

// LLVMCIR-LABEL: define internal void @__cxx_global_var_init.1()
// LLVMCIR:         call void @__cxa_atexit(ptr @_ZN10NonTrivialD1Ev, ptr @_ZGR10static_ref_, ptr @__dso_handle)

// OGCG-LABEL: define internal void @__cxx_global_var_init()
// OGCG:         call void @llvm.memset.{{[^(]+}}(ptr {{.*}}@_ZGR10static_ref_, i8 0, i64 4, i1 false)
// OGCG:         call i32 @__cxa_atexit(ptr @_ZN10NonTrivialD1Ev, ptr @_ZGR10static_ref_, ptr @__dso_handle)
// OGCG:         store ptr @_ZGR10static_ref_, ptr @static_ref

// Thread, non-array.

// LLVMCIR-LABEL: define internal void @__cxx_global_var_init.2()
// LLVMCIR:         call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@thread_ref)
// LLVMCIR:         store %struct.NonTrivial zeroinitializer, ptr @_ZGR10thread_ref_
// LLVMCIR:         store ptr @_ZGR10thread_ref_

// LLVMCIR-LABEL: define internal void @__cxx_global_var_init.3()
// LLVMCIR:         call void @__cxa_thread_atexit(ptr @_ZN10NonTrivialD1Ev, ptr @_ZGR10thread_ref_, ptr @__dso_handle)

// OGCG-LABEL: define internal void @__cxx_global_var_init.1()
// OGCG:         call void @llvm.memset.{{[^(]+}}(ptr {{.*}}@_ZGR10thread_ref_, i8 0, i64 4, i1 false)
// OGCG:         call i32 @__cxa_thread_atexit(ptr @_ZN10NonTrivialD1Ev, ptr @_ZGR10thread_ref_, ptr @__dso_handle)
// OGCG:         %[[THREAD_REF_ADDR:.*]] = call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@thread_ref)
// OGCG:         store ptr @_ZGR10thread_ref_, ptr %[[THREAD_REF_ADDR]]

// Static, array: a generated array-destroy helper is registered with
// __cxa_atexit instead of the destructor itself. CIR passes the array
// pointer as the second argument; OGCG passes null and the helper hard-codes
// the global reference.

// LLVMCIR-LABEL: define internal void @__cxx_global_array_dtor(ptr {{.*}})
// LLVMCIR:         call void @_ZN10NonTrivialD1Ev(ptr

// LLVMCIR-LABEL: define internal void @__cxx_global_var_init.5()
// LLVMCIR:         call void @__cxa_atexit(ptr @__cxx_global_array_dtor, ptr @_ZGR14static_arr_ref_, ptr @__dso_handle)

// OGCG-LABEL: define internal void @__cxx_global_var_init.2()
// OGCG:         call i32 @__cxa_atexit(ptr @__cxx_global_array_dtor, ptr null, ptr @__dso_handle)
// OGCG:         store ptr @_ZGR14static_arr_ref_, ptr @static_arr_ref

// OGCG-LABEL: define internal void @__cxx_global_array_dtor(ptr {{.*}})
// OGCG:         call void @_ZN10NonTrivialD1Ev(ptr

// Thread, array.

// LLVMCIR-LABEL: define internal void @__cxx_global_array_dtor.1(ptr {{.*}})
// LLVMCIR:         call void @_ZN10NonTrivialD1Ev(ptr

// LLVMCIR-LABEL: define internal void @__cxx_global_var_init.7()
// LLVMCIR:         call void @__cxa_thread_atexit(ptr @__cxx_global_array_dtor.1, ptr @_ZGR14thread_arr_ref_, ptr @__dso_handle)

// OGCG-LABEL: define internal void @__cxx_global_var_init.3()
// OGCG:         call i32 @__cxa_thread_atexit(ptr @__cxx_global_array_dtor.4, ptr null, ptr @__dso_handle)
// OGCG:         %[[THREAD_ARR_ADDR:.*]] = call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@thread_arr_ref)
// OGCG:         store ptr @_ZGR14thread_arr_ref_, ptr %[[THREAD_ARR_ADDR]]

// OGCG-LABEL: define internal void @__cxx_global_array_dtor.4(ptr {{.*}})
// OGCG:         call void @_ZN10NonTrivialD1Ev(ptr
