// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --input-file=%t-before.cir %s --check-prefixes=CIR-BEFORE
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=LLVM,OGCG
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -fclangir -emit-llvm %s -o %t-cir-eh.ll
// RUN: FileCheck --input-file=%t-cir-eh.ll %s --check-prefix=LLVM-EH
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -emit-llvm %s -o %t-eh.ll
// RUN: FileCheck --input-file=%t-eh.ll %s --check-prefix=LLVM-EH

// Exercises lifetime-extended reference temporaries with non-trivial
// destructors where the extending declaration has static or thread storage
// duration, for both non-array and array temporary types.

struct NonTrivial {
  NonTrivial();
  ~NonTrivial();
  int x;
};

const NonTrivial &static_ref = NonTrivial();
thread_local const NonTrivial &thread_ref = NonTrivial();

typedef NonTrivial NonTrivialArr[2];

const NonTrivialArr &static_arr_ref = NonTrivialArr{};
thread_local const NonTrivialArr &thread_arr_ref = NonTrivialArr{};

// CIR-BEFORE: cir.global external @static_ref = ctor : !cir.ptr<!rec_NonTrivial> {
// CIR-BEFORE:   %[[STATIC_REF:.*]] = cir.get_global @static_ref
// CIR-BEFORE:   %[[REF_TEMP:.*]] = cir.get_global @_ZGR10static_ref_
// CIR-BEFORE:   cir.call @_ZN10NonTrivialC1Ev(%[[REF_TEMP]])
// CIR-BEFORE:   cir.register_exit_dtor @_ZGR10static_ref_ {
// CIR-BEFORE:     %[[REF_TEMP_DTOR:.*]] = cir.get_global @_ZGR10static_ref_
// CIR-BEFORE:     cir.call @_ZN10NonTrivialD1Ev(%[[REF_TEMP_DTOR]])
// CIR-BEFORE:   }
// CIR-BEFORE:   cir.store{{.*}} %[[REF_TEMP]], %[[STATIC_REF]]
// CIR-BEFORE: }
// CIR-BEFORE: cir.global "private" internal @_ZGR10static_ref_ = #cir.zero : !rec_NonTrivial

// CIR: cir.global external @static_ref = #cir.ptr<null> : !cir.ptr<!rec_NonTrivial>
// CIR: cir.func internal private @__cxx_global_var_init()
// CIR:   cir.get_global @static_ref
// CIR:   cir.get_global @_ZGR10static_ref_
// CIR:   cir.call @_ZN10NonTrivialC1Ev
// CIR:   cir.get_global @_ZGR10static_ref_
// CIR:   cir.get_global @_ZN10NonTrivialD1Ev
// CIR:   cir.get_global @__dso_handle
// CIR:   cir.call @__cxa_atexit({{.*}}) nothrow
// CIR:   cir.store
// CIR: cir.global "private" internal @_ZGR10static_ref_ = #cir.zero : !rec_NonTrivial

// CIR-BEFORE: cir.global external tls_model = tls_dyn tls_refs = <"_ZTW10thread_ref", "_ZTH10thread_ref"> @thread_ref = ctor : !cir.ptr<!rec_NonTrivial> {
// CIR-BEFORE:   %[[THREAD_REF:.*]] = cir.get_global thread_local @thread_ref
// CIR-BEFORE:   %[[REF_TEMP:.*]] = cir.get_global @_ZGR10thread_ref_
// CIR-BEFORE:   cir.call @_ZN10NonTrivialC1Ev(%[[REF_TEMP]])
// CIR-BEFORE:   cir.register_exit_dtor @_ZGR10thread_ref_ {
// CIR-BEFORE:     %[[REF_TEMP_DTOR:.*]] = cir.get_global @_ZGR10thread_ref_
// CIR-BEFORE:     cir.call @_ZN10NonTrivialD1Ev(%[[REF_TEMP_DTOR]])
// CIR-BEFORE:   }
// CIR-BEFORE:   cir.store{{.*}} %[[REF_TEMP]], %[[THREAD_REF]]
// CIR-BEFORE: }
// CIR-BEFORE: cir.global "private" internal tls_model = tls_dyn @_ZGR10thread_ref_ = #cir.zero : !rec_NonTrivial

// CIR: cir.global external tls_model = tls_dyn tls_refs = <"_ZTW10thread_ref", "_ZTH10thread_ref"> @thread_ref = #cir.ptr<null> : !cir.ptr<!rec_NonTrivial>
// CIR: cir.func internal private @__cxx_global_var_init.1()
// CIR:   cir.get_global thread_local @thread_ref
// CIR:   cir.get_global @_ZGR10thread_ref_
// CIR:   cir.call @_ZN10NonTrivialC1Ev
// CIR:   cir.get_global @_ZGR10thread_ref_
// CIR:   cir.get_global @_ZN10NonTrivialD1Ev
// CIR:   cir.get_global @__dso_handle
// CIR:   cir.call @__cxa_thread_atexit({{.*}}) nothrow
// CIR:   cir.store
// CIR: cir.global "private" internal tls_model = tls_dyn @_ZGR10thread_ref_ = #cir.zero : !rec_NonTrivial

// CIR-BEFORE: cir.global external @static_arr_ref = ctor : !cir.ptr<!cir.array<!rec_NonTrivial x 2>> {
// CIR-BEFORE:   %[[ARRAY_INIT_TEMP:.*]] = cir.alloca {{.*}}"arrayinit.temp"
// CIR-BEFORE:   %[[STATIC_ARR_REF:.*]] = cir.get_global @static_arr_ref
// CIR-BEFORE:   %[[STATIC_ARR_REF_TEMP:.*]] = cir.get_global @_ZGR14static_arr_ref_
// CIR-BEFORE:   %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[STATIC_ARR_REF_TEMP]]
// CIR-BEFORE:   cir.store{{.*}} %[[DECAY]], %[[ARRAY_INIT_TEMP]]
// CIR-BEFORE:   %[[TWO:.*]] = cir.const #cir.int<2> : !s64i
// CIR-BEFORE:   %[[NEXT:.*]] = cir.ptr_stride %[[DECAY]], %[[TWO]] : (!cir.ptr<!rec_NonTrivial>, !s64i) -> !cir.ptr<!rec_NonTrivial>
// CIR-BEFORE:   cir.do {
// CIR-BEFORE:     cir.call @_ZN10NonTrivialC1Ev
// CIR-BEFORE:   } while {
// CIR-BEFORE:     cir.condition
// CIR-BEFORE:   }
// CIR-BEFORE:   cir.register_exit_dtor @_ZGR14static_arr_ref_ {
// CIR-BEFORE:     %[[STATIC_ARR_REF_TEMP_DTOR:.*]] = cir.get_global @_ZGR14static_arr_ref_
// CIR-BEFORE:     cir.array.dtor %[[STATIC_ARR_REF_TEMP_DTOR]] : !cir.ptr<!cir.array<!rec_NonTrivial x 2>> {
// CIR-BEFORE:     ^bb0(%[[ELEMENT:.*]]: !cir.ptr<!rec_NonTrivial>):
// CIR-BEFORE:       cir.call @_ZN10NonTrivialD1Ev(%[[ELEMENT]])
// CIR-BEFORE:     }
// CIR-BEFORE:   }
// CIR-BEFORE:   cir.store{{.*}} %[[STATIC_ARR_REF_TEMP]], %[[STATIC_ARR_REF]] : !cir.ptr<!cir.array<!rec_NonTrivial x 2>>, !cir.ptr<!cir.ptr<!cir.array<!rec_NonTrivial x 2>>>
// CIR-BEFORE: }
// CIR-BEFORE: cir.global "private" internal @_ZGR14static_arr_ref_ = #cir.zero : !cir.array<!rec_NonTrivial x 2>

// CIR: cir.global external @static_arr_ref = #cir.ptr<null> : !cir.ptr<!cir.array<!rec_NonTrivial x 2>>
// CIR: cir.func internal private @__cxx_global_var_init.2()
// CIR:   cir.call @_ZN10NonTrivialC1Ev
// CIR:   cir.get_global @__cxx_global_array_dtor
// CIR:   cir.get_global @__dso_handle
// CIR:   cir.call @__cxa_atexit({{.*}}) nothrow
// CIR:   cir.store
// CIR: cir.global "private" internal @_ZGR14static_arr_ref_ = #cir.zero : !cir.array<!rec_NonTrivial x 2>
// CIR: cir.func internal private @__cxx_global_array_dtor(
// CIR:   cir.do {
// CIR:     cir.call @_ZN10NonTrivialD1Ev
// CIR:   } while {
// CIR:     cir.condition
// CIR:   }

// CIR-BEFORE: cir.global external tls_model = tls_dyn tls_refs = <"_ZTW14thread_arr_ref", "_ZTH14thread_arr_ref"> @thread_arr_ref = ctor : !cir.ptr<!cir.array<!rec_NonTrivial x 2>> {
// CIR-BEFORE:   %[[ARRAY_INIT_TEMP:.*]] = cir.alloca {{.*}}"arrayinit.temp"
// CIR-BEFORE:   %[[THREAD_ARR_REF:.*]] = cir.get_global thread_local @thread_arr_ref
// CIR-BEFORE:   %[[THREAD_ARR_REF_TEMP:.*]] = cir.get_global @_ZGR14thread_arr_ref_
// CIR-BEFORE:   %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[THREAD_ARR_REF_TEMP]]
// CIR-BEFORE:   cir.store{{.*}} %[[DECAY]], %[[ARRAY_INIT_TEMP]]
// CIR-BEFORE:   %[[TWO:.*]] = cir.const #cir.int<2> : !s64i
// CIR-BEFORE:   %[[NEXT:.*]] = cir.ptr_stride %[[DECAY]], %[[TWO]] : (!cir.ptr<!rec_NonTrivial>, !s64i) -> !cir.ptr<!rec_NonTrivial>
// CIR-BEFORE:   cir.do {
// CIR-BEFORE:     cir.call @_ZN10NonTrivialC1Ev
// CIR-BEFORE:   } while {
// CIR-BEFORE:     cir.condition
// CIR-BEFORE:   }
// CIR-BEFORE:   cir.register_exit_dtor @_ZGR14thread_arr_ref_ {
// CIR-BEFORE:     %[[THREAD_ARR_REF_TEMP_DTOR:.*]] = cir.get_global @_ZGR14thread_arr_ref_
// CIR-BEFORE:     cir.array.dtor %[[THREAD_ARR_REF_TEMP_DTOR]] : !cir.ptr<!cir.array<!rec_NonTrivial x 2>> {
// CIR-BEFORE:     ^bb0(%[[ELEMENT:.*]]: !cir.ptr<!rec_NonTrivial>):
// CIR-BEFORE:       cir.call @_ZN10NonTrivialD1Ev(%[[ELEMENT]])
// CIR-BEFORE:     }
// CIR-BEFORE:   }
// CIR-BEFORE:   cir.store{{.*}} %[[THREAD_ARR_REF_TEMP]], %[[THREAD_ARR_REF]] : !cir.ptr<!cir.array<!rec_NonTrivial x 2>>, !cir.ptr<!cir.ptr<!cir.array<!rec_NonTrivial x 2>>>
// CIR-BEFORE: }
// CIR-BEFORE: cir.global "private" internal tls_model = tls_dyn @_ZGR14thread_arr_ref_ = #cir.zero : !cir.array<!rec_NonTrivial x 2>

// CIR: cir.global external tls_model = tls_dyn tls_refs = <"_ZTW14thread_arr_ref", "_ZTH14thread_arr_ref"> @thread_arr_ref = #cir.ptr<null> : !cir.ptr<!cir.array<!rec_NonTrivial x 2>>
// CIR: cir.func internal private @__cxx_global_var_init.3()
// CIR:   cir.call @_ZN10NonTrivialC1Ev
// CIR:   cir.get_global @__cxx_global_array_dtor.1
// CIR:   cir.get_global @__dso_handle
// CIR:   cir.call @__cxa_thread_atexit({{.*}}) nothrow
// CIR:   cir.store
// CIR: cir.global "private" internal tls_model = tls_dyn @_ZGR14thread_arr_ref_ = #cir.zero : !cir.array<!rec_NonTrivial x 2>
// CIR: cir.func internal private @__cxx_global_array_dtor.1(
// CIR:   cir.do {
// CIR:     cir.call @_ZN10NonTrivialD1Ev
// CIR:   } while {
// CIR:     cir.condition
// CIR:   }

// LLVM-DAG: @static_ref = global ptr null
// LLVM-DAG: @_ZGR10static_ref_ = internal global %struct.NonTrivial zeroinitializer
// LLVM-DAG: @thread_ref = thread_local global ptr null
// LLVM-DAG: @_ZGR10thread_ref_ = internal thread_local global %struct.NonTrivial zeroinitializer
// LLVM-DAG: @static_arr_ref = global ptr null
// LLVM-DAG: @_ZGR14static_arr_ref_ = internal global [2 x %struct.NonTrivial] zeroinitializer
// LLVM-DAG: @thread_arr_ref = thread_local global ptr null
// LLVM-DAG: @_ZGR14thread_arr_ref_ = internal thread_local global [2 x %struct.NonTrivial] zeroinitializer

// Static, non-array.

// LLVM-LABEL: define internal void @__cxx_global_var_init()
// LLVM:         call void @_ZN10NonTrivialC1Ev(ptr noundef nonnull align 4 dereferenceable(4) @_ZGR10static_ref_)
// LLVM:         call i32 @__cxa_atexit(ptr @_ZN10NonTrivialD1Ev, ptr @_ZGR10static_ref_, ptr @__dso_handle)
// LLVM:         store ptr @_ZGR10static_ref_, ptr @static_ref

// Thread, non-array.  CIR takes the variable's thread-local address before
// running the initializer, and without classic's align 8 on the call.

// LLVM-LABEL: define internal void @__cxx_global_var_init.1()
// LLVMCIR:      %[[TLS_ADDR:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @thread_ref)
// LLVM:         call void @_ZN10NonTrivialC1Ev(ptr noundef nonnull align 4 dereferenceable(4) @_ZGR10thread_ref_)
// LLVM:         call i32 @__cxa_thread_atexit(ptr @_ZN10NonTrivialD1Ev, ptr @_ZGR10thread_ref_, ptr @__dso_handle)
// OGCG:         %[[TLS_ADDR:.*]] = call align 8 ptr @llvm.threadlocal.address.p0(ptr align 8 @thread_ref)
// LLVM:         store ptr @_ZGR10thread_ref_, ptr %[[TLS_ADDR]]

// Static, array.  The two pipelines build the construction loop differently.

// LLVM-LABEL: define internal void @__cxx_global_var_init.2()
// LLVMCIR:       [[LOOP_CONDITION_BLOCK:.*]]:
// LLVMCIR:         %[[DONE:.*]] = icmp ne ptr
// LLVMCIR:         br i1 %[[DONE]], label %[[LOOP_BODY_BLOCK:.*]], label %[[LOOP_EXIT_BLOCK:.*]]
// LLVMCIR:       [[LOOP_BODY_BLOCK]]:
// LLVMCIR:         call void @_ZN10NonTrivialC1Ev
// LLVMCIR:         br label %[[LOOP_CONDITION_BLOCK]]
// OGCG:          br label %[[LOOP_BODY_BLOCK:.*]]
// OGCG:        [[LOOP_BODY_BLOCK]]:
// OGCG:          call void @_ZN10NonTrivialC1Ev
// OGCG:          %[[DONE:.*]] = icmp eq ptr
// OGCG:          br i1 %[[DONE]], label %[[LOOP_EXIT_BLOCK:.*]], label %[[LOOP_BODY_BLOCK]]
// LLVM:        [[LOOP_EXIT_BLOCK]]:
// LLVM:          call i32 @__cxa_atexit(ptr @__cxx_global_array_dtor, ptr null, ptr @__dso_handle)
// LLVM:          store ptr @_ZGR14static_arr_ref_, ptr @static_arr_ref

// The helper destroys the array through the global, not its argument.
// LLVM-LABEL: define internal void @__cxx_global_array_dtor(ptr noundef %{{.+}})
// LLVM:          getelementptr inbounds nuw (i8, ptr @_ZGR14static_arr_ref_, i64 8)
// LLVM:          call void @_ZN10NonTrivialD1Ev(ptr

// Thread, array.  CIR also reaches the variable through its thread wrapper,
// and the two pipelines number the helper differently.

// LLVM-LABEL: define internal void @__cxx_global_var_init.3()
// LLVMCIR:         %[[THREAD_ARR_REF:.*]] = call ptr @_ZTW14thread_arr_ref()
// LLVMCIR:       [[LOOP_CONDITION_BLOCK:.*]]:
// LLVMCIR:         %[[DONE:.*]] = icmp ne ptr
// LLVMCIR:         br i1 %[[DONE]], label %[[LOOP_BODY_BLOCK:.*]], label %[[LOOP_EXIT_BLOCK:.*]]
// LLVMCIR:       [[LOOP_BODY_BLOCK]]:
// LLVMCIR:         call void @_ZN10NonTrivialC1Ev
// LLVMCIR:         br label %[[LOOP_CONDITION_BLOCK]]
// OGCG:          br label %[[LOOP_BODY_BLOCK:.*]]
// OGCG:        [[LOOP_BODY_BLOCK]]:
// OGCG:          call void @_ZN10NonTrivialC1Ev
// OGCG:          %[[DONE:.*]] = icmp eq ptr
// OGCG:          br i1 %[[DONE]], label %[[LOOP_EXIT_BLOCK:.*]], label %[[LOOP_BODY_BLOCK]]
// LLVM:        [[LOOP_EXIT_BLOCK]]:
// LLVMCIR:         call i32 @__cxa_thread_atexit(ptr @__cxx_global_array_dtor.1, ptr null, ptr @__dso_handle)
// LLVMCIR:         store ptr @_ZGR14thread_arr_ref_, ptr %[[THREAD_ARR_REF]]
// OGCG:          call i32 @__cxa_thread_atexit(ptr @__cxx_global_array_dtor.4, ptr null, ptr @__dso_handle)
// OGCG:          %[[THREAD_ARR_ADDR:.*]] = call align 8 ptr @llvm.threadlocal.address.p0(ptr align 8 @thread_arr_ref)
// OGCG:          store ptr @_ZGR14thread_arr_ref_, ptr %[[THREAD_ARR_ADDR]]

// LLVMCIR-LABEL: define internal void @__cxx_global_array_dtor.1(ptr noundef %{{.+}})
// OGCG-LABEL:    define internal void @__cxx_global_array_dtor.4(ptr noundef %{{.+}})
// LLVM:          getelementptr inbounds nuw (i8, ptr @_ZGR14thread_arr_ref_, i64 8)
// LLVM:          call void @_ZN10NonTrivialD1Ev(ptr

struct TwoRefs {
  const NonTrivial &a, &b;
};

TwoRefs two_refs{NonTrivial(), NonTrivial()};

struct Holder {
  const NonTrivial &r;
  ~Holder();
};

Holder holder{NonTrivial()};

// CIR-BEFORE: cir.global external @two_refs = ctor : !rec_TwoRefs {
// CIR-BEFORE:   %[[A:.*]] = cir.get_global @_ZGR8two_refs_
// CIR-BEFORE:   cir.call @_ZN10NonTrivialC1Ev(%[[A]])
// CIR-BEFORE:   cir.register_exit_dtor @_ZGR8two_refs_ {
// CIR-BEFORE:     %[[A_DTOR:.*]] = cir.get_global @_ZGR8two_refs_
// CIR-BEFORE:     cir.call @_ZN10NonTrivialD1Ev(%[[A_DTOR]])
// CIR-BEFORE:   }
// CIR-BEFORE:   cir.store{{.*}} %[[A]]
// CIR-BEFORE:   %[[B:.*]] = cir.get_global @_ZGR8two_refs0_
// CIR-BEFORE:   cir.call @_ZN10NonTrivialC1Ev(%[[B]])
// CIR-BEFORE:   cir.register_exit_dtor @_ZGR8two_refs0_ {
// CIR-BEFORE:     %[[B_DTOR:.*]] = cir.get_global @_ZGR8two_refs0_
// CIR-BEFORE:     cir.call @_ZN10NonTrivialD1Ev(%[[B_DTOR]])
// CIR-BEFORE:   }
// CIR-BEFORE:   cir.store{{.*}} %[[B]]
// CIR-BEFORE: }

// CIR-BEFORE: cir.global external @holder = ctor : !rec_Holder {
// CIR-BEFORE:   %[[T:.*]] = cir.get_global @_ZGR6holder_
// CIR-BEFORE:   cir.call @_ZN10NonTrivialC1Ev(%[[T]])
// CIR-BEFORE:   cir.register_exit_dtor @_ZGR6holder_ {
// CIR-BEFORE:     %[[T_DTOR:.*]] = cir.get_global @_ZGR6holder_
// CIR-BEFORE:     cir.call @_ZN10NonTrivialD1Ev(%[[T_DTOR]])
// CIR-BEFORE:   }
// CIR-BEFORE:   cir.store{{.*}} %[[T]]
// CIR-BEFORE: } dtor {
// CIR-BEFORE:   %[[HOLDER:.*]] = cir.get_global @holder
// CIR-BEFORE:   cir.call @_ZN6HolderD1Ev(%[[HOLDER]])
// CIR-BEFORE: }

// Two temporaries extended by one variable.
// LLVM-LABEL: define internal void @__cxx_global_var_init.{{[0-9]+}}()
// LLVM:         call void @_ZN10NonTrivialC1Ev(ptr noundef nonnull align 4 dereferenceable(4) @_ZGR8two_refs_)
// LLVM:         call i32 @__cxa_atexit(ptr @_ZN10NonTrivialD1Ev, ptr @_ZGR8two_refs_, ptr @__dso_handle)
// LLVM:         store ptr @_ZGR8two_refs_, ptr @two_refs
// LLVM:         call void @_ZN10NonTrivialC1Ev(ptr noundef nonnull align 4 dereferenceable(4) @_ZGR8two_refs0_)
// LLVM:         call i32 @__cxa_atexit(ptr @_ZN10NonTrivialD1Ev, ptr @_ZGR8two_refs0_, ptr @__dso_handle)
// LLVM:         store ptr @_ZGR8two_refs0_, ptr getelementptr inbounds nuw (i8, ptr @two_refs, i64 8)

// An extending declaration with its own destructor.
// LLVM-LABEL: define internal void @__cxx_global_var_init.{{[0-9]+}}()
// LLVM:         call void @_ZN10NonTrivialC1Ev(ptr noundef nonnull align 4 dereferenceable(4) @_ZGR6holder_)
// LLVM:         call i32 @__cxa_atexit(ptr @_ZN10NonTrivialD1Ev, ptr @_ZGR6holder_, ptr @__dso_handle)
// LLVM:         store ptr @_ZGR6holder_, ptr @holder
// LLVM:         call i32 @__cxa_atexit(ptr @_ZN6HolderD1Ev, ptr @holder, ptr @__dso_handle)

bool pick();

const NonTrivial &cond_ref =
    pick() ? static_cast<const NonTrivial &>(NonTrivial()) : static_ref;

// CIR-BEFORE: cir.global external @cond_ref = ctor : !cir.ptr<!rec_NonTrivial> {
// CIR-BEFORE:   cir.ternary(%{{.+}}, true {
// CIR-BEFORE:     %[[TEMP:.*]] = cir.get_global @_ZGR8cond_ref_
// CIR-BEFORE:     cir.call @_ZN10NonTrivialC1Ev(%[[TEMP]])
// CIR-BEFORE:     cir.register_exit_dtor @_ZGR8cond_ref_ {
// CIR-BEFORE:       %[[TEMP_DTOR:.*]] = cir.get_global @_ZGR8cond_ref_
// CIR-BEFORE:       cir.call @_ZN10NonTrivialD1Ev(%[[TEMP_DTOR]])
// CIR-BEFORE:     }
// CIR-BEFORE:     cir.yield %[[TEMP]]
// CIR-BEFORE:   }, false {
// CIR-BEFORE:     cir.get_global @static_ref
// CIR-BEFORE-NOT:   cir.register_exit_dtor
// CIR-BEFORE:   })

// Only the arm that builds the temporary registers its destruction.
// LLVM-LABEL: define internal void @__cxx_global_var_init.{{[0-9]+}}()
// LLVM:         call noundef zeroext i1 @_Z4pickv()
// LLVM:         call void @_ZN10NonTrivialC1Ev(ptr noundef nonnull align 4 dereferenceable(4) @_ZGR8cond_ref_)
// LLVM-NEXT:    call i32 @__cxa_atexit(ptr @_ZN10NonTrivialD1Ev, ptr @_ZGR8cond_ref_, ptr @__dso_handle)
// LLVM-NEXT:    br label
// LLVM-NOT:     __cxa_atexit
// LLVM:         phi ptr
// LLVM-NOT:     __cxa_atexit
// LLVM:         store ptr %{{.+}}, ptr @cond_ref
// LLVM-NOT:     __cxa_atexit
// LLVM:         ret void

struct Arg {
  Arg();
  ~Arg();
};

struct Built {
  Built(const Arg &);
  ~Built();
};

const Built &built_ref = Built(Arg());

// CIR-BEFORE: cir.global external @built_ref = ctor : !cir.ptr<!rec_Built> {
// CIR-BEFORE:   cir.call @_ZN3ArgC1Ev
// CIR-BEFORE:   cir.cleanup.scope {
// CIR-BEFORE:     cir.call @_ZN5BuiltC1ERK3Arg
// CIR-BEFORE:     cir.register_exit_dtor @_ZGR9built_ref_ {
// CIR-BEFORE:       %[[TEMP_DTOR:.*]] = cir.get_global @_ZGR9built_ref_
// CIR-BEFORE:       cir.call @_ZN5BuiltD1Ev(%[[TEMP_DTOR]])
// CIR-BEFORE:     }
// CIR-BEFORE:   } cleanup normal {
// CIR-BEFORE:     cir.call @_ZN3ArgD1Ev

// The extended temporary is registered before the full-expression temporary
// is destroyed.
// LLVM-LABEL: define internal void @__cxx_global_var_init.{{[0-9]+}}()
// LLVM:         call void @_ZN3ArgC1Ev(ptr noundef nonnull align 1 dereferenceable(1) %[[ARG:.+]])
// LLVM:         call void @_ZN5BuiltC1ERK3Arg(ptr noundef nonnull align 1 dereferenceable(1) @_ZGR9built_ref_, ptr noundef nonnull align 1 dereferenceable(1) %[[ARG]])
// LLVM-NEXT:    call i32 @__cxa_atexit(ptr @_ZN5BuiltD1Ev, ptr @_ZGR9built_ref_, ptr @__dso_handle)
// LLVM:         call void @_ZN3ArgD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %[[ARG]])

// With exceptions enabled, the registration is still a call, not an invoke.
// LLVM-EH:      invoke void @_ZN5BuiltC1ERK3Arg(
// LLVM-EH-NEXT:   to label %{{.+}} unwind label %{{.+}}
// LLVM-EH:      call i32 @__cxa_atexit(ptr @_ZN5BuiltD1Ev, ptr @_ZGR9built_ref_, ptr @__dso_handle)
// LLVM-EH:      call void @_ZN3ArgD1Ev(
