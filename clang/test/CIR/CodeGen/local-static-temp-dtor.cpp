// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --input-file=%t-before.cir %s --check-prefix=CIR-BEFORE
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=HELPER
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=HELPER
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -fclangir -emit-llvm %s -o %t-cir-eh.ll
// RUN: FileCheck --input-file=%t-cir-eh.ll %s --check-prefix=LLVM-EH
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -emit-llvm %s -o %t-eh.ll
// RUN: FileCheck --input-file=%t-eh.ll %s --check-prefix=LLVM-EH
// RUN: %clang_cc1 -std=c++17 -triple arm64-apple-macosx14.0.0 -fclangir -emit-llvm %s -o %t-cir-darwin.ll
// RUN: FileCheck --input-file=%t-cir-darwin.ll %s --check-prefixes=DARWIN,DARWIN-CIR
// RUN: %clang_cc1 -std=c++17 -triple arm64-apple-macosx14.0.0 -emit-llvm %s -o %t-darwin.ll
// RUN: FileCheck --input-file=%t-darwin.ll %s --check-prefixes=DARWIN,DARWIN-OGCG

struct NonTrivial {
  NonTrivial();
  ~NonTrivial();
  int x;
};

typedef NonTrivial NonTrivialArr[2];

void use(const NonTrivial &);

void ref() {
  static const NonTrivial &r = NonTrivial();
  use(r);
}

// CIR-BEFORE-LABEL: cir.func {{.*}}@_Z3refv
// CIR-BEFORE:         cir.local_init static_local @_ZZ3refvE1r ctor {
// CIR-BEFORE:           %[[VAR:.*]] = cir.get_global static_local @_ZZ3refvE1r
// CIR-BEFORE:           %[[TEMP:.*]] = cir.get_global @_ZGRZ3refvE1r_
// CIR-BEFORE:           cir.call @_ZN10NonTrivialC1Ev(%[[TEMP]])
// CIR-BEFORE:           cir.register_exit_dtor @_ZGRZ3refvE1r_ {
// CIR-BEFORE:             %[[TEMP_DTOR:.*]] = cir.get_global @_ZGRZ3refvE1r_
// CIR-BEFORE:             cir.call @_ZN10NonTrivialD1Ev(%[[TEMP_DTOR]])
// CIR-BEFORE:           }
// CIR-BEFORE:           cir.store{{.*}} %[[TEMP]], %[[VAR]]
// CIR-BEFORE:         }

// LLVM-DAG: @_ZGRZ3refvE1r_ = internal global %struct.NonTrivial zeroinitializer
// LLVM-DAG: @_ZGRZ3arrvE1a_ = internal global [2 x %struct.NonTrivial] zeroinitializer

// The array helper is emitted ahead of the functions.
// CIR-LABEL: cir.func internal private @__cxx_global_array_dtor(%{{.+}}: !cir.ptr<!void> {llvm.noundef}
// CIR-NEXT:    cir.get_global @_ZGRZ3arrvE1a_ : !cir.ptr<!cir.array<!rec_NonTrivial x 2>>
// CIR:         cir.call @_ZN10NonTrivialD1Ev

// CIR-LABEL: cir.func {{.*}}@_Z3refv
// CIR:         cir.call @__cxa_guard_acquire
// CIR:         %[[TEMP:.*]] = cir.get_global @_ZGRZ3refvE1r_
// CIR:         cir.call @_ZN10NonTrivialC1Ev(%[[TEMP]])
// CIR:         %[[OBJ:.*]] = cir.get_global @_ZGRZ3refvE1r_
// CIR:         %[[DTOR:.*]] = cir.get_global @_ZN10NonTrivialD1Ev
// CIR:         %[[DTOR_CAST:.*]] = cir.cast bitcast %[[DTOR]]
// CIR:         %[[OBJ_CAST:.*]] = cir.cast bitcast %[[OBJ]] : !cir.ptr<!rec_NonTrivial> -> !cir.ptr<!void>
// CIR:         %[[HANDLE:.*]] = cir.get_global @__dso_handle
// CIR:         cir.call @__cxa_atexit(%[[DTOR_CAST]], %[[OBJ_CAST]], %[[HANDLE]]) nothrow
// CIR:         cir.store{{.*}} %[[TEMP]]
// CIR:         cir.call @__cxa_guard_release

// LLVM-LABEL: define dso_local void @_Z3refv()
// LLVM:         call i32 @__cxa_guard_acquire(ptr @_ZGVZ3refvE1r)
// LLVM:         call void @_ZN10NonTrivialC1Ev(ptr noundef nonnull align 4 dereferenceable(4) @_ZGRZ3refvE1r_)
// LLVM:         call i32 @__cxa_atexit(ptr @_ZN10NonTrivialD1Ev, ptr @_ZGRZ3refvE1r_, ptr @__dso_handle)
// LLVM:         store ptr @_ZGRZ3refvE1r_, ptr @_ZZ3refvE1r
// LLVM:         call void @__cxa_guard_release(ptr @_ZGVZ3refvE1r)

void arr() {
  static const NonTrivialArr &a = NonTrivialArr{};
  use(a[0]);
}

// CIR-BEFORE-LABEL: cir.func {{.*}}@_Z3arrv
// CIR-BEFORE:         cir.local_init static_local @_ZZ3arrvE1a ctor {
// CIR-BEFORE:           cir.do {
// CIR-BEFORE:             cir.call @_ZN10NonTrivialC1Ev
// CIR-BEFORE:           } while {
// CIR-BEFORE:           cir.register_exit_dtor @_ZGRZ3arrvE1a_ {
// CIR-BEFORE:             %[[ARR_TEMP:.*]] = cir.get_global @_ZGRZ3arrvE1a_
// CIR-BEFORE:             cir.array.dtor %[[ARR_TEMP]] : !cir.ptr<!cir.array<!rec_NonTrivial x 2>> {
// CIR-BEFORE:             ^bb0(%[[ELEMENT:.*]]: !cir.ptr<!rec_NonTrivial>):
// CIR-BEFORE:               cir.call @_ZN10NonTrivialD1Ev(%[[ELEMENT]])
// CIR-BEFORE:             }
// CIR-BEFORE:           }
// CIR-BEFORE:           cir.store
// CIR-BEFORE:         }

// CIR-LABEL: cir.func {{.*}}@_Z3arrv
// CIR:         cir.call @_ZN10NonTrivialC1Ev
// CIR:         %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CIR:         %[[ARR_DTOR:.*]] = cir.get_global @__cxx_global_array_dtor
// CIR:         %[[ARR_DTOR_CAST:.*]] = cir.cast bitcast %[[ARR_DTOR]]
// CIR:         %[[NULL_CAST:.*]] = cir.cast bitcast %[[NULL]]
// CIR:         %[[HANDLE:.*]] = cir.get_global @__dso_handle
// CIR:         cir.call @__cxa_atexit(%[[ARR_DTOR_CAST]], %[[NULL_CAST]], %[[HANDLE]]) nothrow

// The helper destroys the array through the global, not its argument.
// HELPER-LABEL: define internal void @__cxx_global_array_dtor(ptr noundef %{{.+}})
// HELPER:         getelementptr inbounds nuw (i8, ptr @_ZGRZ3arrvE1a_, i64 8)
// HELPER:         call void @_ZN10NonTrivialD1Ev(ptr

// LLVM-LABEL: define dso_local void @_Z3arrv()
// LLVM:         call i32 @__cxa_guard_acquire(ptr @_ZGVZ3arrvE1a)
// LLVM:         call void @_ZN10NonTrivialC1Ev
// LLVM:         call i32 @__cxa_atexit(ptr @__cxx_global_array_dtor, ptr null, ptr @__dso_handle)
// LLVM:         store ptr @_ZGRZ3arrvE1a_, ptr @_ZZ3arrvE1a
// LLVM:         call void @__cxa_guard_release(ptr @_ZGVZ3arrvE1a)

struct TwoRefs {
  const NonTrivial &a, &b;
};

void two() {
  static TwoRefs t{NonTrivial(), NonTrivial()};
  use(t.a);
}

struct Holder {
  const NonTrivial &r;
  ~Holder();
};

void holder() {
  static Holder h{NonTrivial()};
  use(h.r);
}

// CIR-BEFORE-LABEL: cir.func {{.*}}@_Z3twov
// CIR-BEFORE:         cir.local_init static_local @_ZZ3twovE1t ctor {
// CIR-BEFORE:           %[[A:.*]] = cir.get_global @_ZGRZ3twovE1t_
// CIR-BEFORE:           cir.call @_ZN10NonTrivialC1Ev(%[[A]])
// CIR-BEFORE:           cir.register_exit_dtor @_ZGRZ3twovE1t_ {
// CIR-BEFORE:             %[[A_DTOR:.*]] = cir.get_global @_ZGRZ3twovE1t_
// CIR-BEFORE:             cir.call @_ZN10NonTrivialD1Ev(%[[A_DTOR]])
// CIR-BEFORE:           }
// CIR-BEFORE:           cir.store{{.*}} %[[A]]
// CIR-BEFORE:           %[[B:.*]] = cir.get_global @_ZGRZ3twovE1t0_
// CIR-BEFORE:           cir.call @_ZN10NonTrivialC1Ev(%[[B]])
// CIR-BEFORE:           cir.register_exit_dtor @_ZGRZ3twovE1t0_ {
// CIR-BEFORE:             %[[B_DTOR:.*]] = cir.get_global @_ZGRZ3twovE1t0_
// CIR-BEFORE:             cir.call @_ZN10NonTrivialD1Ev(%[[B_DTOR]])
// CIR-BEFORE:           }
// CIR-BEFORE:           cir.store{{.*}} %[[B]]
// CIR-BEFORE:         }

// LLVM-LABEL: define dso_local void @_Z3twov()
// LLVM:         call i32 @__cxa_guard_acquire(ptr @_ZGVZ3twovE1t)
// LLVM:         call void @_ZN10NonTrivialC1Ev(ptr noundef nonnull align 4 dereferenceable(4) @_ZGRZ3twovE1t_)
// LLVM:         call i32 @__cxa_atexit(ptr @_ZN10NonTrivialD1Ev, ptr @_ZGRZ3twovE1t_, ptr @__dso_handle)
// LLVM:         store ptr @_ZGRZ3twovE1t_, ptr @_ZZ3twovE1t
// LLVM:         call void @_ZN10NonTrivialC1Ev(ptr noundef nonnull align 4 dereferenceable(4) @_ZGRZ3twovE1t0_)
// LLVM:         call i32 @__cxa_atexit(ptr @_ZN10NonTrivialD1Ev, ptr @_ZGRZ3twovE1t0_, ptr @__dso_handle)
// LLVM:         store ptr @_ZGRZ3twovE1t0_, ptr getelementptr inbounds nuw (i8, ptr @_ZZ3twovE1t, i64 8)
// LLVM:         call void @__cxa_guard_release(ptr @_ZGVZ3twovE1t)

// LLVM-EH-LABEL: define dso_local void @_Z3twov()
// LLVM-EH:         call i32 @__cxa_guard_acquire(ptr @_ZGVZ3twovE1t)
// LLVM-EH:         invoke void @_ZN10NonTrivialC1Ev(ptr noundef nonnull align 4 dereferenceable(4) @_ZGRZ3twovE1t_)
// LLVM-EH-NEXT:      to label %{{.+}} unwind label %[[LPAD:.+]]
// LLVM-EH:         call i32 @__cxa_atexit(ptr @_ZN10NonTrivialD1Ev, ptr @_ZGRZ3twovE1t_, ptr @__dso_handle)
// LLVM-EH:         invoke void @_ZN10NonTrivialC1Ev(ptr noundef nonnull align 4 dereferenceable(4) @_ZGRZ3twovE1t0_)
// LLVM-EH-NEXT:      to label %{{.+}} unwind label %[[LPAD]]
// LLVM-EH:         call i32 @__cxa_atexit(ptr @_ZN10NonTrivialD1Ev, ptr @_ZGRZ3twovE1t0_, ptr @__dso_handle)
// LLVM-EH:       [[LPAD]]:
// LLVM-EH:         landingpad
// LLVM-EH:         call void @__cxa_guard_abort(ptr @_ZGVZ3twovE1t)

// CIR-BEFORE-LABEL: cir.func {{.*}}@_Z6holderv
// CIR-BEFORE:         cir.local_init static_local @_ZZ6holdervE1h ctor {
// CIR-BEFORE:           %[[T:.*]] = cir.get_global @_ZGRZ6holdervE1h_
// CIR-BEFORE:           cir.call @_ZN10NonTrivialC1Ev(%[[T]])
// CIR-BEFORE:           cir.register_exit_dtor @_ZGRZ6holdervE1h_ {
// CIR-BEFORE:             %[[T_DTOR:.*]] = cir.get_global @_ZGRZ6holdervE1h_
// CIR-BEFORE:             cir.call @_ZN10NonTrivialD1Ev(%[[T_DTOR]])
// CIR-BEFORE:           }
// CIR-BEFORE:           cir.store{{.*}} %[[T]]
// CIR-BEFORE:         } dtor {
// CIR-BEFORE:           %[[H:.*]] = cir.get_global static_local @_ZZ6holdervE1h
// CIR-BEFORE:           cir.call @_ZN6HolderD1Ev(%[[H]])
// CIR-BEFORE:         }

// LLVM-LABEL: define dso_local void @_Z6holderv()
// LLVM:         call i32 @__cxa_guard_acquire(ptr @_ZGVZ6holdervE1h)
// LLVM:         call void @_ZN10NonTrivialC1Ev(ptr noundef nonnull align 4 dereferenceable(4) @_ZGRZ6holdervE1h_)
// LLVM:         call i32 @__cxa_atexit(ptr @_ZN10NonTrivialD1Ev, ptr @_ZGRZ6holdervE1h_, ptr @__dso_handle)
// LLVM:         store ptr @_ZGRZ6holdervE1h_, ptr @_ZZ6holdervE1h
// LLVM:         call i32 @__cxa_atexit(ptr @_ZN6HolderD1Ev, ptr @_ZZ6holdervE1h, ptr @__dso_handle)
// LLVM:         call void @__cxa_guard_release(ptr @_ZGVZ6holdervE1h)

void tls() {
  thread_local const NonTrivial &r = NonTrivial();
  use(r);
}

void tls_arr() {
  thread_local const NonTrivialArr &a = NonTrivialArr{};
  use(a[0]);
}

extern NonTrivial nt;

void cond(bool c) {
  static const NonTrivial &r =
      c ? static_cast<const NonTrivial &>(NonTrivial()) : nt;
  use(r);
}

// CIR-BEFORE-LABEL: cir.func {{.*}}@_Z3tlsv
// CIR-BEFORE:         cir.local_init thread_local @_ZZ3tlsvE1r ctor {
// CIR-BEFORE:           %[[TEMP:.*]] = cir.get_global @_ZGRZ3tlsvE1r_
// CIR-BEFORE:           cir.call @_ZN10NonTrivialC1Ev(%[[TEMP]])
// CIR-BEFORE:           cir.register_exit_dtor @_ZGRZ3tlsvE1r_ {
// CIR-BEFORE:             %[[TEMP_DTOR:.*]] = cir.get_global @_ZGRZ3tlsvE1r_
// CIR-BEFORE:             cir.call @_ZN10NonTrivialD1Ev(%[[TEMP_DTOR]])
// CIR-BEFORE:           }

// LLVM-LABEL: define dso_local void @_Z3tlsv()
// LLVM:         call void @_ZN10NonTrivialC1Ev(ptr noundef nonnull align 4 dereferenceable(4) @_ZGRZ3tlsvE1r_)
// LLVM:         call i32 @__cxa_thread_atexit(ptr @_ZN10NonTrivialD1Ev, ptr @_ZGRZ3tlsvE1r_, ptr @__dso_handle)
// LLVM:         store ptr @_ZGRZ3tlsvE1r_, ptr %{{.+}}

// DARWIN-LABEL: define void @_Z3tlsv()
// Apple arm64 constructors return this in classic but not in CIR.
// DARWIN-CIR:     call void @_ZN10NonTrivialC1Ev(ptr noundef nonnull align 4 dereferenceable(4) @_ZGRZ3tlsvE1r_)
// DARWIN-OGCG:    %{{.+}} = call noundef ptr @_ZN10NonTrivialC1Ev(ptr noundef nonnull align 4 dereferenceable(4) @_ZGRZ3tlsvE1r_)
// DARWIN-NEXT:    call i32 @_tlv_atexit(ptr @_ZN10NonTrivialD1Ev, ptr @_ZGRZ3tlsvE1r_, ptr @__dso_handle)
// DARWIN:         store ptr @_ZGRZ3tlsvE1r_, ptr %{{.+}}

// CIR-BEFORE-LABEL: cir.func {{.*}}@_Z7tls_arrv
// CIR-BEFORE:         cir.local_init thread_local @_ZZ7tls_arrvE1a ctor {
// CIR-BEFORE:           cir.do {
// CIR-BEFORE:             cir.call @_ZN10NonTrivialC1Ev
// CIR-BEFORE:           } while {
// CIR-BEFORE:           cir.register_exit_dtor @_ZGRZ7tls_arrvE1a_ {
// CIR-BEFORE:             cir.array.dtor

// HELPER-LABEL: define internal void @__cxx_global_array_dtor.1(ptr noundef %{{.+}})
// HELPER:         getelementptr inbounds nuw (i8, ptr @_ZGRZ7tls_arrvE1a_, i64 8)
// HELPER:         call void @_ZN10NonTrivialD1Ev(ptr

// LLVM-LABEL: define dso_local void @_Z7tls_arrv()
// LLVM:         call void @_ZN10NonTrivialC1Ev
// LLVM:         call i32 @__cxa_thread_atexit(ptr @__cxx_global_array_dtor.1, ptr null, ptr @__dso_handle)
// LLVM:         store ptr @_ZGRZ7tls_arrvE1a_, ptr %{{.+}}

// CIR-BEFORE-LABEL: cir.func {{.*}}@_Z4condb
// CIR-BEFORE:         cir.local_init static_local @_ZZ4condbE1r ctor {
// CIR-BEFORE:           cir.ternary(%{{.+}}, true {
// CIR-BEFORE:             %[[TEMP:.*]] = cir.get_global @_ZGRZ4condbE1r_
// CIR-BEFORE:             cir.call @_ZN10NonTrivialC1Ev(%[[TEMP]])
// CIR-BEFORE:             cir.register_exit_dtor @_ZGRZ4condbE1r_ {
// CIR-BEFORE:               %[[TEMP_DTOR:.*]] = cir.get_global @_ZGRZ4condbE1r_
// CIR-BEFORE:               cir.call @_ZN10NonTrivialD1Ev(%[[TEMP_DTOR]])
// CIR-BEFORE:             }
// CIR-BEFORE:             cir.yield %[[TEMP]]
// CIR-BEFORE:           }, false {
// CIR-BEFORE:             cir.get_global @nt
// CIR-BEFORE-NOT:         cir.register_exit_dtor
// CIR-BEFORE:           })

// Only the arm that builds the temporary registers its destruction.
// LLVM-LABEL: define dso_local void @_Z4condb(i1 noundef zeroext %{{.+}})
// LLVM:         call i32 @__cxa_guard_acquire(ptr @_ZGVZ4condbE1r)
// LLVM:         call void @_ZN10NonTrivialC1Ev(ptr noundef nonnull align 4 dereferenceable(4) @_ZGRZ4condbE1r_)
// LLVM-NEXT:    call i32 @__cxa_atexit(ptr @_ZN10NonTrivialD1Ev, ptr @_ZGRZ4condbE1r_, ptr @__dso_handle)
// LLVM-NEXT:    br label
// LLVM-NOT:     __cxa_atexit
// LLVM:         phi ptr
// LLVM-NOT:     __cxa_atexit
// LLVM:         call void @__cxa_guard_release(ptr @_ZGVZ4condbE1r)
