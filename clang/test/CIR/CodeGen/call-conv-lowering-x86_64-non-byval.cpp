// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-CIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t.ll %s

struct WithDtor {
  int x;
  ~WithDtor();
};

struct Big {
  long a, b, c, d;
};

struct WithCopyCtor {
  int x;
  WithCopyCtor();
  WithCopyCtor(const WithCopyCtor &);
};

void takeNonByval(WithDtor t);
void takeTwoNonByval(WithDtor a, WithDtor b);
void takeByval(Big b);
void takeCopyCtorNonByval(WithCopyCtor c);

// The callee must receive the temporary the caller destroys, not a copy of it.
void callNonByval() {
  WithDtor t;
  takeNonByval(t);
}

// CIR-LABEL: cir.func {{.*}}@_Z12callNonByvalv
// CIR:         %[[T:.*]] = cir.alloca "t" align(4) : !cir.ptr<!rec_WithDtor>
// CIR:         %[[TMP:.*]] = cir.alloca "agg.tmp0" align(4) : !cir.ptr<!rec_WithDtor>
// CIR:         cir.copy %[[T]] align(4) to %[[TMP]] align(4) : !cir.ptr<!rec_WithDtor>
// CIR-NOT:     cir.load
// CIR:         cir.call @_Z12takeNonByval8WithDtor(%[[TMP]]) : (!cir.ptr<!rec_WithDtor> {llvm.align = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nofreeobj, llvm.noundef}) -> ()
// CIR:         cir.call @_ZN8WithDtorD1Ev(%[[TMP]])
// CIR:         cir.call @_ZN8WithDtorD1Ev(%[[T]])

// LLVM-LABEL: define dso_local void @_Z12callNonByvalv()
// LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[TMP:[^,]+]], ptr align 4 %[[T:[^,]+]], i64 4, i1 false)
// LLVM:         call void @_Z12takeNonByval8WithDtor(ptr nofreeobj noundef align 4 dereferenceable(4) %[[TMP]])
// LLVM-CIR:     call void @_ZN8WithDtorD1Ev(ptr noundef nonnull align 4 dereferenceable(4) %[[TMP]])
// LLVM-CIR:     call void @_ZN8WithDtorD1Ev(ptr noundef nonnull align 4 dereferenceable(4) %[[T]])
// OGCG:         call void @_ZN8WithDtorD1Ev(ptr noundef nonnull align 4 dead_on_return(4) dereferenceable(4) %[[TMP]])
// OGCG:         call void @_ZN8WithDtorD1Ev(ptr noundef nonnull align 4 dead_on_return(4) dereferenceable(4) %[[T]])

// Each non-byval argument forwards its own temporary.
void callTwoNonByval() {
  WithDtor a, b;
  takeTwoNonByval(a, b);
}

// CIR-LABEL: cir.func {{.*}}@_Z15callTwoNonByvalv
// CIR:         %[[TMP_A:.*]] = cir.alloca "agg.tmp0" align(4) : !cir.ptr<!rec_WithDtor>
// CIR:         %[[TMP_B:.*]] = cir.alloca "agg.tmp1" align(4) : !cir.ptr<!rec_WithDtor>
// CIR-NOT:     cir.load
// CIR:         cir.call @_Z15takeTwoNonByval8WithDtorS_(%[[TMP_A]], %[[TMP_B]]) : (!cir.ptr<!rec_WithDtor> {llvm.align = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nofreeobj, llvm.noundef}, !cir.ptr<!rec_WithDtor> {llvm.align = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nofreeobj, llvm.noundef}) -> ()
// CIR:         cir.call @_ZN8WithDtorD1Ev(%[[TMP_B]])
// CIR:         cir.call @_ZN8WithDtorD1Ev(%[[TMP_A]])

// LLVM-LABEL: define dso_local void @_Z15callTwoNonByvalv()
// LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[TMP_A:[^,]+]], ptr align 4 %{{[^,]+}}, i64 4, i1 false)
// LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[TMP_B:[^,]+]], ptr align 4 %{{[^,]+}}, i64 4, i1 false)
// LLVM:         call void @_Z15takeTwoNonByval8WithDtorS_(ptr nofreeobj noundef align 4 dereferenceable(4) %[[TMP_A]], ptr nofreeobj noundef align 4 dereferenceable(4) %[[TMP_B]])
// LLVM-CIR:     call void @_ZN8WithDtorD1Ev(ptr noundef nonnull align 4 dereferenceable(4) %[[TMP_B]])
// LLVM-CIR:     call void @_ZN8WithDtorD1Ev(ptr noundef nonnull align 4 dereferenceable(4) %[[TMP_A]])
// OGCG:         call void @_ZN8WithDtorD1Ev(ptr noundef nonnull align 4 dead_on_return(4) dereferenceable(4) %[[TMP_B]])
// OGCG:         call void @_ZN8WithDtorD1Ev(ptr noundef nonnull align 4 dead_on_return(4) dereferenceable(4) %[[TMP_A]])

// A non-trivial copy constructor also classifies non-byval: the constructor
// call populates the forwarded temporary directly, with no load in between.
void callCopyCtorNonByval() {
  WithCopyCtor c;
  takeCopyCtorNonByval(c);
}

// CIR-LABEL: cir.func {{.*}}@_Z20callCopyCtorNonByvalv
// CIR:         %[[C:.*]] = cir.alloca "c" align(4) init : !cir.ptr<!rec_WithCopyCtor>
// CIR:         %[[TMP:.*]] = cir.alloca "agg.tmp0" align(4) : !cir.ptr<!rec_WithCopyCtor>
// CIR:         cir.call @_ZN12WithCopyCtorC1Ev(%[[C]])
// CIR:         cir.call @_ZN12WithCopyCtorC1ERKS_(%[[TMP]], %[[C]])
// CIR-NOT:     cir.load
// CIR:         cir.call @_Z20takeCopyCtorNonByval12WithCopyCtor(%[[TMP]]) : (!cir.ptr<!rec_WithCopyCtor> {llvm.align = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nofreeobj, llvm.noundef}) -> ()

// LLVM-LABEL: define dso_local void @_Z20callCopyCtorNonByvalv()
// LLVM:         call void @_ZN12WithCopyCtorC1Ev(ptr noundef nonnull align 4 dereferenceable(4) %[[C:[^)]+]])
// LLVM:         call void @_ZN12WithCopyCtorC1ERKS_(ptr noundef nonnull align 4 dereferenceable(4) %[[TMP:[^,]+]], ptr noundef nonnull align 4 dereferenceable(4) %[[C]])
// LLVM-CIR:     call void @_Z20takeCopyCtorNonByval12WithCopyCtor(ptr nofreeobj noundef align 4 dereferenceable(4) %[[TMP]])
// OGCG:         call void @_Z20takeCopyCtorNonByval12WithCopyCtor(ptr nofreeobj noundef align 4 dead_on_return dereferenceable(4) %[[TMP]])

// byval keeps the fresh copy the callee owns.
void callByval() {
  Big b;
  takeByval(b);
}

// CIR-LABEL: cir.func {{.*}}@_Z9callByvalv
// CIR:         %[[TMP:.*]] = cir.alloca "agg.tmp0" align(8) : !cir.ptr<!rec_Big>
// CIR:         %[[V:.*]] = cir.load align(8) %[[TMP]] : !cir.ptr<!rec_Big>, !rec_Big
// CIR:         %[[SLOT:.*]] = cir.alloca "byval" align(8) : !cir.ptr<!rec_Big>
// CIR:         cir.store %[[V]], %[[SLOT]] : !rec_Big, !cir.ptr<!rec_Big>
// CIR:         cir.call @_Z9takeByval3Big(%[[SLOT]]) : (!cir.ptr<!rec_Big> {llvm.align = 8 : i64, llvm.byval = !rec_Big, llvm.noundef}) -> ()

// LLVM-LABEL: define dso_local void @_Z9callByvalv()
// LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[TMP:[^,]+]], ptr align 8 %{{[^,]+}}, i64 32, i1 false)
// LLVM-CIR:     %[[V:.*]] = load %struct.Big, ptr %[[TMP]], align 8
// LLVM-CIR:     store %struct.Big %[[V]], ptr %[[SLOT:.*]], align 8
// LLVM-CIR:     call void @_Z9takeByval3Big(ptr noundef byval(%struct.Big) align 8 %[[SLOT]])
// OGCG:         call void @_Z9takeByval3Big(ptr noundef byval(%struct.Big) align 8 %[[TMP]])

// An inherited constructor forwards its by-value parameter with no temporary
// of its own, so the base constructor operates on the object the caller
// destroys.
struct Base { Base(WithDtor t); };
struct Derived : Base { using Base::Base; };
void callInheritedCtor(WithDtor t) { Derived d(t); }

// The caller materializes a temporary and passes that, which is the boundary
// the forwarding below sits against.
// CIR-LABEL: cir.func {{.*}}@_Z17callInheritedCtor8WithDtor
// CIR:         %[[TMP:.*]] = cir.alloca "agg.tmp0" align(4) : !cir.ptr<!rec_WithDtor>
// CIR:         cir.copy %{{.*}} to %[[TMP]]
// CIR:         cir.call @_ZN7DerivedCI14BaseE8WithDtor(%{{.*}}, %[[TMP]])
// CIR-SAME:      llvm.dereferenceable = 4 : i64, llvm.nofreeobj, llvm.noundef

// LLVM:         define dso_local void @_Z17callInheritedCtor8WithDtor(ptr nofreeobj noundef align 4 dereferenceable(4) %[[INHARG:[^,)]+]])
// LLVM:          %[[INHTMP:.+]] = alloca %struct.WithDtor, align 4
// LLVM:          call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[INHTMP]], ptr align 4 %[[INHARG]], i64 4, i1 false)
// LLVM:          call void @_ZN7DerivedCI14BaseE8WithDtor(ptr noundef nonnull align 1 dereferenceable(1) %{{.+}}, ptr nofreeobj noundef align 4 dereferenceable(4) %[[INHTMP]])
// LLVM-CIR:      call void @_ZN8WithDtorD1Ev(ptr noundef nonnull align 4 dereferenceable(4) %[[INHTMP]])
// OGCG:          call void @_ZN8WithDtorD1Ev(ptr noundef nonnull align 4 dead_on_return(4) dereferenceable(4) %[[INHTMP]])

// Both inheriting constructor variants hand their own parameter on unchanged.
// CIR-LABEL: cir.func {{.*}}@_ZN7DerivedCI14BaseE8WithDtor
// CIR-SAME:      %[[CI1ARG:[^:]*]]: !cir.ptr<!rec_WithDtor> {llvm.align = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nofreeobj, llvm.noundef}
// CIR-NOT:     cir.copy
// CIR:         cir.call @_ZN7DerivedCI24BaseE8WithDtor(%{{.*}}, %[[CI1ARG]])

// CIR-LABEL: cir.func {{.*}}@_ZN7DerivedCI24BaseE8WithDtor
// CIR-SAME:      %[[ARG:[^:]*]]: !cir.ptr<!rec_WithDtor> {llvm.align = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nofreeobj, llvm.noundef}
// CIR-NOT:     cir.copy
// CIR:         cir.call @_ZN4BaseC2E8WithDtor(%{{.*}}, %[[ARG]])

// LLVM-CIR:     define {{.*}}@_ZN7DerivedCI14BaseE8WithDtor(ptr noundef nonnull align 1 dereferenceable(1) %{{[^,]+}}, ptr nofreeobj noundef align 4 dereferenceable(4) %[[CI1ARG:[^,)]+]])
// OGCG:         define {{.*}}@_ZN7DerivedCI14BaseE8WithDtor(ptr noundef nonnull align 1 dereferenceable(1) %{{[^,]+}}, ptr nofreeobj noundef align 4 dereferenceable(4) %[[CI1ARG:[^,)]+]])
// LLVM-NOT:     alloca %struct.WithDtor
// LLVM:         call void @_ZN7DerivedCI24BaseE8WithDtor(ptr noundef nonnull align 1 dereferenceable(1) %{{[^,]+}}, ptr nofreeobj noundef align 4 dereferenceable(4) %[[CI1ARG]])
// LLVM-CIR:     define {{.*}}@_ZN7DerivedCI24BaseE8WithDtor(ptr noundef nonnull align 1 dereferenceable(1) %{{[^,]+}}, ptr nofreeobj noundef align 4 dereferenceable(4) %[[ARG:[^,)]+]])
// OGCG:         define {{.*}}@_ZN7DerivedCI24BaseE8WithDtor(ptr noundef nonnull align 1 dereferenceable(1) %{{[^,]+}}, ptr nofreeobj noundef align 4 dereferenceable(4) %[[ARG:[^,)]+]])
// LLVM-NOT:     alloca %struct.WithDtor
// LLVM:         call void @_ZN4BaseC2E8WithDtor(ptr noundef nonnull align 1 dereferenceable(1) %{{[^,]+}}, ptr nofreeobj noundef align 4 dereferenceable(4) %[[ARG]])
