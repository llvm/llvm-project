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

void takeByref(WithDtor t);
void takeTwoByref(WithDtor a, WithDtor b);
void takeByval(Big b);
void takeCopyCtorByref(WithCopyCtor c);

// The callee must receive the temporary the caller destroys, not a copy of it.
void callByref() {
  WithDtor t;
  takeByref(t);
}

// CIR-LABEL: cir.func {{.*}}@_Z9callByrefv
// CIR:         %[[T:.*]] = cir.alloca "t" align(4) : !cir.ptr<!rec_WithDtor>
// CIR:         %[[TMP:.*]] = cir.alloca "agg.tmp0" align(4) : !cir.ptr<!rec_WithDtor>
// CIR:         cir.copy %[[T]] align(4) to %[[TMP]] align(4) : !cir.ptr<!rec_WithDtor>
// CIR-NOT:     cir.load
// CIR:         cir.call @_Z9takeByref8WithDtor(%[[TMP]]) : (!cir.ptr<!rec_WithDtor> {llvm.align = 4 : i64, llvm.byref = !rec_WithDtor}) -> ()
// CIR:         cir.call @_ZN8WithDtorD1Ev(%[[TMP]])
// CIR:         cir.call @_ZN8WithDtorD1Ev(%[[T]])

// LLVM-LABEL: define dso_local void @_Z9callByrefv()
// LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[TMP:[^,]+]], ptr align 4 %[[T:[^,]+]], i64 4, i1 false)
// CIR marks the byref argument and drops the ownership and dereferenceability
// attrs classic emits, and classic adds dead_on_return on the destructor calls.
// LLVM-CIR:     call void @_Z9takeByref8WithDtor(ptr byref(%struct.WithDtor) align 4 %[[TMP]])
// LLVM-CIR:     call void @_ZN8WithDtorD1Ev(ptr noundef nonnull align 4 dereferenceable(4) %[[TMP]])
// LLVM-CIR:     call void @_ZN8WithDtorD1Ev(ptr noundef nonnull align 4 dereferenceable(4) %[[T]])
// OGCG:         call void @_Z9takeByref8WithDtor(ptr nofreeobj noundef align 4 dereferenceable(4) %[[TMP]])
// OGCG:         call void @_ZN8WithDtorD1Ev(ptr noundef nonnull align 4 dead_on_return(4) dereferenceable(4) %[[TMP]])
// OGCG:         call void @_ZN8WithDtorD1Ev(ptr noundef nonnull align 4 dead_on_return(4) dereferenceable(4) %[[T]])

// Each byref argument forwards its own temporary.
void callTwoByref() {
  WithDtor a, b;
  takeTwoByref(a, b);
}

// CIR-LABEL: cir.func {{.*}}@_Z12callTwoByrefv
// CIR:         %[[TMP_A:.*]] = cir.alloca "agg.tmp0" align(4) : !cir.ptr<!rec_WithDtor>
// CIR:         %[[TMP_B:.*]] = cir.alloca "agg.tmp1" align(4) : !cir.ptr<!rec_WithDtor>
// CIR-NOT:     cir.load
// CIR:         cir.call @_Z12takeTwoByref8WithDtorS_(%[[TMP_A]], %[[TMP_B]]) : (!cir.ptr<!rec_WithDtor> {llvm.align = 4 : i64, llvm.byref = !rec_WithDtor}, !cir.ptr<!rec_WithDtor> {llvm.align = 4 : i64, llvm.byref = !rec_WithDtor}) -> ()
// CIR:         cir.call @_ZN8WithDtorD1Ev(%[[TMP_B]])
// CIR:         cir.call @_ZN8WithDtorD1Ev(%[[TMP_A]])

// LLVM-LABEL: define dso_local void @_Z12callTwoByrefv()
// LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[TMP_A:[^,]+]], ptr align 4 %{{[^,]+}}, i64 4, i1 false)
// LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[TMP_B:[^,]+]], ptr align 4 %{{[^,]+}}, i64 4, i1 false)
// LLVM-CIR:     call void @_Z12takeTwoByref8WithDtorS_(ptr byref(%struct.WithDtor) align 4 %[[TMP_A]], ptr byref(%struct.WithDtor) align 4 %[[TMP_B]])
// LLVM-CIR:     call void @_ZN8WithDtorD1Ev(ptr noundef nonnull align 4 dereferenceable(4) %[[TMP_B]])
// LLVM-CIR:     call void @_ZN8WithDtorD1Ev(ptr noundef nonnull align 4 dereferenceable(4) %[[TMP_A]])
// OGCG:         call void @_Z12takeTwoByref8WithDtorS_(ptr nofreeobj noundef align 4 dereferenceable(4) %[[TMP_A]], ptr nofreeobj noundef align 4 dereferenceable(4) %[[TMP_B]])
// OGCG:         call void @_ZN8WithDtorD1Ev(ptr noundef nonnull align 4 dead_on_return(4) dereferenceable(4) %[[TMP_B]])
// OGCG:         call void @_ZN8WithDtorD1Ev(ptr noundef nonnull align 4 dead_on_return(4) dereferenceable(4) %[[TMP_A]])

// A non-trivial copy constructor also classifies byref: the constructor call
// populates the forwarded temporary directly, with no load in between.
void callCopyCtorByref() {
  WithCopyCtor c;
  takeCopyCtorByref(c);
}

// CIR-LABEL: cir.func {{.*}}@_Z17callCopyCtorByrefv
// CIR:         %[[C:.*]] = cir.alloca "c" align(4) init : !cir.ptr<!rec_WithCopyCtor>
// CIR:         %[[TMP:.*]] = cir.alloca "agg.tmp0" align(4) : !cir.ptr<!rec_WithCopyCtor>
// CIR:         cir.call @_ZN12WithCopyCtorC1Ev(%[[C]])
// CIR:         cir.call @_ZN12WithCopyCtorC1ERKS_(%[[TMP]], %[[C]])
// CIR-NOT:     cir.load
// CIR:         cir.call @_Z17takeCopyCtorByref12WithCopyCtor(%[[TMP]]) : (!cir.ptr<!rec_WithCopyCtor> {llvm.align = 4 : i64, llvm.byref = !rec_WithCopyCtor}) -> ()

// LLVM-LABEL: define dso_local void @_Z17callCopyCtorByrefv()
// LLVM:         call void @_ZN12WithCopyCtorC1Ev(ptr noundef nonnull align 4 dereferenceable(4) %[[C:[^)]+]])
// LLVM:         call void @_ZN12WithCopyCtorC1ERKS_(ptr noundef nonnull align 4 dereferenceable(4) %[[TMP:[^,]+]], ptr noundef nonnull align 4 dereferenceable(4) %[[C]])
// LLVM-CIR:     call void @_Z17takeCopyCtorByref12WithCopyCtor(ptr byref(%struct.WithCopyCtor) align 4 %[[TMP]])
// OGCG:         call void @_Z17takeCopyCtorByref12WithCopyCtor(ptr nofreeobj noundef align 4 dead_on_return dereferenceable(4) %[[TMP]])

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
// CIR:         cir.call @_Z9takeByval3Big(%[[SLOT]]) : (!cir.ptr<!rec_Big> {llvm.align = 8 : i64, llvm.byval = !rec_Big, llvm.noalias, llvm.noundef}) -> ()

// LLVM-LABEL: define dso_local void @_Z9callByvalv()
// LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[TMP:[^,]+]], ptr align 8 %{{[^,]+}}, i64 32, i1 false)
// LLVM-CIR:     %[[V:.*]] = load %struct.Big, ptr %[[TMP]], align 8
// LLVM-CIR:     store %struct.Big %[[V]], ptr %[[SLOT:.*]], align 8
// LLVM-CIR:     call void @_Z9takeByval3Big(ptr noalias noundef byval(%struct.Big) align 8 %[[SLOT]])
// OGCG:         call void @_Z9takeByval3Big(ptr noundef byval(%struct.Big) align 8 %[[TMP]])
