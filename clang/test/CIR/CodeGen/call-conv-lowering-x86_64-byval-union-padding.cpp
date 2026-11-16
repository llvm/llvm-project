// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-CIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t.ll %s

struct DA {
  unsigned a : 31;
  unsigned b : 1;
  void *p;
  void *q;
};

struct I {
  unsigned a : 31;
  unsigned b : 1;
  unsigned w : 31;
  unsigned u : 1;
  unsigned long v;
  void *t;
};

union U {
  DA d;
  I i;
};

unsigned readW(U x) { return x.i.w; }

// CIR:      cir.func{{.*}} @_Z5readW1U(%arg0: !cir.ptr<!rec_U> {llvm.align = 8 : i64, llvm.byval = !rec_U, llvm.noundef} loc({{.+}})) -> (!u32i {llvm.noundef})
// CIR:        %[[X:.*]] = cir.alloca "x" align(8) init : !cir.ptr<!rec_U>
// CIR:        cir.copy %arg0 align(8) to %[[X]] align(8) : !cir.ptr<!rec_U>

// LLVM:       define dso_local noundef i32 @_Z5readW1U(ptr noundef byval(%union.U) align 8 %[[ARG:.+]])
// LLVM-CIR:     %[[X:.+]] = alloca %union.U, align 8
// LLVM-CIR:     call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[X]], ptr align 8 %[[ARG]], i64 24, i1 false)
// The reference reads the incoming slot in place and needs no copy.
// OGCG-NOT:     memcpy
// OGCG:         load i64, ptr %[[ARG]], align 8

unsigned callReadW() {
  U u = {};
  u.i.w = 21845;
  return readW(u);
}

// C++ materializes agg.tmp for the by-value argument, and byval is copied at
// the call boundary, so that temporary is the byval pointer.
// CIR:      cir.func{{.*}} @_Z9callReadWv() -> (!u32i {llvm.noundef})
// CIR:        %[[U:.*]] = cir.alloca "u" align(8) init : !cir.ptr<!rec_U>
// CIR:        %[[TMP:.*]] = cir.alloca "agg.tmp0" align(8) : !cir.ptr<!rec_U>
// CIR-NOT:    cir.alloca "byval"
// CIR:        cir.copy %[[U]] align(8) to %[[TMP]] align(8) : !cir.ptr<!rec_U>
// CIR-NEXT:   %{{.*}} = cir.call @_Z5readW1U(%[[TMP]]) : (!cir.ptr<!rec_U> {llvm.align = 8 : i64, llvm.byval = !rec_U, llvm.noundef}) -> (!u32i {llvm.noundef})

// LLVM:       define dso_local noundef i32 @_Z9callReadWv()
// LLVM:         %[[U:.+]] = alloca %union.U, align 8
// LLVM:         %[[TMP:.+]] = alloca %union.U, align 8
// LLVM-NOT:     alloca %union.U
// LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[TMP]], ptr align 8 %[[U]], i64 24, i1 false)
// LLVM-NEXT:    %{{.+}} = call noundef i32 @_Z5readW1U(ptr noundef byval(%union.U) align 8 %[[TMP]])

unsigned callReadWFromRef(const U &src) { return readW(src); }

// The same shape with no local object behind the temporary.
// CIR:      cir.func{{.*}} @_Z16callReadWFromRefRK1U
// CIR:        %[[TMP:.*]] = cir.alloca "agg.tmp0" align(8) : !cir.ptr<!rec_U>
// CIR-NOT:    cir.alloca "byval"
// CIR:        cir.copy %{{.*}} align(8) to %[[TMP]] align(8) : !cir.ptr<!rec_U>
// CIR-NEXT:   %{{.*}} = cir.call @_Z5readW1U(%[[TMP]]) : (!cir.ptr<!rec_U> {llvm.align = 8 : i64, llvm.byval = !rec_U, llvm.noundef}) -> (!u32i {llvm.noundef})

// LLVM:       define dso_local noundef i32 @_Z16callReadWFromRefRK1U(ptr noundef nonnull align 8 dereferenceable(24) %{{.+}})
// LLVM:         %[[TMP:.+]] = alloca %union.U, align 8
// LLVM-NOT:     alloca %union.U
// LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[TMP]], ptr align 8 %{{.+}}, i64 24, i1 false)
// LLVM-NEXT:    %{{.+}} = call noundef i32 @_Z5readW1U(ptr noundef byval(%union.U) align 8 %[[TMP]])
