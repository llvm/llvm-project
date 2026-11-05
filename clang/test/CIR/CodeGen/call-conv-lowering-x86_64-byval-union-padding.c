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
  struct DA d;
  struct I i;
};

unsigned readW(union U x) { return x.i.w; }

// CIR:      cir.func{{.*}} @readW(%arg0: !cir.ptr<!rec_U> {llvm.align = 8 : i64, llvm.byval = !rec_U, llvm.noundef} loc({{.+}})) -> !u32i
// CIR:        %[[X:.*]] = cir.alloca "x" align(8) init : !cir.ptr<!rec_U>
// CIR:        cir.copy %arg0 align(8) to %[[X]] align(8) : !cir.ptr<!rec_U>

// LLVM:       define dso_local i32 @readW(ptr noundef byval(%union.U) align 8 %[[ARG:.+]])
// LLVM-CIR:     %[[X:.+]] = alloca %union.U, align 8
// LLVM-CIR:     call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[X]], ptr align 8 %[[ARG]], i64 24, i1 false)
// The reference reads the incoming slot in place and needs no copy.
// OGCG-NOT:     memcpy
// OGCG:         load i64, ptr %[[ARG]], align 8

unsigned callReadW(void) {
  union U u = {0};
  u.i.w = 21845;
  return readW(u);
}

// `u` is the operand's storage, so it is handed on as the byval pointer.
// CIR:      cir.func{{.*}} @callReadW() -> !u32i
// CIR:        %[[U:.*]] = cir.alloca "u" align(8) init : !cir.ptr<!rec_U>
// CIR-NOT:    cir.alloca "byval"
// CIR:        %{{.*}} = cir.call @readW(%[[U]]) : (!cir.ptr<!rec_U> {llvm.align = 8 : i64, llvm.byval = !rec_U, llvm.noundef}) -> !u32i

// LLVM:       define dso_local i32 @callReadW()
// LLVM:         %[[U:.+]] = alloca %union.U, align 8
// LLVM-NOT:     alloca %union.U
// LLVM:         %{{.+}} = call i32 @readW(ptr noundef byval(%union.U) align 8 %[[U]])

// Read through a pointer, so the operand's load has no alloca behind it and
// the byval argument is filled by a copy instead.
unsigned readWThroughPtr(union U *src) { return readW(*src); }

// CIR:      cir.func{{.*}} @readWThroughPtr(%arg0: !cir.ptr<!rec_U> {llvm.noundef} loc({{.+}})) -> !u32i
// CIR:        %[[SRC:.*]] = cir.load deref align(8) %{{.*}} : !cir.ptr<!cir.ptr<!rec_U>>, !cir.ptr<!rec_U>
// CIR-NEXT:   %[[SLOT:.*]] = cir.alloca "byval" align(8) : !cir.ptr<!rec_U>
// CIR-NEXT:   cir.copy %[[SRC]] align(8) to %[[SLOT]] align(8) : !cir.ptr<!rec_U>
// CIR-NEXT:   %{{.*}} = cir.call @readW(%[[SLOT]]) : (!cir.ptr<!rec_U> {llvm.align = 8 : i64, llvm.byval = !rec_U, llvm.noundef}) -> !u32i

// LLVM:       define dso_local i32 @readWThroughPtr(ptr noundef %{{.+}})
// LLVM:         %[[SRC:.+]] = load ptr, ptr %{{.+}}, align 8
// LLVM-CIR-NEXT: %[[SLOT:.+]] = alloca %union.U, align 8
// LLVM-CIR-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[SLOT]], ptr align 8 %[[SRC]], i64 24, i1 false)
// LLVM-CIR-NEXT: %{{.+}} = call i32 @readW(ptr noundef byval(%union.U) align 8 %[[SLOT]])
// OGCG-NEXT:    %{{.+}} = call i32 @readW(ptr noundef byval(%union.U) align 8 %[[SRC]])

// Forwarding the operand's own storage rests on nothing writing that storage
// between the load and the call.  The assignment here is sequenced before the
// argument is taken, so the callee has to see 7.
unsigned commaSeqW(void) {
  union U u = {0};
  u.i.w = 21845;
  return readW((u.i.w = 7, u));
}

// CIR:      cir.func{{.*}} @commaSeqW() -> !u32i
// CIR:        %[[U:.*]] = cir.alloca "u" align(8) init : !cir.ptr<!rec_U>
// CIR:        %[[TMP:.*]] = cir.alloca "agg.tmp0" align(8) : !cir.ptr<!rec_U>
// CIR-NOT:    cir.alloca "byval"
// CIR:        %[[SEVEN:.*]] = cir.const #cir.int<7> : !u32i
// CIR:        %{{.*}} = cir.set_bitfield align(8) (#bfi_w, %{{.*}} : !cir.ptr<!u64i>, %[[SEVEN]] : !u32i) -> !u32i
// CIR-NEXT:   cir.copy %[[U]] align(8) to %[[TMP]] align(8) : !cir.ptr<!rec_U>
// CIR-NEXT:   %{{.*}} = cir.call @readW(%[[TMP]]) : (!cir.ptr<!rec_U> {llvm.align = 8 : i64, llvm.byval = !rec_U, llvm.noundef}) -> !u32i

// LLVM:       define dso_local i32 @commaSeqW()
// LLVM:         %[[U:.+]] = alloca %union.U, align 8
// LLVM:         %[[TMP:.+]] = alloca %union.U, align 8
// 7 shifted into w's bit position.  The store of it precedes the copy the
// argument is taken from, so the callee reads 7 rather than 21845.  Only the
// destination pointer varies: CIR keeps the zero-offset member GEP that the
// reference folds away.
// LLVM:         %[[SEVEN:.+]] = or i64 %{{.+}}, 30064771072
// LLVM-NEXT:    store i64 %[[SEVEN]], ptr %{{.+}}, align 8
// LLVM-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[TMP]], ptr align 8 %[[U]], i64 24, i1 false)
// LLVM-NEXT:    %{{.+}} = call i32 @readW(ptr noundef byval(%union.U) align 8 %[[TMP]])

void bumpW(union U *p);

// The same ordering with the write done by an intervening call.  Here the
// byval pointer is `u` itself, so the write has to be the one the callee sees.
unsigned callSeqW(void) {
  union U u = {0};
  u.i.w = 21845;
  return (bumpW(&u), readW(u));
}

// CIR:      cir.func{{.*}} @callSeqW() -> !u32i
// CIR:        %[[U:.*]] = cir.alloca "u" align(8) init : !cir.ptr<!rec_U>
// CIR-NOT:    cir.alloca "byval"
// CIR:        cir.call @bumpW(%[[U]]) : (!cir.ptr<!rec_U> {llvm.noundef}) -> ()
// CIR-NEXT:   %{{.*}} = cir.call @readW(%[[U]]) : (!cir.ptr<!rec_U> {llvm.align = 8 : i64, llvm.byval = !rec_U, llvm.noundef}) -> !u32i

// LLVM:       define dso_local i32 @callSeqW()
// LLVM:         %[[U:.+]] = alloca %union.U, align 8
// LLVM-NOT:     alloca %union.U
// LLVM:         call void @bumpW(ptr noundef %[[U]])
// LLVM-NEXT:    %{{.+}} = call i32 @readW(ptr noundef byval(%union.U) align 8 %[[U]])

struct T {
  unsigned long a;
  unsigned long b;
  unsigned c;
};

struct V {
  unsigned long a;
  unsigned long b;
  unsigned c;
  unsigned w;
};

union UT {
  struct T t;
  struct V v;
};

// The storage type's fields span 20 bytes, so the copy must be sized from the
// union's 24.
unsigned readTail(union UT x) { return x.v.w; }

// CIR:      cir.func{{.*}} @readTail(%arg0: !cir.ptr<!rec_UT> {llvm.align = 8 : i64, llvm.byval = !rec_UT, llvm.noundef} loc({{.+}})) -> !u32i
// CIR:        %[[X:.*]] = cir.alloca "x" align(8) init : !cir.ptr<!rec_UT>
// CIR:        cir.copy %arg0 align(8) to %[[X]] align(8) : !cir.ptr<!rec_UT>

// LLVM:       define dso_local i32 @readTail(ptr noundef byval(%union.UT) align 8 %[[ARG:.+]])
// LLVM-CIR:     %[[X:.+]] = alloca %union.UT, align 8
// LLVM-CIR:     call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[X]], ptr align 8 %[[ARG]], i64 24, i1 false)
// The reference reads the incoming slot in place and needs no copy.
// OGCG-NOT:     memcpy
// OGCG:         %[[W:.+]] = getelementptr inbounds nuw %struct.V, ptr %[[ARG]], i32 0, i32 3
// OGCG-NEXT:    %{{.+}} = load i32, ptr %[[W]], align 4

unsigned callReadTail(void) {
  union UT u = {0};
  u.v.w = 21845;
  return readTail(u);
}

// CIR:      cir.func{{.*}} @callReadTail() -> !u32i
// CIR:        %[[U:.*]] = cir.alloca "u" align(8) init : !cir.ptr<!rec_UT>
// CIR-NOT:    cir.alloca "byval"
// CIR:        %{{.*}} = cir.call @readTail(%[[U]]) : (!cir.ptr<!rec_UT> {llvm.align = 8 : i64, llvm.byval = !rec_UT, llvm.noundef}) -> !u32i

// LLVM:       define dso_local i32 @callReadTail()
// LLVM:         %[[U:.+]] = alloca %union.UT, align 8
// LLVM-NOT:     alloca %union.UT
// LLVM:         %{{.+}} = call i32 @readTail(ptr noundef byval(%union.UT) align 8 %[[U]])

// The parameter's own slot is the operand's storage, so the call forwards it
// rather than building a second one.
unsigned forwardTail(union UT x) { return readTail(x); }

// CIR:      cir.func{{.*}} @forwardTail(%arg0: !cir.ptr<!rec_UT> {llvm.align = 8 : i64, llvm.byval = !rec_UT, llvm.noundef} loc({{.+}})) -> !u32i
// CIR:        %[[X:.*]] = cir.alloca "x" align(8) init : !cir.ptr<!rec_UT>
// CIR-NOT:    cir.alloca "byval"
// CIR:        cir.copy %arg0 align(8) to %[[X]] align(8) : !cir.ptr<!rec_UT>
// CIR-NEXT:   %{{.*}} = cir.call @readTail(%[[X]]) : (!cir.ptr<!rec_UT> {llvm.align = 8 : i64, llvm.byval = !rec_UT, llvm.noundef}) -> !u32i

// LLVM:       define dso_local i32 @forwardTail(ptr noundef byval(%union.UT) align 8 %[[ARG:.+]])
// LLVM-CIR:     %[[X:.+]] = alloca %union.UT, align 8
// LLVM-CIR:     call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[X]], ptr align 8 %[[ARG]], i64 24, i1 false)
// LLVM-CIR:     %{{.+}} = call i32 @readTail(ptr noundef byval(%union.UT) align 8 %[[X]])
// The reference hands the incoming slot on with no copy of its own.
// OGCG-NOT:     memcpy
// OGCG:         %{{.+}} = call i32 @readTail(ptr noundef byval(%union.UT) align 8 %[[ARG]])
