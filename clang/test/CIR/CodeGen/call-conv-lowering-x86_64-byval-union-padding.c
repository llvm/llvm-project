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

// CIR:      cir.func{{.*}} @callReadW() -> !u32i
// CIR:        %[[U:.*]] = cir.alloca "u" align(8) init : !cir.ptr<!rec_U>
// CIR:        %[[SLOT:.*]] = cir.alloca "byval" align(8) : !cir.ptr<!rec_U>
// CIR-NEXT:   cir.copy %[[U]] align(8) to %[[SLOT]] align(8) : !cir.ptr<!rec_U>
// CIR-NEXT:   %{{.*}} = cir.call @readW(%[[SLOT]]) : (!cir.ptr<!rec_U> {llvm.align = 8 : i64, llvm.byval = !rec_U, llvm.noundef}) -> !u32i

// LLVM:       define dso_local i32 @callReadW()
// LLVM-CIR:     %[[U:.+]] = alloca %union.U, align 8
// LLVM-CIR:     %[[SLOT:.+]] = alloca %union.U, align 8
// LLVM-CIR:     call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[SLOT]], ptr align 8 %[[U]], i64 24, i1 false)
// LLVM-CIR:     %{{.+}} = call i32 @readW(ptr noundef byval(%union.U) align 8 %[[SLOT]])
// OGCG:         %[[U:.+]] = alloca %union.U, align 8
// OGCG:         %{{.+}} = call i32 @readW(ptr noundef byval(%union.U) align 8 %[[U]])

// Read through a pointer, so the operand's load has no alloca behind it.
unsigned readWThroughPtr(union U *src) { return readW(*src); }

// CIR:      cir.func{{.*}} @readWThroughPtr(%arg0: !cir.ptr<!rec_U> {llvm.noundef} loc({{.+}})) -> !u32i
// CIR:        %[[SRC:.*]] = cir.load deref align(8) %{{.*}} : !cir.ptr<!cir.ptr<!rec_U>>, !cir.ptr<!rec_U>
// CIR-NEXT:   %[[SLOT:.*]] = cir.alloca "byval" align(8) : !cir.ptr<!rec_U>
// CIR-NEXT:   cir.copy %[[SRC]] align(8) to %[[SLOT]] align(8) : !cir.ptr<!rec_U>
// CIR-NEXT:   %{{.*}} = cir.call @readW(%[[SLOT]]) : (!cir.ptr<!rec_U> {llvm.align = 8 : i64, llvm.byval = !rec_U, llvm.noundef}) -> !u32i

// LLVM:       define dso_local i32 @readWThroughPtr(ptr noundef %{{.+}})
// LLVM-CIR:     %[[SRC:.+]] = load ptr, ptr %{{.+}}, align 8
// LLVM-CIR:     %[[SLOT:.+]] = alloca %union.U, align 8
// LLVM-CIR-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[SLOT]], ptr align 8 %[[SRC]], i64 24, i1 false)
// LLVM-CIR:     %{{.+}} = call i32 @readW(ptr noundef byval(%union.U) align 8 %[[SLOT]])
// OGCG:         %[[SRC:.+]] = load ptr, ptr %{{.+}}, align 8
// OGCG:         %{{.+}} = call i32 @readW(ptr noundef byval(%union.U) align 8 %[[SRC]])

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
