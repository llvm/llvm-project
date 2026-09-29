// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// A union passed byval is copied as bytes. Its LLVM type is built from one
// member, so a record load/store would drop bytes that are padding in that
// member but data in another: bytes 4-7, y.b, here.
union U {
  struct { int a; void *p, *q; } x;
  struct { int a, b; void *p, *q; } y;
};

int get_b(union U u) { return u.y.b; }

void pass(void) {
  union U u;
  u.y.b = 42;
  get_b(u);
}

// CIR-LABEL: cir.func {{.*}}@get_b(%arg0: !cir.ptr<!rec_U> {llvm.align = 8 : i64, llvm.byval = !rec_U, llvm.noundef}
// CIR:         %[[U:.*]] = cir.alloca "u" align(8) init : !cir.ptr<!rec_U>
// CIR:         cir.copy %arg0 align(8) to %[[U]] : !cir.ptr<!rec_U>
// CIR-NOT:     cir.load {{.*}} !rec_U

// CIR-LABEL: cir.func {{.*}}@pass()
// CIR:         %[[U:.*]] = cir.alloca "u" align(8) : !cir.ptr<!rec_U>
// CIR:         %[[SLOT:.*]] = cir.alloca "byval" align(8) : !cir.ptr<!rec_U>
// CIR-NEXT:    cir.copy %[[U]] align(8) to %[[SLOT]] align(8) : !cir.ptr<!rec_U>
// CIR-NEXT:    cir.call @get_b(%[[SLOT]])

// LLVM-LABEL: define {{.*}}i32 @get_b(ptr noundef byval(%union.U) align 8 %0)
// LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr align 8 %{{.+}}, ptr align 8 %0, i64 24, i1 false)
// LLVM-NOT:     load %union.U

// LLVM-LABEL: define {{.*}}void @pass()
// LLVM:         %[[U:.+]] = alloca %union.U, align 8
// LLVM:         %[[SLOT:.+]] = alloca %union.U, align 8
// LLVM-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[SLOT]], ptr align 8 %[[U]], i64 24, i1 false)
// LLVM-NEXT:    call i32 @get_b(ptr noundef byval(%union.U) align 8 %[[SLOT]])

// FIXME: Classic codegen copies neither the parameter nor the argument, so
// LLVM and OGCG differ at -O0 (not at -O2, where the copies go away). To match:
// - callee: use the byval pointer as the parameter's storage instead of
//   copying it into the spill slot, as CallConvLowering already does for
//   non-byval indirect parameters; copy only when the slot needs more
//   alignment than the byval pointer has.
// - caller: pass the address the argument was loaded from straight to the
//   byval call (byval already gives the callee its own copy), when it is
//   aligned enough and nothing writes that memory between the load and call.

// OGCG-LABEL: define {{.*}}i32 @get_b(
// OGCG-SAME:    ptr noundef byval(%union.U) align 8 %[[U:.+]])
// OGCG-NOT:     call void @llvm.memcpy
// OGCG:         %[[B:.+]] = getelementptr inbounds nuw %struct.anon.0, ptr %[[U]], i32 0, i32 1
// OGCG-NEXT:    load i32, ptr %[[B]], align 4

// OGCG-LABEL: define {{.*}}void @pass()
// OGCG:         %[[U:.+]] = alloca %union.U, align 8
// OGCG-NOT:     call void @llvm.memcpy
// OGCG:         call i32 @get_b(ptr noundef byval(%union.U) align 8 %[[U]])
