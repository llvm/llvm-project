// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-rtti -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-rtti -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-rtti -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// Note: -fno-rtti is used to simplify vtable output (no typeinfo references).

namespace Test1 {
// Non-virtual this-adjusting thunk for a void method.
// When C overrides f() from both A and B, the vtable for B-in-C needs a thunk
// that adjusts 'this' from B* to C* before calling C::f().

struct A {
  virtual void f();
};

struct B {
  virtual void f();
};

struct C : A, B {
  void f() override;
};

void C::f() {}

} // namespace Test1

namespace Test2 {
// Non-virtual this-adjusting thunk for a method with a non-void return type.
// The thunk should forward the return value from the adjusted call.

struct A {
  virtual int g();
};

struct B {
  virtual int g();
};

struct C : A, B {
  int g() override;
};

int C::g() { return 42; }

} // namespace Test2

namespace Test3 {
// Non-virtual this-adjusting thunks for destructors.
// When D inherits from both E and F, the complete and deleting destructors for
// F-in-D need thunks that adjust 'this'.

struct E {
  virtual ~E();
};

struct F {
  virtual ~F();
};

struct D : E, F {
  ~D() override;
};

D::~D() {}

} // namespace Test3

namespace Test4 {
// Non-virtual this-adjusting thunk for a method with a by-value parameter.
// This exercises the case where emitDelegateCallArg is called with an invalid
// SourceLocation during thunk generation, which requires getLoc to handle
// a null currSrcLoc gracefully.

struct A {
  virtual void f();
};

struct B {
  virtual void g(int x);
};

struct C : A, B {
  void g(int x) override;
};

void C::g(int x) {}

} // namespace Test4

namespace CovariantReturn {
// Covariant return with virtual inheritance: return-adjusting thunks use a
// null check for pointer returns (classic PerformReturnAdjustment).
struct A {
  virtual A *f();
};
struct B : virtual A {
  virtual A *f();
};
struct C : B {
  virtual C *f();
};
C *C::f() { return 0; }
} // namespace CovariantReturn

namespace VarargThunk {
// Variadic this-adjusting thunk.  On x86_64, the thunk forwards arguments
// via musttail (classic codegen) or direct argument forwarding (CIR).
struct A {
  virtual void f(int x, ...);
};
struct B {
  virtual void f(int x, ...);
};
struct C : A, B {
  void f(int x, ...) override;
};
void C::f(int x, ...) {}
} // namespace VarargThunk

// In CIR, all globals are emitted before functions.

// Test1 vtable: C's vtable references the thunk for B's entry.
// CIR-DAG: cir.global "private" external @_ZTVN5Test11CE = #cir.vtable<{
// CIR-DAG:   #cir.global_view<@_ZN5Test11C1fEv> : !cir.ptr<!u8i>
// CIR-DAG:   #cir.global_view<@_ZThn8_N5Test11C1fEv> : !cir.ptr<!u8i>

// Test2 vtable: C's vtable references the thunk for B's entry.
// CIR-DAG: cir.global "private" external @_ZTVN5Test21CE = #cir.vtable<{
// CIR-DAG:   #cir.global_view<@_ZN5Test21C1gEv> : !cir.ptr<!u8i>
// CIR-DAG:   #cir.global_view<@_ZThn8_N5Test21C1gEv> : !cir.ptr<!u8i>

// Test4 vtable: C's vtable references the thunk for B's entry.
// CIR-DAG: cir.global "private" external @_ZTVN5Test41CE = #cir.vtable<{
// CIR-DAG:   #cir.global_view<@_ZN5Test41C1gEi> : !cir.ptr<!u8i>
// CIR-DAG:   #cir.global_view<@_ZThn8_N5Test41C1gEi> : !cir.ptr<!u8i>

// Test3 vtable: D's vtable references D1, D0, and their thunks.
// CIR-DAG: cir.global "private" external @_ZTVN5Test31DE = #cir.vtable<{
// CIR-DAG:   #cir.global_view<@_ZN5Test31DD1Ev> : !cir.ptr<!u8i>
// CIR-DAG:   #cir.global_view<@_ZN5Test31DD0Ev> : !cir.ptr<!u8i>
// CIR-DAG:   #cir.global_view<@_ZThn8_N5Test31DD1Ev> : !cir.ptr<!u8i>
// CIR-DAG:   #cir.global_view<@_ZThn8_N5Test31DD0Ev> : !cir.ptr<!u8i>

// --- Test1: void method thunk ---

// CIR: cir.func {{.*}} @_ZN5Test11C1fEv

// The thunk adjusts 'this' by -8 bytes and calls C::f().
// CIR: cir.func {{.*}} @_ZThn8_N5Test11C1fEv(%arg0: !cir.ptr<
// CIR:   %[[T1_THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CIR:   cir.store %arg0, %[[T1_THIS_ADDR]]
// CIR:   %[[T1_THIS:.*]] = cir.load %[[T1_THIS_ADDR]]
// CIR:   %[[T1_CAST:.*]] = cir.cast bitcast %[[T1_THIS]] : !cir.ptr<{{.*}}> -> !cir.ptr<!u8i>
// CIR:   %[[T1_OFFSET:.*]] = cir.const #cir.int<-8> : !s64i
// CIR:   %[[T1_ADJUSTED:.*]] = cir.ptr_stride %[[T1_CAST]], %[[T1_OFFSET]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR:   %[[T1_RESULT:.*]] = cir.cast bitcast %[[T1_ADJUSTED]] : !cir.ptr<!u8i> -> !cir.ptr<
// CIR:   cir.call @_ZN5Test11C1fEv(%[[T1_RESULT]])
// CIR:   cir.return

// --- Test2: non-void return type thunk ---

// CIR: cir.func {{.*}} @_ZN5Test21C1gEv

// CIR: cir.func {{.*}} @_ZThn8_N5Test21C1gEv(%arg0: !cir.ptr<
// CIR:   %[[T2_THIS:.*]] = cir.load
// CIR:   %[[T2_CAST:.*]] = cir.cast bitcast %[[T2_THIS]] : !cir.ptr<{{.*}}> -> !cir.ptr<!u8i>
// CIR:   %[[T2_OFFSET:.*]] = cir.const #cir.int<-8> : !s64i
// CIR:   %[[T2_ADJUSTED:.*]] = cir.ptr_stride %[[T2_CAST]], %[[T2_OFFSET]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR:   %[[T2_RESULT:.*]] = cir.cast bitcast %[[T2_ADJUSTED]] : !cir.ptr<!u8i> -> !cir.ptr<
// CIR:   %[[T2_RET:.*]] = cir.call @_ZN5Test21C1gEv(%[[T2_RESULT]])
// CIR:   cir.store {{.*}} %[[T2_RET]]
// CIR:   %[[T2_RET_VAL:.*]] = cir.load
// CIR:   cir.return %[[T2_RET_VAL]]

// --- Test3: destructor thunks ---

// Complete destructor thunk: adjusts 'this' and calls D1.
// CIR: cir.func {{.*}} @_ZThn8_N5Test31DD1Ev(%arg0: !cir.ptr<
// CIR:   %[[T3A_THIS:.*]] = cir.load
// CIR:   %[[T3A_CAST:.*]] = cir.cast bitcast %[[T3A_THIS]] : !cir.ptr<{{.*}}> -> !cir.ptr<!u8i>
// CIR:   %[[T3A_OFFSET:.*]] = cir.const #cir.int<-8> : !s64i
// CIR:   %[[T3A_ADJUSTED:.*]] = cir.ptr_stride %[[T3A_CAST]], %[[T3A_OFFSET]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR:   %[[T3A_RESULT:.*]] = cir.cast bitcast %[[T3A_ADJUSTED]] : !cir.ptr<!u8i> -> !cir.ptr<
// CIR:   cir.call @_ZN5Test31DD1Ev(%[[T3A_RESULT]])
// CIR:   cir.return

// Deleting destructor thunk: adjusts 'this' and calls D0.
// CIR: cir.func {{.*}} @_ZThn8_N5Test31DD0Ev(%arg0: !cir.ptr<
// CIR:   %[[T3B_THIS:.*]] = cir.load
// CIR:   %[[T3B_CAST:.*]] = cir.cast bitcast %[[T3B_THIS]] : !cir.ptr<{{.*}}> -> !cir.ptr<!u8i>
// CIR:   %[[T3B_OFFSET:.*]] = cir.const #cir.int<-8> : !s64i
// CIR:   %[[T3B_ADJUSTED:.*]] = cir.ptr_stride %[[T3B_CAST]], %[[T3B_OFFSET]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR:   %[[T3B_RESULT:.*]] = cir.cast bitcast %[[T3B_ADJUSTED]] : !cir.ptr<!u8i> -> !cir.ptr<
// CIR:   cir.call @_ZN5Test31DD0Ev(%[[T3B_RESULT]])
// CIR:   cir.return

// --- Test4: by-value parameter thunk ---

// CIR: cir.func {{.*}} @_ZN5Test41C1gEi

// The thunk adjusts 'this' by -8 bytes, forwards the int arg, and calls C::g().
// CIR: cir.func {{.*}} @_ZThn8_N5Test41C1gEi(%arg0: !cir.ptr<
// CIR:   %[[T4_THIS:.*]] = cir.load
// CIR:   %[[T4_CAST:.*]] = cir.cast bitcast %[[T4_THIS]] : !cir.ptr<{{.*}}> -> !cir.ptr<!u8i>
// CIR:   %[[T4_OFFSET:.*]] = cir.const #cir.int<-8> : !s64i
// CIR:   %[[T4_ADJUSTED:.*]] = cir.ptr_stride %[[T4_CAST]], %[[T4_OFFSET]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR:   %[[T4_RESULT:.*]] = cir.cast bitcast %[[T4_ADJUSTED]] : !cir.ptr<!u8i> -> !cir.ptr<
// CIR:   %[[T4_ARG:.*]] = cir.load
// CIR:   cir.call @_ZN5Test41C1gEi(%[[T4_RESULT]], %[[T4_ARG]])
// CIR:   cir.return

// --- CovariantReturn: return adjustment with null check on pointer return ---

// CIR-LABEL: cir.func {{.*}} @_ZTch0_v0_n32_N15CovariantReturn1C1fEv
// CIR:       cir.call @_ZN15CovariantReturn1C1fEv
// CIR:       cir.ternary

// --- VarargThunk: variadic this-adjusting thunk ---

// CIR: cir.func {{.*}} @_ZThn8_N11VarargThunk1C1fEiz(%arg0: !cir.ptr<
// CIR:   %[[VT_THIS:.*]] = cir.load
// CIR:   %[[VT_CAST:.*]] = cir.cast bitcast %[[VT_THIS]] : !cir.ptr<{{.*}}> -> !cir.ptr<!u8i>
// CIR:   %[[VT_OFFSET:.*]] = cir.const #cir.int<-8> : !s64i
// CIR:   %[[VT_ADJUSTED:.*]] = cir.ptr_stride %[[VT_CAST]], %[[VT_OFFSET]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR:   %[[VT_RESULT:.*]] = cir.cast bitcast %[[VT_ADJUSTED]] : !cir.ptr<!u8i> -> !cir.ptr<
// CIR:   cir.call @_ZN11VarargThunk1C1fEiz(%[[VT_RESULT]], %arg1) musttail
// CIR:   cir.return

// --- LLVM checks ---

// LLVM: @_ZTVN5Test11CE = global { [3 x ptr], [3 x ptr] } {
// LLVM-SAME: [3 x ptr] [ptr null, ptr null, ptr @_ZN5Test11C1fEv],
// LLVM-SAME: [3 x ptr] [ptr inttoptr (i64 -8 to ptr), ptr null, ptr @_ZThn8_N5Test11C1fEv]
// LLVM-SAME: }

// LLVM: @_ZTVN5Test21CE = global { [3 x ptr], [3 x ptr] } {
// LLVM-SAME: [3 x ptr] [ptr null, ptr null, ptr @_ZN5Test21C1gEv],
// LLVM-SAME: [3 x ptr] [ptr inttoptr (i64 -8 to ptr), ptr null, ptr @_ZThn8_N5Test21C1gEv]
// LLVM-SAME: }

// LLVM: @_ZTVN5Test31DE = global { [4 x ptr], [4 x ptr] } {
// LLVM-SAME: [4 x ptr] [ptr null, ptr null, ptr @_ZN5Test31DD1Ev, ptr @_ZN5Test31DD0Ev],
// LLVM-SAME: [4 x ptr] [ptr inttoptr (i64 -8 to ptr), ptr null, ptr @_ZThn8_N5Test31DD1Ev, ptr @_ZThn8_N5Test31DD0Ev]
// LLVM-SAME: }

// LLVM: @_ZTVN5Test41CE = global { [4 x ptr], [3 x ptr] } {
// LLVM-SAME: [4 x ptr] [ptr null, ptr null, ptr @_ZN5Test41A1fEv, ptr @_ZN5Test41C1gEi],
// LLVM-SAME: [3 x ptr] [ptr inttoptr (i64 -8 to ptr), ptr null, ptr @_ZThn8_N5Test41C1gEi]
// LLVM-SAME: }

// LLVM: define {{.*}} void @_ZThn8_N5Test11C1fEv(ptr{{.*}})
// LLVM:   %[[L1_THIS:.*]] = load ptr, ptr
// LLVM:   %[[L1_ADJ:.*]] = getelementptr i8, ptr %[[L1_THIS]], i64 -8
// LLVM:   call void @_ZN5Test11C1fEv(ptr{{.*}} %[[L1_ADJ]])

// LLVM: define {{.*}} i32 @_ZThn8_N5Test21C1gEv(ptr{{.*}})
// LLVM:   %[[L2_THIS:.*]] = load ptr, ptr
// LLVM:   %[[L2_ADJ:.*]] = getelementptr i8, ptr %[[L2_THIS]], i64 -8
// LLVM:   %[[L2_RET:.*]] = call {{.*}} i32 @_ZN5Test21C1gEv(ptr{{.*}} %[[L2_ADJ]])

// LLVM: define {{.*}} void @_ZThn8_N5Test31DD1Ev(ptr{{.*}})
// LLVM:   %[[L3A_THIS:.*]] = load ptr, ptr
// LLVM:   %[[L3A_ADJ:.*]] = getelementptr i8, ptr %[[L3A_THIS]], i64 -8
// LLVM:   call void @_ZN5Test31DD1Ev(ptr{{.*}} %[[L3A_ADJ]])

// LLVM: define {{.*}} void @_ZThn8_N5Test31DD0Ev(ptr{{.*}})
// LLVM:   %[[L3B_THIS:.*]] = load ptr, ptr
// LLVM:   %[[L3B_ADJ:.*]] = getelementptr i8, ptr %[[L3B_THIS]], i64 -8
// LLVM:   call void @_ZN5Test31DD0Ev(ptr{{.*}} %[[L3B_ADJ]])

// LLVM: define {{.*}} void @_ZThn8_N5Test41C1gEi(ptr{{.*}}, i32{{.*}})
// LLVM:   %[[L4_THIS:.*]] = load ptr, ptr
// LLVM:   %[[L4_ADJ:.*]] = getelementptr i8, ptr %[[L4_THIS]], i64 -8
// LLVM:   %[[L4_ARG:.*]] = load i32, ptr
// LLVM:   call void @_ZN5Test41C1gEi(ptr{{.*}} %[[L4_ADJ]], i32{{.*}} %[[L4_ARG]])

// LLVM-LABEL: define {{.*}} @_ZTch0_v0_n32_N15CovariantReturn1C1fEv
// LLVM:       call {{.*}} @_ZN15CovariantReturn1C1fEv
// LLVM:       phi ptr

// LLVM-LABEL: define {{.*}} void @_ZThn8_N11VarargThunk1C1fEiz(ptr{{.*}}, i32{{.*}}, ...)
// LLVM:   getelementptr i8, ptr {{.*}}, i64 -8
// LLVM:   musttail call void (ptr, i32, ...) @_ZN11VarargThunk1C1fEiz(ptr{{.*}}, i32{{.*}}, ...)

// --- OGCG checks ---

// OGCG: @_ZTVN5Test11CE = unnamed_addr constant { [3 x ptr], [3 x ptr] } {
// OGCG-SAME: [3 x ptr] [ptr null, ptr null, ptr @_ZN5Test11C1fEv],
// OGCG-SAME: [3 x ptr] [ptr inttoptr (i64 -8 to ptr), ptr null, ptr @_ZThn8_N5Test11C1fEv]
// OGCG-SAME: }

// OGCG: @_ZTVN5Test21CE = unnamed_addr constant { [3 x ptr], [3 x ptr] } {
// OGCG-SAME: [3 x ptr] [ptr null, ptr null, ptr @_ZN5Test21C1gEv],
// OGCG-SAME: [3 x ptr] [ptr inttoptr (i64 -8 to ptr), ptr null, ptr @_ZThn8_N5Test21C1gEv]
// OGCG-SAME: }

// OGCG: @_ZTVN5Test31DE = unnamed_addr constant { [4 x ptr], [4 x ptr] } {
// OGCG-SAME: [4 x ptr] [ptr null, ptr null, ptr @_ZN5Test31DD1Ev, ptr @_ZN5Test31DD0Ev],
// OGCG-SAME: [4 x ptr] [ptr inttoptr (i64 -8 to ptr), ptr null, ptr @_ZThn8_N5Test31DD1Ev, ptr @_ZThn8_N5Test31DD0Ev]
// OGCG-SAME: }

// OGCG: @_ZTVN5Test41CE = unnamed_addr constant { [4 x ptr], [3 x ptr] } {
// OGCG-SAME: [4 x ptr] [ptr null, ptr null, ptr @_ZN5Test41A1fEv, ptr @_ZN5Test41C1gEi],
// OGCG-SAME: [3 x ptr] [ptr inttoptr (i64 -8 to ptr), ptr null, ptr @_ZThn8_N5Test41C1gEi]
// OGCG-SAME: }

// OGCG: define {{.*}} void @_ZThn8_N5Test11C1fEv(ptr{{.*}})
// OGCG:   %[[O1_THIS:.*]] = load ptr, ptr
// OGCG:   %[[O1_ADJ:.*]] = getelementptr inbounds i8, ptr %[[O1_THIS]], i64 -8
// OGCG:   call void @_ZN5Test11C1fEv(ptr{{.*}} %[[O1_ADJ]])

// OGCG: define {{.*}} i32 @_ZThn8_N5Test21C1gEv(ptr{{.*}})
// OGCG:   %[[O2_THIS:.*]] = load ptr, ptr
// OGCG:   %[[O2_ADJ:.*]] = getelementptr inbounds i8, ptr %[[O2_THIS]], i64 -8
// OGCG:   {{.*}}call {{.*}} i32 @_ZN5Test21C1gEv(ptr{{.*}} %[[O2_ADJ]])

// OGCG: define {{.*}} void @_ZThn8_N5Test31DD1Ev(ptr{{.*}})
// OGCG:   %[[O3A_THIS:.*]] = load ptr, ptr
// OGCG:   %[[O3A_ADJ:.*]] = getelementptr inbounds i8, ptr %[[O3A_THIS]], i64 -8
// OGCG:   call void @_ZN5Test31DD1Ev(ptr{{.*}} %[[O3A_ADJ]])

// OGCG: define {{.*}} void @_ZThn8_N5Test31DD0Ev(ptr{{.*}})
// OGCG:   %[[O3B_THIS:.*]] = load ptr, ptr
// OGCG:   %[[O3B_ADJ:.*]] = getelementptr inbounds i8, ptr %[[O3B_THIS]], i64 -8
// OGCG:   call void @_ZN5Test31DD0Ev(ptr{{.*}} %[[O3B_ADJ]])

// OGCG: define {{.*}} void @_ZThn8_N5Test41C1gEi(ptr{{.*}}, i32{{.*}})
// OGCG:   %[[O4_THIS:.*]] = load ptr, ptr
// OGCG:   %[[O4_ADJ:.*]] = getelementptr inbounds i8, ptr %[[O4_THIS]], i64 -8
// OGCG:   %[[O4_ARG:.*]] = load i32, ptr
// OGCG:   {{.*}}call void @_ZN5Test41C1gEi(ptr{{.*}} %[[O4_ADJ]], i32{{.*}} %[[O4_ARG]])

// OGCG-LABEL: define {{.*}} @_ZTch0_v0_n32_N15CovariantReturn1C1fEv
// OGCG:       {{.*}}call {{.*}} @_ZN15CovariantReturn1C1fEv
// OGCG:       phi ptr

// OGCG-LABEL: define {{.*}} void @_ZThn8_N11VarargThunk1C1fEiz(ptr{{.*}}, i32{{.*}}, ...)
// OGCG:   getelementptr inbounds i8, ptr {{.*}}, i64 -8
// OGCG:   musttail call void (ptr, i32, ...) @_ZN11VarargThunk1C1fEiz(ptr{{.*}}, i32{{.*}}, ...)
