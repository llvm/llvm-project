// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -O1 -disable-llvm-passes %s -o %t-opt.ll
// RUN: FileCheck --check-prefix=LLVM-OPT --input-file=%t-opt.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm -O1 -disable-llvm-passes %s -o %t-opt-ogcg.ll
// RUN: FileCheck --check-prefix=OGCG-OPT --input-file=%t-opt-ogcg.ll %s

// Test for global with constant storage - const object with constructor but no destructor
// Check that we add an llvm.invariant.start to mark when a global becomes read-only.

struct A {
  A();
  int n;
};

// Should emit invariant.start - has constructor, no destructor, no mutable
extern const A a = A();

struct A2 {
  A2();
  constexpr ~A2() {}
  int n;
};

// Should emit invariant.start - constexpr destructor doesn't prevent constant storage
extern const A2 a2 = A2();

struct B {
  B();
  mutable int n;
};

// Should NOT emit invariant.start - has mutable member
extern const B b = B();

// Simple case - just const C c; (no initializer) - Andy's suggestion
class C {
public:
  C();
  int a;
  int b;
};

const C c;

// CIR checks for 'a' - should have constant storage
// CIR: cir.global external @a = #cir.zero : !rec_A
// CIR: cir.func internal private @__cxx_global_var_init() {
// CIR:   %[[OBJ:.*]] = cir.get_global @a : !cir.ptr<!rec_A>
// CIR:   cir.call @_ZN1AC1Ev(%[[OBJ]]) : (!cir.ptr<!rec_A>) -> ()
// CIR:   cir.return
// CIR: }

// CIR checks for 'a2' - should have constant storage (constexpr dtor)
// CIR: cir.global external @a2 = #cir.zero : !rec_A2
// CIR: cir.func internal private @__cxx_global_var_init.1() {
// CIR:   %[[OBJ:.*]] = cir.get_global @a2 : !cir.ptr<!rec_A2>
// CIR:   cir.call @_ZN2A2C1Ev(%[[OBJ]]) : (!cir.ptr<!rec_A2>) -> ()
// CIR:   cir.return
// CIR: }

// CIR checks for 'b' - should NOT have constant storage (mutable member)
// CIR: cir.global external @b = #cir.zero : !rec_B
// CIR: cir.func internal private @__cxx_global_var_init.2() {
// CIR:   %[[OBJ:.*]] = cir.get_global @b : !cir.ptr<!rec_B>
// CIR:   cir.call @_ZN1BC1Ev(%[[OBJ]]) : (!cir.ptr<!rec_B>) -> ()
// CIR:   cir.return
// CIR: }

// CIR checks for 'c' - Andy's simple case, should have constant storage (internal linkage)
// CIR: cir.global {{.*}} internal {{.*}} @_ZL1c = #cir.zero : !rec_C
// CIR: cir.func internal private @__cxx_global_var_init.3() {
// CIR:   %[[OBJ:.*]] = cir.get_global @_ZL1c : !cir.ptr<!rec_C>
// CIR:   cir.call @_ZN1CC1Ev(%[[OBJ]]) : (!cir.ptr<!rec_C>) -> ()
// CIR:   cir.return
// CIR: }

// LLVM checks (no optimization)
// Check all globals first (they appear at the top)
// LLVM: @a ={{.*}} global {{.*}} zeroinitializer
// LLVM: @a2 ={{.*}} global {{.*}} zeroinitializer
// LLVM: @b ={{.*}} global {{.*}} zeroinitializer
// LLVM: @_ZL1c ={{.*}} global {{.*}} zeroinitializer

// Then check the init functions
// LLVM: define internal void @__cxx_global_var_init() {
// LLVM:   call void @_ZN1AC1Ev(ptr @a)
// LLVM:   ret void
// LLVM: }

// LLVM: define internal void @__cxx_global_var_init.1() {
// LLVM:   call void @_ZN2A2C1Ev(ptr @a2)
// LLVM:   ret void
// LLVM: }

// LLVM: define internal void @__cxx_global_var_init.2() {
// LLVM:   call void @_ZN1BC1Ev(ptr @b)
// LLVM:   ret void
// LLVM: }

// LLVM: define internal void @__cxx_global_var_init.3() {
// LLVM:   call void @_ZN1CC1Ev(ptr @_ZL1c)
// LLVM:   ret void
// LLVM: }

// OGCG checks (no optimization)
// Check all globals first (they appear at the top)
// OGCG: @a ={{.*}} global {{.*}} zeroinitializer
// OGCG: @a2 ={{.*}} global {{.*}} zeroinitializer
// OGCG: @b ={{.*}} global {{.*}} zeroinitializer
// OGCG: @_ZL1c ={{.*}} global {{.*}} zeroinitializer

// Then check the init functions
// OGCG: define internal void @__cxx_global_var_init() {{.*}} section ".text.startup" {
// OGCG:   call void @_ZN1AC1Ev(ptr noundef nonnull align 4 dereferenceable(4) @a)
// OGCG:   ret void
// OGCG: }

// OGCG: define internal void @__cxx_global_var_init.1() {{.*}} section ".text.startup" {
// OGCG:   call void @_ZN2A2C1Ev(ptr noundef nonnull align 4 dereferenceable(4) @a2)
// OGCG:   ret void
// OGCG: }

// OGCG: define internal void @__cxx_global_var_init.2() {{.*}} section ".text.startup" {
// OGCG:   call void @_ZN1BC1Ev(ptr noundef nonnull align 4 dereferenceable(4) @b)
// OGCG:   ret void
// OGCG: }

// OGCG: define internal void @__cxx_global_var_init.3() {{.*}} section ".text.startup" {
// OGCG:   call void @_ZN1CC1Ev(ptr noundef nonnull align 4 dereferenceable(8) @_ZL1c)
// OGCG:   ret void
// OGCG: }

// With optimization enabled, should emit invariant.start intrinsic for constant storage cases
// Check all globals first (they appear at the top)
// LLVM-OPT: @a ={{.*}} global {{.*}} zeroinitializer
// LLVM-OPT: @a2 ={{.*}} global {{.*}} zeroinitializer
// LLVM-OPT: @b ={{.*}} global {{.*}} zeroinitializer
// LLVM-OPT: @_ZL1c ={{.*}} global {{.*}} zeroinitializer

// Then check the init functions with invariant.start
// LLVM-OPT: define internal void @__cxx_global_var_init() {
// LLVM-OPT:   call void @_ZN1AC1Ev(ptr @a)
// LLVM-OPT:   call {{.*}}@llvm.invariant.start.p0(i64 4, ptr @a)
// LLVM-OPT:   ret void
// LLVM-OPT: }

// LLVM-OPT: define internal void @__cxx_global_var_init.1() {
// LLVM-OPT:   call void @_ZN2A2C1Ev(ptr @a2)
// LLVM-OPT:   call {{.*}}@llvm.invariant.start.p0(i64 4, ptr @a2)
// LLVM-OPT:   ret void
// LLVM-OPT: }

// LLVM-OPT: define internal void @__cxx_global_var_init.2() {
// LLVM-OPT:   call void @_ZN1BC1Ev(ptr @b)
// LLVM-OPT-NOT: call {{.*}}@llvm.invariant.start.p0(i64 {{.*}}, ptr @b)
// LLVM-OPT:   ret void
// LLVM-OPT: }

// LLVM-OPT: define internal void @__cxx_global_var_init.3() {
// LLVM-OPT:   call void @_ZN1CC1Ev(ptr @_ZL1c)
// LLVM-OPT:   call {{.*}}@llvm.invariant.start.p0(i64 8, ptr @_ZL1c)
// LLVM-OPT:   ret void
// LLVM-OPT: }

// OGCG-OPT checks (with optimization)
// Check all globals first (they appear at the top)
// OGCG-OPT: @a ={{.*}} global {{.*}} zeroinitializer
// OGCG-OPT: @a2 ={{.*}} global {{.*}} zeroinitializer
// OGCG-OPT: @b ={{.*}} global {{.*}} zeroinitializer
// OGCG-OPT: @_ZL1c ={{.*}} global {{.*}} zeroinitializer

// Then check the init functions with invariant.start
// OGCG-OPT: define internal void @__cxx_global_var_init() {{.*}} section ".text.startup" {
// OGCG-OPT:   call void @_ZN1AC1Ev(ptr noundef nonnull align 4 dereferenceable(4) @a)
// OGCG-OPT:   call {{.*}}@llvm.invariant.start.p0(i64 4, ptr @a)
// OGCG-OPT:   ret void
// OGCG-OPT: }

// OGCG-OPT: define internal void @__cxx_global_var_init.1() {{.*}} section ".text.startup" {
// OGCG-OPT:   call void @_ZN2A2C1Ev(ptr noundef nonnull align 4 dereferenceable(4) @a2)
// OGCG-OPT:   call {{.*}}@llvm.invariant.start.p0(i64 4, ptr @a2)
// OGCG-OPT:   ret void
// OGCG-OPT: }

// OGCG-OPT: define internal void @__cxx_global_var_init.2() {{.*}} section ".text.startup" {
// OGCG-OPT:   call void @_ZN1BC1Ev(ptr noundef nonnull align 4 dereferenceable(4) @b)
// OGCG-OPT-NOT: call {{.*}}@llvm.invariant.start.p0(i64 {{.*}}, ptr @b)
// OGCG-OPT:   ret void
// OGCG-OPT: }

// OGCG-OPT: define internal void @__cxx_global_var_init.3() {{.*}} section ".text.startup" {
// OGCG-OPT:   call void @_ZN1CC1Ev(ptr noundef nonnull align 4 dereferenceable(8) @_ZL1c)
// OGCG-OPT:   call {{.*}}@llvm.invariant.start.p0(i64 8, ptr @_ZL1c)
// OGCG-OPT:   ret void
// OGCG-OPT: }

