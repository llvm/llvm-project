// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -mconstructor-aliases -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -mconstructor-aliases -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -mconstructor-aliases -emit-llvm %s -o %t-og.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t-og.ll %s

enum class EFMode { Always, Verbose };

class PSEvent {
 public:
  PSEvent(
      EFMode m,
      const char* n);
  ~PSEvent();

 private:
  const char* n;
  EFMode m;
};

void blue() {
  PSEvent p(EFMode::Verbose, __FUNCTION__);
}

class A
{
public:
    A() noexcept {}
    A(const A&) noexcept = default;

    virtual ~A() noexcept;
    virtual const char* quack() const noexcept;
};

class B : public A
{
public:
    virtual ~B() noexcept {}
};

// Class A
// CHECK: ![[ClassA:rec_.*]] = !cir.record<class "A" {!cir.vptr} #cir.record.decl.ast>

// Class B
// CHECK: ![[ClassB:rec_.*]] = !cir.record<class "B" {![[ClassA]]}>

// CHECK: cir.func {{.*}} @_Z4bluev()
// CHECK:   %0 = cir.alloca !rec_PSEvent, !cir.ptr<!rec_PSEvent>, ["p", init] {alignment = 8 : i64}
// CHECK:   %1 = cir.const #cir.int<1> : !s32i
// CHECK:   %2 = cir.get_global @".str" : !cir.ptr<!cir.array<!s8i x 5>>
// CHECK:   %3 = cir.cast array_to_ptrdecay %2 : !cir.ptr<!cir.array<!s8i x 5>> -> !cir.ptr<!s8i>
// CHECK:   cir.call @_ZN7PSEventC1E6EFModePKc(%0, %1, %3) : (!cir.ptr<!rec_PSEvent>, !s32i, !cir.ptr<!s8i>) -> ()
// CHECK:   cir.return
// CHECK: }

struct X {
  int a;
  X(int a) : a(a) {}
  ~X() {}
};

bool foo(const X &) { return false; }
bool bar() { return foo(1) || foo(2); }

// CHECK: cir.func {{.*}} @_Z3barv()
// CHECK:   %[[V0:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["__retval"] {alignment = 1 : i64}
// CHECK:   cir.scope {
// CHECK:     %[[V2:.*]] = cir.alloca !rec_X, !cir.ptr<!rec_X>, ["ref.tmp0"] {alignment = 4 : i64}
// CHECK:     %[[V3:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:     cir.call @_ZN1XC2Ei(%[[V2]], %[[V3]]) : (!cir.ptr<!rec_X>, !s32i) -> ()
// CHECK:     %[[V4:.*]] = cir.call @_Z3fooRK1X(%[[V2]]) : (!cir.ptr<!rec_X>) -> !cir.bool
// CHECK:     %[[V5:.*]] = cir.ternary(%[[V4]], true {
// CHECK:       %[[V6:.*]] = cir.const #true
// CHECK:       cir.yield %[[V6]] : !cir.bool
// CHECK:     }, false {
// CHECK:       %[[V6:.*]] = cir.alloca !rec_X, !cir.ptr<!rec_X>, ["ref.tmp1"] {alignment = 4 : i64}
// CHECK:       %[[V7:.*]] = cir.const #cir.int<2> : !s32i
// CHECK:       cir.call @_ZN1XC2Ei(%[[V6]], %[[V7]]) : (!cir.ptr<!rec_X>, !s32i) -> ()
// CHECK:       %[[V8:.*]] = cir.call @_Z3fooRK1X(%[[V6]]) : (!cir.ptr<!rec_X>) -> !cir.bool
// CHECK:       %[[V9:.*]] = cir.ternary(%[[V8]], true {
// CHECK:         %[[V10:.*]] = cir.const #true
// CHECK:         cir.yield %[[V10]] : !cir.bool
// CHECK:       }, false {
// CHECK:         %[[V10:.*]] = cir.const #false
// CHECK:         cir.yield %[[V10]] : !cir.bool
// CHECK:       }) : (!cir.bool) -> !cir.bool
// CHECK:       cir.call @_ZN1XD2Ev(%[[V6]]) : (!cir.ptr<!rec_X>) -> ()
// CHECK:       cir.yield %[[V9]] : !cir.bool
// CHECK:     }) : (!cir.bool) -> !cir.bool
// CHECK:     cir.store{{.*}} %[[V5]], %[[V0]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:     cir.call @_ZN1XD2Ev(%[[V2]]) : (!cir.ptr<!rec_X>) -> ()
// CHECK:   }
// CHECK:   %[[V1:.*]] = cir.load{{.*}} %[[V0]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK:   cir.return %[[V1]] : !cir.bool
// CHECK: }

bool bar2() { return foo(1) && foo(2); }

// CHECK:  cir.func {{.*}} @_Z4bar2v()
// CHECK:     cir.alloca !rec_X, !cir.ptr<!rec_X>
// CHECK:       {{.*}} = cir.ternary({{.*}}, true {
// CHECK:         cir.alloca !rec_X, !cir.ptr<!rec_X>
// CHECK:         cir.call @_ZN1XD2Ev
// CHECK:         cir.yield
// CHECK:       }, false {
// CHECK:         {{.*}} = cir.const #false
// CHECK:         cir.yield
// CHECK:       }) : (!cir.bool) -> !cir.bool
// CHECK:     cir.call @_ZN1XD2Ev

typedef int I;
void pseudo_dtor() {
  I x = 10;
  x.I::~I();
}
// CHECK: cir.func {{.*}} @_Z11pseudo_dtorv()
// CHECK:   %[[INT:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>
// CHECK:   %[[TEN:.*]] = cir.const #cir.int<10> : !s32i
// CHECK:   cir.store{{.*}} %[[TEN]], %[[INT]] : !s32i, !cir.ptr<!s32i>
// CHECK:   cir.return

// @B::~B() #1 definition call into base @A::~A()
// CHECK:  cir.func {{.*}} @_ZN1BD2Ev{{.*}}{
// CHECK:    cir.call @_ZN1AD2Ev(

// void foo()
// CHECK: cir.func {{.*}} @_Z3foov()
// CHECK:   cir.scope {
// CHECK:     cir.call @_ZN1BC2Ev(%0) : (!cir.ptr<!rec_B>) -> ()
// CHECK:     cir.call @_ZN1BD2Ev(%0) : (!cir.ptr<!rec_B>) -> ()

// operator delete(void*) declaration
// CHECK:   cir.func {{.*}} @_ZdlPvm(!cir.ptr<!void>, !u64i)

// B dtor => @B::~B() #2
// Calls dtor #1
// Calls operator delete
//
// CHECK:   cir.func {{.*}} @_ZN1BD0Ev(%arg0: !cir.ptr<![[ClassB]]>
// CHECK:     %0 = cir.alloca !cir.ptr<![[ClassB]]>, !cir.ptr<!cir.ptr<![[ClassB]]>>, ["this", init] {alignment = 8 : i64}
// CHECK:     cir.store %arg0, %0 : !cir.ptr<![[ClassB]]>, !cir.ptr<!cir.ptr<![[ClassB]]>>
// CHECK:     %1 = cir.load{{.*}} %0 : !cir.ptr<!cir.ptr<![[ClassB]]>>, !cir.ptr<![[ClassB]]>
// CHECK:     cir.call @_ZN1BD2Ev(%1) : (!cir.ptr<![[ClassB]]>) -> ()
// CHECK:     %2 = cir.cast bitcast %1 : !cir.ptr<![[ClassB]]> -> !cir.ptr<!void>
// CHECK:     cir.call @_ZdlPvm(%2, %3) : (!cir.ptr<!void>, !u64i) -> ()
// CHECK:     cir.return
// CHECK:   }

void foo() { B(); }

class A2 {
public:
  ~A2();
};

struct B2 {
  template <typename> using C = A2;
};

struct E {
  typedef B2::C<int> D;
};

struct F {
  F(long, A2);
};

class G : F {
public:
  A2 h;
  G(long) : F(i(), h) {}
  long i() { k(E::D()); };
  long k(E::D);
};

int j;
void m() { G l(j); }

// CHECK: cir.func {{.*}} @_ZN1G1kE2A2(!cir.ptr<!rec_G>, !rec_A2) -> !s64i
// CHECK: cir.func {{.*}} @_ZN1G1iEv(%arg0: !cir.ptr<!rec_G>
// CHECK:   %[[V0:.*]] = cir.alloca !cir.ptr<!rec_G>, !cir.ptr<!cir.ptr<!rec_G>>, ["this", init] {alignment = 8 : i64}
// CHECK:   %[[V1:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["__retval"] {alignment = 8 : i64}
// CHECK:   cir.store %arg0, %[[V0]] : !cir.ptr<!rec_G>, !cir.ptr<!cir.ptr<!rec_G>>
// CHECK:   %[[V2:.*]] = cir.load{{.*}} %[[V0]] : !cir.ptr<!cir.ptr<!rec_G>>, !cir.ptr<!rec_G>
// Trivial default constructor call is lowered away.
// CHECK:   %[[V3:.*]] = cir.scope {
// CHECK:     %[[V4:.*]] = cir.alloca !rec_A2, !cir.ptr<!rec_A2>, ["agg.tmp0"] {alignment = 1 : i64}
// CHECK:     %[[V5:.*]] = cir.load{{.*}} %[[V4]] : !cir.ptr<!rec_A2>, !rec_A2
// CHECK:     %[[V6:.*]] = cir.call @_ZN1G1kE2A2(%[[V2]], %[[V5]]) : (!cir.ptr<!rec_G>, !rec_A2) -> !s64i
// CHECK:     cir.call @_ZN2A2D1Ev(%[[V4]]) : (!cir.ptr<!rec_A2>) -> ()
// CHECK:     cir.yield %[[V6]] : !s64i
// CHECK:   } : !s64i
// CHECK:   cir.trap
// CHECK: }

// LLVM-LABEL: define {{.*}} @_ZN1G1iEv
// LLVM:         alloca %class.A2
// LLVM-NOT:     call {{.*}} @_ZN2A2C2Ev
// LLVM:         call {{.*}} @_ZN1G1kE2A2
// LLVM:         call {{.*}} @_ZN2A2D1Ev

// OGCG-LABEL: define {{.*}} @_ZN1G1iEv
// OGCG:         %agg.tmp = alloca %class.A2
// OGCG-NOT:     call {{.*}} @_ZN2A2C2Ev
// OGCG:         call {{.*}} @_ZN1G1kE2A2
// OGCG:         call {{.*}} @_ZN2A2D1Ev
