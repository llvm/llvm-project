// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -emit-llvm -o - -mconstructor-aliases -fcxx-exceptions -fexceptions | FileCheck %s

// CHECK: @_ZN5test01AD1Ev = alias {{.*}} @_ZN5test01AD2Ev
// CHECK: @_ZN5test11MD2Ev = alias {{.*}} @_ZN5test11AD2Ev
// CHECK: @_ZN5test11ND2Ev = alias {{.*}} @_ZN5test11AD2Ev
// CHECK: @_ZN5test11OD2Ev = alias {{.*}} @_ZN5test11AD2Ev
// CHECK: @_ZN5test11SD2Ev = alias bitcast {{.*}} @_ZN5test11AD2Ev

// CHECK: @_ZN5test312_GLOBAL__N_11DD1Ev = alias internal {{.*}} @_ZN5test312_GLOBAL__N_11DD2Ev
// CHECK: @_ZN5test312_GLOBAL__N_11DD2Ev = alias internal bitcast {{.*}} @_ZN5test312_GLOBAL__N_11CD2Ev
// CHECK: @_ZN5test312_GLOBAL__N_11CD1Ev = alias internal {{.*}} @_ZN5test312_GLOBAL__N_11CD2Ev

struct A {
  int a;
  
  ~A();
};

// Base with non-trivial destructor
struct B : A {
  ~B();
};

B::~B() { }

// Field with non-trivial destructor
struct C {
  A a;
  
  ~C();
};

C::~C() { }

namespace PR7526 {
  extern void foo();
  struct allocator {
    ~allocator() throw();
  };

  struct allocator_derived : allocator { };

  // CHECK: define void @_ZN6PR75269allocatorD2Ev(%"struct.PR7526::allocator"* %this) unnamed_addr
  // CHECK: call void @__cxa_call_unexpected
  allocator::~allocator() throw() { foo(); }

  // CHECK: define linkonce_odr void @_ZN6PR752617allocator_derivedD1Ev(%"struct.PR7526::allocator_derived"* %this) unnamed_addr
  // CHECK-NOT: call void @__cxa_call_unexpected
  // CHECK:     }
  void foo() {
    allocator_derived ad;
  }
}

// PR5084
template<typename T>
class A1 {
  ~A1();
};

template<> A1<char>::~A1();

// PR5529
namespace PR5529 {
  struct A {
    ~A();
  };
  
  A::~A() { }
  struct B : A {
    virtual ~B();
  };
  
  B::~B()  {}
}

// FIXME: there's a known problem in the codegen here where, if one
// destructor throws, the remaining destructors aren't run.  Fix it,
// then make this code check for it.
namespace test0 {
  void foo();
  struct VBase { ~VBase(); };
  struct Base { ~Base(); };
  struct Member { ~Member(); };

  struct A : Base {
    Member M;
    ~A();
  };

  // The function-try-block won't suppress -mconstructor-aliases here.
  A::~A() try { } catch (int i) {}

// complete destructor alias tested above

// CHECK: define void @_ZN5test01AD2Ev(%"struct.test0::A"* %this) unnamed_addr
// CHECK: invoke void @_ZN5test06MemberD1Ev
// CHECK:   unwind label [[MEM_UNWIND:%[a-zA-Z0-9.]+]]
// CHECK: invoke void @_ZN5test04BaseD2Ev
// CHECK:   unwind label [[BASE_UNWIND:%[a-zA-Z0-9.]+]]

  struct B : Base, virtual VBase {
    Member M;
    ~B();
  };
  B::~B() try { } catch (int i) {}
  // It will suppress the delegation optimization here, though.

// CHECK: define void @_ZN5test01BD1Ev(%"struct.test0::B"* %this) unnamed_addr
// CHECK: invoke void @_ZN5test06MemberD1Ev
// CHECK:   unwind label [[MEM_UNWIND:%[a-zA-Z0-9.]+]]
// CHECK: invoke void @_ZN5test04BaseD2Ev
// CHECK:   unwind label [[BASE_UNWIND:%[a-zA-Z0-9.]+]]
// CHECK: invoke void @_ZN5test05VBaseD2Ev
// CHECK:   unwind label [[VBASE_UNWIND:%[a-zA-Z0-9.]+]]

// CHECK: define void @_ZN5test01BD2Ev(%"struct.test0::B"* %this, i8** %vtt) unnamed_addr
// CHECK: invoke void @_ZN5test06MemberD1Ev
// CHECK:   unwind label [[MEM_UNWIND:%[a-zA-Z0-9.]+]]
// CHECK: invoke void @_ZN5test04BaseD2Ev
// CHECK:   unwind label [[BASE_UNWIND:%[a-zA-Z0-9.]+]]
}

// Test base-class aliasing.
namespace test1 {
  struct A { ~A(); char ***m; }; // non-trivial destructor
  struct B { ~B(); }; // non-trivial destructor
  struct Empty { }; // trivial destructor, empty
  struct NonEmpty { int x; }; // trivial destructor, non-empty

  // There must be a definition in this translation unit for the alias
  // optimization to apply.
  A::~A() { delete m; }

  struct M : A { ~M(); };
  M::~M() {} // alias tested above

  struct N : A, Empty { ~N(); };
  N::~N() {} // alias tested above

  struct O : Empty, A { ~O(); };
  O::~O() {} // alias tested above

  struct P : NonEmpty, A { ~P(); };
  P::~P() {} // CHECK: define void @_ZN5test11PD2Ev(%"struct.test1::P"* %this) unnamed_addr

  struct Q : A, B { ~Q(); };
  Q::~Q() {} // CHECK: define void @_ZN5test11QD2Ev(%"struct.test1::Q"* %this) unnamed_addr

  struct R : A { ~R(); };
  R::~R() { A a; } // CHECK: define void @_ZN5test11RD2Ev(%"struct.test1::R"* %this) unnamed_addr

  struct S : A { ~S(); int x; };
  S::~S() {} // alias tested above

  struct T : A { ~T(); B x; };
  T::~T() {} // CHECK: define void @_ZN5test11TD2Ev(%"struct.test1::T"* %this) unnamed_addr

  // The VTT parameter prevents this.  We could still make this work
  // for calling conventions that are safe against extra parameters.
  struct U : A, virtual B { ~U(); };
  U::~U() {} // CHECK: define void @_ZN5test11UD2Ev(%"struct.test1::U"* %this, i8** %vtt) unnamed_addr
}

// PR6471
namespace test2 {
  struct A { ~A(); char ***m; };
  struct B : A { ~B(); };

  B::~B() {}
  // CHECK: define void @_ZN5test21BD2Ev(%"struct.test2::B"* %this) unnamed_addr
  // CHECK: call void @_ZN5test21AD2Ev
}

// PR7142
namespace test3 {
  struct A { virtual ~A(); };
  struct B { virtual ~B(); };
  namespace { // internal linkage => deferred
    struct C : A, B {}; // ~B() in D requires a this-adjustment thunk
    struct D : C {};    // D::~D() is an alias to C::~C()
  }

  void test() {
    new D; // Force emission of D's vtable
  }

  // Checked at top of file:
  // @_ZN5test312_GLOBAL__N_11CD1Ev = alias internal {{.*}} @_ZN5test312_GLOBAL__N_11CD2Ev

  // More checks at end of file.

}

namespace test4 {
  struct A { ~A(); };

  // CHECK: define void @_ZN5test43fooEv()
  // CHECK: call void @_ZN5test41AD1Ev
  // CHECK: ret void
  void foo() {
    {
      A a;
      goto failure;
    }

  failure:
    return;
  }

  // CHECK: define void @_ZN5test43barEi(
  // CHECK:      [[X:%.*]] = alloca i32
  // CHECK-NEXT: [[A:%.*]] = alloca
  // CHECK:      br label
  // CHECK:      [[TMP:%.*]] = load i32* [[X]]
  // CHECK-NEXT: [[CMP:%.*]] = icmp ne i32 [[TMP]], 0
  // CHECK-NEXT: br i1
  // CHECK:      call void @_ZN5test41AD1Ev(
  // CHECK:      br label
  // CHECK:      [[TMP:%.*]] = load i32* [[X]]
  // CHECK:      [[TMP2:%.*]] = add nsw i32 [[TMP]], -1
  // CHECK:      store i32 [[TMP2]], i32* [[X]]
  // CHECK:      br label
  // CHECK:      ret void
  void bar(int x) {
    for (A a; x; ) {
      x--;
    }
  }
}

// PR7575
namespace test5 {
  struct A { ~A(); };

  // CHECK: define void @_ZN5test53fooEv()
  // CHECK:      [[ELEMS:%.*]] = alloca [5 x [[A:%.*]]], align
  // CHECK-NEXT: [[EXN:%.*]] = alloca i8*
  // CHECK-NEXT: [[SEL:%.*]] = alloca i32
  // CHECK-NEXT: [[BEGIN:%.*]] = getelementptr inbounds [5 x [[A]]]* [[ELEMS]], i32 0, i32 0
  // CHECK-NEXT: [[END:%.*]] = getelementptr inbounds [[A]]* [[BEGIN]], i64 5
  // CHECK-NEXT: br label
  // CHECK:      [[POST:%.*]] = phi [[A]]* [ [[END]], {{%.*}} ], [ [[ELT:%.*]], {{%.*}} ]
  // CHECK-NEXT: [[ELT]] = getelementptr inbounds [[A]]* [[POST]], i64 -1
  // CHECK-NEXT: invoke void @_ZN5test51AD1Ev([[A]]* [[ELT]])
  // CHECK:      [[T0:%.*]] = icmp eq [[A]]* [[ELT]], [[BEGIN]]
  // CHECK-NEXT: br i1 [[T0]],
  // CHECK:      ret void
  // lpad
  // CHECK:      [[EMPTY:%.*]] = icmp eq [[A]]* [[BEGIN]], [[ELT]]
  // CHECK-NEXT: br i1 [[EMPTY]]
  // CHECK:      [[AFTER:%.*]] = phi [[A]]* [ [[ELT]], {{%.*}} ], [ [[CUR:%.*]], {{%.*}} ]
  // CHECK-NEXT: [[CUR:%.*]] = getelementptr inbounds [[A]]* [[AFTER]], i64 -1
  // CHECK-NEXT: invoke void @_ZN5test51AD1Ev([[A]]* [[CUR]])
  // CHECK:      [[DONE:%.*]] = icmp eq [[A]]* [[CUR]], [[BEGIN]]
  // CHECK-NEXT: br i1 [[DONE]],
  void foo() {
    A elems[5];
  }
}

namespace test6 {
  void opaque();

  struct A { ~A(); };
  template <unsigned> struct B { B(); ~B(); int _; };
  struct C : B<0>, B<1>, virtual B<2>, virtual B<3> {
    A x, y, z;

    C();
    ~C();
  };

  C::C() { opaque(); }
  // CHECK: define void @_ZN5test61CC1Ev(%"struct.test6::C"* %this) unnamed_addr
  // CHECK:   call void @_ZN5test61BILj2EEC2Ev
  // CHECK:   invoke void @_ZN5test61BILj3EEC2Ev
  // CHECK:   invoke void @_ZN5test61BILj0EEC2Ev
  // CHECK:   invoke void @_ZN5test61BILj1EEC2Ev
  // CHECK:   invoke void @_ZN5test66opaqueEv
  // CHECK:   ret void
  // FIXME: way too much EH cleanup code follows

  C::~C() { opaque(); }
  // CHECK: define void @_ZN5test61CD1Ev(%"struct.test6::C"* %this) unnamed_addr
  // CHECK:   invoke void @_ZN5test61CD2Ev
  // CHECK:   invoke void @_ZN5test61BILj3EED2Ev
  // CHECK:   call void @_ZN5test61BILj2EED2Ev
  // CHECK:   ret void
  // CHECK:   invoke void @_ZN5test61BILj3EED2Ev
  // CHECK:   invoke void @_ZN5test61BILj2EED2Ev

  // CHECK: define void @_ZN5test61CD2Ev(%"struct.test6::C"* %this, i8** %vtt) unnamed_addr
  // CHECK:   invoke void @_ZN5test66opaqueEv
  // CHECK:   invoke void @_ZN5test61AD1Ev
  // CHECK:   invoke void @_ZN5test61AD1Ev
  // CHECK:   invoke void @_ZN5test61AD1Ev
  // CHECK:   invoke void @_ZN5test61BILj1EED2Ev
  // CHECK:   call void @_ZN5test61BILj0EED2Ev
  // CHECK:   ret void
  // CHECK:   invoke void @_ZN5test61AD1Ev
  // CHECK:   invoke void @_ZN5test61AD1Ev
  // CHECK:   invoke void @_ZN5test61AD1Ev
  // CHECK:   invoke void @_ZN5test61BILj1EED2Ev
  // CHECK:   invoke void @_ZN5test61BILj0EED2Ev
}

// PR 9197
namespace test7 {
  struct D { ~D(); };

  struct A { ~A(); };
  A::~A() { }

  struct B : public A {
    ~B();
    D arr[1];
  };

  // Verify that this doesn't get emitted as an alias
  // CHECK: define void @_ZN5test71BD2Ev(
  // CHECK:   invoke void @_ZN5test71DD1Ev(
  // CHECK:   call void @_ZN5test71AD2Ev(
  B::~B() {}
}

// PR10467
namespace test8 {
  struct A { A(); ~A(); };

  void die() __attribute__((noreturn));
  void test() {
    A x;
    while (1) {
      A y;
      goto l;
    }
  l: die();
  }

  // CHECK:    define void @_ZN5test84testEv()
  // CHECK:      [[X:%.*]] = alloca [[A:%.*]], align 1
  // CHECK-NEXT: [[Y:%.*]] = alloca [[A:%.*]], align 1
  // CHECK:      call void @_ZN5test81AC1Ev([[A]]* [[X]])
  // CHECK-NEXT: br label
  // CHECK:      invoke void @_ZN5test81AC1Ev([[A]]* [[Y]])
  // CHECK:      invoke void @_ZN5test81AD1Ev([[A]]* [[Y]])
  // CHECK-NOT:  switch
  // CHECK:      invoke void @_ZN5test83dieEv()
  // CHECK:      unreachable
}

// PR12710
namespace test9 {
  struct ArgType {
    ~ArgType();
  };
  template<typename T>
  void f1(const ArgType& = ArgType());
  void f2();
  void bar() {
    f1<int>();
    f2();
  }
  // CHECK: call void @_ZN5test97ArgTypeD1Ev(%"struct.test9::ArgType"* %
  // CHECK: call void @_ZN5test92f2Ev()
}

// Checks from test3:

  // CHECK: define internal void @_ZN5test312_GLOBAL__N_11DD0Ev(%"struct.test3::<anonymous namespace>::D"* %this) unnamed_addr
  // CHECK: invoke void @_ZN5test312_GLOBAL__N_11DD1Ev(
  // CHECK: call void @_ZdlPv({{.*}}) nounwind
  // CHECK: ret void
  // CHECK: landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
  // CHECK-NEXT: cleanup
  // CHECK: call void @_ZdlPv({{.*}}) nounwind
  // CHECK: resume { i8*, i32 }

  // Checked at top of file:
  // @_ZN5test312_GLOBAL__N_11DD1Ev = alias internal {{.*}} @_ZN5test312_GLOBAL__N_11DD2Ev
  // @_ZN5test312_GLOBAL__N_11DD2Ev = alias internal bitcast {{.*}} @_ZN5test312_GLOBAL__N_11CD2Ev

  // CHECK: define internal void @_ZThn8_N5test312_GLOBAL__N_11DD1Ev(
  // CHECK: getelementptr inbounds i8* {{.*}}, i64 -8
  // CHECK: call void @_ZN5test312_GLOBAL__N_11DD1Ev(
  // CHECK: ret void

  // CHECK: define internal void @_ZThn8_N5test312_GLOBAL__N_11DD0Ev(
  // CHECK: getelementptr inbounds i8* {{.*}}, i64 -8
  // CHECK: call void @_ZN5test312_GLOBAL__N_11DD0Ev(
  // CHECK: ret void

  // CHECK: define internal void @_ZN5test312_GLOBAL__N_11CD2Ev(%"struct.test3::<anonymous namespace>::C"* %this) unnamed_addr
  // CHECK: invoke void @_ZN5test31BD2Ev(
  // CHECK: call void @_ZN5test31AD2Ev(
  // CHECK: ret void

  // CHECK: declare void @_ZN5test31BD2Ev(
  // CHECK: declare void @_ZN5test31AD2Ev(

  // CHECK: define internal void @_ZN5test312_GLOBAL__N_11CD0Ev(%"struct.test3::<anonymous namespace>::C"* %this) unnamed_addr
  // CHECK: invoke void @_ZN5test312_GLOBAL__N_11CD1Ev(
  // CHECK: call void @_ZdlPv({{.*}}) nounwind
  // CHECK: ret void
  // CHECK: landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
  // CHECK-NEXT: cleanup
  // CHECK: call void @_ZdlPv({{.*}}) nounwind
  // CHECK: resume { i8*, i32 }

  // CHECK: define internal void @_ZThn8_N5test312_GLOBAL__N_11CD1Ev(
  // CHECK: getelementptr inbounds i8* {{.*}}, i64 -8
  // CHECK: call void @_ZN5test312_GLOBAL__N_11CD1Ev(
  // CHECK: ret void

  // CHECK: define internal void @_ZThn8_N5test312_GLOBAL__N_11CD0Ev(
  // CHECK: getelementptr inbounds i8* {{.*}}, i64 -8
  // CHECK: call void @_ZN5test312_GLOBAL__N_11CD0Ev(
  // CHECK: ret void
