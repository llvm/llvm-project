// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -emit-llvm -o - -mconstructor-aliases -fcxx-exceptions -fexceptions -O1 -disable-llvm-passes -std=c++03 > %t
// RUN: FileCheck --check-prefix=CHECK1 --input-file=%t %s
// RUN: FileCheck --check-prefix=CHECK2 --input-file=%t %s
// RUN: FileCheck --check-prefix=CHECK3 --input-file=%t %s
// RUN: FileCheck --check-prefixes=CHECK4,CHECK4v03 --input-file=%t %s
// RUN: FileCheck --check-prefixes=CHECK5,CHECK5v03 --input-file=%t %s
// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -emit-llvm -o - -mconstructor-aliases -fcxx-exceptions -fexceptions -O1 -disable-llvm-passes -std=c++11 > %t2
// RUN: FileCheck --check-prefix=CHECK1    --input-file=%t2 %s
// RUN: FileCheck --check-prefix=CHECK2v11 --input-file=%t2 %s
// RUN: FileCheck --check-prefix=CHECK3    --input-file=%t2 %s
// RUN: FileCheck --check-prefixes=CHECK4,CHECK4v11 --input-file=%t2 %s
// RUN: FileCheck --check-prefixes=CHECK5,CHECK5v11 --input-file=%t2 %s
// RUN: FileCheck --check-prefix=CHECK6    --input-file=%t2 %s
// REQUIRES: asserts

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

  // CHECK1-LABEL: define{{.*}} void @_ZN6PR75263fooEv()
  // CHECK1: call void {{.*}} @_ZN6PR75269allocatorD2Ev

  // CHECK1-LABEL: define{{.*}} void @_ZN6PR75269allocatorD2Ev(%"struct.PR7526::allocator"* {{[^,]*}} %this) unnamed_addr
  // CHECK1: call void @__cxa_call_unexpected
  allocator::~allocator() throw() { foo(); }

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

// CHECK2-LABEL: @_ZN5test01AD1Ev ={{.*}} unnamed_addr alias {{.*}} @_ZN5test01AD2Ev
// CHECK2-LABEL: define{{.*}} void @_ZN5test01AD2Ev(%"struct.test0::A"* {{[^,]*}} %this) unnamed_addr
// CHECK2: invoke void @_ZN5test06MemberD1Ev
// CHECK2:   unwind label [[MEM_UNWIND:%[a-zA-Z0-9.]+]]
// CHECK2: invoke void @_ZN5test04BaseD2Ev
// CHECK2:   unwind label [[BASE_UNWIND:%[a-zA-Z0-9.]+]]

// In C++11, the destructors are often known not to throw.
// CHECK2v11-LABEL: @_ZN5test01AD1Ev ={{.*}} unnamed_addr alias {{.*}} @_ZN5test01AD2Ev
// CHECK2v11-LABEL: define{{.*}} void @_ZN5test01AD2Ev(%"struct.test0::A"* {{[^,]*}} %this) unnamed_addr
// CHECK2v11: call void @_ZN5test06MemberD1Ev
// CHECK2v11: call void @_ZN5test04BaseD2Ev

  struct B : Base, virtual VBase {
    Member M;
    ~B();
  };
  B::~B() try { } catch (int i) {}
  // It will suppress the delegation optimization here, though.

// CHECK2-LABEL: define{{.*}} void @_ZN5test01BD2Ev(%"struct.test0::B"* {{[^,]*}} %this, i8** %vtt) unnamed_addr
// CHECK2: invoke void @_ZN5test06MemberD1Ev
// CHECK2:   unwind label [[MEM_UNWIND:%[a-zA-Z0-9.]+]]
// CHECK2: invoke void @_ZN5test04BaseD2Ev
// CHECK2:   unwind label [[BASE_UNWIND:%[a-zA-Z0-9.]+]]

// CHECK2v11-LABEL: define{{.*}} void @_ZN5test01BD2Ev(%"struct.test0::B"* {{[^,]*}} %this, i8** %vtt) unnamed_addr
// CHECK2v11: call void @_ZN5test06MemberD1Ev
// CHECK2v11: call void @_ZN5test04BaseD2Ev

// CHECK2-LABEL: define{{.*}} void @_ZN5test01BD1Ev(%"struct.test0::B"* {{[^,]*}} %this) unnamed_addr
// CHECK2: invoke void @_ZN5test06MemberD1Ev
// CHECK2:   unwind label [[MEM_UNWIND:%[a-zA-Z0-9.]+]]
// CHECK2: invoke void @_ZN5test04BaseD2Ev
// CHECK2:   unwind label [[BASE_UNWIND:%[a-zA-Z0-9.]+]]
// CHECK2: invoke void @_ZN5test05VBaseD2Ev
// CHECK2:   unwind label [[VBASE_UNWIND:%[a-zA-Z0-9.]+]]

// CHECK2v11-LABEL: define{{.*}} void @_ZN5test01BD1Ev(%"struct.test0::B"* {{[^,]*}} %this) unnamed_addr
// CHECK2v11: call void @_ZN5test06MemberD1Ev
// CHECK2v11: call void @_ZN5test04BaseD2Ev
// CHECK2v11: call void @_ZN5test05VBaseD2Ev
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
  M::~M() {}
  // CHECK3: @_ZN5test11MD2Ev ={{.*}} unnamed_addr alias {{.*}} @_ZN5test11AD2Ev

  struct N : A, Empty { ~N(); };
  N::~N() {}
  // CHECK3: @_ZN5test11ND2Ev ={{.*}} unnamed_addr alias {{.*}} @_ZN5test11AD2Ev

  struct O : Empty, A { ~O(); };
  O::~O() {}
  // CHECK3: @_ZN5test11OD2Ev ={{.*}} unnamed_addr alias {{.*}} @_ZN5test11AD2Ev

  struct P : NonEmpty, A { ~P(); };
  P::~P() {} // CHECK3-LABEL: define{{.*}} void @_ZN5test11PD2Ev(%"struct.test1::P"* {{[^,]*}} %this) unnamed_addr

  struct Q : A, B { ~Q(); };
  Q::~Q() {} // CHECK3-LABEL: define{{.*}} void @_ZN5test11QD2Ev(%"struct.test1::Q"* {{[^,]*}} %this) unnamed_addr

  struct R : A { ~R(); };
  R::~R() { A a; } // CHECK3-LABEL: define{{.*}} void @_ZN5test11RD2Ev(%"struct.test1::R"* {{[^,]*}} %this) unnamed_addr

  struct S : A { ~S(); int x; };
  S::~S() {}
  // CHECK4: @_ZN5test11SD2Ev ={{.*}} unnamed_addr alias {{.*}}, bitcast {{.*}} @_ZN5test11AD2Ev

  struct T : A { ~T(); B x; };
  T::~T() {} // CHECK4-LABEL: define{{.*}} void @_ZN5test11TD2Ev(%"struct.test1::T"* {{[^,]*}} %this) unnamed_addr

  // The VTT parameter prevents this.  We could still make this work
  // for calling conventions that are safe against extra parameters.
  struct U : A, virtual B { ~U(); };
  U::~U() {} // CHECK4-LABEL: define{{.*}} void @_ZN5test11UD2Ev(%"struct.test1::U"* {{[^,]*}} %this, i8** %vtt) unnamed_addr
}

// PR6471
namespace test2 {
  struct A { ~A(); char ***m; };
  struct B : A { ~B(); };

  B::~B() {}
  // CHECK4-LABEL: define{{.*}} void @_ZN5test21BD2Ev(%"struct.test2::B"* {{[^,]*}} %this) unnamed_addr
  // CHECK4: call void @_ZN5test21AD2Ev
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

  // CHECK4-LABEL: define internal void @_ZN5test312_GLOBAL__N_11CD2Ev(%"struct.test3::(anonymous namespace)::C"* {{[^,]*}} %this) unnamed_addr
  // CHECK4v03: invoke void @_ZN5test31BD2Ev(
  // CHECK4v11: call   void @_ZN5test31BD2Ev(
  // CHECK4: call void @_ZN5test31AD2Ev(
  // CHECK4: ret void

  // CHECK4-LABEL: define internal void @_ZN5test312_GLOBAL__N_11DD0Ev(%"struct.test3::(anonymous namespace)::D"* {{[^,]*}} %this) unnamed_addr
  // CHECK4v03-SAME:  personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
  // CHECK4v03: invoke void {{.*}} @_ZN5test312_GLOBAL__N_11CD2Ev
  // CHECK4v11: call   void {{.*}} @_ZN5test312_GLOBAL__N_11CD2Ev
  // CHECK4: call void @_ZdlPv({{.*}}) [[NUW:#[0-9]+]]
  // CHECK4: ret void
  // CHECK4v03: landingpad { i8*, i32 }
  // CHECK4v03-NEXT: cleanup
  // CHECK4v03: call void @_ZdlPv({{.*}}) [[NUW]]
  // CHECK4v03: resume { i8*, i32 }
  // CHECK4v11-NOT: landingpad

  // CHECK4-LABEL: define internal void @_ZThn8_N5test312_GLOBAL__N_11DD1Ev(
  // CHECK4: getelementptr inbounds i8, i8* {{.*}}, i64 -8
  // CHECK4: call void {{.*}} @_ZN5test312_GLOBAL__N_11CD2Ev
  // CHECK4: ret void

  // CHECK4-LABEL: define internal void @_ZThn8_N5test312_GLOBAL__N_11DD0Ev(
  // CHECK4: getelementptr inbounds i8, i8* {{.*}}, i64 -8
  // CHECK4: call void @_ZN5test312_GLOBAL__N_11DD0Ev(
  // CHECK4: ret void

  // CHECK4-LABEL: define internal void @_ZN5test312_GLOBAL__N_11CD0Ev(%"struct.test3::(anonymous namespace)::C"* {{[^,]*}} %this) unnamed_addr
  // CHECK4v03-SAME:  personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
  // CHECK4v03: invoke void @_ZN5test312_GLOBAL__N_11CD2Ev(
  // CHECK4v11: call   void @_ZN5test312_GLOBAL__N_11CD2Ev(
  // CHECK4: call void @_ZdlPv({{.*}}) [[NUW]]
  // CHECK4: ret void
  // CHECK4v03: landingpad { i8*, i32 }
  // CHECK4v03-NEXT: cleanup
  // CHECK4v03: call void @_ZdlPv({{.*}}) [[NUW]]
  // CHECK4v03: resume { i8*, i32 }

  // CHECK4-LABEL: define internal void @_ZThn8_N5test312_GLOBAL__N_11CD1Ev(
  // CHECK4: getelementptr inbounds i8, i8* {{.*}}, i64 -8
  // CHECK4: call void @_ZN5test312_GLOBAL__N_11CD2Ev(
  // CHECK4: ret void

  // CHECK4-LABEL: define internal void @_ZThn8_N5test312_GLOBAL__N_11CD0Ev(
  // CHECK4: getelementptr inbounds i8, i8* {{.*}}, i64 -8
  // CHECK4: call void @_ZN5test312_GLOBAL__N_11CD0Ev(
  // CHECK4: ret void

  // CHECK4-LABEL: declare void @_ZN5test31BD2Ev(
  // CHECK4-LABEL: declare void @_ZN5test31AD2Ev(

  // CHECK4: attributes [[NUW]] = {{[{].*}} nounwind {{.*[}]}}
}

namespace test4 {
  struct A { ~A(); };

  // CHECK5-LABEL: define{{.*}} void @_ZN5test43fooEv()
  // CHECK5: call void @_ZN5test41AD1Ev
  // CHECK5: ret void
  void foo() {
    {
      A a;
      goto failure;
    }

  failure:
    return;
  }

  // CHECK5-LABEL: define{{.*}} void @_ZN5test43barEi(
  // CHECK5:      [[X:%.*]] = alloca i32
  // CHECK5-NEXT: [[A:%.*]] = alloca
  // CHECK5:      br label
  // CHECK5:      [[TMP:%.*]] = load i32, i32* [[X]]
  // CHECK5-NEXT: [[CMP:%.*]] = icmp ne i32 [[TMP]], 0
  // CHECK5-NEXT: br i1
  // CHECK5:      call void @_ZN5test41AD1Ev(
  // CHECK5:      br label
  // CHECK5:      [[TMP:%.*]] = load i32, i32* [[X]]
  // CHECK5:      [[TMP2:%.*]] = add nsw i32 [[TMP]], -1
  // CHECK5:      store i32 [[TMP2]], i32* [[X]]
  // CHECK5:      br label
  // CHECK5:      ret void
  void bar(int x) {
    for (A a; x; ) {
      x--;
    }
  }
}

// PR7575
namespace test5 {
  struct A { ~A(); };

  // CHECK5-LABEL: define{{.*}} void @_ZN5test53fooEv()
  // CHECK5:      [[ELEMS:%.*]] = alloca [5 x [[A:%.*]]], align
  // CHECK5v03-NEXT: [[EXN:%.*]] = alloca i8*
  // CHECK5v03-NEXT: [[SEL:%.*]] = alloca i32
  // CHECK5-NEXT: [[PELEMS:%.*]] = bitcast [5 x [[A]]]* [[ELEMS]] to i8*
  // CHECK5-NEXT: call void @llvm.lifetime.start.p0i8(i64 5, i8* [[PELEMS]])
  // CHECK5-NEXT: [[BEGIN:%.*]] = getelementptr inbounds [5 x [[A]]], [5 x [[A]]]* [[ELEMS]], i32 0, i32 0
  // CHECK5-NEXT: [[END:%.*]] = getelementptr inbounds [[A]], [[A]]* [[BEGIN]], i64 5
  // CHECK5-NEXT: br label
  // CHECK5:      [[POST:%.*]] = phi [[A]]* [ [[END]], {{%.*}} ], [ [[ELT:%.*]], {{%.*}} ]
  // CHECK5-NEXT: [[ELT]] = getelementptr inbounds [[A]], [[A]]* [[POST]], i64 -1
  // CHECK5v03-NEXT: invoke void @_ZN5test51AD1Ev([[A]]* {{[^,]*}} [[ELT]])
  // CHECK5v11-NEXT: call   void @_ZN5test51AD1Ev([[A]]* {{[^,]*}} [[ELT]])
  // CHECK5:      [[T0:%.*]] = icmp eq [[A]]* [[ELT]], [[BEGIN]]
  // CHECK5-NEXT: br i1 [[T0]],
  // CHECK5:      call void @llvm.lifetime.end
  // CHECK5-NEXT: ret void
  // lpad
  // CHECK5v03:      [[EMPTY:%.*]] = icmp eq [[A]]* [[BEGIN]], [[ELT]]
  // CHECK5v03-NEXT: br i1 [[EMPTY]]
  // CHECK5v03:      [[AFTER:%.*]] = phi [[A]]* [ [[ELT]], {{%.*}} ], [ [[CUR:%.*]], {{%.*}} ]
  // CHECK5v03-NEXT: [[CUR:%.*]] = getelementptr inbounds [[A]], [[A]]* [[AFTER]], i64 -1
  // CHECK5v03-NEXT: invoke void @_ZN5test51AD1Ev([[A]]* {{[^,]*}} [[CUR]])
  // CHECK5v03:      [[DONE:%.*]] = icmp eq [[A]]* [[CUR]], [[BEGIN]]
  // CHECK5v03-NEXT: br i1 [[DONE]],
  // CHECK5v11-NOT: landingpad
  // CHECK5v11:   }
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
  // CHECK5-LABEL: define{{.*}} void @_ZN5test61CC1Ev(%"struct.test6::C"* {{[^,]*}} %this) unnamed_addr
  // CHECK5:   call void @_ZN5test61BILj2EEC2Ev
  // CHECK5:   invoke void @_ZN5test61BILj3EEC2Ev
  // CHECK5:   invoke void @_ZN5test61BILj0EEC2Ev
  // CHECK5:   invoke void @_ZN5test61BILj1EEC2Ev
  // CHECK5:   invoke void @_ZN5test66opaqueEv
  // CHECK5:   ret void
  // FIXME: way too much EH cleanup code follows

  C::~C() { opaque(); }
  // CHECK5-LABEL: define{{.*}} void @_ZN5test61CD2Ev(%"struct.test6::C"* {{[^,]*}} %this, i8** %vtt) unnamed_addr
  // CHECK5:   invoke void @_ZN5test66opaqueEv
  // CHECK5v03:   invoke void @_ZN5test61AD1Ev
  // CHECK5v03:   invoke void @_ZN5test61AD1Ev
  // CHECK5v03:   invoke void @_ZN5test61AD1Ev
  // CHECK5v03:   invoke void @_ZN5test61BILj1EED2Ev
  // CHECK5v11:   call   void @_ZN5test61AD1Ev
  // CHECK5v11:   call   void @_ZN5test61AD1Ev
  // CHECK5v11:   call   void @_ZN5test61AD1Ev
  // CHECK5v11:   call   void @_ZN5test61BILj1EED2Ev
  // CHECK5:   call void @_ZN5test61BILj0EED2Ev
  // CHECK5:   ret void
  // CHECK5v03:   invoke void @_ZN5test61AD1Ev
  // CHECK5v03:   invoke void @_ZN5test61AD1Ev
  // CHECK5v03:   invoke void @_ZN5test61AD1Ev
  // CHECK5v03:   invoke void @_ZN5test61BILj1EED2Ev
  // CHECK5v03:   invoke void @_ZN5test61BILj0EED2Ev

  // CHECK5-LABEL: define{{.*}} void @_ZN5test61CD1Ev(%"struct.test6::C"* {{[^,]*}} %this) unnamed_addr
  // CHECK5v03:   invoke void @_ZN5test61CD2Ev
  // CHECK5v03:   invoke void @_ZN5test61BILj3EED2Ev
  // CHECK5v03:   call void @_ZN5test61BILj2EED2Ev
  // CHECK5v03:   ret void
  // CHECK5v03:   invoke void @_ZN5test61BILj3EED2Ev
  // CHECK5v03:   invoke void @_ZN5test61BILj2EED2Ev

  // CHECK5v11:   call void @_ZN5test61CD2Ev
  // CHECK5v11:   call void @_ZN5test61BILj3EED2Ev
  // CHECK5v11:   call void @_ZN5test61BILj2EED2Ev
  // CHECK5v11:   ret void
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
  // CHECK5-LABEL: define{{.*}} void @_ZN5test71BD2Ev(
  // CHECK5v03:   invoke void @_ZN5test71DD1Ev(
  // CHECK5v11:   call   void @_ZN5test71DD1Ev(
  // CHECK5:   call void @_ZN5test71AD2Ev(
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

  // CHECK5-LABEL:    define{{.*}} void @_ZN5test84testEv()
  // CHECK5:      [[X:%.*]] = alloca [[A:%.*]], align 1
  // CHECK5-NEXT: [[Y:%.*]] = alloca [[A:%.*]], align 1
  // CHECK5:      call void @_ZN5test81AC1Ev([[A]]* {{[^,]*}} [[X]])
  // CHECK5-NEXT: br label
  // CHECK5:      invoke void @_ZN5test81AC1Ev([[A]]* {{[^,]*}} [[Y]])
  // CHECK5v03:   invoke void @_ZN5test81AD1Ev([[A]]* {{[^,]*}} [[Y]])
  // CHECK5v11:   call   void @_ZN5test81AD1Ev([[A]]* {{[^,]*}} [[Y]])
  // CHECK5-NOT:  switch
  // CHECK5:      invoke void @_ZN5test83dieEv()
  // CHECK5:      unreachable
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
  // CHECK5: call void @_ZN5test97ArgTypeD1Ev(%"struct.test9::ArgType"* {{[^,]*}} %
  // CHECK5: call void @_ZN5test92f2Ev()
}

namespace test10 {
  // We used to crash trying to replace _ZN6test106OptionD1Ev with
  // _ZN6test106OptionD2Ev twice.
  struct Option {
    virtual ~Option() {}
  };
  template <class DataType> class opt : public Option {};
  template class opt<int>;
  // CHECK5-LABEL: define{{.*}} zeroext i1 @_ZN6test1016handleOccurrenceEv(
  bool handleOccurrence() {
    // CHECK5: call void @_ZN6test106OptionD2Ev(
    Option x;
    return true;
  }
}

#if __cplusplus >= 201103L
namespace test11 {

// Check that lifetime.end is emitted in the landing pad.

// CHECK6-LABEL: define{{.*}} void @_ZN6test1115testLifetimeEndEi(
// CHECK6: entry:
// CHECK6: [[T1:%[a-z0-9]+]] = alloca %"struct.test11::S1"
// CHECK6: [[T2:%[a-z0-9]+]] = alloca %"struct.test11::S1"
// CHECK6: [[T3:%[a-z0-9]+]] = alloca %"struct.test11::S1"

// CHECK6: {{^}}invoke.cont
// CHECK6: call void @_ZN6test112S1D1Ev(%"struct.test11::S1"* {{[^,]*}} [[T1]])
// CHECK6: [[BC1:%[a-z0-9]+]] = bitcast %"struct.test11::S1"* [[T1]] to i8*
// CHECK6: call void @llvm.lifetime.end.p0i8(i64 32, i8* [[BC1]])
// CHECK6: {{^}}lpad
// CHECK6: call void @_ZN6test112S1D1Ev(%"struct.test11::S1"* {{[^,]*}} [[T1]])
// CHECK6: [[BC2:%[a-z0-9]+]] = bitcast %"struct.test11::S1"* [[T1]] to i8*
// CHECK6: call void @llvm.lifetime.end.p0i8(i64 32, i8* [[BC2]])

// CHECK6: {{^}}invoke.cont
// CHECK6: call void @_ZN6test112S1D1Ev(%"struct.test11::S1"* {{[^,]*}} [[T2]])
// CHECK6: [[BC3:%[a-z0-9]+]] = bitcast %"struct.test11::S1"* [[T2]] to i8*
// CHECK6: call void @llvm.lifetime.end.p0i8(i64 32, i8* [[BC3]])
// CHECK6: {{^}}lpad
// CHECK6: call void @_ZN6test112S1D1Ev(%"struct.test11::S1"* {{[^,]*}} [[T2]])
// CHECK6: [[BC4:%[a-z0-9]+]] = bitcast %"struct.test11::S1"* [[T2]] to i8*
// CHECK6: call void @llvm.lifetime.end.p0i8(i64 32, i8* [[BC4]])

// CHECK6: {{^}}invoke.cont
// CHECK6: call void @_ZN6test112S1D1Ev(%"struct.test11::S1"* {{[^,]*}} [[T3]])
// CHECK6: [[BC5:%[a-z0-9]+]] = bitcast %"struct.test11::S1"* [[T3]] to i8*
// CHECK6: call void @llvm.lifetime.end.p0i8(i64 32, i8* [[BC5]])
// CHECK6: {{^}}lpad
// CHECK6: call void @_ZN6test112S1D1Ev(%"struct.test11::S1"* {{[^,]*}} [[T3]])
// CHECK6: [[BC6:%[a-z0-9]+]] = bitcast %"struct.test11::S1"* [[T3]] to i8*
// CHECK6: call void @llvm.lifetime.end.p0i8(i64 32, i8* [[BC6]])

  struct S1 {
    ~S1();
    int a[8];
  };

  void func1(S1 &) noexcept(false);

  void testLifetimeEnd(int n) {
    if (n < 10) {
      S1 t1;
      func1(t1);
    } else if (n < 100) {
      S1 t2;
      func1(t2);
    } else if (n < 1000) {
      S1 t3;
      func1(t3);
    }
  }

}

namespace final_dtor {
  struct A {
    virtual void f();
    // CHECK6-LABEL: define {{.*}} @_ZN10final_dtor1AD2Ev(
    // CHECK6: store {{.*}} @_ZTV
    // CHECK6-LABEL: {{^}}}
    virtual ~A() { f(); }
  };
  struct B : A {
    // CHECK6-LABEL: define {{.*}} @_ZN10final_dtor1BD2Ev(
    // CHECK6: store {{.*}} @_ZTV
    // CHECK6-LABEL: {{^}}}
    virtual ~B() { f(); }
  };
  struct C final : A {
    // CHECK6-LABEL: define {{.*}} @_ZN10final_dtor1CD2Ev(
    // CHECK6-NOT: store {{.*}} @_ZTV
    // CHECK6-LABEL: {{^}}}
    virtual ~C() { f(); }
  };
  struct D : A {
    // CHECK6-LABEL: define {{.*}} @_ZN10final_dtor1DD2Ev(
    // CHECK6-NOT: store {{.*}} @_ZTV
    // CHECK6-LABEL: {{^}}}
    virtual ~D() final { f(); }
  };
  void use() {
    {A a;}
    {B b;}
    {C c;}
    {D d;}
  }
}
#endif
