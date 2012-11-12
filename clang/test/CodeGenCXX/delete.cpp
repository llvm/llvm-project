// RUN: %clang_cc1 -triple x86_64-apple-darwin10 %s -emit-llvm -o - | FileCheck %s

void t1(int *a) {
  delete a;
}

struct S {
  int a;
};

// POD types.
void t3(S *s) {
  delete s;
}

// Non-POD
struct T {
  ~T();
  int a;
};

// CHECK: define void @_Z2t4P1T
void t4(T *t) {
  // CHECK: call void @_ZN1TD1Ev
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: call void @_ZdlPv
  delete t;
}

// PR5102
template <typename T>
class A {
  public: operator T *() const;
};

void f() {
  A<char*> a;
  
  delete a;
}

namespace test0 {
  struct A {
    void *operator new(__SIZE_TYPE__ sz);
    void operator delete(void *p) { ::operator delete(p); }
    ~A() {}
  };

  // CHECK: define void @_ZN5test04testEPNS_1AE(
  void test(A *a) {
    // CHECK: call void @_ZN5test01AD1Ev
    // CHECK-NEXT: bitcast
    // CHECK-NEXT: call void @_ZN5test01AdlEPv
    delete a;
  }

  // CHECK: define linkonce_odr void @_ZN5test01AD1Ev(%"struct.test0::A"* %this) unnamed_addr
  // CHECK: define linkonce_odr void @_ZN5test01AdlEPv
}

namespace test1 {
  struct A {
    int x;
    ~A();
  };

  // CHECK: define void @_ZN5test14testEPA10_A20_NS_1AE(
  void test(A (*arr)[10][20]) {
    delete [] arr;
    // CHECK:      icmp eq [10 x [20 x [[A:%.*]]]]* [[PTR:%.*]], null
    // CHECK-NEXT: br i1

    // CHECK:      [[BEGIN:%.*]] = getelementptr inbounds [10 x [20 x [[A]]]]* [[PTR]], i32 0, i32 0, i32 0
    // CHECK-NEXT: [[T0:%.*]] = bitcast [[A]]* [[BEGIN]] to i8*
    // CHECK-NEXT: [[ALLOC:%.*]] = getelementptr inbounds i8* [[T0]], i64 -8
    // CHECK-NEXT: [[T1:%.*]] = bitcast i8* [[ALLOC]] to i64*
    // CHECK-NEXT: [[COUNT:%.*]] = load i64* [[T1]]
    // CHECK:      [[END:%.*]] = getelementptr inbounds [[A]]* [[BEGIN]], i64 [[COUNT]]
    // CHECK-NEXT: [[ISEMPTY:%.*]] = icmp eq [[A]]* [[BEGIN]], [[END]]
    // CHECK-NEXT: br i1 [[ISEMPTY]],
    // CHECK:      [[PAST:%.*]] = phi [[A]]* [ [[END]], {{%.*}} ], [ [[CUR:%.*]], {{%.*}} ]
    // CHECK-NEXT: [[CUR:%.*]] = getelementptr inbounds [[A]]* [[PAST]], i64 -1
    // CHECK-NEXT: call void @_ZN5test11AD1Ev([[A]]* [[CUR]])
    // CHECK-NEXT: [[ISDONE:%.*]] = icmp eq [[A]]* [[CUR]], [[BEGIN]]
    // CHECK-NEXT: br i1 [[ISDONE]]
    // CHECK:      call void @_ZdaPv(i8* [[ALLOC]])
  }
}

namespace test2 {
  // CHECK: define void @_ZN5test21fEPb
  void f(bool *b) {
    // CHECK: call void @_ZdlPv(i8*
    delete b;
    // CHECK: call void @_ZdaPv(i8*
    delete [] b;
  }
}

namespace test3 {
  void f(int a[10][20]) {
    // CHECK: call void @_ZdaPv(i8*
    delete a;
  }
}

namespace test4 {
  // PR10341: ::delete with a virtual destructor
  struct X {
    virtual ~X();
    void operator delete (void *);
  };

  // CHECK: define void @_ZN5test421global_delete_virtualEPNS_1XE
  void global_delete_virtual(X *xp) {
    //   Load the offset-to-top from the vtable and apply it.
    //   This has to be done first because the dtor can mess it up.
    // CHECK:      [[T0:%.*]] = bitcast [[X:%.*]]* [[XP:%.*]] to i64**
    // CHECK-NEXT: [[VTABLE:%.*]] = load i64** [[T0]]
    // CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds i64* [[VTABLE]], i64 -2
    // CHECK-NEXT: [[OFFSET:%.*]] = load i64* [[T0]], align 8
    // CHECK-NEXT: [[T0:%.*]] = bitcast [[X]]* [[XP]] to i8*
    // CHECK-NEXT: [[ALLOCATED:%.*]] = getelementptr inbounds i8* [[T0]], i64 [[OFFSET]]
    //   Load the complete-object destructor (not the deleting destructor)
    //   and call it.
    // CHECK-NEXT: [[T0:%.*]] = bitcast [[X:%.*]]* [[XP:%.*]] to void ([[X]]*)***
    // CHECK-NEXT: [[VTABLE:%.*]] = load void ([[X]]*)*** [[T0]]
    // CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds void ([[X]]*)** [[VTABLE]], i64 0
    // CHECK-NEXT: [[DTOR:%.*]] = load void ([[X]]*)** [[T0]]
    // CHECK-NEXT: call void [[DTOR]]([[X]]* [[OBJ:%.*]])
    //   Call the global operator delete.
    // CHECK-NEXT: call void @_ZdlPv(i8* [[ALLOCATED]]) nounwind
    ::delete xp;
  }
}

namespace test5 {
  struct Incomplete;
  // CHECK: define void @_ZN5test523array_delete_incompleteEPNS_10IncompleteES1_
  void array_delete_incomplete(Incomplete *p1, Incomplete *p2) {
    // CHECK: call void @_ZdlPv
    delete p1;
    // CHECK: call void @_ZdaPv
    delete [] p2;
  }
}
