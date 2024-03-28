// RUN: %clang_cc1 -triple x86_64-apple-darwin10 %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NOSIZE
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 %s -emit-llvm -o - -Oz -disable-llvm-passes | FileCheck %s --check-prefixes=CHECK,CHECK-SIZE

void t1(int *a) {
  delete a;
}

struct S {
  int a;
};

// POD types.

// CHECK-LABEL: define{{.*}} void @_Z2t3P1S
void t3(S *s) {
  // CHECK: icmp {{.*}} null
  // CHECK: br i1

  // CHECK: call void @_ZdlPvm

  // Check the delete is inside the 'if !null' check unless we're optimizing
  // for size. FIXME: We could omit the branch entirely in this case.
  // CHECK-NOSIZE-NEXT: br
  // CHECK-SIZE-NEXT: ret
  delete s;
}

// Non-POD
struct T {
  ~T();
  int a;
};

// CHECK-LABEL: define{{.*}} void @_Z2t4P1T
void t4(T *t) {
  // CHECK: call void @_ZN1TD1Ev
  // CHECK-SIZE-NEXT: br
  // CHECK: call void @_ZdlPvm
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

  // CHECK-LABEL: define{{.*}} void @_ZN5test04testEPNS_1AE(
  void test(A *a) {
    // CHECK: call void @_ZN5test01AD1Ev
    // CHECK-SIZE-NEXT: br
    // CHECK: call void @_ZN5test01AdlEPv
    delete a;
  }

  // CHECK-LABEL: define linkonce_odr void @_ZN5test01AD1Ev(ptr {{[^,]*}} %this) unnamed_addr
  // CHECK-LABEL: define linkonce_odr void @_ZN5test01AdlEPv
}

namespace test1 {
  struct A {
    int x;
    ~A();
  };

  // CHECK-LABEL: define{{.*}} void @_ZN5test14testEPA10_A20_NS_1AE(
  void test(A (*arr)[10][20]) {
    delete [] arr;
    // CHECK:      icmp eq ptr [[PTR:%.*]], null
    // CHECK-NEXT: br i1

    // CHECK:      [[BEGIN:%.*]] = getelementptr inbounds [10 x [20 x [[A:%.*]]]], ptr [[PTR]], i32 0, i32 0, i32 0
    // CHECK-NEXT: [[ALLOC:%.*]] = getelementptr inbounds i8, ptr [[BEGIN]], i64 -8
    // CHECK-NEXT: [[COUNT:%.*]] = load i64, ptr [[ALLOC]]
    // CHECK:      [[END:%.*]] = getelementptr inbounds [[A]], ptr [[BEGIN]], i64 [[COUNT]]
    // CHECK-NEXT: [[ISEMPTY:%.*]] = icmp eq ptr [[BEGIN]], [[END]]
    // CHECK-NEXT: br i1 [[ISEMPTY]],
    // CHECK:      [[PAST:%.*]] = phi ptr [ [[END]], {{%.*}} ], [ [[CUR:%.*]], {{%.*}} ]
    // CHECK-NEXT: [[CUR:%.*]] = getelementptr inbounds [[A]], ptr [[PAST]], i64 -1
    // CHECK-NEXT: call void @_ZN5test11AD1Ev(ptr {{[^,]*}} [[CUR]])
    // CHECK-NEXT: [[ISDONE:%.*]] = icmp eq ptr [[CUR]], [[BEGIN]]
    // CHECK-NEXT: br i1 [[ISDONE]]
    // CHECK:      [[MUL:%.*]] = mul i64 4, [[COUNT]]
    // CHECK-NEXT: [[SIZE:%.*]] = add i64 [[MUL]], 8
    // CHECK-NEXT: call void @_ZdaPvm(ptr noundef [[ALLOC]], i64 noundef [[SIZE]])
  }
}

namespace test2 {
  // CHECK-LABEL: define{{.*}} void @_ZN5test21fEPb
  void f(bool *b) {
    // CHECK: call void @_ZdlPvm(ptr{{.*}}i64
    delete b;
    // CHECK: call void @_ZdaPv(ptr
    delete [] b;
  }
}

namespace test3 {
  void f(int a[10][20]) {
    // CHECK: call void @_ZdaPv(ptr
    delete a;
  }
}

namespace test4 {
  // PR10341: ::delete with a virtual destructor
  struct X {
    virtual ~X();
    void operator delete (void *);
  };

  // CHECK-LABEL: define{{.*}} void @_ZN5test421global_delete_virtualEPNS_1XE
  void global_delete_virtual(X *xp) {
    //   Load the offset-to-top from the vtable and apply it.
    //   This has to be done first because the dtor can mess it up.
    // CHECK: [[XP:%.*]] = load ptr, ptr [[XP_ADDR:%.*]]
    // CHECK: [[VTABLE:%.*]] = load ptr, ptr [[XP]]
    // CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds i64, ptr [[VTABLE]], i64 -2
    // CHECK-NEXT: [[OFFSET:%.*]] = load i64, ptr [[T0]], align 8
    // CHECK-NEXT: [[ALLOCATED:%.*]] = getelementptr inbounds i8, ptr [[XP]], i64 [[OFFSET]]
    //   Load the complete-object destructor (not the deleting destructor)
    //   and call noundef it.
    // CHECK-NEXT: [[VTABLE:%.*]] = load ptr, ptr [[XP:%.*]]
    // CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds ptr, ptr [[VTABLE]], i64 0
    // CHECK-NEXT: [[DTOR:%.*]] = load ptr, ptr [[T0]]
    // CHECK-NEXT: call void [[DTOR]](ptr {{[^,]*}} [[OBJ:%.*]])
    //   Call the global operator delete.
    // CHECK-NEXT: call void @_ZdlPvm(ptr noundef [[ALLOCATED]], i64 noundef 8) [[NUW:#[0-9]+]]
    ::delete xp;
  }
}

namespace test5 {
  struct Incomplete;
  // CHECK-LABEL: define{{.*}} void @_ZN5test523array_delete_incompleteEPNS_10IncompleteES1_
  void array_delete_incomplete(Incomplete *p1, Incomplete *p2) {
    // CHECK: call void @_ZdlPv
    delete p1;
    // CHECK: call void @_ZdaPv
    delete [] p2;
  }
}

// CHECK: attributes [[NUW]] = {{[{].*}} nounwind {{.*[}]}}
