// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s


struct Spacer { int x; };
struct A { double array[2]; };
struct B : Spacer, A { };

B &getB();

// CHECK-LABEL: define{{.*}} nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_Z4getAv()
// CHECK: call noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_Z4getBv()
// CHECK-NEXT: getelementptr inbounds i8, ptr
// CHECK-NEXT: ret ptr
A &&getA() { return static_cast<A&&>(getB()); }

int &getIntLValue();
int &&getIntXValue();
int getIntPRValue();

// CHECK-LABEL: define{{.*}} nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_Z2f0v()
// CHECK: call noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_Z12getIntLValuev()
// CHECK-NEXT: ret ptr
int &&f0() { return static_cast<int&&>(getIntLValue()); }

// CHECK-LABEL: define{{.*}} nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_Z2f1v()
// CHECK: call noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_Z12getIntXValuev()
// CHECK-NEXT: ret ptr
int &&f1() { return static_cast<int&&>(getIntXValue()); }

// CHECK-LABEL: define{{.*}} nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_Z2f2v
// CHECK: call noundef i32 @_Z13getIntPRValuev()
// CHECK-NEXT: store i32 {{.*}}, ptr
// CHECK-NEXT: ret ptr
int &&f2() { return static_cast<int&&>(getIntPRValue()); }

bool ok;

class C
{
   int* state_;

   C(const C&) = delete;
   C& operator=(const C&) = delete;
public:
  C(int state) : state_(new int(state)) { }
  
  C(C&& a) {
    state_ = a.state_; 
    a.state_ = 0;
  }

  ~C() {
    delete state_; 
    state_ = 0;
  }
};

C test();

// CHECK-LABEL: define{{.*}} void @_Z15elide_copy_initv
void elide_copy_init() {
  ok = false;
  // CHECK: call void @_Z4testv
  C a = test();
  // CHECK-NEXT: call void @_ZN1CD1Ev
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define{{.*}} void @_Z16test_move_returnv
C test_move_return() {
  // CHECK: call void @_ZN1CC1Ei
  C a1(3);
  // CHECK: call void @_ZN1CC1Ei
  C a2(4);
  if (ok)
    // CHECK: call void @_ZN1CC1EOS_
    return a1;
  // CHECK: call void @_ZN1CC1EOS_
  return a2;
  // CHECK: call void @_ZN1CD1Ev
  // CHECK: call void @_ZN1CD1Ev
  //CHECK:  ret void
}

// PR10800: don't crash
namespace test1 {
  int &&move(int&);

  struct A { A(int); };
  struct B {
    A a;
    B(int i);
  };

  // CHECK-LABEL:    define{{.*}} void @_ZN5test11BC2Ei(
  // CHECK:      [[T0:%.*]] = call noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZN5test14moveERi(
  // CHECK-NEXT: [[T1:%.*]] = load i32, ptr [[T0]]
  // CHECK-NEXT: call void @_ZN5test11AC1Ei({{.*}}, i32 noundef [[T1]])
  // CHECK-NEXT: ret void
  B::B(int i) : a(move(i)) {}
}

// PR11009
struct MoveConvertible {
  operator int&& () const;
};
void moveConstruct() {
  (void)(int)MoveConvertible();
}
